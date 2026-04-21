from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader

from config_utils import apply_overrides, load_json_config, resolve_path
from dataset import create_dataset_bundle
from model import GPT, GPTConfig
from run_utils import RunLogger, make_run_dir


def make_grad_scaler(*, enabled: bool):
    # torch.cuda.amp.GradScaler is deprecated in recent torch versions.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def load_checkpoint(path: Path) -> dict[str, Any]:
    for extra_kwargs in (
        {"weights_only": True, "mmap": True},
        {"weights_only": True},
        {},
    ):
        try:
            checkpoint = torch.load(path, map_location="cpu", **extra_kwargs)
            if not isinstance(checkpoint, dict):
                raise RuntimeError("Checkpoint payload must be a dictionary.")
            return checkpoint
        except TypeError:
            continue
    raise RuntimeError(f"Failed to load checkpoint: {path}")


def nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def resolve_split_cache_name(
    config: dict[str, Any],
    *,
    shared_key: str,
    fallback_train_key: str,
    fallback_val_key: str,
    default: str,
) -> str:
    explicit = config.get(shared_key)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    for fallback_key in (fallback_train_key, fallback_val_key):
        candidate = config.get(fallback_key)
        if not isinstance(candidate, str):
            continue
        candidate = candidate.strip()
        if not candidate:
            continue
        if candidate.endswith("_train") or candidate.endswith("_val"):
            return candidate.rsplit("_", 1)[0]
        return candidate

    return default


def clamp_batch_size(desired: int, available_blocks: int, *, name: str) -> int:
    if available_blocks < 1:
        raise RuntimeError(
            f"{name} dataset produced zero blocks. Lower block_size, enable include_remainder, or add more data."
        )
    return max(1, min(desired, available_blocks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune GPT with JSON configuration.")
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_config.json"))
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (supports dotted keys, e.g. model.dropout=0.0).",
    )
    return parser.parse_args()


def cycle(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def mixed_batch(
    astro_batch: Tuple[torch.Tensor, torch.Tensor],
    general_batch: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    xa, ya = astro_batch
    xg, yg = general_batch
    x = torch.cat([xa, xg], dim=0)
    y = torch.cat([ya, yg], dim=0)
    perm = torch.randperm(x.size(0))
    return x[perm], y[perm]


def cosine_lr(
    step: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    if warmup_iters > 0 and step < warmup_iters:
        return learning_rate * float(step) / float(warmup_iters)
    if step >= lr_decay_iters:
        return min_lr
    if lr_decay_iters <= warmup_iters:
        return min_lr
    decay_ratio = float(step - warmup_iters) / float(lr_decay_iters - warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def scheduled_lr(
    step: int,
    *,
    lr_schedule: str,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    if lr_schedule == "cosine":
        return cosine_lr(
            step,
            learning_rate=learning_rate,
            min_lr=min_lr,
            warmup_iters=warmup_iters,
            lr_decay_iters=lr_decay_iters,
        )
    raise ValueError(f"Unsupported lr_schedule: {lr_schedule}.")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda but CUDA is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(dtype_name: str, device: torch.device) -> Tuple[torch.dtype, bool]:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}. Use one of {sorted(dtype_map)}.")
    amp_dtype = dtype_map[dtype_name]

    if device.type == "cpu" and amp_dtype == torch.float16:
        raise ValueError("dtype=float16 is not supported on cpu.")
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("dtype=bfloat16 requested but CUDA bf16 is not supported on this GPU.")

    use_autocast = amp_dtype != torch.float32 and device.type in {"cuda", "cpu"}
    return amp_dtype, use_autocast


@torch.no_grad()
def evaluate_mixed(
    model: GPT,
    astro_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    general_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    *,
    eval_iters: int,
    device: torch.device,
    use_autocast: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    losses: List[float] = []
    for _ in range(eval_iters):
        xb, yb = mixed_batch(next(astro_iter), next(general_iter))
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
            _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / len(losses))


@torch.no_grad()
def generate_sample(
    model: GPT,
    *,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eos_token_id: int | None,
) -> str:
    prompt_text = f"<bos> {prompt.strip()}"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    if not prompt_ids:
        raise RuntimeError("Sample prompt encoded to an empty sequence.")

    was_training = model.training
    model.eval()
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )
    if was_training:
        model.train()
    return tokenizer.decode(y[0].tolist(), skip_special_tokens=False)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config, base_dir=Path.cwd())
    finetune_config = load_json_config(config_path)
    apply_overrides(finetune_config, args.set)
    explicit_model_config = args.model_config is not None or "model_config_path" in finetune_config
    if args.model_config is not None:
        finetune_config["model_config_path"] = str(args.model_config)

    model_config_path = resolve_path(
        finetune_config.get("model_config_path", "configs/model_config.json"),
        base_dir=Path.cwd(),
    )
    model_config_data = load_json_config(model_config_path)
    model_overrides = finetune_config.pop("model", {})
    if model_overrides and not isinstance(model_overrides, dict):
        raise ValueError("model overrides must be an object (e.g. --set model.n_layer=24).")

    seed = int(finetune_config.get("seed", 1337))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requested_device = str(finetune_config.get("device", "cuda"))
    device = resolve_device(requested_device)
    amp_dtype, use_autocast = resolve_dtype(str(finetune_config.get("dtype", "float32")), device)
    use_grad_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = make_grad_scaler(enabled=use_grad_scaler)

    resume_checkpoint_path = None
    if args.resume_checkpoint is not None:
        resume_checkpoint_path = resolve_path(args.resume_checkpoint, base_dir=Path.cwd())
        checkpoint_path = resume_checkpoint_path
    else:
        checkpoint_path = resolve_path(finetune_config["pretrained_checkpoint"], base_dir=Path.cwd())

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain model weights.")

    checkpoint_model_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_config, dict):
        checkpoint_model_config = nested_get(checkpoint, "config", "model_config")
    if isinstance(checkpoint_model_config, dict):
        if explicit_model_config:
            merged_model_config = dict(checkpoint_model_config)
            merged_model_config.update(model_config_data)
            model_config_data = merged_model_config
        else:
            model_config_data.update(checkpoint_model_config)
    if model_overrides:
        model_config_data.update(model_overrides)

    tokenizer_path_value = finetune_config.get("tokenizer_path")
    if tokenizer_path_value is None:
        for keys in (
            ("tokenizer_path",),
            ("dataset", "tokenizer_path"),
            ("training_config", "tokenizer_path"),
            ("config", "dataset", "tokenizer_path"),
            ("config", "training_config", "tokenizer_path"),
        ):
            candidate = nested_get(checkpoint, *keys)
            if isinstance(candidate, str) and candidate.strip():
                tokenizer_path_value = candidate
                break
    if tokenizer_path_value is None:
        raise RuntimeError("tokenizer_path missing from finetune config and checkpoint.")
    tokenizer_path = resolve_path(tokenizer_path_value, base_dir=Path.cwd())
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    resume_optimizer_state = None
    resume_scaler_state = None
    resume_loaded_step = -1
    if resume_checkpoint_path is not None:
        resume_optimizer_state = checkpoint.get("optimizer_state")
        resume_scaler_state = checkpoint.get("scaler_state")
        resume_loaded_step = int(checkpoint.get("step", checkpoint.get("iter_num", -1)))

    context_length = int(model_config_data["block_size"])
    cache_dir = resolve_path(finetune_config.get("cache_dir", "data/nanogpt/streams"), base_dir=Path.cwd())
    include_remainder = bool(finetune_config.get("include_remainder", False))
    force_rebuild = bool(finetune_config.get("force_rebuild_streams", False))

    astro_train_dirs = [resolve_path(finetune_config["astronomy_data_path"], base_dir=Path.cwd())]
    general_train_dirs = [resolve_path(finetune_config["general_data_path"], base_dir=Path.cwd())]
    configured_astro_val = finetune_config.get("val_astronomy_data_path")
    if configured_astro_val is not None:
        astro_val_dirs = [resolve_path(configured_astro_val, base_dir=Path.cwd())]
        if [str(path) for path in astro_val_dirs] != [str(path) for path in astro_train_dirs]:
            raise ValueError(
                "Finetune now uses per-document train/validation splitting from astronomy_data_path. "
                "Set val_astronomy_data_path equal to astronomy_data_path or remove val_astronomy_data_path."
            )
    configured_general_val = finetune_config.get("val_general_data_path")
    if configured_general_val is not None:
        general_val_dirs = [resolve_path(configured_general_val, base_dir=Path.cwd())]
        if [str(path) for path in general_val_dirs] != [str(path) for path in general_train_dirs]:
            raise ValueError(
                "Finetune now uses per-document train/validation splitting from general_data_path. "
                "Set val_general_data_path equal to general_data_path or remove val_general_data_path."
            )

    astro_cache_name = resolve_split_cache_name(
        finetune_config,
        shared_key="astro_cache_name",
        fallback_train_key="astro_train_cache_name",
        fallback_val_key="astro_val_cache_name",
        default="astro_finetune",
    )
    general_cache_name = resolve_split_cache_name(
        finetune_config,
        shared_key="general_cache_name",
        fallback_train_key="general_train_cache_name",
        fallback_val_key="general_val_cache_name",
        default="general_finetune",
    )

    astro_bundle = create_dataset_bundle(
        tokenizer_path=tokenizer_path,
        corpus_dirs=astro_train_dirs,
        cache_name=astro_cache_name,
        context_length=context_length,
        cache_dir=cache_dir,
        seed=seed,
        include_remainder=include_remainder,
        force_rebuild=force_rebuild,
    )
    general_bundle = create_dataset_bundle(
        tokenizer_path=tokenizer_path,
        corpus_dirs=general_train_dirs,
        cache_name=general_cache_name,
        context_length=context_length,
        cache_dir=cache_dir,
        seed=seed,
        include_remainder=include_remainder,
        force_rebuild=force_rebuild,
    )

    vocab_size = int(model_config_data.get("vocab_size", astro_bundle.vocab_size))
    max_dataset_vocab = max(
        astro_bundle.vocab_size,
        general_bundle.vocab_size,
    )
    if vocab_size < max_dataset_vocab:
        raise ValueError(f"Configured vocab_size ({vocab_size}) is smaller than tokenizer vocab ({max_dataset_vocab}).")

    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=context_length,
        n_layer=int(model_config_data["n_layer"]),
        n_head=int(model_config_data["n_head"]),
        n_embd=int(model_config_data["n_embd"]),
        dropout=float(model_config_data["dropout"]),
        bias=bool(model_config_data["bias"]),
        weight_tying=bool(model_config_data.get("weight_tying", True)),
        layer_norm_eps=float(model_config_data.get("layer_norm_eps", 1e-5)),
        pad_token_id=astro_bundle.pad_token_id,
    )

    batch_size = int(finetune_config["batch_size"])
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 for astronomy/general mixing.")
    eval_batch_size = int(finetune_config.get("eval_batch_size", batch_size))
    if eval_batch_size < 2:
        raise ValueError("eval_batch_size must be >= 2 for astronomy/general evaluation mixing.")
    astronomy_ratio = float(finetune_config["astronomy_ratio"])
    astro_bs = int(round(batch_size * astronomy_ratio))
    astro_bs = max(1, min(batch_size - 1, astro_bs))
    general_bs = batch_size - astro_bs
    astro_eval_bs = int(round(eval_batch_size * astronomy_ratio))
    astro_eval_bs = max(1, min(eval_batch_size - 1, astro_eval_bs))
    general_eval_bs = eval_batch_size - astro_eval_bs

    num_workers = int(finetune_config.get("num_workers", 0))
    astro_train_blocks = len(astro_bundle.train_dataset)
    general_train_blocks = len(general_bundle.train_dataset)
    astro_val_blocks = len(astro_bundle.val_dataset)
    general_val_blocks = len(general_bundle.val_dataset)

    astro_bs = clamp_batch_size(astro_bs, astro_train_blocks, name="astronomy train")
    general_bs = clamp_batch_size(general_bs, general_train_blocks, name="general train")
    astro_eval_bs = clamp_batch_size(astro_eval_bs, astro_val_blocks, name="astronomy val")
    general_eval_bs = clamp_batch_size(general_eval_bs, general_val_blocks, name="general val")
    print(
        f"Finetune batch mix: train(astronomy={astro_bs}, general={general_bs}) "
        f"| eval(astronomy={astro_eval_bs}, general={general_eval_bs})"
    )

    astro_train_loader = DataLoader(
        astro_bundle.train_dataset,
        batch_size=astro_bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    general_train_loader = DataLoader(
        general_bundle.train_dataset,
        batch_size=general_bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    astro_val_loader = DataLoader(
        astro_bundle.val_dataset,
        batch_size=astro_eval_bs,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    general_val_loader = DataLoader(
        general_bundle.val_dataset,
        batch_size=general_eval_bs,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if min(len(astro_train_loader), len(general_train_loader), len(astro_val_loader), len(general_val_loader)) == 0:
        raise RuntimeError("One or more finetune loaders are empty. Lower block_size/batch_size or add more data.")

    raw_model = GPT(model_config).to(device)
    raw_model.load_state_dict(state_dict, strict=True)
    del state_dict
    print(f"Model parameters (non-embedding): {raw_model.get_num_params(non_embedding=True):,}")

    learning_rate = float(finetune_config["learning_rate"])
    min_lr = float(finetune_config["min_lr"])
    warmup_iters = int(finetune_config["warmup_iters"])
    lr_decay_iters = int(finetune_config["lr_decay_iters"])
    weight_decay = float(finetune_config["weight_decay"])
    beta1 = float(finetune_config["beta1"])
    beta2 = float(finetune_config["beta2"])
    grad_clip = float(finetune_config["grad_clip"])
    gradient_accumulation_steps = int(finetune_config["gradient_accumulation_steps"])
    max_iters = int(finetune_config["max_iters"])
    lr_schedule = str(finetune_config.get("lr_schedule", "cosine")).lower()
    adam_eps = float(finetune_config.get("adam_eps", 1e-8))
    eval_interval = int(finetune_config["eval_interval"])
    eval_iters = int(finetune_config["eval_iters"])
    log_interval = int(finetune_config["log_interval"])
    checkpoint_interval = int(finetune_config.get("checkpoint_interval", eval_interval))
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1.")
    if eval_interval < 1:
        raise ValueError("eval_interval must be >= 1.")
    if eval_iters < 1:
        raise ValueError("eval_iters must be >= 1.")
    if log_interval < 1:
        raise ValueError("log_interval must be >= 1.")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1.")

    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(beta1, beta2),
        eps=adam_eps,
    )

    start_step = 0
    best_val_loss = float("inf")
    tokens_processed = 0
    if resume_checkpoint_path is not None:
        if resume_optimizer_state is not None:
            optimizer.load_state_dict(resume_optimizer_state)
        if resume_scaler_state is not None and use_grad_scaler:
            scaler.load_state_dict(resume_scaler_state)
        start_step = resume_loaded_step + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        tokens_processed = int(checkpoint.get("tokens_processed", 0))
        print(f"Resuming from {resume_checkpoint_path} at step {start_step}.")
    del checkpoint

    compile_model = bool(finetune_config.get("compile", False))
    model: GPT | torch.nn.Module
    if compile_model:
        if hasattr(torch, "compile"):
            model = torch.compile(raw_model)
        else:
            print("compile=true ignored: torch.compile is unavailable in this runtime.")
            model = raw_model
    else:
        model = raw_model

    experiment_name = args.experiment_name or finetune_config.get("experiment_name")
    runs_root = resolve_path(finetune_config.get("runs_dir", "runs"), base_dir=Path.cwd())
    run_dir = make_run_dir(runs_root=runs_root, experiment_name=experiment_name)
    logger = RunLogger(run_dir)
    print(f"Run directory: {run_dir}")

    run_config = {
        "script": "pipeline/train_finetune.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        "experiment_name": experiment_name,
        "config_path": str(config_path),
        "model_config_path": str(model_config_path),
        "cli_overrides": args.set,
        "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        "initial_checkpoint": str(checkpoint_path),
        "model_config": asdict(model_config),
        "training_config": finetune_config,
        "dataset": {
            "astronomy_data_path": [str(path) for path in astro_train_dirs],
            "general_data_path": [str(path) for path in general_train_dirs],
            "split_strategy": "per_document",
            "train_fraction": 0.9,
            "val_fraction": 0.1,
            "tokenizer_path": str(tokenizer_path),
            "cache_dir": str(cache_dir),
            "cache_names": {
                "astronomy": astro_cache_name,
                "general": general_cache_name,
            },
            "astronomy_ratio": astronomy_ratio,
            "mixed_batch_sizes": {
                "train": {"astronomy": astro_bs, "general": general_bs, "total": batch_size},
                "eval": {"astronomy": astro_eval_bs, "general": general_eval_bs, "total": eval_batch_size},
            },
        },
        "system": {
            "requested_device": requested_device,
            "resolved_device": str(device),
            "dtype": str(finetune_config.get("dtype", "float32")),
            "compile": compile_model,
        },
    }
    logger.save_config(run_config)

    sample_prompt = str(finetune_config.get("sample_prompt", "Explain why the planets move backwards in the sky"))
    sample_max_new_tokens = int(finetune_config.get("sample_max_new_tokens", 120))
    sample_temperature = float(finetune_config.get("sample_temperature", 0.8))
    sample_top_k_raw = finetune_config.get("sample_top_k", 50)
    sample_top_k = int(sample_top_k_raw) if sample_top_k_raw is not None else None
    if sample_top_k is not None and sample_top_k <= 0:
        sample_top_k = None

    astro_train_iter = cycle(astro_train_loader)
    general_train_iter = cycle(general_train_loader)
    astro_val_iter = cycle(astro_val_loader)
    general_val_iter = cycle(general_val_loader)
    steps_per_epoch = max(1, min(len(astro_train_loader), len(general_train_loader)))
    tokens_per_step = batch_size * context_length * gradient_accumulation_steps
    start_time = time.perf_counter()
    latest_val_loss: float | None = None

    if start_step >= max_iters:
        print(f"start_step ({start_step}) >= max_iters ({max_iters}); nothing to train.")
        return 0

    for step in range(start_step, max_iters):
        lr = scheduled_lr(
            step,
            lr_schedule=lr_schedule,
            learning_rate=learning_rate,
            min_lr=min_lr,
            warmup_iters=warmup_iters,
            lr_decay_iters=lr_decay_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss_accum = 0.0
        for _ in range(gradient_accumulation_steps):
            xb, yb = mixed_batch(next(astro_train_iter), next(general_train_iter))
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                _, loss = model(xb, yb)
                loss_to_backprop = loss / gradient_accumulation_steps

            train_loss_accum += float(loss.item())
            if use_grad_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

        if grad_clip > 0:
            if use_grad_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

        if use_grad_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        tokens_processed += tokens_per_step
        epoch = (step // steps_per_epoch) + 1
        time_elapsed = time.perf_counter() - start_time
        train_loss = train_loss_accum / gradient_accumulation_steps

        should_eval = (step % eval_interval == 0) or (step == max_iters - 1)
        val_loss: float | None = None
        if should_eval:
            val_loss = evaluate_mixed(
                model=raw_model,
                astro_iter=astro_val_iter,
                general_iter=general_val_iter,
                eval_iters=eval_iters,
                device=device,
                use_autocast=use_autocast,
                amp_dtype=amp_dtype,
            )
            best_val_loss = min(best_val_loss, val_loss)
            latest_val_loss = val_loss
            print(f"step {step} | epoch {epoch} | val_loss {val_loss:.4f} | best {best_val_loss:.4f} | lr {lr:.6e}")

            try:
                generated = generate_sample(
                    model=raw_model,
                    tokenizer=astro_bundle.tokenizer,
                    prompt=sample_prompt,
                    device=device,
                    max_new_tokens=sample_max_new_tokens,
                    temperature=sample_temperature,
                    top_k=sample_top_k,
                    eos_token_id=astro_bundle.eos_token_id,
                )
            except Exception as exc:  # noqa: BLE001
                generated = f"[sample generation failed] {exc}"
            logger.append_sample(step=step, prompt=sample_prompt, generated_text=generated)

            checkpoint_payload = {
                "step": step,
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_grad_scaler else None,
                "tokens_processed": tokens_processed,
                "best_val_loss": best_val_loss,
                "val_loss": latest_val_loss,
                "config": run_config,
            }
            versioned_checkpoint_path = logger.checkpoint_path_for_step(step)
            torch.save(checkpoint_payload, versioned_checkpoint_path)
            torch.save(checkpoint_payload, logger.checkpoint_path)
            print(f"Saved checkpoint: {versioned_checkpoint_path}")
            print(f"Updated latest checkpoint: {logger.checkpoint_path}")

        should_checkpoint = ((step + 1) % checkpoint_interval == 0) or (step == max_iters - 1)
        if should_checkpoint and not should_eval:
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if use_grad_scaler else None,
                    "tokens_processed": tokens_processed,
                    "best_val_loss": best_val_loss,
                    "val_loss": latest_val_loss,
                    "config": run_config,
                },
                logger.checkpoint_path,
            )
            print(f"Saved checkpoint: {logger.checkpoint_path}")

        if (step % log_interval == 0) and not should_eval:
            print(f"step {step} | epoch {epoch} | train_loss {train_loss:.4f} | lr {lr:.6e}")

        logger.log_metrics(
            {
                "step": step,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loss is not None else "",
                "learning_rate": lr,
                "tokens_processed": tokens_processed,
                "time_elapsed": round(time_elapsed, 3),
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
