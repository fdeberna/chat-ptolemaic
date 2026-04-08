from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader

from config_utils import apply_overrides, load_json_config, resolve_path
from dataset import create_dataset_bundle
from model import GPT, GPTConfig
from run_utils import RunLogger, make_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain GPT with JSON configuration.")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain_config.json"))
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (supports dotted keys, e.g. model.n_layer=24).",
    )
    return parser.parse_args()


def as_path_list(value: object, *, field_name: str) -> List[Path]:
    if isinstance(value, str):
        return [Path(value)]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return [Path(item) for item in value]
    raise ValueError(f"{field_name} must be a string path or list of string paths.")


def cycle(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


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
def estimate_loss(
    model: GPT,
    iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    *,
    eval_iters: int,
    device: torch.device,
    use_autocast: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    losses: List[float] = []
    for _ in range(eval_iters):
        xb, yb = next(iterator)
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
    train_config = load_json_config(config_path)
    apply_overrides(train_config, args.set)
    if args.model_config is not None:
        train_config["model_config_path"] = str(args.model_config)

    model_config_path = resolve_path(
        train_config.get("model_config_path", "configs/model_config.json"),
        base_dir=Path.cwd(),
    )
    model_config_data = load_json_config(model_config_path)

    model_overrides = train_config.pop("model", {})
    if model_overrides:
        if not isinstance(model_overrides, dict):
            raise ValueError("model overrides must be an object (e.g. --set model.n_layer=24).")
        model_config_data.update(model_overrides)

    seed = int(train_config.get("seed", 1337))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requested_device = str(train_config.get("device", "cuda"))
    device = resolve_device(requested_device)
    amp_dtype, use_autocast = resolve_dtype(str(train_config.get("dtype", "float32")), device)
    use_grad_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    tokenizer_path = resolve_path(train_config["tokenizer_path"], base_dir=Path.cwd())
    cache_dir = resolve_path(train_config.get("cache_dir", "data/nanogpt/streams"), base_dir=Path.cwd())
    context_length = int(model_config_data["block_size"])
    include_remainder = bool(train_config.get("include_remainder", False))
    force_rebuild = bool(train_config.get("force_rebuild_streams", False))

    train_dirs = [resolve_path(path, base_dir=Path.cwd()) for path in as_path_list(train_config["train_data_path"], field_name="train_data_path")]
    configured_val = train_config.get("val_data_path")
    if configured_val is not None:
        configured_val_dirs = [resolve_path(path, base_dir=Path.cwd()) for path in as_path_list(configured_val, field_name="val_data_path")]
        if [str(path) for path in configured_val_dirs] != [str(path) for path in train_dirs]:
            raise ValueError(
                "Pretrain now uses per-document train/validation splitting from train_data_path. "
                "Set val_data_path equal to train_data_path or remove val_data_path."
            )
    # Pretrain uses per-document 90/10 split inside create_dataset_bundle.
    bundle = create_dataset_bundle(
        tokenizer_path=tokenizer_path,
        corpus_dirs=train_dirs,
        cache_name=str(train_config.get("cache_name", "general_pretrain")),
        context_length=context_length,
        cache_dir=cache_dir,
        seed=seed,
        include_remainder=include_remainder,
        force_rebuild=force_rebuild,
    )

    vocab_size = int(model_config_data.get("vocab_size", bundle.vocab_size))
    if vocab_size < bundle.vocab_size:
        raise ValueError(
            f"Configured vocab_size ({vocab_size}) is smaller than tokenizer vocab ({bundle.vocab_size})."
        )

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
        pad_token_id=bundle.pad_token_id,
    )

    batch_size = int(train_config["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    eval_batch_size = int(train_config.get("eval_batch_size", batch_size))
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be >= 1.")
    num_workers = int(train_config.get("num_workers", 0))

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("Train/val loader is empty. Lower block_size or batch_size, or add more data.")

    raw_model = GPT(model_config).to(device)
    print(f"Model parameters (non-embedding): {raw_model.get_num_params(non_embedding=True):,}")

    learning_rate = float(train_config["learning_rate"])
    min_lr = float(train_config["min_lr"])
    warmup_iters = int(train_config["warmup_iters"])
    lr_decay_iters = int(train_config["lr_decay_iters"])
    weight_decay = float(train_config["weight_decay"])
    beta1 = float(train_config["beta1"])
    beta2 = float(train_config["beta2"])
    grad_clip = float(train_config["grad_clip"])
    gradient_accumulation_steps = int(train_config["gradient_accumulation_steps"])
    max_iters = int(train_config["max_iters"])
    lr_schedule = str(train_config.get("lr_schedule", "cosine")).lower()
    adam_eps = float(train_config.get("adam_eps", 1e-8))
    eval_interval = int(train_config["eval_interval"])
    eval_iters = int(train_config["eval_iters"])
    log_interval = int(train_config["log_interval"])
    checkpoint_interval = int(train_config.get("checkpoint_interval", eval_interval))

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
    resume_checkpoint_path = None
    if args.resume_checkpoint is not None:
        resume_checkpoint_path = resolve_path(args.resume_checkpoint, base_dir=Path.cwd())
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
        if state_dict is None:
            raise RuntimeError("Resume checkpoint does not contain model weights.")
        raw_model.load_state_dict(state_dict, strict=True)
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scaler_state = checkpoint.get("scaler_state")
        if scaler_state is not None and use_grad_scaler:
            scaler.load_state_dict(scaler_state)
        loaded_step = int(checkpoint.get("step", checkpoint.get("iter_num", -1)))
        start_step = loaded_step + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        tokens_processed = int(checkpoint.get("tokens_processed", 0))
        print(f"Resuming from {resume_checkpoint_path} at step {start_step}.")

    compile_model = bool(train_config.get("compile", False))
    model: GPT | torch.nn.Module
    if compile_model:
        if hasattr(torch, "compile"):
            model = torch.compile(raw_model)
        else:
            print("compile=true ignored: torch.compile is unavailable in this runtime.")
            model = raw_model
    else:
        model = raw_model

    experiment_name = args.experiment_name or train_config.get("experiment_name")
    runs_root = resolve_path(train_config.get("runs_dir", "runs"), base_dir=Path.cwd())
    run_dir = make_run_dir(runs_root=runs_root, experiment_name=experiment_name)
    logger = RunLogger(run_dir)
    print(f"Run directory: {run_dir}")

    run_config = {
        "script": "pipeline/train_pretrain.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        "experiment_name": experiment_name,
        "config_path": str(config_path),
        "model_config_path": str(model_config_path),
        "cli_overrides": args.set,
        "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        "model_config": asdict(model_config),
        "training_config": train_config,
        "dataset": {
            "corpus_dirs": [str(path) for path in train_dirs],
            "split_strategy": "per_document",
            "train_fraction": 0.9,
            "val_fraction": 0.1,
            "tokenizer_path": str(tokenizer_path),
            "cache_dir": str(cache_dir),
        },
        "system": {
            "requested_device": requested_device,
            "resolved_device": str(device),
            "dtype": str(train_config.get("dtype", "float32")),
            "compile": compile_model,
        },
    }
    logger.save_config(run_config)

    sample_prompt = str(train_config.get("sample_prompt", "Explain why the planets move backwards in the sky"))
    sample_max_new_tokens = int(train_config.get("sample_max_new_tokens", 120))
    sample_temperature = float(train_config.get("sample_temperature", 0.8))
    sample_top_k_raw = train_config.get("sample_top_k", 50)
    sample_top_k = int(sample_top_k_raw) if sample_top_k_raw is not None else None
    if sample_top_k is not None and sample_top_k <= 0:
        sample_top_k = None

    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)
    steps_per_epoch = max(1, len(train_loader))
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
            xb, yb = next(train_iter)
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
            val_loss = estimate_loss(
                model=raw_model,
                iterator=val_iter,
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
                    tokenizer=bundle.tokenizer,
                    prompt=sample_prompt,
                    device=device,
                    max_new_tokens=sample_max_new_tokens,
                    temperature=sample_temperature,
                    top_k=sample_top_k,
                    eos_token_id=bundle.eos_token_id,
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
