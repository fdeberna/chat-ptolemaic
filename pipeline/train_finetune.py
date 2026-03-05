from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import create_dataset_bundle
from model import GPT, GPTConfig


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


def evaluate_mixed(
    model: GPT,
    astro_loader: DataLoader,
    general_loader: DataLoader,
    steps: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    losses = []
    astro_iter = cycle(astro_loader)
    general_iter = cycle(general_loader)
    with torch.no_grad():
        for _ in range(steps):
            xb, yb = mixed_batch(next(astro_iter), next(general_iter))
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def append_log(log_path: Path, row: Dict[str, object]) -> None:
    exists = log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "train_loss", "val_loss", "lr"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Finetune GPT on astronomy with 80/20 astro/general mixing.")
    parser.add_argument("--checkpoint-in", type=Path, default=Path("checkpoints/pretrain.pt"))
    parser.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/astro_model.pt"))
    parser.add_argument("--tokenizer", type=Path, default=None, help="Optional override for tokenizer path.")
    parser.add_argument("--astro-dir", type=Path, default=Path("data/corpus_astronomy"))
    parser.add_argument("--general-dir", type=Path, default=Path("data/corpus_general"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/nanogpt/streams"))
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--astro-ratio", type=float, default=0.8)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--force-rebuild-streams", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 2:
        raise ValueError("batch-size must be >= 2 for 80/20 astro/general mixing.")

    if not args.checkpoint_in.exists():
        raise FileNotFoundError(f"Missing pretrain checkpoint: {args.checkpoint_in}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    checkpoint = torch.load(args.checkpoint_in, map_location="cpu")
    tokenizer_path = args.tokenizer or Path(checkpoint["tokenizer_path"])
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    astro_bundle = create_dataset_bundle(
        tokenizer_path=tokenizer_path,
        corpus_dirs=[args.astro_dir],
        cache_name="astro_finetune",
        context_length=args.context_length,
        cache_dir=args.cache_dir,
        seed=args.seed,
        include_remainder=False,
        force_rebuild=args.force_rebuild_streams,
    )
    general_bundle = create_dataset_bundle(
        tokenizer_path=tokenizer_path,
        corpus_dirs=[args.general_dir],
        cache_name="general_finetune",
        context_length=args.context_length,
        cache_dir=args.cache_dir,
        seed=args.seed,
        include_remainder=False,
        force_rebuild=args.force_rebuild_streams,
    )

    astro_bs = int(round(args.batch_size * args.astro_ratio))
    astro_bs = max(1, min(args.batch_size - 1, astro_bs))
    general_bs = args.batch_size - astro_bs
    print(f"Finetune batch mix: astro={astro_bs}, general={general_bs}")

    astro_train_loader = _make_loader(astro_bundle.train_dataset, astro_bs, True, args.num_workers, device)
    astro_val_loader = _make_loader(astro_bundle.val_dataset, astro_bs, False, args.num_workers, device)
    general_train_loader = _make_loader(general_bundle.train_dataset, general_bs, True, args.num_workers, device)
    general_val_loader = _make_loader(general_bundle.val_dataset, general_bs, False, args.num_workers, device)

    if len(astro_train_loader) == 0 or len(general_train_loader) == 0:
        raise RuntimeError("One of the finetune loaders is empty. Lower batch size or verify corpus size.")

    if "model_config" in checkpoint:
        model_config = GPTConfig(**checkpoint["model_config"])
    else:
        model_config = GPTConfig(
            vocab_size=astro_bundle.vocab_size,
            block_size=args.context_length,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.1,
            pad_token_id=astro_bundle.pad_token_id,
        )
    model_config.pad_token_id = astro_bundle.pad_token_id
    model = GPT(model_config).to(device)
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain model weights.")
    model.load_state_dict(state_dict, strict=True)

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = args.steps_per_epoch or len(astro_train_loader)
    val_steps = max(1, args.val_steps)

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.checkpoint_out.parent / "finetune_log.csv"

    astro_train_iter = cycle(astro_train_loader)
    general_train_iter = cycle(general_train_loader)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for _ in range(steps_per_epoch):
            xb, yb = mixed_batch(next(astro_train_iter), next(general_train_iter))
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"step {global_step} | train_loss {loss.item():.4f} | lr {lr:.6e}")
                append_log(
                    log_path,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": f"{loss.item():.6f}",
                        "val_loss": "nan",
                        "lr": f"{lr:.8e}",
                    },
                )

        val_loss = evaluate_mixed(
            model=model,
            astro_loader=astro_val_loader,
            general_loader=general_val_loader,
            steps=val_steps,
            device=device,
            use_amp=use_amp,
        )
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch} | step {global_step} | val_loss {val_loss:.4f} | lr {lr:.6e}")
        append_log(
            log_path,
            {
                "step": global_step,
                "epoch": epoch,
                "train_loss": "nan",
                "val_loss": f"{val_loss:.6f}",
                "lr": f"{lr:.8e}",
            },
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": asdict(model_config),
                "tokenizer_path": str(tokenizer_path),
                "pad_token_id": astro_bundle.pad_token_id,
                "global_step": global_step,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            args.checkpoint_out,
        )
        print(f"Saved checkpoint: {args.checkpoint_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
