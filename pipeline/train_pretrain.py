from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from dataset import create_dataset_bundle
from model import GPT, GPTConfig


def evaluate(model: GPT, loader: DataLoader, device: torch.device, use_amp: bool, max_batches: Optional[int]) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(xb, yb)
            losses.append(loss.item())
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    model.train()
    if not losses:
        raise RuntimeError("Validation loader produced no batches.")
    return float(sum(losses) / len(losses))


def append_log(log_path: Path, row: Dict[str, object]) -> None:
    exists = log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "train_loss", "val_loss", "lr"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain GPT on data/corpus_general.")
    parser.add_argument("--tokenizer", type=Path, default=Path("data/nanogpt/tokenizer/tokenizer.json"))
    parser.add_argument("--general-dir", type=Path, default=Path("data/corpus_general"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/nanogpt/streams"))
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pretrain.pt"))
    parser.add_argument("--force-rebuild-streams", action="store_true")
    args = parser.parse_args()

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    bundle = create_dataset_bundle(
        tokenizer_path=args.tokenizer,
        corpus_dirs=[args.general_dir],
        cache_name="general_pretrain",
        context_length=args.context_length,
        cache_dir=args.cache_dir,
        seed=args.seed,
        include_remainder=False,
        force_rebuild=args.force_rebuild_streams,
    )

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_config = GPTConfig(
        vocab_size=bundle.vocab_size,
        block_size=args.context_length,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        pad_token_id=bundle.pad_token_id,
    )
    model = GPT(model_config).to(device)
    print(f"Model parameters (non-embedding): {model.get_num_params(non_embedding=True):,}")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.checkpoint.parent / "pretrain_log.csv"

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for xb, yb in train_loader:
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

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            max_batches=args.eval_max_batches,
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
                "tokenizer_path": str(args.tokenizer),
                "pad_token_id": bundle.pad_token_id,
                "global_step": global_step,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            args.checkpoint,
        )
        print(f"Saved checkpoint: {args.checkpoint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
