from __future__ import annotations

import argparse
import importlib
import math
import os
from pathlib import Path

import numpy as np
import torch

from nanogpt.model import GPT, GPTConfig


def set_seed(seed: int = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_meta(data_dir: Path):
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {data_dir}. Run prepare_corpus.py first.")
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta


def get_batch(split, data, block_size, batch_size, device):
    data_split = data[split]
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data_split[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data_split[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def estimate_loss(model, data, eval_iters, block_size, batch_size, device):
    model.eval()
    out = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            losses = []
            for _ in range(eval_iters):
                xb, yb = get_batch(split, data, block_size, batch_size, device)
                logits, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(description="Train nanoGPT on pre-Copernican corpus.")
    parser.add_argument("--config", default="nanogpt.config.pre_copernican_char", help="Config module path")
    parser.add_argument("--data-dir", type=Path, help="Override data dir")
    parser.add_argument("--out-dir", type=Path, help="Override output dir")
    parser.add_argument("--device", help="Override device (cpu/cuda)")
    args = parser.parse_args()

    cfg_mod = importlib.import_module(args.config)
    cfg = cfg_mod.config.copy()
    if args.data_dir:
        cfg["data_dir"] = str(args.data_dir)
    if args.out_dir:
        cfg["out_dir"] = str(args.out_dir)
    if args.device:
        cfg["device"] = args.device

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed()

    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    data = {"train": train_data, "val": val_data}
    meta = load_meta(data_dir)
    vocab_size = meta["vocab_size"]
    block_size = cfg["block_size"]

    # model
    model = GPT(
        GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            dropout=cfg["dropout"],
        )
    ).to(device)

    optimizer = model.configure_optimizers(
        weight_decay=cfg["weight_decay"],
        learning_rate=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )

    max_iters = cfg["max_iters"]
    eval_interval = cfg["eval_interval"]
    eval_iters = cfg["eval_iters"]
    batch_size = cfg["batch_size"]
    grad_clip = cfg["grad_clip"]

    for iter_num in range(max_iters):
        xb, yb = get_batch("train", data, block_size, batch_size, device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, data, eval_iters, block_size, cfg["eval_batch_size"], device)
            print(
                f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            ckpt_path = out_dir / "ckpt.pt"
            torch.save({"model_state_dict": model.state_dict(), "config": cfg, "iter": iter_num}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
