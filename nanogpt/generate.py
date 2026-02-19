from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanogpt.model import GPT, GPTConfig


def load_meta(meta_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    return stoi, itos


def encode(text: str, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(indices, itos):
    return "".join([itos[int(i)] for i in indices])


def sample(model, idx, max_new_tokens, temperature, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained nanoGPT checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint ckpt.pt")
    parser.add_argument("--meta", type=Path, required=True, help="Path to meta.json")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    stoi, itos = load_meta(args.meta)
    vocab_size = len(stoi)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config")
    if cfg is None:
        raise RuntimeError("Checkpoint missing config.")

    model = GPT(
        GPTConfig(
            vocab_size=vocab_size,
            block_size=cfg["block_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            dropout=cfg.get("dropout", 0.0),
        )
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if not args.prompt:
        prompt = ""
    else:
        prompt = args.prompt
    if any(ch not in stoi for ch in prompt):
        missing = {ch for ch in prompt if ch not in stoi}
        raise ValueError(f"Prompt contains characters not in vocabulary: {missing}")

    idx = encode(prompt, stoi).unsqueeze(0).to(device)
    out_idx = sample(model, idx, args.max_new_tokens, args.temperature, args.top_k)
    out_text = decode(out_idx[0].tolist(), itos)
    print(out_text)


if __name__ == "__main__":
    main()
