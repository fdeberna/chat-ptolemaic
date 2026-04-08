from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import torch
from tokenizers import Tokenizer

from nanogpt.model import GPT, GPTConfig


def load_meta(meta_path: Path):
    return json.loads(meta_path.read_text(encoding="utf-8"))


def apply_repetition_penalty(logits, generated_tokens, repetition_penalty):
    if repetition_penalty <= 1.0:
        return logits

    for batch_idx in range(logits.size(0)):
        unique_tokens = torch.unique(generated_tokens[batch_idx])
        token_logits = logits[batch_idx, unique_tokens]
        token_logits = torch.where(
            token_logits > 0,
            token_logits / repetition_penalty,
            token_logits * repetition_penalty,
        )
        logits[batch_idx, unique_tokens] = token_logits
    return logits


def get_banned_tokens_ngram(tokens, ngram_size):
    if ngram_size <= 0:
        return set()
    if len(tokens) + 1 < ngram_size:
        return set()
    if ngram_size == 1:
        return set(tokens)

    ngram_prefixes = {}
    for i in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[i : i + ngram_size])
        prefix = ngram[:-1]
        ngram_prefixes.setdefault(prefix, set()).add(ngram[-1])

    current_prefix = tuple(tokens[-(ngram_size - 1) :])
    return ngram_prefixes.get(current_prefix, set())


def top_p_filtering(logits, top_p):
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("inf")
    return logits


def has_repeated_ngram(tokens, ngram_size):
    if len(tokens) < ngram_size:
        return False
    last_ngram = tuple(tokens[-ngram_size:])
    for i in range(len(tokens) - ngram_size):
        if tuple(tokens[i : i + ngram_size]) == last_ngram:
            return True
    return False


def sample(
    model,
    idx,
    max_new_tokens,
    temperature,
    top_k=None,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    top_p=1.0,
):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        logits = apply_repetition_penalty(logits, idx, repetition_penalty)

        if no_repeat_ngram_size > 0:
            for batch_idx in range(idx.size(0)):
                banned_tokens = get_banned_tokens_ngram(idx[batch_idx].tolist(), no_repeat_ngram_size)
                if banned_tokens:
                    logits[batch_idx, list(banned_tokens)] = -float("inf")

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        logits = top_p_filtering(logits, top_p)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        if has_repeated_ngram(idx[0].tolist(), 3):
            break
    return idx


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained nanoGPT checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint ckpt.pt")
    parser.add_argument("--meta", type=Path, required=True, help="Path to meta.json")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    meta = load_meta(args.meta)
    tokenizer_type = meta.get("tokenizer", "char")
    vocab_size = meta["vocab_size"]
    dtype = meta.get("dtype", "uint16")

    # Suppress the FutureWarning about weights_only default flip; we need the full
    # dict (config + state dict) saved during training.
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
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

    if tokenizer_type == "char":
        stoi = meta["stoi"]
        itos = {int(k): v for k, v in meta["itos"].items()}
        if any(ch not in stoi for ch in prompt):
            missing = {ch for ch in prompt if ch not in stoi}
            raise ValueError(f"Prompt contains characters not in vocabulary: {missing}")
        idx = torch.tensor([stoi[ch] for ch in prompt], dtype=torch.long).unsqueeze(0).to(device)
        out_idx = sample(
            model,
            idx,
            args.max_new_tokens,
            args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_p=args.top_p,
        )
        out_text = "".join([itos[int(i)] for i in out_idx[0].tolist()])
    else:
        tok_path = args.meta.parent / meta["tokenizer_file"]
        tokenizer = Tokenizer.from_file(str(tok_path))
        idx_ids = tokenizer.encode(prompt).ids
        idx = torch.tensor(idx_ids, dtype=torch.long).unsqueeze(0).to(device)
        out_idx = sample(
            model,
            idx,
            args.max_new_tokens,
            args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_p=args.top_p,
        )
        out_text = tokenizer.decode(out_idx[0].tolist())

    print(out_text)


if __name__ == "__main__":
    main()
