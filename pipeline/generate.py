from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a GPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text (without <bos>).")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Optional tokenizer override.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    tokenizer_path = args.tokenizer or Path(checkpoint["tokenizer_path"])
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    if bos_token_id is None:
        raise RuntimeError("Tokenizer is missing <bos> token.")

    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config)
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain model weights.")
    model.load_state_dict(state_dict, strict=True)

    if args.device:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    prompt_text = f"<bos> {args.prompt.strip()}"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    if not prompt_ids:
        raise RuntimeError("Prompt encoded to an empty token sequence.")

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=eos_token_id,
        )

    token_ids = y[0].tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
