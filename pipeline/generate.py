from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older torch versions that do not support weights_only.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint payload must be a dictionary.")
    return checkpoint


def _nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _resolve_tokenizer_path(checkpoint: dict[str, Any], tokenizer_override: Path | None) -> Path:
    if tokenizer_override is not None:
        return tokenizer_override

    candidates = (
        ("tokenizer_path",),
        ("dataset", "tokenizer_path"),
        ("training_config", "tokenizer_path"),
        ("config", "dataset", "tokenizer_path"),
        ("config", "training_config", "tokenizer_path"),
    )
    for keys in candidates:
        tokenizer_path_value = _nested_get(checkpoint, *keys)
        if isinstance(tokenizer_path_value, str) and tokenizer_path_value.strip():
            return Path(tokenizer_path_value)

    raise RuntimeError(
        "tokenizer_path missing from checkpoint; pass --tokenizer explicitly."
    )


def _resolve_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, dict):
        return model_config

    nested_model_config = _nested_get(checkpoint, "config", "model_config")
    if isinstance(nested_model_config, dict):
        return nested_model_config

    raise RuntimeError("model_config missing from checkpoint.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a GPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text (without <bos>).")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Optional tokenizer override.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.checkpoint)
    tokenizer_path = _resolve_tokenizer_path(checkpoint, args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    if bos_token_id is None:
        raise RuntimeError("Tokenizer is missing <bos> token.")

    model_config = GPTConfig(**_resolve_model_config(checkpoint))
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
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_token_id,
        )

    token_ids = y[0].tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
