from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def read_corpus(input_dir: Path) -> str:
    texts = []
    for path in sorted(input_dir.glob("*.txt")):
        texts.append(path.read_text(encoding="utf-8", errors="ignore").strip())
    return "\n\n".join(texts)


def build_vocab(text: str) -> Dict[str, int]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return stoi


def encode(text: str, stoi: Dict[str, int]) -> np.ndarray:
    data = np.array([stoi[ch] for ch in text], dtype=np.uint16)
    return data


def save_meta_char(meta_path: Path, stoi: Dict[str, int], dtype: str) -> None:
    itos = {i: ch for ch, i in stoi.items()}
    meta = {
        "tokenizer": "char",
        "vocab_size": len(stoi),
        "dtype": dtype,
        "stoi": stoi,
        "itos": itos,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def train_bpe_tokenizer(text: str, vocab_size: int, min_frequency: int, special_tokens) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer


def save_meta_bpe(meta_path: Path, tokenizer_path: Path, vocab_size: int, dtype: str, special_tokens) -> None:
    meta = {
        "tokenizer": "byte_level_bpe",
        "tokenizer_file": tokenizer_path.name,
        "vocab_size": vocab_size,
        "dtype": dtype,
        "special_tokens": special_tokens,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for nanoGPT (char-level or byte-level BPE)."
    )
    parser.add_argument("--input", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=Path("data/nanogpt/pre_copernican"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="bpe",
        help="Tokenizer type: char-level (legacy) or byte-level BPE",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="BPE vocabulary size (only used when --tokenizer bpe)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="BPE minimum token frequency (only used when --tokenizer bpe)",
    )
    parser.add_argument(
        "--write-text",
        action="store_true",
        help="Also write concatenated plain text corpus.txt to the output directory.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {args.input}")

    print(f"Found {len(files)} files in {args.input}")

    # Concatenated text is still used for tokenizer training (and optional export)
    print("Reading corpus for tokenizer training ...")
    concat_text = "\n\n".join(
        path.read_text(encoding="utf-8", errors="ignore").strip() for path in files
    )
    print(f"Total characters (concat): {len(concat_text):,}")

    if args.write_text:
        (args.out / "corpus.txt").write_text(concat_text, encoding="utf-8")
        print(f"Wrote concatenated text to {args.out / 'corpus.txt'}")

    if args.tokenizer == "char":
        print("Building character vocabulary ...")
        stoi = build_vocab(concat_text)
        vocab_size = len(stoi)
        print(f"Vocab size: {vocab_size}")
        dtype = "uint16"
    else:
        print("Training byte-level BPE tokenizer ...")
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        tokenizer = train_bpe_tokenizer(
            concat_text,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
        )
        vocab_size = tokenizer.get_vocab_size()
        tokenizer_path = args.out / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        print(f"Trained BPE vocab size: {vocab_size}")
        dtype = "uint32"

    print("Encoding files and splitting per-file ...")
    train_parts = []
    val_parts = []

    for path in files:
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue
        # preserve file boundaries with a blank line separator
        raw_with_sep = raw + "\n\n"

        if args.tokenizer == "char":
            ids = np.array([stoi[ch] for ch in raw_with_sep], dtype=np.uint16)
        else:
            ids = np.array(tokenizer.encode(raw_with_sep).ids, dtype=np.uint32)

        split_idx = int(len(ids) * (1 - args.val_fraction))
        train_parts.append(ids[:split_idx])
        val_parts.append(ids[split_idx:])

    if not train_parts:
        raise SystemExit("No data encoded; aborting.")

    train_data = np.concatenate(train_parts)
    val_data = np.concatenate(val_parts) if val_parts else np.array([], dtype=train_data.dtype)

    np.memmap(args.out / "train.bin", dtype=data.dtype, mode="w+", shape=train_data.shape)[:] = train_data
    np.memmap(args.out / "val.bin", dtype=data.dtype, mode="w+", shape=val_data.shape)[:] = val_data
    if args.tokenizer == "char":
        save_meta_char(args.out / "meta.json", stoi, dtype)
    else:
        save_meta_bpe(args.out / "meta.json", tokenizer_path, vocab_size, dtype, special_tokens)

    print(f"Wrote train.bin ({train_data.shape[0]:,}) and val.bin ({val_data.shape[0]:,}) to {args.out}")
    print(f"Meta saved to {args.out / 'meta.json'}")


if __name__ == "__main__":
    main()
