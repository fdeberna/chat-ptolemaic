from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


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


def save_meta(meta_path: Path, stoi: Dict[str, int]) -> None:
    itos = {i: ch for ch, i in stoi.items()}
    meta = {"vocab_size": len(stoi), "stoi": stoi, "itos": itos}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare character-level dataset for nanoGPT."
    )
    parser.add_argument("--input", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=Path("data/nanogpt/pre_copernican"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--write-text",
        action="store_true",
        help="Also write concatenated plain text corpus.txt to the output directory.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Reading corpus from {args.input} ...")
    text = read_corpus(args.input)
    print(f"Total characters: {len(text):,}")

    if args.write_text:
        (args.out / "corpus.txt").write_text(text, encoding="utf-8")
        print(f"Wrote concatenated text to {args.out / 'corpus.txt'}")

    print("Building vocabulary ...")
    stoi = build_vocab(text)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    print("Encoding data ...")
    data = encode(text, stoi)

    split_idx = int(len(data) * (1 - args.val_fraction))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    np.memmap(args.out / "train.bin", dtype=np.uint16, mode="w+", shape=train_data.shape)[:] = train_data
    np.memmap(args.out / "val.bin", dtype=np.uint16, mode="w+", shape=val_data.shape)[:] = val_data
    save_meta(args.out / "meta.json", stoi)

    print(f"Wrote train.bin ({train_data.shape[0]:,}) and val.bin ({val_data.shape[0]:,}) to {args.out}")
    print(f"Meta saved to {args.out / 'meta.json'}")


if __name__ == "__main__":
    main()
