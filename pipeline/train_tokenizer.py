from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = ["<doc>", "<pad>", "<bos>", "<eos>"]


def iter_text_documents(corpus_dirs: List[Path]) -> Iterable[str]:
    for corpus_dir in corpus_dirs:
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
        for path in sorted(corpus_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                yield text


def build_tokenizer(vocab_size: int) -> Tuple[Tokenizer, trainers.BpeTrainer]:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special = [
        AddedToken(token, special=True, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for token in SPECIAL_TOKENS
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    return tokenizer, trainer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on corpus_general + corpus_astronomy.")
    parser.add_argument(
        "--corpus-dirs",
        nargs="+",
        default=["data/corpus_general_training", "data/corpus_astronomy_training"],
        help="One or more directories containing .txt documents.",
    )
    parser.add_argument("--vocab-size", type=int, default=32000, help="BPE vocabulary size.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/nanogpt/tokenizer/tokenizer.json"),
        help="Tokenizer output path.",
    )
    args = parser.parse_args()

    corpus_dirs = [Path(p) for p in args.corpus_dirs]
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, trainer = build_tokenizer(vocab_size=args.vocab_size)
    print("Training tokenizer...")
    tokenizer.train_from_iterator(iter_text_documents(corpus_dirs), trainer=trainer)

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"Special token was not added to vocabulary: {token}")

    tokenizer.save(str(out_path))
    print(f"Saved tokenizer: {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    for token in SPECIAL_TOKENS:
        print(f"{token}: {tokenizer.token_to_id(token)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
