from __future__ import annotations

import argparse
from pathlib import Path

from dataset import create_dataset_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect tokenized dataset blocks.")
    parser.add_argument("--tokenizer", type=Path, default=Path("data/nanogpt/tokenizer/tokenizer.json"))
    parser.add_argument(
        "--mode",
        choices=["all", "general", "astronomy"],
        default="all",
        help="Which corpus split to inspect.",
    )
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/nanogpt/streams"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--force-rebuild-streams", action="store_true")
    args = parser.parse_args()

    if args.mode == "all":
        corpus_dirs = [Path("data/corpus_general_training"), Path("data/corpus_astronomy_training")]
        cache_name = "inspect_all"
    elif args.mode == "general":
        corpus_dirs = [Path("data/corpus_general_training")]
        cache_name = "inspect_general"
    else:
        corpus_dirs = [Path("data/corpus_astronomy_training")]
        cache_name = "inspect_astronomy"

    bundle = create_dataset_bundle(
        tokenizer_path=args.tokenizer,
        corpus_dirs=corpus_dirs,
        cache_name=cache_name,
        context_length=args.context_length,
        cache_dir=args.cache_dir,
        seed=args.seed,
        include_remainder=True,
        force_rebuild=args.force_rebuild_streams,
    )

    print(f"tokenizer vocab: {bundle.vocab_size}")
    print(f"special token ids: doc={bundle.doc_token_id}, pad={bundle.pad_token_id}, bos={bundle.bos_token_id}, eos={bundle.eos_token_id}")
    print(f"documents: {bundle.artifacts.doc_count}")
    print(f"train tokens: {bundle.artifacts.train_tokens}")
    print(f"val tokens: {bundle.artifacts.val_tokens}")
    print(f"train blocks: {len(bundle.train_dataset)}")
    print(f"val blocks: {len(bundle.val_dataset)}")

    index = min(max(args.sample_index, 0), len(bundle.train_dataset) - 1)
    x, y = bundle.train_dataset[index]
    print(f"sample index: {index}")
    print(f"input shape: {tuple(x.shape)}, target shape: {tuple(y.shape)}")
    print("input token ids (first 64):")
    print(x[:64].tolist())

    decoded = bundle.tokenizer.decode(x.tolist(), skip_special_tokens=False)
    print("\ndecoded input text snippet:")
    print(decoded[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
