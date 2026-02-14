from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

DATA_DIR = Path("data")
TRANSLATED_DIR = DATA_DIR / "translated"
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_PATH = CORPUS_DIR / "corpus.txt"

DEFAULT_DROP_PATTERNS = [
    r"^\s*page\s+\d+\s*$",
    r"^\s*\d+\s*$",
]


def strip_boilerplate(text: str, drop_patterns: Iterable[str]) -> str:
    patterns = [re.compile(p, re.IGNORECASE) for p in drop_patterns]
    lines = []
    for line in text.splitlines():
        if any(p.match(line) for p in patterns):
            continue
        lines.append(line.rstrip())
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def build_corpus(
    input_dir: Path, drop_patterns: Iterable[str]
) -> List[str]:
    documents: List[str] = []
    for path in sorted(input_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = strip_boilerplate(text, drop_patterns)
        if cleaned:
            documents.append(cleaned)
    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge translated texts into a single corpus."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TRANSLATED_DIR,
        help="Directory with translated text files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CORPUS_PATH,
        help="Output corpus path.",
    )
    parser.add_argument(
        "--drop-pattern",
        action="append",
        default=[],
        help="Regex pattern to drop boilerplate lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drop_patterns = DEFAULT_DROP_PATTERNS + args.drop_pattern

    if not args.input_dir.exists():
        print(
            f"Missing {args.input_dir}. Run translate.py first."
        )
        return

    documents = build_corpus(args.input_dir, drop_patterns)
    if not documents:
        print(f"No documents found in {args.input_dir}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus = "\n\n".join(documents).strip() + "\n"
    args.output.write_text(corpus, encoding="utf-8")
    print(f"Wrote corpus: {args.output}")


if __name__ == "__main__":
    main()
