from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "translated"
OUTPUT_DIR = DATA_DIR / "corpus"


def _find_header_span(lines):
    """
    Look for the ICA header block that starts with 'Provided by The Internet Classics Archive.'
    and ends at the dashed separator line.
    Returns a tuple (start, end_exclusive) or None.
    """
    start = None
    for i, line in enumerate(lines[:20]):  # header appears near the top
        if line.strip().lower().startswith("provided by the internet classics archive"):
            start = i
            break
    if start is None:
        return None

    end = None
    for j in range(start, min(len(lines), start + 30)):
        if re.match(r"^-{5,}$", lines[j].strip()):
            end = j + 1
            break
    if end is None:
        end = start + 1
    return start, end


def _find_footer_span(lines):
    """
    Footer starts at 'THE END' or 'Copyright statement:' and continues to EOF.
    Returns (start, len(lines)) or None.
    """
    start = None
    for i in range(len(lines) - 1, max(-1, len(lines) - 200), -1):
        stripped = lines[i].strip().lower()
        if stripped.startswith("copyright statement"):
            start = i
            break
        if stripped == "the end":
            start = i
            break
    if start is None:
        return None
    return start, len(lines)


def strip_boilerplate(text: str) -> str:
    lines = text.splitlines()

    header_span = _find_header_span(lines)
    if header_span:
        s, e = header_span
        lines = lines[:s] + lines[e:]

    footer_span = _find_footer_span(lines)
    if footer_span:
        s, e = footer_span
        lines = lines[:s]

    cleaned = "\n".join(lines).strip()
    return cleaned + "\n" if cleaned else ""


def process_files(input_dir: Path, output_dir: Path, overwrite: bool, recursive: bool):
    pattern = "**/*.txt" if recursive else "*.txt"
    files: Iterable[Path] = input_dir.glob(pattern)

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(files):
        if not path.is_file():
            continue
        out_path = output_dir / path.name
        if out_path.exists() and not overwrite:
            print(f"Skipping existing: {out_path}")
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = strip_boilerplate(text)

        out_path.write_text(cleaned, encoding="utf-8")
        print(f"Wrote: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove Internet Classics Archive header/footer boilerplate."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory with source .txt files (default: data/translated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write cleaned files (default: data/corpus)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning input-dir",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_files(args.input_dir, args.output_dir, args.overwrite, args.recursive)


if __name__ == "__main__":
    main()
