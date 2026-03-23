from pathlib import Path


RAW_DIR = Path("data/translated")
OUT_DIR = Path("data/translated_cleaned")

HEADER_MARKERS = [
    "provided by the internet classics archive",
]

FOOTER_MARKERS = [
    "copyright statement:",
    "the internet classics archive by daniel c. stevenson",
    "world wide web presentation is copyright",
    "all rights reserved under international",
    "classics@classics.mit.edu",
]


def strip_header(lines: list[str]) -> list[str]:
    """Remove the standard ICA header block if present."""
    lower_lines = [l.lower() for l in lines]
    start_idx = None
    for i, text in enumerate(lower_lines):
        if any(marker in text for marker in HEADER_MARKERS):
            start_idx = i
            break
    if start_idx is None:
        return lines

    # Skip until the next blank line after the marker block.
    for j in range(start_idx, len(lines)):
        if not lines[j].strip():
            return lines[j + 1 :]
    return lines  # fallback if no blank line found


def strip_footer(lines: list[str]) -> list[str]:
    """Remove the trailing copyright block."""
    lower_lines = [l.lower() for l in lines]
    cut_idx = None

    for i, text in enumerate(lower_lines):
        if any(marker in text for marker in FOOTER_MARKERS):
            cut_idx = i
            break

    # Also catch a trailing hyphen divider line near the end.
    if cut_idx is None:
        for i, text in enumerate(lower_lines[-200:], start=len(lower_lines) - 200):
            if text.strip().startswith("-") and len(text.strip()) >= 10:
                cut_idx = i
                break

    if cut_idx is None:
        return lines

    return lines[:cut_idx]


def clean_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    lines = strip_header(lines)
    lines = strip_footer(lines)

    # Trim leading/trailing blank lines.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    cleaned = "\n".join(lines) + "\n"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / path.name
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"Cleaned {path.name} -> {out_path}")


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(f"Missing source directory: {RAW_DIR}")

    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        print("No translated texts found.")
        return

    for file in files:
        clean_file(file)


if __name__ == "__main__":
    main()
