from pathlib import Path
from utils import clean_latin_ocr, rebuild_paragraphs

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def clean_file(filepath: Path):
    print(f"Cleaning: {filepath.name}")

    raw_text = filepath.read_text(encoding="utf-8", errors="ignore")

    cleaned = clean_latin_ocr(raw_text)
    cleaned = rebuild_paragraphs(cleaned)

    output_path = CLEAN_DIR / filepath.name
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"Saved cleaned text -> {output_path}")


def main():
    files = list(RAW_DIR.glob("*.txt"))

    if not files:
        print("No raw files found. Run download.py first.")
        return

    for file in files:
        clean_file(file)


if __name__ == "__main__":
    main()

