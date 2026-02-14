from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Add sources here
SOURCES = {
    "sphaera_mundi_1501.txt":
        "https://archive.org/stream/sphaeramundi00sacr_0/sphaeramundi00sacr_0_djvu.txt",
}

def download_file(filename, url):
    filepath = RAW_DIR / filename

    if filepath.exists():
        print(f"✓ Skipping (exists): {filename}")
        return

    print(f"↓ Downloading {filename}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    filepath.write_text(response.text, encoding="utf-8")
    print(f"✓ Saved to {filepath}")

def main():
    for filename, url in SOURCES.items():
        download_file(filename, url)

if __name__ == "__main__":
    main()

