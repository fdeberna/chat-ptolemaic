import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import re

BOOK_IDS = [17611, 17897, 18755, 19950, 22295]  # ← add your IDs here

BASE_URL = "https://www.gutenberg.org"
OUTPUT_DIR = Path("data/gutenberg/")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_plain_text_link(book_id):
    """Find the Plain Text UTF-8 link from the ebook page."""
    url = f"{BASE_URL}/ebooks/{book_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.select("a.link"):
        if "Plain Text UTF-8" in a.get_text():
            return BASE_URL + a["href"]

    return None


import re

import re

def extract_core_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Match marker lines; allow THE/THIS, allow extra spaces, allow anything after EBOOK, allow trailing whitespace
    start_re = re.compile(
        r"^\*{3}\s*START OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG\s+EBOOK.*?\*{3}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # END lines sometimes vary a bit; still anchored to a line starting with "*** END OF"
    end_re = re.compile(
        r"^\*{3}\s*END OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG(?:\s+EBOOK)?.*?\*{3}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    start_match = start_re.search(text)
    if not start_match:
        # Fallback: any START line mentioning PROJECT GUTENBERG
        start_re2 = re.compile(
            r"^\*{3}\s*START OF.*?PROJECT\s+GUTENBERG.*?\*{3}\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        start_match = start_re2.search(text)

    if start_match:
        text_after_start = text[start_match.end():]
    else:
        text_after_start = text

    # IMPORTANT: search END only after START (prevents accidental early match/miss)
    end_match = end_re.search(text_after_start)
    if not end_match:
        # Fallback: any END line mentioning PROJECT GUTENBERG
        end_re2 = re.compile(
            r"^\*{3}\s*END OF.*?PROJECT\s+GUTENBERG.*?\*{3}\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        end_match = end_re2.search(text_after_start)

    if end_match:
        core = text_after_start[:end_match.start()]
    else:
        core = text_after_start  # if no END found, keep everything after START

    return core.strip()

def download_book(book_id):
    txt_link = get_plain_text_link(book_id)

    if not txt_link:
        print(f"{book_id}: UTF-8 text not found")
        return

    print(f"Downloading {book_id}...")
    r = requests.get(txt_link, headers=HEADERS)
    r.raise_for_status()

    clean_text = extract_core_text(r.text)

    out_path = OUTPUT_DIR / f"{book_id}.txt"
    out_path.write_text(clean_text, encoding="utf-8")

    print(f"Saved -> {out_path}")


def main():
    for book_id in BOOK_IDS:
        download_book(book_id)
        time.sleep(1)  # be polite to Gutenberg


if __name__ == "__main__":
    main()