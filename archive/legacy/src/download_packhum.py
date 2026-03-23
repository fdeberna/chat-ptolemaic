"""
Downloader for PHI Latin texts at https://latin.packhum.org/browse

Features:
- Crawls the author list from /browse
- For each author, discovers works from /author/<anum>
- For each work, fetches all pages via the /dx/text endpoint and
  concatenates the first-column text into a single plaintext file.
- Optional Latin -> English translation (best-effort) using deep_translator;
  if unavailable, you can enable Selenium/Chrome with --use-selenium.

Outputs:
- Latin plaintext: data/packhum/latin/<author>__<work>.txt
- If --translate: English plaintext: data/packhum/english/<author>__<work>.txt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://latin.packhum.org"

LATIN_DIR = Path("data/packhum/latin")
EN_DIR = Path("data/packhum/english")

HEADERS = {
    "User-Agent": "packhum-scraper/0.1 (+https://latin.packhum.org)",
}


@dataclass
class Work:
    anum: int
    wnum: int
    name: str


def fetch(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_authors(html: str) -> List[tuple[int, str]]:
    soup = BeautifulSoup(html, "html.parser")
    ul = soup.find("ul", class_="authors")
    authors: List[tuple[int, str]] = []
    if not ul:
        return authors
    for li in ul.find_all("li", class_="anam"):
        a = li.find("a", class_="link")
        if not a or not a.get("href"):
            continue
        href = a["href"].strip("/")  # author/2000
        parts = href.split("/")
        if len(parts) != 2 or parts[0] != "author":
            continue
        try:
            anum = int(parts[1])
        except ValueError:
            continue
        name = a.get_text(strip=True).replace("\u00a0", " ")
        authors.append((anum, name))
    return authors


def parse_works(author_html: str, anum: int) -> List[Work]:
    soup = BeautifulSoup(author_html, "html.parser")
    ul = soup.find("ul", class_="works")
    works: List[Work] = []
    if not ul:
        return works
    for li in ul.find_all("li", class_="work"):
        a = li.find("a", class_="link")
        if not a or not a.get("href"):
            continue
        href = a["href"].strip("/")  # loc/400/1/0
        parts = href.split("/")
        if len(parts) < 3:
            continue
        try:
            wnum = int(parts[2])
        except ValueError:
            continue
        name = a.get_text(strip=True).replace("\u00a0", " ")
        works.append(Work(anum=anum, wnum=wnum, name=name))
    return works


def parse_loc_info(loc_html: str) -> dict:
    # locInfo JSON is embedded as: var locInfo = {...};
    marker = "var locInfo = "
    start = loc_html.find(marker)
    if start == -1:
        raise RuntimeError("locInfo not found")
    start += len(marker)
    end = loc_html.find(";", start)
    raw = loc_html[start:end]
    return json.loads(raw)


def fetch_page_text(anum: int, wnum: int, pnum: int, offsets: Optional[list] = None) -> List[str]:
    offsets = offsets or []
    url = f"{BASE}/dx/text/{anum}/{wnum}/{pnum}/{json.dumps(offsets)}"
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    lines: List[str] = []
    for tr in rows:
        tds = tr.find_all("td")
        if not tds:
            continue
        latin = tds[0].get_text(" ", strip=True)
        if latin:
            lines.append(latin)
    return lines


def translate_lines(lines: List[str], use_selenium: bool = False, sleep_sec: float = 0.0):
    """
    Translate Latin -> English. Prefers the free Google endpoint; falls back to deep_translator or Selenium.
    """
    def google_free_batch(chunk: List[str]) -> List[str]:
        url = "https://translate.googleapis.com/translate_a/single"
        translated = []
        for line in chunk:
            if not line.strip():
                translated.append("")
                continue
            params = {"client": "gtx", "sl": "la", "tl": "en", "dt": "t", "q": line}
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            translated.append(" ".join(part[0] for part in data[0] if part[0]))
            if sleep_sec:
                time.sleep(sleep_sec)
        return translated

    try:
        return google_free_batch(lines)
    except Exception:
        pass

    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="la", target="en")
        return [translator.translate(line) if line.strip() else "" for line in lines]
    except Exception as exc:  # noqa: BLE001
        if not use_selenium:
            raise RuntimeError(
                "Translation failed and --use-selenium not set; install deep-translator or pass --use-selenium"
            ) from exc

    # Selenium fallback (headless Chrome needed)
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=opts)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Selenium translation requested but Chrome driver is unavailable") from exc

    translated: List[str] = []
    try:
        for line in lines:
            if not line.strip():
                translated.append("")
                continue
            url = (
                "https://translate.google.com/?sl=la&tl=en&op=translate&text="
                + requests.utils.quote(line)
            )
            driver.get(url)
            time.sleep(2.0)
            elems = driver.find_elements("css selector", 'div[data-result-index="0"] span[jsname="W297wb"]')
            text = " ".join(e.text for e in elems if e.text)
            translated.append(text or "")
    finally:
        driver.quit()
    return translated


def save_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(",", "")
        .replace(".", "")
        .replace(";", "")
    )


def main():
    parser = argparse.ArgumentParser(description="Download PHI Latin texts from latin.packhum.org.")
    parser.add_argument(
        "--authors",
        nargs="*",
        type=int,
        help="Author IDs to include (default: all). Use one ID to test a single author.",
    )
    parser.add_argument("--translate", action="store_true", help="Translate Latin to English.")
    parser.add_argument("--use-selenium", action="store_true", help="Force Selenium-based translation fallback.")
    parser.add_argument(
        "--translate-sleep",
        type=float,
        default=0.0,
        help="Sleep between translation requests (seconds) to avoid throttling.",
    )
    args = parser.parse_args()

    browse_html = fetch(urljoin(BASE, "/browse"))
    authors = parse_authors(browse_html)
    if args.authors:
        authors = [a for a in authors if a[0] in args.authors]
    print(f"Found {len(authors)} authors to process")

    for anum, aname in authors:
        print(f"[author] {aname} ({anum})")
        author_html = fetch(f"{BASE}/author/{anum}")
        works = parse_works(author_html, anum)
        if not works:
            print("  no works found, skipping")
            continue
        for work in works:
            print(f"  [work] {work.name} ({work.wnum})")
            loc_html = fetch(f"{BASE}/loc/{work.anum}/{work.wnum}/0")
            loc_info = parse_loc_info(loc_html)
            pages = loc_info.get("pages", [])
            offsets = loc_info.get("offsets", [])

            all_lines: List[str] = []
            for pnum in range(len(pages)):
                page_lines = fetch_page_text(work.anum, work.wnum, pnum, offsets=offsets)
                all_lines.extend(page_lines)

            latin_path = LATIN_DIR / f"{slugify(aname)}__{slugify(work.name)}.txt"
            save_text(latin_path, all_lines)

            if args.translate:
                print("    translating ...")
                eng_lines = translate_lines(all_lines, use_selenium=args.use_selenium, sleep_sec=args.translate_sleep)
                eng_path = EN_DIR / latin_path.name
                save_text(eng_path, eng_lines)


if __name__ == "__main__":
    main()
