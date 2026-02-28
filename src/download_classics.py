from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

DATA_DIR = Path("data")
TRANSLATED_DIR = DATA_DIR / "translated"
BASE_URL = "http://classics.mit.edu/"

# DEFAULT_AUTHORS = [
#     "Aristotle",
#     "Plato",
#     "Marcus Aurelius",
#     "Homer",
#     "Virgil",
#     "Herodotus",
#     "Thucydides",
#     "Sophocles",
#     "Euripides",
#     "Aeschylus",
#     "Galen",
#     "Euclid",
# ]

DEFAULT_AUTHORS = [
    "Aesop",
    "Apollonius",
    "Apuleius",
    "Aristophanes",
    "Augustus","Julius Caesar","Epictetus","Euripides",
    "Hippocrates","Thucydides","Tacitus","Sophocles",
    "Quintus","Porphyry","Plutarch","Plotinus","Ovid"
]

#unsafe: lucretius, epicurus

# -------- HTML helpers --------
@dataclass
class Anchor:
    href: str
    text: str


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: List[Anchor] = []
        self._current_href: Optional[str] = None
        self._buffer: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self._current_href = href
                self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            text = "".join(self._buffer).strip()
            self.anchors.append(Anchor(self._current_href, text))
            self._current_href = None
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._buffer.append(data)


class TextExtractor(HTMLParser):
    """
    Minimal tag stripper that keeps rough paragraph breaks.
    """

    BLOCK_TAGS = {
        "p",
        "div",
        "br",
        "hr",
        "li",
        "ul",
        "ol",
        "table",
        "tr",
        "td",
        "th",
        "blockquote",
        "pre",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__()
        self.chunks: List[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip = True
        if tag.lower() in self.BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"}:
            self._skip = False
        if tag.lower() in self.BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.chunks.append(data)

    def get_text(self) -> str:
        text = "".join(self.chunks)
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


# -------- Fetching --------
def fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req) as resp:
            content_bytes = resp.read()
        return content_bytes.decode("utf-8", errors="ignore")
    except URLError as exc:
        # Some environments lack SSL support; try http fallback.
        if url.startswith("https://"):
            fallback = url.replace("https://", "http://", 1)
            req = Request(fallback, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req) as resp:
                content_bytes = resp.read()
            return content_bytes.decode("utf-8", errors="ignore")
        raise exc


# -------- Parsing utilities --------
def parse_anchors(html_text: str) -> List[Anchor]:
    parser = AnchorParser()
    parser.feed(html_text)
    return parser.anchors


def norm_label(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w]+", "_", text.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "text"


def crop_content_region(html_text: str) -> str:
    start = re.search(r'<a name="start"', html_text, flags=re.IGNORECASE)
    if start:
        html_text = html_text[start.end() :]
    end = re.search(r"&copy;", html_text, flags=re.IGNORECASE)
    if end:
        html_text = html_text[: end.start()]
    return html_text


def html_to_text(html_text: str) -> str:
    extractor = TextExtractor()
    extractor.feed(html_text)
    return extractor.get_text()


def strip_preformatted(raw: str) -> str:
    pre_match = re.search(
        r"<pre[^>]*>(.*?)</pre>", raw, flags=re.IGNORECASE | re.DOTALL
    )
    if pre_match:
        return html.unescape(pre_match.group(1)).strip()
    # fallback: strip tags if no <pre>
    return html_to_text(raw)


# -------- Site-specific logic --------
def load_author_links(base_url: str) -> Dict[str, str]:
    authors_url = urljoin(base_url, "Browse/authors.html")
    html_text = fetch(authors_url)
    links: Dict[str, str] = {}
    for a in parse_anchors(html_text):
        if a.href and a.href.startswith("browse-"):
            # author links are relative to /Browse/authors.html, not site root
            links[norm_label(a.text)] = urljoin(authors_url, a.href)
    return links


def is_work_href(href: str) -> bool:
    if not href.startswith("/"):
        return False
    parts = href.lstrip("/").split("/")
    if len(parts) != 2:
        return False
    directory, filename = parts
    if not filename.endswith(".html"):
        return False
    if filename.startswith("index"):
        return False
    if directory.lower() in {"browse", "search", "help", "images", "buy"}:
        return False
    return True


def is_section_href(href: str) -> bool:
    if "://" in href:
        return False
    if href.startswith("/"):
        return False
    if not href.endswith(".html"):
        return False
    if href.startswith("index"):
        return False
    return True


def collect_works(author_url: str) -> List[Tuple[str, str]]:
    """
    Returns list of (title, absolute_url).
    """
    html_text = fetch(author_url)
    works: List[Tuple[str, str]] = []
    for a in parse_anchors(html_text):
        if is_work_href(a.href):
            title = a.text or a.href
            works.append((title.strip(), urljoin(author_url, a.href)))
    return works


def collect_download_link(work_html: str, work_url: str) -> Optional[str]:
    for a in parse_anchors(work_html):
        if a.href and a.href.lower().endswith(".txt"):
            return urljoin(work_url, a.href)
    return None


def collect_section_links(work_html: str, work_url: str) -> List[str]:
    links: List[str] = []
    for a in parse_anchors(work_html):
        if is_section_href(a.href):
            links.append(urljoin(work_url, a.href))
    return links


def fetch_work_text(work_url: str) -> str:
    """
    Fetch a work page. Prefer the provided .txt download; otherwise
    fallback to HTML (including multi-part section pages).
    """
    work_html = fetch(work_url)

    txt_link = collect_download_link(work_html, work_url)
    if txt_link:
        try:
            raw_txt = fetch(txt_link)
            return strip_preformatted(raw_txt)
        except (HTTPError, URLError):
            print(f"Failed to download text version {txt_link}; falling back to HTML.")

    sections = collect_section_links(work_html, work_url)
    if sections:
        parts = []
        for section_url in sections:
            section_html = fetch(section_url)
            section_body = crop_content_region(section_html)
            parts.append(html_to_text(section_body))
        return "\n\n".join(p for p in parts if p.strip())

    content = crop_content_region(work_html)
    return html_to_text(content)


# -------- CLI --------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download works for selected authors from classics.mit.edu"
    )
    parser.add_argument(
        "--authors",
        nargs="*",
        help="Author names to download. Defaults to a fixed list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRANSLATED_DIR,
        help="Where to write downloaded texts.",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Base URL for the Internet Classics Archive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    authors = args.authors or DEFAULT_AUTHORS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching author index from {args.base_url} ...")
    author_links = load_author_links(args.base_url)

    for author in authors:
        key = norm_label(author)
        author_url = author_links.get(key)
        if not author_url:
            print(f"[warn] Author not found on site: {author}")
            continue

        print(f"\nAuthor: {author} -> {author_url}")
        works = collect_works(author_url)
        if not works:
            print(f"[warn] No works found for {author}")
            continue

        for title, work_url in works:
            safe_name = f"{slugify(author)}_{slugify(title)}.txt"
            out_path = args.output_dir / safe_name
            if out_path.exists() and not args.overwrite:
                print(f"Skipping existing: {out_path}")
                continue

            print(f"  Downloading {title} ...")
            try:
                text = fetch_work_text(work_url)
            except (HTTPError, URLError) as exc:
                print(f"  [error] Failed to fetch {work_url}: {exc}")
                continue

            if not text.strip():
                print(f"  [warn] Empty content for {title}")
                continue

            out_path.write_text(text.strip() + "\n", encoding="utf-8")
            print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
