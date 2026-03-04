import argparse
import csv
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET


NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dcterms": "http://purl.org/dc/terms/",
    "pgterms": "http://www.gutenberg.org/2009/pgterms/",
}

RDF_ABOUT_ATTR = f"{{{NS['rdf']}}}about"
RDF_RESOURCE_ATTR = f"{{{NS['rdf']}}}resource"

TAG_RDF_VALUE = f"{{{NS['rdf']}}}value"
TAG_RDF_DESCRIPTION = f"{{{NS['rdf']}}}Description"

TAG_EBOOK = f"{{{NS['pgterms']}}}ebook"
TAG_FILE = f"{{{NS['pgterms']}}}file"
TAG_BOOKSHELF = f"{{{NS['pgterms']}}}bookshelf"

TAG_TITLE = f"{{{NS['dcterms']}}}title"
TAG_LANGUAGE = f"{{{NS['dcterms']}}}language"
TAG_SUBJECT = f"{{{NS['dcterms']}}}subject"
TAG_HAS_FORMAT = f"{{{NS['dcterms']}}}hasFormat"
TAG_FORMAT = f"{{{NS['dcterms']}}}format"
TAG_EXTENT = f"{{{NS['dcterms']}}}extent"

SCIENCE_KEYWORDS = [
    "science",
    "astronomy",
    "physics",
    "cosmology",
    "heliocentr",
    "copernic",
    "newton",
    "kepler",
    "telescope",
    "gravitation",
    "astronomical",
    "astronomer","orb", "mechanics"
]

START_MARKER_RE = re.compile(r"\*\*\*\s*START", re.IGNORECASE)
END_MARKER_RE = re.compile(r"\*\*\*\s*END", re.IGNORECASE)
PARAGRAPH_BREAK_RE = re.compile(r"((?:\r?\n[ \t]*){2,})")


@dataclass
class PlainTextFormat:
    url: str
    mime: str
    extent: int
    charset: str


@dataclass
class BookCandidate:
    book_id: str
    title: str
    url: str
    charset: str
    extent: int


def normalize_id(text: str) -> str:
    match = re.search(r"(\d+)", text or "")
    if not match:
        return ""
    return str(int(match.group(1)))


def clean_text(value: Optional[str]) -> str:
    return value.strip() if value else ""


def parse_int(value: Optional[str]) -> int:
    if not value:
        return 0
    match = re.search(r"\d+", value.replace(",", ""))
    if not match:
        return 0
    return int(match.group(0))


def parse_mime(mime: str) -> Tuple[str, str]:
    if not mime:
        return "", ""
    parts = [part.strip() for part in mime.split(";") if part.strip()]
    if not parts:
        return "", ""
    base_mime = parts[0].lower()
    charset = ""
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip().lower() == "charset":
            charset = value.strip().strip('"').replace("_", "-").lower()
            break
    return base_mime, charset


def iter_rdf_files(rdf_dir: Path) -> Iterable[Path]:
    for root, _, files in os.walk(rdf_dir):
        for filename in files:
            if filename.lower().endswith(".rdf"):
                yield Path(root) / filename


def extract_value_from_structured_node(node: ET.Element) -> str:
    for child in node:
        if child.tag == TAG_RDF_VALUE:
            value = clean_text(child.text)
            if value:
                return value
        elif child.tag == TAG_RDF_DESCRIPTION:
            for grandchild in child:
                if grandchild.tag == TAG_RDF_VALUE:
                    value = clean_text(grandchild.text)
                    if value:
                        return value
    return ""


def parse_file_node(file_node: ET.Element) -> Optional[PlainTextFormat]:
    url = clean_text(file_node.attrib.get(RDF_ABOUT_ATTR, ""))
    mime = ""
    extent = 0
    for child in file_node:
        if child.tag == TAG_FORMAT:
            mime = extract_value_from_structured_node(child).lower()
        elif child.tag == TAG_EXTENT:
            extent = parse_int(child.text)
    if not mime or not url:
        return None
    _, charset = parse_mime(mime)
    return PlainTextFormat(url=url, mime=mime, extent=extent, charset=charset)


def choose_best_plain_text(formats: List[PlainTextFormat]) -> Optional[PlainTextFormat]:
    best = None
    best_key = (-1, -1, "")
    for fmt in formats:
        base_mime, charset = parse_mime(fmt.mime)
        if base_mime != "text/plain":
            continue
        priority = 2 if charset == "utf-8" else 1
        key = (priority, fmt.extent, fmt.url)
        if key > best_key:
            best = fmt
            best_key = key
    return best


def is_english(languages: List[str]) -> bool:
    return any(lang == "en" or lang.startswith("en-") for lang in languages)


def contains_science_keywords(title: str, subjects: List[str], bookshelves: List[str]) -> bool:
    text = " ".join([title] + subjects + bookshelves).lower()
    return any(keyword in text for keyword in SCIENCE_KEYWORDS)


def parse_candidate_from_rdf(path: Path, bypass_filters: bool = False) -> Optional[BookCandidate]:
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return None

    root = tree.getroot()

    ebook = None
    for child in root:
        if child.tag == TAG_EBOOK:
            ebook = child
            break
    if ebook is None:
        return None

    file_index: Dict[str, ET.Element] = {}
    for child in root:
        if child.tag == TAG_FILE:
            about = clean_text(child.attrib.get(RDF_ABOUT_ATTR, ""))
            if about:
                file_index[about] = child

    book_id = normalize_id(clean_text(ebook.attrib.get(RDF_ABOUT_ATTR, "")))
    if not book_id:
        book_id = normalize_id(path.stem)
    if not book_id:
        return None

    title = ""
    languages: List[str] = []
    subjects: List[str] = []
    bookshelves: List[str] = []
    formats: List[PlainTextFormat] = []

    for node in ebook:
        tag = node.tag
        if tag == TAG_TITLE:
            title = clean_text(node.text)
        elif tag == TAG_LANGUAGE:
            value = extract_value_from_structured_node(node).lower()
            if value:
                languages.append(value)
        elif tag == TAG_SUBJECT:
            value = extract_value_from_structured_node(node)
            if value:
                subjects.append(value)
        elif tag == TAG_BOOKSHELF:
            value = extract_value_from_structured_node(node)
            if value:
                bookshelves.append(value)
        elif tag == TAG_HAS_FORMAT:
            inline_found = False
            for child in node:
                if child.tag == TAG_FILE:
                    inline_found = True
                    parsed = parse_file_node(child)
                    if parsed is not None:
                        formats.append(parsed)
            if inline_found:
                continue

            resource = clean_text(node.attrib.get(RDF_RESOURCE_ATTR, ""))
            if resource:
                external_file = file_index.get(resource)
                if external_file is not None:
                    parsed = parse_file_node(external_file)
                    if parsed is not None:
                        formats.append(parsed)

    if not bypass_filters:
        if not is_english(languages):
            return None
        if contains_science_keywords(title, subjects, bookshelves):
            return None

    best = choose_best_plain_text(formats)
    if best is None:
        return None

    return BookCandidate(
        book_id=book_id,
        title=title,
        url=best.url,
        charset=best.charset,
        extent=best.extent,
    )


def strip_gutenberg_boilerplate(text: str) -> str:
    start_match = START_MARKER_RE.search(text)
    if start_match:
        newline_pos = text.find("\n", start_match.start())
        text = "" if newline_pos == -1 else text[newline_pos + 1 :]

    end_match = END_MARKER_RE.search(text)
    if end_match:
        text = text[: end_match.start()]

    return text.strip("\ufeff\r\n ")


def is_mostly_uppercase(text: str, threshold: float = 0.6) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    uppercase = sum(1 for ch in letters if ch.isupper())
    return (uppercase / len(letters)) >= threshold


def clean_gutenberg_text(raw_text: str) -> str:
    """
    Trim leading front-matter by finding the first substantial prose paragraph.

    Paragraphs are defined by blank lines. We skip leading paragraphs that are
    short, all-caps-ish, or not prose-like, then keep the first paragraph that
    looks like real body text and everything after it unchanged.
    """
    parts = PARAGRAPH_BREAK_RE.split(raw_text)
    start_idx = None

    for idx in range(0, len(parts), 2):
        paragraph = parts[idx]
        candidate = paragraph.strip()
        if not candidate:
            continue
        if is_mostly_uppercase(candidate):
            continue
        if "." not in candidate:
            continue
        if not any(ch.islower() for ch in candidate):
            continue

        normalized = " ".join(candidate.split())
        if len(normalized) < 200:
            continue

        start_idx = idx
        break

    if start_idx is None:
        return raw_text

    return "".join(parts[start_idx:])


def decode_text(data: bytes, preferred_charset: str) -> str:
    tried = set()
    candidates = []
    if preferred_charset:
        candidates.append(preferred_charset)
    candidates.extend(["utf-8", "utf-8-sig", "latin-1"])

    for encoding in candidates:
        encoding = encoding.lower()
        if encoding in tried:
            continue
        tried.add(encoding)
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def download_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "gutenberg-safe-corpus-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def build_candidates(rdf_dir: Path) -> Dict[str, BookCandidate]:
    candidates: Dict[str, BookCandidate] = {}
    scanned = 0
    for rdf_path in iter_rdf_files(rdf_dir):
        scanned += 1
        if scanned % 5000 == 0:
            print(f"[scan] processed {scanned} RDF files, candidates so far: {len(candidates)}")

        candidate = parse_candidate_from_rdf(rdf_path)
        if candidate is None:
            continue

        existing = candidates.get(candidate.book_id)
        if existing is None:
            candidates[candidate.book_id] = candidate
            continue

        # Keep the "better" plain text candidate if duplicate book IDs exist.
        current_key = (existing.charset == "utf-8", existing.extent, existing.url)
        new_key = (candidate.charset == "utf-8", candidate.extent, candidate.url)
        if new_key > current_key:
            candidates[candidate.book_id] = candidate

    print(f"[scan] done. total RDF files: {scanned}, total candidates: {len(candidates)}")
    return candidates


def save_candidates_csv(candidates: Dict[str, BookCandidate], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["book_id", "title", "url", "extent", "charset"])
        for candidate in sorted(candidates.values(), key=lambda c: int(c.book_id)):
            writer.writerow(
                [
                    candidate.book_id,
                    candidate.title,
                    candidate.url,
                    candidate.extent,
                    candidate.charset,
                ]
            )


def load_candidates_csv(csv_path: Path) -> Dict[str, BookCandidate]:
    candidates: Dict[str, BookCandidate] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            book_id = normalize_id(clean_text(row.get("book_id", "")))
            if not book_id:
                continue
            candidates[book_id] = BookCandidate(
                book_id=book_id,
                title=clean_text(row.get("title", "")),
                url=clean_text(row.get("url", "")),
                extent=parse_int(row.get("extent")),
                charset=clean_text(row.get("charset", "")).lower(),
            )
    return candidates


def get_or_build_candidates(rdf_dir: Path, candidates_csv: Path, refresh: bool) -> Dict[str, BookCandidate]:
    if candidates_csv.exists() and not refresh:
        print(f"Loading candidates from CSV: {candidates_csv}")
        loaded = load_candidates_csv(candidates_csv)
        if loaded:
            print(f"[scan] loaded {len(loaded)} candidates from CSV")
            return loaded
        print("[scan] candidate CSV was empty or invalid; rebuilding from RDF")

    print("Scanning RDF to build candidates...")
    built = build_candidates(rdf_dir)
    save_candidates_csv(built, candidates_csv)
    print(f"[scan] wrote candidates CSV: {candidates_csv}")
    return built


def parse_book_ids_arg(raw_ids: str) -> List[str]:
    if not raw_ids:
        return []

    ordered: List[str] = []
    seen = set()
    for chunk in raw_ids.split(","):
        book_id = normalize_id(clean_text(chunk))
        if not book_id or book_id in seen:
            continue
        seen.add(book_id)
        ordered.append(book_id)
    return ordered


def find_rdf_path_for_book_id(rdf_dir: Path, book_id: str) -> Optional[Path]:
    common_path = rdf_dir / "cache" / "epub" / book_id / f"pg{book_id}.rdf"
    if common_path.exists():
        return common_path

    filename = f"pg{book_id}.rdf"
    for path in rdf_dir.rglob(filename):
        if path.is_file():
            return path
    return None


def build_forced_candidates(book_ids: List[str], rdf_dir: Path) -> Tuple[List[BookCandidate], List[str]]:
    forced: List[BookCandidate] = []
    missing: List[str] = []

    for book_id in book_ids:
        rdf_path = find_rdf_path_for_book_id(rdf_dir, book_id)
        if rdf_path is None:
            missing.append(book_id)
            continue

        candidate = parse_candidate_from_rdf(rdf_path, bypass_filters=True)
        if candidate is None:
            missing.append(book_id)
            continue

        forced.append(candidate)

    return forced, missing


def load_priority_ids(priority_csv: Path) -> List[str]:
    if not priority_csv.exists():
        return []

    priority_ids: List[str] = []
    seen = set()
    with priority_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        id_key = "book_id" if "book_id" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            book_id = normalize_id(clean_text(row.get(id_key, "")))
            if not book_id or book_id in seen:
                continue
            seen.add(book_id)
            priority_ids.append(book_id)
    return priority_ids


def order_candidates_with_priority(
    candidates: Dict[str, BookCandidate], priority_ids: List[str]
) -> Tuple[List[BookCandidate], int]:
    if not priority_ids:
        return sorted(candidates.values(), key=lambda c: int(c.book_id)), 0

    priority_candidates: List[BookCandidate] = []
    priority_set = set(priority_ids)
    for book_id in priority_ids:
        candidate = candidates.get(book_id)
        if candidate is not None:
            priority_candidates.append(candidate)

    others = [candidate for book_id, candidate in candidates.items() if book_id not in priority_set]
    others.sort(key=lambda c: int(c.book_id))
    return priority_candidates + others, len(priority_candidates)


def run(args: argparse.Namespace) -> None:
    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    rdf_dir = Path(args.rdf_dir)
    if not rdf_dir.exists():
        raise FileNotFoundError(f"RDF directory not found: {rdf_dir}")

    books_dir = Path(args.books_dir)
    books_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = base_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = Path(args.candidates_csv)
    candidates_csv.parent.mkdir(parents=True, exist_ok=True)

    existing_book_paths = {p.stem: p for p in books_dir.glob("*.txt") if p.is_file()}
    current_bytes = sum(p.stat().st_size for p in existing_book_paths.values())
    target_bytes = int(args.target_bytes)
    max_books = int(args.max_books)

    print(f"Current corpus bytes across {books_dir}: {current_bytes}")
    print(f"Target bytes: {target_bytes}")
    print(f"Already saved books: {len(existing_book_paths)}")

    candidates = get_or_build_candidates(rdf_dir, candidates_csv, args.refresh_candidates)
    priority_csv = Path(args.priority_csv)
    priority_ids = load_priority_ids(priority_csv)
    ordered_candidates, priority_hits = order_candidates_with_priority(candidates, priority_ids)
    forced_book_ids = parse_book_ids_arg(args.book_ids)
    only_book_ids = args.only_book_ids

    if priority_ids:
        print(
            f"Priority list: {priority_csv} (ids in file: {len(priority_ids)}, "
            f"present in candidates: {priority_hits})"
        )
    else:
        print(f"Priority list not used or empty: {priority_csv}")

    if forced_book_ids:
        forced_candidates, forced_missing = build_forced_candidates(forced_book_ids, rdf_dir)
        forced_set = {candidate.book_id for candidate in forced_candidates}
        trailing_candidates = [c for c in ordered_candidates if c.book_id not in forced_set]

        if only_book_ids:
            ordered_candidates = forced_candidates
        else:
            ordered_candidates = forced_candidates + trailing_candidates

        print(
            f"Explicit book IDs requested: {len(forced_book_ids)} "
            f"(resolved: {len(forced_candidates)}, unresolved: {len(forced_missing)})"
        )
        if forced_missing:
            print(f"[warn] unresolved explicit IDs: {', '.join(forced_missing)}")
    elif only_book_ids:
        print("[warn] --only-book-ids was set but no --book-ids were provided.")

    saved_this_run = 0
    downloaded_this_run = 0
    skipped_already_saved = 0
    skipped_failed = 0

    for candidate in ordered_candidates:
        if (not only_book_ids) and current_bytes >= target_bytes:
            break
        if max_books > 0 and saved_this_run >= max_books:
            break

        book_path = books_dir / f"{candidate.book_id}.txt"
        if book_path.exists() and book_path.stat().st_size > 0:
            skipped_already_saved += 1
            continue

        cache_path = downloads_dir / f"{candidate.book_id}.txt"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            raw = cache_path.read_bytes()
        else:
            try:
                raw = download_bytes(candidate.url)
                downloaded_this_run += 1
                if downloaded_this_run % args.progress_every == 0:
                    print(
                        f"[download] new downloads: {downloaded_this_run}, "
                        f"saved this run: {saved_this_run}, corpus bytes: {current_bytes}"
                    )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                skipped_failed += 1
                print(f"[warn] failed download for {candidate.book_id} ({candidate.url}): {exc}")
                continue
            cache_path.write_bytes(raw)
            time.sleep(args.sleep_seconds)

        text = decode_text(raw, candidate.charset)
        stripped = strip_gutenberg_boilerplate(text)
        cleaned = clean_gutenberg_text(stripped)
        if not cleaned:
            skipped_failed += 1
            continue

        payload = (cleaned + "\n").encode("utf-8")
        book_path.write_bytes(payload)

        current_bytes += len(payload)
        saved_this_run += 1

        if saved_this_run % args.progress_every == 0:
            print(
                f"[save] saved this run: {saved_this_run}, "
                f"new downloads: {downloaded_this_run}, corpus bytes: {current_bytes}"
            )

    print("\nDone")
    print(f"Books directory: {books_dir}")
    print(f"Corpus bytes: {current_bytes}")
    print(f"Newly saved books: {saved_this_run}")
    print(f"New network downloads: {downloaded_this_run}")
    print(f"Skipped already saved: {skipped_already_saved}")
    print(f"Skipped failed/empty: {skipped_failed}")
    if only_book_ids:
        print("Book-ID-only run complete.")
    elif current_bytes >= target_bytes:
        print("Target reached.")
    else:
        print("Target not reached with available filtered candidates.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-science English Gutenberg corpus with one cleaned text file per book."
    )
    parser.add_argument("--rdf-dir", default="data/rdf-files", help="Directory containing Gutenberg RDF files")
    parser.add_argument("--out-dir", default="data/gutenberg", help="Base output directory (used for download cache)")
    parser.add_argument(
        "--books-dir",
        default="data/gutenberg/books",
        help="Directory where one cleaned .txt file per book will be saved",
    )
    parser.add_argument(
        "--candidates-csv",
        default="data/gutenberg/candidates_safe.csv",
        help="CSV cache of filtered candidates (book_id,title,url,extent,charset)",
    )
    parser.add_argument(
        "--refresh-candidates",
        action="store_true",
        help="Force a fresh RDF scan and rewrite candidates CSV",
    )
    parser.add_argument(
        "--priority-csv",
        default="qualifying_pre1543_en.csv",
        help="CSV file whose book IDs should be downloaded first (expects a book_id column)",
    )
    parser.add_argument(
        "--book-ids",
        default="",
        help="Comma-separated Gutenberg book IDs to force in front of queue (bypasses language/science filters)",
    )
    parser.add_argument(
        "--only-book-ids",
        action="store_true",
        help="Process only --book-ids instead of continuing with the general candidate queue",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=1_000_000_000,
        help="Stop when corpus size reaches at least this many bytes",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=0,
        help="Optional hard cap on newly saved books this run (0 means no cap)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N downloads/saves (recommended 100-500)",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Delay between HTTP requests")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
