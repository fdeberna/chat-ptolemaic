import argparse
import csv
import os
import re
from collections import Counter
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
TAG_AGENT = f"{{{NS['pgterms']}}}agent"
TAG_FILE = f"{{{NS['pgterms']}}}file"
TAG_DEATHDATE = f"{{{NS['pgterms']}}}deathdate"
TAG_BOOKSHELF = f"{{{NS['pgterms']}}}bookshelf"

TAG_TITLE = f"{{{NS['dcterms']}}}title"
TAG_CREATOR = f"{{{NS['dcterms']}}}creator"
TAG_LANGUAGE = f"{{{NS['dcterms']}}}language"
TAG_SUBJECT = f"{{{NS['dcterms']}}}subject"
TAG_HAS_FORMAT = f"{{{NS['dcterms']}}}hasFormat"
TAG_FORMAT = f"{{{NS['dcterms']}}}format"
TAG_EXTENT = f"{{{NS['dcterms']}}}extent"

ASTRO_KEYWORDS = [
    "astronomy",
    "astronomical",
    "astrolog",
    "sphere",
    "sphaera",
    "celestial",
    "heavens",
    "planets",
    "planet",
    "sun",
    "moon",
    "stars",
    "zodiac",
    "eclipse",
    "epicycle",
    "equant",
    "deferent",
    "ptolemy",
    "almagest",
    "cosmology",
    "cosmos",
    "orb",
    "orbit",
    "comet",
    "calendar",
    "almanac",
    "chronology",
    "navigation",
    "latitude",
    "longitude",
    "meteorology",
]

NATPHIL_KEYWORDS = [
    "physics",
    "nature",
    "natural philosophy",
    "meteor",
    "elements",
    "motion",
    "de caelo",
    "de mundo",
]

BANNED_KEYWORDS = [
    "copernic",
    "heliocentr",
    "kepler",
    "newton",
    "galileo",
    "telescope",
    "gravitation",
]


@dataclass
class FileFormat:
    url: str
    mime: str
    extent: int


@dataclass
class ParsedBook:
    book_id: str
    title: str
    death_year: Optional[int]
    languages: List[str]
    subjects: List[str]
    bookshelves: List[str]
    formats: List[FileFormat]


def clean_text(value: Optional[str]) -> str:
    return value.strip() if value else ""


def parse_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"-?\d+", value.replace(",", ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def normalize_id(value: str) -> str:
    match = re.search(r"(\d+)", value)
    if not match:
        return ""
    return str(int(match.group(1)))


def extract_book_id(ebook_about: str, path: Path) -> str:
    from_about = normalize_id(ebook_about)
    if from_about:
        return from_about
    return normalize_id(path.stem)


def normalize_mime(mime: str) -> Tuple[str, Dict[str, str]]:
    if not mime:
        return "", {}
    parts = [part.strip() for part in mime.split(";") if part.strip()]
    if not parts:
        return "", {}
    base = parts[0].lower()
    params: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        params[key.strip().lower()] = val.strip().strip('"')
    return base, params


def choose_best_plain_text(formats: List[FileFormat]) -> Optional[FileFormat]:
    best: Optional[FileFormat] = None
    best_key = (-1, -1, "")
    for fmt in formats:
        base_mime, params = normalize_mime(fmt.mime)
        if base_mime != "text/plain":
            continue
        charset = params.get("charset", "").replace("_", "-").lower()
        priority = 2 if charset == "utf-8" else 1
        sort_key = (priority, fmt.extent, fmt.url)
        if sort_key > best_key:
            best = fmt
            best_key = sort_key
    return best


def find_keyword_matches(text: str, keywords: List[str]) -> List[str]:
    return [keyword for keyword in keywords if keyword in text]


def categorize_book(title: str, subjects: List[str], bookshelves: List[str]) -> Tuple[str, List[str]]:
    searchable_text = " ".join([title] + subjects + bookshelves).lower()
    if any(keyword in searchable_text for keyword in BANNED_KEYWORDS):
        return "other", []

    astro_matches = find_keyword_matches(searchable_text, ASTRO_KEYWORDS)
    natphil_matches = find_keyword_matches(searchable_text, NATPHIL_KEYWORDS)
    all_matches = sorted(set(astro_matches + natphil_matches))

    if astro_matches:
        return "astro", all_matches
    if natphil_matches:
        return "natphil", all_matches
    return "other", []


def extract_value_from_node(node: ET.Element) -> str:
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


def extract_agent_death_year(agent: ET.Element) -> Optional[int]:
    for child in agent:
        if child.tag == TAG_DEATHDATE:
            year = parse_int(child.text)
            if year is not None:
                return year
    return None


def build_agent_index(root: ET.Element) -> Tuple[Dict[str, ET.Element], Dict[str, ET.Element]]:
    by_about: Dict[str, ET.Element] = {}
    by_id: Dict[str, ET.Element] = {}
    for child in root:
        if child.tag != TAG_AGENT:
            continue
        about = clean_text(child.attrib.get(RDF_ABOUT_ATTR, ""))
        if about:
            by_about[about] = child
            agent_id = normalize_id(about)
            if agent_id:
                by_id[agent_id] = child
    return by_about, by_id


def build_file_index(root: ET.Element) -> Dict[str, ET.Element]:
    by_about: Dict[str, ET.Element] = {}
    for child in root:
        if child.tag != TAG_FILE:
            continue
        about = clean_text(child.attrib.get(RDF_ABOUT_ATTR, ""))
        if about:
            by_about[about] = child
    return by_about


def iter_creator_agents(
    creator: ET.Element, agent_by_about: Dict[str, ET.Element], agent_by_id: Dict[str, ET.Element]
) -> Iterable[ET.Element]:
    has_direct_agent = False
    for child in creator:
        if child.tag == TAG_AGENT:
            has_direct_agent = True
            yield child
    if has_direct_agent:
        return

    resource = clean_text(creator.attrib.get(RDF_RESOURCE_ATTR, ""))
    if not resource:
        return

    agent = agent_by_about.get(resource)
    if agent is None:
        agent = agent_by_id.get(normalize_id(resource))
    if agent is not None:
        yield agent


def extract_file_format(file_node: ET.Element) -> Optional[FileFormat]:
    url = clean_text(file_node.attrib.get(RDF_ABOUT_ATTR, ""))
    mime = ""
    extent = 0
    for child in file_node:
        if child.tag == TAG_FORMAT:
            mime = extract_value_from_node(child).lower()
        elif child.tag == TAG_EXTENT:
            parsed_extent = parse_int(child.text)
            if parsed_extent is not None:
                extent = parsed_extent
    if not mime:
        return None
    return FileFormat(url=url, mime=mime, extent=extent)


def parse_book(
    path: Path,
    need_languages: bool,
    need_subjects: bool,
    need_bookshelves: bool,
    need_formats: bool,
) -> Optional[ParsedBook]:
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return None

    root = tree.getroot()
    ebook: Optional[ET.Element] = None
    for child in root:
        if child.tag == TAG_EBOOK:
            ebook = child
            break
    if ebook is None:
        return None

    agent_by_about, agent_by_id = build_agent_index(root)
    file_by_about = build_file_index(root) if need_formats else {}

    book_id = extract_book_id(clean_text(ebook.attrib.get(RDF_ABOUT_ATTR, "")), path)
    title = ""
    death_year: Optional[int] = None
    languages: List[str] = []
    subjects: List[str] = []
    bookshelves: List[str] = []
    formats: List[FileFormat] = []

    for node in ebook:
        tag = node.tag
        if tag == TAG_TITLE:
            title = clean_text(node.text)
        elif tag == TAG_CREATOR:
            for agent in iter_creator_agents(node, agent_by_about, agent_by_id):
                year = extract_agent_death_year(agent)
                if year is not None and (death_year is None or year < death_year):
                    death_year = year
        elif need_languages and tag == TAG_LANGUAGE:
            lang = extract_value_from_node(node).lower()
            if lang:
                languages.append(lang)
        elif need_subjects and tag == TAG_SUBJECT:
            subject = extract_value_from_node(node)
            if subject:
                subjects.append(subject)
        elif need_bookshelves and tag == TAG_BOOKSHELF:
            shelf = extract_value_from_node(node)
            if shelf:
                bookshelves.append(shelf)
        elif need_formats and tag == TAG_HAS_FORMAT:
            found_inline = False
            for child in node:
                if child.tag == TAG_FILE:
                    found_inline = True
                    fmt = extract_file_format(child)
                    if fmt is not None:
                        formats.append(fmt)
            if found_inline:
                continue

            resource = clean_text(node.attrib.get(RDF_RESOURCE_ATTR, ""))
            if resource:
                resource_file = file_by_about.get(resource)
                if resource_file is not None:
                    fmt = extract_file_format(resource_file)
                    if fmt is not None:
                        formats.append(fmt)

    return ParsedBook(
        book_id=book_id,
        title=title,
        death_year=death_year,
        languages=languages,
        subjects=subjects,
        bookshelves=bookshelves,
        formats=formats,
    )


def iter_rdf_files(rdf_dir: Path) -> Iterable[Path]:
    for root, _, files in os.walk(rdf_dir):
        for filename in files:
            if filename.lower().endswith(".rdf"):
                yield Path(root) / filename


def is_english_book(languages: List[str]) -> bool:
    return any(lang == "en" or lang.startswith("en-") for lang in languages)


def run_stage1(rdf_dir: Path, output_csv: Path, progress_interval: int) -> None:
    processed = 0
    qualifying = 0
    total_bytes = 0

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "book_id",
                "title",
                "death_year",
                "plain_text_url",
                "plain_text_extent_bytes",
            ]
        )

        for path in iter_rdf_files(rdf_dir):
            processed += 1
            if processed % progress_interval == 0:
                print(f"[stage1] processed {processed} RDF files")

            book = parse_book(
                path,
                need_languages=True,
                need_subjects=False,
                need_bookshelves=False,
                need_formats=True,
            )
            if book is None or not book.book_id:
                continue
            if book.death_year is None or book.death_year >= 1543:
                continue
            if not is_english_book(book.languages):
                continue

            best_plain_text = choose_best_plain_text(book.formats)
            if best_plain_text is None:
                continue

            qualifying += 1
            total_bytes += best_plain_text.extent
            writer.writerow(
                [
                    book.book_id,
                    book.title,
                    book.death_year,
                    best_plain_text.url,
                    best_plain_text.extent,
                ]
            )

    print("\nStage 1 complete")
    print(f"wrote: {output_csv}")
    print(f"total qualifying books: {qualifying}")
    print(f"total bytes: {total_bytes}")
    print(f"estimated tokens (bytes / 4): {total_bytes // 4}")


def load_qualifying_books(stage1_csv: Path) -> Dict[str, Tuple[int, int]]:
    qualifying: Dict[str, Tuple[int, int]] = {}
    with stage1_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            book_id = normalize_id(clean_text(row.get("book_id", "")))
            death_year = parse_int(row.get("death_year"))
            extent = parse_int(row.get("plain_text_extent_bytes"))
            if not book_id or death_year is None:
                continue
            qualifying[book_id] = (death_year, extent or 0)
    return qualifying


def run_stage2(rdf_dir: Path, stage1_csv: Path, output_csv: Path, progress_interval: int) -> None:
    qualifying = load_qualifying_books(stage1_csv)
    qualifying_ids = set(qualifying.keys())
    print(f"\nStage 2 input books: {len(qualifying_ids)}")

    counts = Counter()
    byte_totals = Counter()
    processed = 0
    matched_files = 0

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "book_id",
                "title",
                "death_year",
                "extent_bytes",
                "category",
                "matched_keywords",
            ]
        )

        for path in iter_rdf_files(rdf_dir):
            processed += 1
            if processed % progress_interval == 0:
                print(f"[stage2] processed {processed} RDF files")

            file_id = normalize_id(path.stem)
            if not file_id or file_id not in qualifying_ids:
                continue

            matched_files += 1
            death_year, extent_bytes = qualifying[file_id]
            book = parse_book(
                path,
                need_languages=False,
                need_subjects=True,
                need_bookshelves=True,
                need_formats=False,
            )
            if book is None:
                continue

            category, matches = categorize_book(book.title, book.subjects, book.bookshelves)
            counts[category] += 1
            byte_totals[category] += extent_bytes

            writer.writerow(
                [
                    file_id,
                    book.title,
                    death_year,
                    extent_bytes,
                    category,
                    ";".join(matches),
                ]
            )

    print("\nStage 2 complete")
    print(f"wrote: {output_csv}")
    print(f"matched qualifying IDs to RDF files: {matched_files}")

    for category in ("astro", "natphil", "other"):
        category_bytes = byte_totals[category]
        category_count = counts[category]
        category_tokens = category_bytes // 4
        print(
            f"{category}: books={category_count}, bytes={category_bytes}, "
            f"estimated_tokens={category_tokens}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage Gutenberg RDF analysis for English books with author death year < 1543 "
            "and available plain text format."
        )
    )
    parser.add_argument("--rdf-dir", default="data/rdf-files", help="Directory containing Gutenberg .rdf files")
    parser.add_argument(
        "--stage1-out",
        default="qualifying_pre1543_en.csv",
        help="Output CSV path for Stage 1 qualifying books",
    )
    parser.add_argument(
        "--stage2-out",
        default="categorization_pre1543_en.csv",
        help="Output CSV path for Stage 2 categorization results",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=500,
        help="Progress print interval (files processed) for both stages",
    )

    args = parser.parse_args()
    rdf_dir = Path(args.rdf_dir)
    stage1_out = Path(args.stage1_out)
    stage2_out = Path(args.stage2_out)
    progress_interval = max(1, args.progress)

    if not rdf_dir.exists() or not rdf_dir.is_dir():
        print(f"RDF directory not found: {rdf_dir}")
        return 1

    run_stage1(rdf_dir=rdf_dir, output_csv=stage1_out, progress_interval=progress_interval)
    run_stage2(rdf_dir=rdf_dir, stage1_csv=stage1_out, output_csv=stage2_out, progress_interval=progress_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
