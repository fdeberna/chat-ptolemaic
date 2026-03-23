#!/usr/bin/env python3
"""
Clean a text corpus for ML training and write a cleaned copy.

Removes:
1. Leading metadata/header blocks (Project Gutenberg and similar notices).
2. Table-of-contents style sections.
3. Structural artifact lines (chapter headings, standalone page numbers).
4. Inline annotation markers ([Footnote 1], [See Figure 2], [Sidenote: ...], etc.).
5. Excessive whitespace.

Writes:
- Cleaned files under output directory (same relative layout as input).
- JSON and CSV reports with per-file and aggregate removal statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


START_MARKER_RE = re.compile(
    r"^\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG", re.IGNORECASE
)
END_MARKER_RE = re.compile(
    r"^\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG", re.IGNORECASE
)

LEADING_METADATA_RE = re.compile(
    r"(?i)\b("
    r"project gutenberg|"
    r"e-?text|ebook|"
    r"release date|updated editions?|"
    r"copyright|all rights reserved|"
    r"digiti[sz]ed|proofread|transcriber(?:'s)? notes?|"
    r"produced by|distributed proofreaders|"
    r"millennium fulcrum edition|"
    r"character set encoding|language:\s|"
    r"title:\s|author:\s|editor:\s|"
    r"http://|https://|www\."
    r")\b"
)

TOC_HEADER_RE = re.compile(
    r"(?i)^\s*(table of contents|contents?|index)\s*[:.]?\s*$"
)
TOC_ENTRY_RE = re.compile(
    r"(?i)^\s*("
    r"(chapter|book|part|section)\s+[ivxlcdm0-9]+[a-z]?[.:)]?|"
    r"[ivxlcdm0-9]+\.\s+|"
    r"(preface|introduction|appendix|prologue|epilogue)\b"
    r").*?$"
)
TOC_DOTTED_RE = re.compile(r"\.{2,}\s*\d+\s*$")

STRUCTURAL_LINE_RE = re.compile(
    r"(?i)^\s*("
    r"(chapter|book|part|section)\s+[ivxlcdm0-9]+[a-z]?[.:)]?|"
    r"[-–—]?\s*\d{1,4}\s*[-–—]?|"
    r"page\s+\d{1,4}"
    r")\s*$"
)

INLINE_BRACKET_ANNOTATION_RE = re.compile(
    r"""
    \[
      (?:
        \s*(?:footnote|note|notes|sidenote|editor(?:'s)?\s+note|transcriber(?:'s)?\s+note)\b[^\]]*
        |
        \s*(?:see\s+figure|figure|fig\.|illustration|plate)\b[^\]]*
        |
        \s*\d{1,3}\s*
      )
    \]
    """,
    re.IGNORECASE | re.VERBOSE,
)

INLINE_PAREN_ANNOTATION_RE = re.compile(
    r"""
    \(
      \s*(?:see\s+figure|figure|fig\.|footnote|cf\.)\b[^)]*
    \)
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class FileReport:
    relative_path: str
    chars_before: int
    chars_after: int
    metadata_lines_removed: int
    gutenberg_footer_lines_removed: int
    toc_lines_removed: int
    structural_lines_removed: int
    inline_annotations_removed: int
    blank_line_reductions: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean text corpus for ML training and write cleaned copy."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing source text files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for cleaned corpus copy.",
    )
    parser.add_argument(
        "--glob",
        default="**/*.txt",
        help="Glob pattern to select files to clean (default: **/*.txt).",
    )
    parser.add_argument(
        "--metadata-scan-lines",
        type=int,
        default=400,
        help="Max leading lines to scan for metadata removal.",
    )
    parser.add_argument(
        "--max-toc-lines",
        type=int,
        default=300,
        help="Max length of a detected TOC block in lines.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete output directory if it already exists.",
    )
    parser.add_argument(
        "--copy-non-matching",
        action="store_true",
        help="Copy files that do not match --glob as-is.",
    )
    return parser.parse_args()


def _find_marker_index(lines: list[str], pattern: re.Pattern[str]) -> int:
    for idx, line in enumerate(lines):
        if pattern.search(line):
            return idx
    return -1


def strip_project_gutenberg_markers(lines: list[str]) -> tuple[list[str], int, int]:
    metadata_removed = 0
    footer_removed = 0

    start_idx = _find_marker_index(lines, START_MARKER_RE)
    if start_idx >= 0:
        metadata_removed += start_idx + 1
        lines = lines[start_idx + 1 :]

    end_idx = _find_marker_index(lines, END_MARKER_RE)
    if end_idx >= 0:
        footer_removed += len(lines) - end_idx
        lines = lines[:end_idx]

    return lines, metadata_removed, footer_removed


def strip_leading_metadata(lines: list[str], max_scan_lines: int) -> tuple[list[str], int]:
    if not lines:
        return lines, 0

    scan_limit = min(max_scan_lines, len(lines))
    metadata_hits = 0
    candidate_cutoff = 0

    for idx in range(scan_limit):
        stripped = lines[idx].strip()

        if not stripped:
            if metadata_hits > 0:
                candidate_cutoff = idx + 1
            continue

        if LEADING_METADATA_RE.search(stripped):
            metadata_hits += 1
            candidate_cutoff = idx + 1
            continue

        if metadata_hits >= 2:
            # Allow a few continuation lines inside metadata paragraphs.
            if len(stripped) < 120 and not stripped.endswith((".", "!", "?")):
                candidate_cutoff = idx + 1
                continue
            break

        # Stop early if we hit substantive prose without metadata hits.
        if len(stripped) > 40:
            break

    if metadata_hits >= 2 and candidate_cutoff > 0:
        return lines[candidate_cutoff:], candidate_cutoff

    return lines, 0


def looks_like_prose_line(text: str) -> bool:
    if len(text) < 70:
        return False
    words = re.findall(r"[A-Za-z]+", text)
    return len(words) >= 10


def looks_like_toc_entry(text: str) -> bool:
    if TOC_ENTRY_RE.search(text):
        return True
    if TOC_DOTTED_RE.search(text):
        return True
    if re.search(r"\s\d+\s*$", text) and len(text) < 90:
        return True
    return False


def strip_toc_blocks(lines: list[str], max_toc_lines: int) -> tuple[list[str], int]:
    if not lines:
        return lines, 0

    cleaned: list[str] = []
    removed = 0
    idx = 0
    total = len(lines)

    while idx < total:
        current = lines[idx].strip()
        if not TOC_HEADER_RE.match(current):
            cleaned.append(lines[idx])
            idx += 1
            continue

        start = idx
        j = idx + 1
        toc_like = 0
        prose_run = 0

        while j < total and (j - start) <= max_toc_lines:
            probe = lines[j].strip()

            if not probe:
                j += 1
                continue

            if looks_like_toc_entry(probe):
                toc_like += 1
                prose_run = 0
                j += 1
                continue

            if looks_like_prose_line(probe):
                prose_run += 1
                j += 1
                if prose_run >= 2:
                    break
                continue

            if len(probe) <= 60:
                toc_like += 1
                j += 1
                continue

            break

        # Keep prose lines that triggered block termination.
        block_end = j - prose_run if prose_run > 0 else j
        if toc_like >= 2 and block_end > start:
            removed += block_end - start
            idx = block_end
            continue

        cleaned.append(lines[idx])
        idx += 1

    return cleaned, removed


def strip_structural_lines(lines: list[str]) -> tuple[list[str], int]:
    kept: list[str] = []
    removed = 0
    for line in lines:
        if STRUCTURAL_LINE_RE.match(line.strip()):
            removed += 1
            continue
        kept.append(line)
    return kept, removed


def strip_inline_annotations(text: str) -> tuple[str, int]:
    updated, count_a = INLINE_BRACKET_ANNOTATION_RE.subn("", text)
    updated, count_b = INLINE_PAREN_ANNOTATION_RE.subn("", updated)
    return updated, count_a + count_b


def clean_whitespace(text: str) -> tuple[str, int]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    before_blank_runs = len(re.findall(r"\n{3,}", text))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if text:
        text += "\n"
    return text, before_blank_runs


def clean_file_text(raw_text: str, metadata_scan_lines: int, max_toc_lines: int) -> tuple[str, FileReport]:
    chars_before = len(raw_text)
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    lines, pg_header_removed, pg_footer_removed = strip_project_gutenberg_markers(lines)
    lines, leading_metadata_removed = strip_leading_metadata(lines, metadata_scan_lines)
    lines, toc_removed = strip_toc_blocks(lines, max_toc_lines)
    lines, structural_removed = strip_structural_lines(lines)

    text = "\n".join(lines)
    text, inline_removed = strip_inline_annotations(text)
    text, blank_reductions = clean_whitespace(text)

    chars_after = len(text)
    report = FileReport(
        relative_path="",
        chars_before=chars_before,
        chars_after=chars_after,
        metadata_lines_removed=pg_header_removed + leading_metadata_removed,
        gutenberg_footer_lines_removed=pg_footer_removed,
        toc_lines_removed=toc_removed,
        structural_lines_removed=structural_removed,
        inline_annotations_removed=inline_removed,
        blank_line_reductions=blank_reductions,
    )
    return text, report


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite-output to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_reports(output_dir: Path, per_file: list[FileReport]) -> None:
    totals = {
        "files_processed": len(per_file),
        "chars_before": sum(r.chars_before for r in per_file),
        "chars_after": sum(r.chars_after for r in per_file),
        "metadata_lines_removed": sum(r.metadata_lines_removed for r in per_file),
        "gutenberg_footer_lines_removed": sum(r.gutenberg_footer_lines_removed for r in per_file),
        "toc_lines_removed": sum(r.toc_lines_removed for r in per_file),
        "structural_lines_removed": sum(r.structural_lines_removed for r in per_file),
        "inline_annotations_removed": sum(r.inline_annotations_removed for r in per_file),
        "blank_line_reductions": sum(r.blank_line_reductions for r in per_file),
    }

    json_payload = {
        "totals": totals,
        "files": [asdict(r) for r in per_file],
    }
    (output_dir / "_cleaning_report.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    with (output_dir / "_cleaning_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "relative_path",
                "chars_before",
                "chars_after",
                "metadata_lines_removed",
                "gutenberg_footer_lines_removed",
                "toc_lines_removed",
                "structural_lines_removed",
                "inline_annotations_removed",
                "blank_line_reductions",
            ]
        )
        for row in per_file:
            writer.writerow(
                [
                    row.relative_path,
                    row.chars_before,
                    row.chars_after,
                    row.metadata_lines_removed,
                    row.gutenberg_footer_lines_removed,
                    row.toc_lines_removed,
                    row.structural_lines_removed,
                    row.inline_annotations_removed,
                    row.blank_line_reductions,
                ]
            )


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    ensure_output_dir(output_dir, overwrite=args.overwrite_output)

    matched_files = sorted(
        p for p in input_dir.glob(args.glob) if p.is_file()
    )

    reports: list[FileReport] = []
    for src in matched_files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        raw_text = src.read_text(encoding="utf-8", errors="ignore")
        cleaned_text, report = clean_file_text(
            raw_text=raw_text,
            metadata_scan_lines=args.metadata_scan_lines,
            max_toc_lines=args.max_toc_lines,
        )
        report.relative_path = str(rel).replace("\\", "/")
        dst.write_text(cleaned_text, encoding="utf-8")
        reports.append(report)

    if args.copy_non_matching:
        matched_set = {p.resolve() for p in matched_files}
        for src in input_dir.rglob("*"):
            if not src.is_file():
                continue
            if src.resolve() in matched_set:
                continue
            rel = src.relative_to(input_dir)
            dst = output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    write_reports(output_dir, reports)

    print(f"Processed files: {len(reports)}")
    print(f"Cleaned output: {output_dir}")
    print(f"Report JSON: {output_dir / '_cleaning_report.json'}")
    print(f"Report CSV: {output_dir / '_cleaning_report.csv'}")


if __name__ == "__main__":
    main()
