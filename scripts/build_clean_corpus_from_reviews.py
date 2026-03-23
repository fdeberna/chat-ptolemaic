#!/usr/bin/env python3
"""
Build a cleaned copy of corpus_general using existing review summaries.

Behavior:
1. Copy source corpus to a new output directory.
2. Remove files marked REMOVE_FILE in any provided review summary folder.
3. Remove files listed in a plain-text decision file.
4. For files marked REMOVE_SENTENCE, remove sentence_text rows from
   sentence_findings.csv (all rows by default, configurable).

This script does not modify the source corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_REVIEW_DIRS = (
    "data/corpus_general_heliocentric_review",
    "data/corpus_general_heliocentric_review_exact_only",
)

SUMMARIES_SUBDIR = Path("review_package") / "summaries"
FILE_SUMMARY_NAME = "file_summary.csv"
SENTENCE_FINDINGS_NAME = "sentence_findings.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a cleaned copy of corpus_general from review summaries."
    )
    parser.add_argument(
        "--source-dir",
        default="data/corpus_general",
        help="Source corpus directory to copy from.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/corpus_general_cleaned_from_reviews",
        help="Destination directory for the cleaned corpus copy.",
    )
    parser.add_argument(
        "--decision-file",
        default="data/decision.txt",
        help="Text file listing additional books (filenames) to remove.",
    )
    parser.add_argument(
        "--review-dir",
        action="append",
        dest="review_dirs",
        help=(
            "Review folder that contains review_package/summaries. "
            "Pass multiple times to combine reports."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="If set, delete and recreate output-dir when it already exists.",
    )
    parser.add_argument(
        "--sentence-selection",
        choices=("all_mentioned", "remove_only"),
        default="all_mentioned",
        help=(
            "For files with decision=REMOVE_SENTENCE, remove either all sentence_text entries "
            "from sentence_findings.csv (all_mentioned) or only sentence_action=REMOVE rows (remove_only)."
        ),
    )
    return parser.parse_args()


def normalize_filename(value: str) -> str:
    value = value.strip().replace("\\", "/")
    return value.split("/")[-1]


def summaries_dir_for(review_dir: Path) -> Path:
    return review_dir / SUMMARIES_SUBDIR


def resolve_review_dirs(source_dir: Path, explicit_review_dirs: list[str] | None) -> list[Path]:
    if explicit_review_dirs:
        return [Path(p) for p in explicit_review_dirs]

    source_name = source_dir.name
    parent = source_dir.parent if source_dir.parent != Path("") else Path(".")
    pattern = f"{source_name}_heliocentric_review*"
    discovered = sorted(
        p for p in parent.glob(pattern) if p.is_dir()
    )
    if discovered:
        return discovered

    return [Path(p) for p in DEFAULT_REVIEW_DIRS]


def parse_decision_file(decision_file: Path) -> set[str]:
    if not decision_file.exists():
        raise FileNotFoundError(f"Decision file not found: {decision_file}")

    extra_remove: set[str] = set()
    pattern = re.compile(r"([A-Za-z0-9_.-]+\.txt)\b", re.IGNORECASE)

    for raw_line in decision_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if match:
            extra_remove.add(match.group(1))

    return extra_remove


def load_file_decisions(review_dirs: list[Path]) -> tuple[set[str], set[str]]:
    remove_file_books: set[str] = set()
    remove_sentence_books: set[str] = set()

    for review_dir in review_dirs:
        file_summary = summaries_dir_for(review_dir) / FILE_SUMMARY_NAME
        if not file_summary.exists():
            raise FileNotFoundError(f"Missing summary file: {file_summary}")

        with file_summary.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Empty CSV header in {file_summary}")

            required = {"filename", "decision"}
            missing = required - set(reader.fieldnames)
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(f"{file_summary} missing required columns: {missing_str}")

            for row in reader:
                filename = normalize_filename(row.get("filename", ""))
                decision = row.get("decision", "").strip().upper()
                if not filename:
                    continue
                if decision == "REMOVE_FILE":
                    remove_file_books.add(filename)
                elif decision == "REMOVE_SENTENCE":
                    remove_sentence_books.add(filename)

    return remove_file_books, remove_sentence_books


def load_sentences_to_remove(
    review_dirs: list[Path],
    sentence_selection: str,
) -> dict[str, set[str]]:
    by_file: dict[str, set[str]] = defaultdict(set)

    for review_dir in review_dirs:
        sentence_findings = summaries_dir_for(review_dir) / SENTENCE_FINDINGS_NAME
        if not sentence_findings.exists():
            raise FileNotFoundError(f"Missing summary file: {sentence_findings}")

        with sentence_findings.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Empty CSV header in {sentence_findings}")

            required = {"filename", "decision", "sentence_text"}
            missing = required - set(reader.fieldnames)
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"{sentence_findings} missing required columns: {missing_str}"
                )

            for row in reader:
                filename = normalize_filename(row.get("filename", ""))
                decision = row.get("decision", "").strip().upper()
                action = row.get("sentence_action", "").strip().upper()
                sentence_text = row.get("sentence_text", "").strip()

                if not filename or not sentence_text:
                    continue
                if decision != "REMOVE_SENTENCE":
                    continue

                if sentence_selection == "remove_only":
                    if action == "REMOVE":
                        by_file[filename].add(sentence_text)
                else:
                    by_file[filename].add(sentence_text)

    return by_file


def copy_source_corpus(source_dir: Path, output_dir: Path, overwrite_output: bool) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    if output_dir.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite-output to replace it."
            )
        shutil.rmtree(output_dir)

    shutil.copytree(source_dir, output_dir)


def remove_books(output_dir: Path, books_to_remove: set[str]) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    missing: list[str] = []

    for filename in sorted(books_to_remove):
        target = output_dir / filename
        if target.exists() and target.is_file():
            target.unlink()
            removed.append(filename)
        else:
            missing.append(filename)

    return removed, missing


def remove_sentences_from_books(
    output_dir: Path,
    remove_sentence_books: set[str],
    sentences_to_remove: dict[str, set[str]],
    removed_books: set[str],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    stats: dict[str, dict[str, int]] = {}
    missing_files: list[str] = []

    for filename in sorted(remove_sentence_books):
        if filename in removed_books:
            continue

        target = output_dir / filename
        if not target.exists():
            missing_files.append(filename)
            continue
        if not target.is_file():
            missing_files.append(filename)
            continue

        text = target.read_text(encoding="utf-8", errors="ignore")
        updated = text

        sentences = sorted(sentences_to_remove.get(filename, set()), key=len, reverse=True)
        requested = len(sentences)
        found_occurrences = 0
        not_found = 0

        for sentence in sentences:
            count = updated.count(sentence)
            if count > 0:
                updated = updated.replace(sentence, "")
                found_occurrences += count
            else:
                not_found += 1

        if updated != text:
            updated = re.sub(r"[ \t]+\n", "\n", updated)
            updated = re.sub(r"\n{3,}", "\n\n", updated).strip()
            if updated:
                updated += "\n"
            target.write_text(updated, encoding="utf-8")

        stats[filename] = {
            "sentences_requested": requested,
            "sentences_not_found": not_found,
            "occurrences_removed": found_occurrences,
        }

    return stats, missing_files


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    decision_file = Path(args.decision_file)
    review_dirs = resolve_review_dirs(source_dir=source_dir, explicit_review_dirs=args.review_dirs)

    copy_source_corpus(source_dir=source_dir, output_dir=output_dir, overwrite_output=args.overwrite_output)

    remove_file_books, remove_sentence_books = load_file_decisions(review_dirs)
    extra_remove_books = parse_decision_file(decision_file)
    all_remove_books = remove_file_books | extra_remove_books

    removed_books, missing_removed_books = remove_books(output_dir, all_remove_books)

    sentences_to_remove = load_sentences_to_remove(
        review_dirs=review_dirs,
        sentence_selection=args.sentence_selection,
    )
    sentence_stats, missing_sentence_books = remove_sentences_from_books(
        output_dir=output_dir,
        remove_sentence_books=remove_sentence_books,
        sentences_to_remove=sentences_to_remove,
        removed_books=set(removed_books),
    )

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "review_dirs": [str(p) for p in review_dirs],
        "decision_file": str(decision_file),
        "sentence_selection": args.sentence_selection,
        "remove_file_books_from_reports": len(remove_file_books),
        "remove_sentence_books_from_reports": len(remove_sentence_books),
        "additional_remove_books_from_decision_file": len(extra_remove_books),
        "total_books_to_remove": len(all_remove_books),
        "books_removed": len(removed_books),
        "books_missing_in_output": missing_removed_books,
        "sentence_removal_targets": len(sentence_stats),
        "sentence_target_files_missing": missing_sentence_books,
        "sentence_removal_stats": sentence_stats,
    }

    summary_path = output_dir / "_cleanup_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print("Cleaned corpus copy created.")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print("Review dirs: " + ", ".join(str(p) for p in review_dirs))
    print(f"Removed books: {len(removed_books)} / {len(all_remove_books)}")
    print(f"Sentence-cleaned books: {len(sentence_stats)}")
    print(f"Summary report: {summary_path}")
    if missing_removed_books:
        print(f"Books requested for removal but not found: {len(missing_removed_books)}")
    if missing_sentence_books:
        print(f"REMOVE_SENTENCE files not found in output: {len(missing_sentence_books)}")


if __name__ == "__main__":
    main()
