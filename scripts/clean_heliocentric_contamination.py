#!/usr/bin/env python3
"""
Review-only corpus cleaner for heliocentric contamination in data/corpus_general.

It:
1. Parses exact/proximity contamination reports to gather candidate files.
2. Re-runs phrase/regex checks on each candidate and extracts matched sentences.
3. Classifies files as REMOVE_FILE / REMOVE_SENTENCE / KEEP.
4. Writes cleaned copies and quarantine copies to an output folder (no source deletion).
5. Emits CSV/JSON summaries plus a manual-review queue.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


EXACT_PHRASES: Sequence[str] = (
    "earth moves",
    "earth revolves",
    "earth orbits",
    "earth rotates",
    "earth's orbit",
    "earth's motion",
    "earth goes around",
    "earth circles",
    "earth travels around",
    "sun at the center",
    "sun in the center",
    "sun is the center",
    "center of the solar system",
    "heliocentric",
    "copernicus",
    "galileo",
    "kepler",
    "giordano bruno",
)

PROXIMITY_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"earth(\W+\w+){0,5}\W+moves", "earth [0-5 words] moves"),
    (r"earth(\W+\w+){0,5}\W+revolves", "earth [0-5 words] revolves"),
    (r"earth(\W+\w+){0,5}\W+orbits", "earth [0-5 words] orbits"),
    (r"earth(\W+\w+){0,5}\W+rotates", "earth [0-5 words] rotates"),
    (r"earth(\W+\w+){0,5}\W+circles", "earth [0-5 words] circles"),
    (r"earth(\W+\w+){0,5}\W+travels", "earth [0-5 words] travels"),
    (r"earth(\W+\w+){0,5}\W+swinging", "earth [0-5 words] swinging"),
    (r"earth(\W+\w+){0,5}\W+orbit", "earth [0-5 words] orbit"),
    (r"earth(\W+\w+){0,5}\W+motion", "earth [0-5 words] motion"),
    (r"moves(\W+\w+){0,5}\W+sun", "moves [0-5 words] sun"),
    (r"revolves(\W+\w+){0,5}\W+sun", "revolves [0-5 words] sun"),
    (r"orbits(\W+\w+){0,5}\W+sun", "orbits [0-5 words] sun"),
    (r"around(\W+\w+){0,5}\W+sun", "around [0-5 words] sun"),
    (r"sun(\W+\w+){0,5}\W+center", "sun [0-5 words] center"),
    (r"sun(\W+\w+){0,5}\W+middle", "sun [0-5 words] middle"),
)

LINE_FILE_MATCH = re.compile(r"^\s*\[(\d+)\s+matches\]\s+(.+?)\s*$")

# Sentence-level rules
RE_STRONG_DEFINITE = re.compile(
    r"(?is)("
    r"\bheliocentric\b|"
    r"\bearth(?:'s)?\s+(?:orbit|motion|revolution|rotation)\b.{0,140}\b(?:sun|around)\b|"
    r"\bearth\b.{0,140}\b(?:orbits?|revolves?|rotates?|circles?|travels?|goes around)\b.{0,140}\b(?:sun|around)\b|"
    r"\bsun\b.{0,80}\b(?:center|centre|middle)\b.{0,120}\b(?:solar system|universe|system)\b|"
    r"\bcenter of the solar system\b"
    r")"
)
RE_HISTORICAL = re.compile(r"(?i)\b(copernicus|galileo|kepler|giordano\s+bruno)\b")
RE_WEAK_HELIO = re.compile(
    r"(?i)\b("
    r"earth\s+(?:moves|revolves|orbits|rotates|circles|travels)|"
    r"earth(?:'s)?\s+(?:orbit|motion)|"
    r"(?:moves|revolves|orbits)\b.{0,50}\bsun\b|"
    r"around\b.{0,50}\bsun\b|"
    r"sun\b.{0,50}\b(?:center|centre|middle)\b"
    r")"
)
RE_FIGURATIVE = re.compile(
    r"(?i)\b("
    r"beneath\s+my\s+feet|"
    r"with\s+me|"
    r"as\s+if\s+the\s+earth\s+move|"
    r"the\s+earth\s+seemed\s+to\s+move|"
    r"the\s+earth\s+shook|"
    r"my\s+heart|"
    r"poem|"
    r"song"
    r")\b"
)
RE_FALSE_POSITIVE_TERMS = re.compile(
    r"(?i)\b(epicycle|epicycles|deferent|equant|eccentric|ptolemaic|geocentric)\b"
)

RE_STRONG_ASTRONOMY_CONTEXT = re.compile(
    r"(?i)\b(orbit|planet|solar\s+system|heliocentric|geocentric|astronom|sun|earth)\b"
)


@dataclass
class SentenceFinding:
    span_start: int
    span_end: int
    text: str
    triggers: Set[str] = field(default_factory=set)
    source_types: Set[str] = field(default_factory=set)
    label: str = ""
    action: str = ""  # REMOVE / KEEP / REVIEW
    reason: str = ""


@dataclass
class FileAssessment:
    filename: str
    path: Path
    decision: str  # REMOVE_FILE / REMOVE_SENTENCE / KEEP
    manual_review: bool
    total_matched_sentences: int
    removable_sentences: int
    review_sentences: int
    keep_sentences: int
    exact_report_matches: int
    proximity_report_matches: int
    sentence_findings: List[SentenceFinding] = field(default_factory=list)


def parse_report_candidates(
    report_path: Path,
    target_corpus: str,
    report_type: str,
) -> Dict[str, Dict[str, object]]:
    """
    Parse report blocks and return:
      filename -> {
          "report_matches": int,
          "patterns": Counter[str],
          "report_types": set[str],
      }
    """
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    in_target_corpus = False
    current_pattern = None
    out: Dict[str, Dict[str, object]] = {}

    for raw_line in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("CORPUS:"):
            corpus_name = line.split(":", 1)[1].strip()
            in_target_corpus = corpus_name == target_corpus
            current_pattern = None
            continue

        if not in_target_corpus:
            continue

        if line.startswith("Pattern:"):
            current_pattern = line.split(":", 1)[1].strip().strip("'")
            continue
        if line.startswith("Regex:") and current_pattern is None:
            current_pattern = line.split(":", 1)[1].strip()
            continue
        if line.startswith("---") or line.startswith("Summary for"):
            current_pattern = None
            continue

        match = LINE_FILE_MATCH.match(raw_line)
        if not match:
            continue

        count = int(match.group(1))
        filename = match.group(2).strip()
        info = out.setdefault(
            filename,
            {"report_matches": 0, "patterns": Counter(), "report_types": set()},
        )
        info["report_matches"] = int(info["report_matches"]) + count
        if current_pattern:
            cast_counter: Counter = info["patterns"]  # type: ignore[assignment]
            cast_counter[current_pattern] += count
        cast_types: Set[str] = info["report_types"]  # type: ignore[assignment]
        cast_types.add(report_type)

    return out


def build_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    pattern = re.compile(r"[^\n.!?]+(?:[.!?]+(?=\s|$)|$)")
    for line_match in re.finditer(r".*?(?:\n|$)", text):
        line = line_match.group(0)
        if not line.strip():
            continue
        line_start = line_match.start()
        for sent_match in pattern.finditer(line):
            segment = sent_match.group(0)
            if segment.strip():
                spans.append((line_start + sent_match.start(), line_start + sent_match.end()))
    return spans


def locate_sentence_for_index(
    spans: Sequence[Tuple[int, int]], index: int, text: str
) -> Tuple[int, int]:
    for start, end in spans:
        if start <= index < end:
            return start, end

    line_start = text.rfind("\n", 0, index)
    line_end = text.find("\n", index)
    if line_start < 0:
        line_start = 0
    else:
        line_start += 1
    if line_end < 0:
        line_end = len(text)
    return line_start, line_end


def classify_sentence(sentence_text: str) -> Tuple[str, str, str]:
    """
    Returns (label, action, reason)
    label in {DEFINITE,HISTORICAL,FIGURATIVE,FALSE_POSITIVE,BORDERLINE}
    action in {REMOVE,KEEP,REVIEW}
    """
    sentence = sentence_text.strip()
    strong = bool(RE_STRONG_DEFINITE.search(sentence))
    historical = bool(RE_HISTORICAL.search(sentence))
    weak = bool(RE_WEAK_HELIO.search(sentence))
    figurative = bool(RE_FIGURATIVE.search(sentence))
    false_pos = bool(RE_FALSE_POSITIVE_TERMS.search(sentence))
    astro_context = bool(RE_STRONG_ASTRONOMY_CONTEXT.search(sentence))

    if figurative and not (strong or historical):
        return ("FIGURATIVE", "KEEP", "Likely figurative/non-astronomical usage")
    if false_pos and not (strong or historical):
        return ("FALSE_POSITIVE", "KEEP", "Likely geocentric/technical false positive")
    if historical:
        return ("HISTORICAL", "REMOVE", "Historical heliocentric reference")
    if strong:
        return ("DEFINITE", "REMOVE", "Direct heliocentric statement")
    if weak and astro_context:
        return ("BORDERLINE", "REVIEW", "Weak heliocentric signal; needs manual review")
    if weak:
        return ("FIGURATIVE", "KEEP", "Weak pattern without clear astronomy context")
    return ("FALSE_POSITIVE", "KEEP", "Pattern overlap without contamination signal")


def extract_findings_for_file(file_path: Path) -> List[SentenceFinding]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    sentence_spans = build_sentence_spans(text)

    findings: Dict[Tuple[int, int], SentenceFinding] = {}

    for phrase in EXACT_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        for match in pattern.finditer(text):
            s_start, s_end = locate_sentence_for_index(sentence_spans, match.start(), text)
            key = (s_start, s_end)
            entry = findings.get(key)
            if entry is None:
                entry = SentenceFinding(
                    span_start=s_start,
                    span_end=s_end,
                    text=text[s_start:s_end].strip(),
                )
                findings[key] = entry
            entry.triggers.add(phrase)
            entry.source_types.add("exact")

    for regex, description in PROXIMITY_PATTERNS:
        pattern = re.compile(regex, re.IGNORECASE)
        for match in pattern.finditer(text):
            s_start, s_end = locate_sentence_for_index(sentence_spans, match.start(), text)
            key = (s_start, s_end)
            entry = findings.get(key)
            if entry is None:
                entry = SentenceFinding(
                    span_start=s_start,
                    span_end=s_end,
                    text=text[s_start:s_end].strip(),
                )
                findings[key] = entry
            entry.triggers.add(description)
            entry.source_types.add("proximity")

    ordered = sorted(findings.values(), key=lambda f: (f.span_start, f.span_end))
    for finding in ordered:
        label, action, reason = classify_sentence(finding.text)
        finding.label = label
        finding.action = action
        finding.reason = reason
    return ordered


def clean_text_remove_spans(text: str, spans_to_remove: Sequence[Tuple[int, int]]) -> str:
    if not spans_to_remove:
        return text

    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans_to_remove):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    cleaned = "".join(parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def assess_file(
    file_path: Path,
    exact_report_matches: int,
    proximity_report_matches: int,
    remove_file_threshold: int,
    force_remove_if_science_heavy: bool,
) -> FileAssessment:
    findings = extract_findings_for_file(file_path)
    remove_count = sum(1 for f in findings if f.action == "REMOVE")
    review_count = sum(1 for f in findings if f.action == "REVIEW")
    keep_count = sum(1 for f in findings if f.action == "KEEP")

    # A file is "science-heavy contaminated" when several matches are present
    # and most matched sentences are actionable removes.
    science_heavy = (
        len(findings) >= remove_file_threshold
        and remove_count >= max(3, int(0.6 * len(findings)))
    )

    if remove_count >= remove_file_threshold:
        decision = "REMOVE_FILE"
    elif force_remove_if_science_heavy and science_heavy:
        decision = "REMOVE_FILE"
    elif remove_count > 0:
        decision = "REMOVE_SENTENCE"
    else:
        decision = "KEEP"

    manual_review = review_count > 0
    if decision == "REMOVE_FILE" and remove_count < remove_file_threshold and review_count > 0:
        manual_review = True

    return FileAssessment(
        filename=file_path.name,
        path=file_path,
        decision=decision,
        manual_review=manual_review,
        total_matched_sentences=len(findings),
        removable_sentences=remove_count,
        review_sentences=review_count,
        keep_sentences=keep_count,
        exact_report_matches=exact_report_matches,
        proximity_report_matches=proximity_report_matches,
        sentence_findings=findings,
    )


def write_outputs(
    assessments: Sequence[FileAssessment],
    corpus_dir: Path,
    output_dir: Path,
    write_clean_files: bool,
) -> None:
    review_dir = output_dir / "review_package"
    cleaned_dir = review_dir / "cleaned_corpus_general"
    quarantine_dir = review_dir / "quarantine_candidates"
    summaries_dir = review_dir / "summaries"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    file_summary_csv = summaries_dir / "file_summary.csv"
    sentence_summary_csv = summaries_dir / "sentence_findings.csv"
    manual_review_csv = summaries_dir / "manual_review_queue.csv"
    triage_csv = summaries_dir / "triage_buckets.csv"
    quick_summary_md = summaries_dir / "quick_summary.md"
    decisions_json = summaries_dir / "decisions.json"
    run_summary_json = summaries_dir / "run_summary.json"

    with file_summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "decision",
                "manual_review",
                "total_matched_sentences",
                "removable_sentences",
                "review_sentences",
                "keep_sentences",
                "exact_report_matches",
                "proximity_report_matches",
                "path",
            ]
        )
        for a in assessments:
            writer.writerow(
                [
                    a.filename,
                    a.decision,
                    str(a.manual_review).lower(),
                    a.total_matched_sentences,
                    a.removable_sentences,
                    a.review_sentences,
                    a.keep_sentences,
                    a.exact_report_matches,
                    a.proximity_report_matches,
                    str(a.path),
                ]
            )

    with sentence_summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "decision",
                "manual_review",
                "sentence_label",
                "sentence_action",
                "reason",
                "triggers",
                "sentence_text",
            ]
        )
        for a in assessments:
            for finding in a.sentence_findings:
                writer.writerow(
                    [
                        a.filename,
                        a.decision,
                        str(a.manual_review).lower(),
                        finding.label,
                        finding.action,
                        finding.reason,
                        " | ".join(sorted(finding.triggers)),
                        finding.text,
                    ]
                )

    with manual_review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "decision",
                "removable_sentences",
                "review_sentences",
                "path",
            ]
        )
        for a in assessments:
            if a.manual_review:
                writer.writerow(
                    [
                        a.filename,
                        a.decision,
                        a.removable_sentences,
                        a.review_sentences,
                        str(a.path),
                    ]
                )

    with triage_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "triage_bucket",
                "decision",
                "manual_review",
                "removable_sentences",
                "review_sentences",
                "path",
            ]
        )
        for a in assessments:
            if a.decision == "REMOVE_FILE":
                bucket = "AUTO_REMOVE"
            elif a.decision == "REMOVE_SENTENCE" and a.removable_sentences <= 2 and not a.manual_review:
                bucket = "SINGLE_SENTENCE"
            elif a.manual_review:
                bucket = "MANUAL_REVIEW"
            elif a.decision == "REMOVE_SENTENCE":
                bucket = "REMOVE_SENTENCE_BATCH"
            else:
                bucket = "KEEP"
            writer.writerow(
                [
                    a.filename,
                    bucket,
                    a.decision,
                    str(a.manual_review).lower(),
                    a.removable_sentences,
                    a.review_sentences,
                    str(a.path),
                ]
            )

    payload = []
    for a in assessments:
        payload.append(
            {
                "filename": a.filename,
                "path": str(a.path),
                "decision": a.decision,
                "manual_review": a.manual_review,
                "total_matched_sentences": a.total_matched_sentences,
                "removable_sentences": a.removable_sentences,
                "review_sentences": a.review_sentences,
                "keep_sentences": a.keep_sentences,
                "exact_report_matches": a.exact_report_matches,
                "proximity_report_matches": a.proximity_report_matches,
                "findings": [
                    {
                        "sentence": s.text,
                        "label": s.label,
                        "action": s.action,
                        "reason": s.reason,
                        "triggers": sorted(s.triggers),
                    }
                    for s in a.sentence_findings
                ],
            }
        )
    decisions_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    if write_clean_files:
        for a in assessments:
            dst_cleaned = cleaned_dir / a.filename
            dst_quarantine = quarantine_dir / a.filename
            original_text = a.path.read_text(encoding="utf-8", errors="ignore")

            if a.decision == "REMOVE_FILE":
                shutil.copy2(a.path, dst_quarantine)
                shutil.copy2(a.path, dst_cleaned)
                continue

            if a.decision == "REMOVE_SENTENCE":
                remove_spans = [
                    (f.span_start, f.span_end)
                    for f in a.sentence_findings
                    if f.action == "REMOVE"
                ]
                cleaned_text = clean_text_remove_spans(original_text, remove_spans)
                dst_cleaned.write_text(cleaned_text, encoding="utf-8")
                continue

            shutil.copy2(a.path, dst_cleaned)

    decision_counts = Counter(a.decision for a in assessments)
    triage_counts = Counter()
    for a in assessments:
        if a.decision == "REMOVE_FILE":
            triage_counts["AUTO_REMOVE"] += 1
        elif a.decision == "REMOVE_SENTENCE" and a.removable_sentences <= 2 and not a.manual_review:
            triage_counts["SINGLE_SENTENCE"] += 1
        elif a.manual_review:
            triage_counts["MANUAL_REVIEW"] += 1
        elif a.decision == "REMOVE_SENTENCE":
            triage_counts["REMOVE_SENTENCE_BATCH"] += 1
        else:
            triage_counts["KEEP"] += 1

    top_remove = sorted(
        [a for a in assessments if a.removable_sentences > 0],
        key=lambda x: x.removable_sentences,
        reverse=True,
    )[:40]
    quick_lines = [
        "# Heliocentric Contamination Quick Summary",
        "",
        f"- Assessed files: {len(assessments)}",
        "- Decision counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(decision_counts.items())),
        "- Triage buckets: " + ", ".join(f"{k}={v}" for k, v in sorted(triage_counts.items())),
        f"- Manual review files: {sum(1 for a in assessments if a.manual_review)}",
        f"- Total removable sentences: {sum(a.removable_sentences for a in assessments)}",
        "",
        "## Auto Remove (REMOVE_FILE)",
    ]
    for a in sorted((x for x in assessments if x.decision == "REMOVE_FILE"), key=lambda x: x.filename):
        quick_lines.append(
            f"- {a.filename}: removable={a.removable_sentences}, review={a.review_sentences}"
        )
    quick_lines.append("")
    quick_lines.append("## Single Sentence Candidates")
    for a in sorted(
        (
            x
            for x in assessments
            if x.decision == "REMOVE_SENTENCE" and x.removable_sentences <= 2 and not x.manual_review
        ),
        key=lambda x: x.filename,
    ):
        quick_lines.append(
            f"- {a.filename}: removable={a.removable_sentences}, review={a.review_sentences}"
        )
    quick_lines.append("")
    quick_lines.append("## Manual Review Queue")
    for a in sorted((x for x in assessments if x.manual_review), key=lambda x: x.filename):
        quick_lines.append(
            f"- {a.filename}: decision={a.decision}, removable={a.removable_sentences}, review={a.review_sentences}"
        )
    quick_lines.append("")
    quick_lines.append("## Top Files By Removable Sentence Count")
    for a in top_remove:
        quick_lines.append(
            f"- {a.filename}: decision={a.decision}, removable={a.removable_sentences}, review={a.review_sentences}"
        )
    quick_summary_md.write_text("\n".join(quick_lines) + "\n", encoding="utf-8")

    summary = {
        "assessed_files": len(assessments),
        "decision_counts": dict(decision_counts),
        "triage_counts": dict(triage_counts),
        "manual_review_files": sum(1 for a in assessments if a.manual_review),
        "total_removed_sentences": sum(a.removable_sentences for a in assessments),
        "paths": {
            "review_root": str(review_dir),
            "file_summary_csv": str(file_summary_csv),
            "sentence_summary_csv": str(sentence_summary_csv),
            "manual_review_csv": str(manual_review_csv),
            "triage_csv": str(triage_csv),
            "quick_summary_md": str(quick_summary_md),
            "decisions_json": str(decisions_json),
        },
    }
    run_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def gather_candidates(
    exact_report: Path,
    proximity_report: Path,
    corpus_name: str,
    report_scope: str,
) -> Dict[str, Dict[str, object]]:
    exact_info = (
        parse_report_candidates(exact_report, corpus_name, report_type="exact")
        if report_scope in {"both", "exact"}
        else {}
    )
    prox_info = (
        parse_report_candidates(proximity_report, corpus_name, report_type="proximity")
        if report_scope in {"both", "proximity"}
        else {}
    )

    files = set(exact_info.keys()) | set(prox_info.keys())
    combined: Dict[str, Dict[str, object]] = {}
    for fn in files:
        ex = exact_info.get(fn, {"report_matches": 0, "patterns": Counter(), "report_types": set()})
        pr = prox_info.get(fn, {"report_matches": 0, "patterns": Counter(), "report_types": set()})
        combined[fn] = {
            "exact_report_matches": int(ex["report_matches"]),
            "proximity_report_matches": int(pr["report_matches"]),
            "patterns": Counter(ex["patterns"]) + Counter(pr["patterns"]),
            "report_types": set(ex["report_types"]) | set(pr["report_types"]),
        }
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review-only heliocentric contamination cleaner for corpus_general."
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/corpus_general",
        help="Path to source corpus directory.",
    )
    parser.add_argument(
        "--exact-report",
        default="exact_contamination.txt",
        help="Path to exact contamination report.",
    )
    parser.add_argument(
        "--proximity-report",
        default="proximity_contamination.txt",
        help="Path to proximity contamination report.",
    )
    parser.add_argument(
        "--corpus-name-in-report",
        default="corpus_general",
        help="Section name used in report files (after 'CORPUS:').",
    )
    parser.add_argument(
        "--report-scope",
        choices=("both", "exact", "proximity"),
        default="both",
        help="Which report candidates to include.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/corpus_general_heliocentric_review",
        help="Output folder for review package and cleaned copies.",
    )
    parser.add_argument(
        "--remove-file-threshold",
        type=int,
        default=5,
        help="Minimum removable matched sentences to auto-label REMOVE_FILE.",
    )
    parser.add_argument(
        "--no-science-heavy-force-remove",
        action="store_true",
        help="Disable extra REMOVE_FILE promotion for science-heavy contamination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and write summaries only (no cleaned/quarantine files).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_dir = Path(args.corpus_dir)
    exact_report = Path(args.exact_report)
    proximity_report = Path(args.proximity_report)
    output_dir = Path(args.output_dir)

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    candidates = gather_candidates(
        exact_report,
        proximity_report,
        args.corpus_name_in_report,
        args.report_scope,
    )
    if not candidates:
        raise RuntimeError(
            "No candidates were parsed from reports for the requested corpus section."
        )

    missing: List[str] = []
    assessments: List[FileAssessment] = []
    for filename in sorted(candidates.keys()):
        file_path = corpus_dir / filename
        if not file_path.exists():
            missing.append(filename)
            continue

        meta = candidates[filename]
        assessments.append(
            assess_file(
                file_path=file_path,
                exact_report_matches=int(meta["exact_report_matches"]),
                proximity_report_matches=int(meta["proximity_report_matches"]),
                remove_file_threshold=args.remove_file_threshold,
                force_remove_if_science_heavy=not args.no_science_heavy_force_remove,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        assessments=assessments,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        write_clean_files=not args.dry_run,
    )

    decision_counts = Counter(a.decision for a in assessments)
    manual_review = sum(1 for a in assessments if a.manual_review)
    total_remove_sentences = sum(a.removable_sentences for a in assessments)

    print("Heliocentric contamination review complete.")
    print(f"Assessed files: {len(assessments)}")
    print(
        "Decisions: "
        + ", ".join(f"{k}={v}" for k, v in sorted(decision_counts.items()))
    )
    print(f"Manual review queue: {manual_review}")
    print(f"Total removable sentences: {total_remove_sentences}")
    print(f"Output: {(output_dir / 'review_package').resolve()}")
    if missing:
        print(f"Missing files referenced by reports: {len(missing)}")


if __name__ == "__main__":
    main()
