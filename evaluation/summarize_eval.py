import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def safe_div(n, d):
    return n / d if d else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize judged evaluation JSONL outputs into CSV and Markdown reports."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to the judged JSONL file.")
    parser.add_argument("--output-csv", required=True, help="Path to the summary CSV output.")
    parser.add_argument(
        "--output-markdown", required=True, help="Path to the summary Markdown report."
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=1,
        help="Minimum quality_score threshold for quality-filtered metrics.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include records with judge_error or missing judge_result in total counts where possible.",
    )
    return parser.parse_args()


def warn(message):
    print(f"Warning: {message}")


def model_key(record):
    value = record.get("model_id")
    return str(value) if value not in (None, "") else "UNKNOWN_MODEL"


def category_key(record):
    value = record.get("category")
    return str(value) if value not in (None, "") else "UNKNOWN_CATEGORY"


def prompt_key(record):
    value = record.get("prompt_id")
    return str(value) if value not in (None, "") else "UNKNOWN_PROMPT"


def prompt_text_value(record):
    value = record.get("prompt_text")
    return value if isinstance(value, str) else ""


def has_judge_error(record):
    return bool(record.get("judge_error"))


def get_judge_result(record):
    result = record.get("judge_result")
    return result if isinstance(result, dict) else None


def is_usable_judged_record(record):
    return not has_judge_error(record) and get_judge_result(record) is not None


def load_jsonl(input_path, include_errors):
    records = []
    counts = {
        "total_lines": 0,
        "parsed_rows": 0,
        "malformed_rows": 0,
        "excluded_rows": 0,
        "skipped_rows": 0,
        "judge_error_rows": 0,
        "judged_rows": 0,
    }

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            counts["total_lines"] += 1
            stripped = line.strip()
            if not stripped:
                counts["malformed_rows"] += 1
                warn(f"skipping empty line {line_number}")
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                counts["malformed_rows"] += 1
                warn(f"skipping malformed JSON at line {line_number}: {exc}")
                continue

            if not isinstance(record, dict):
                counts["malformed_rows"] += 1
                warn(f"skipping non-object JSON at line {line_number}")
                continue

            counts["parsed_rows"] += 1

            if has_judge_error(record):
                counts["judge_error_rows"] += 1

            if is_usable_judged_record(record):
                counts["judged_rows"] += 1

            should_include = include_errors or is_usable_judged_record(record)
            if should_include:
                records.append(record)
            else:
                counts["excluded_rows"] += 1

    counts["skipped_rows"] = counts["malformed_rows"] + counts["excluded_rows"]
    return records, counts


def summarize_group(records, min_quality):
    summary = {
        "n_total": len(records),
        "n_judged": 0,
        "n_quality_ge_min": 0,
        "n_quality_2": 0,
        "n_explicit_earth_motion": 0,
        "n_explicit_earth_motion_quality_ge_min": 0,
        "n_proto_heliocentric": 0,
        "n_proto_heliocentric_quality_ge_min": 0,
        "n_geocentric": 0,
        "n_ambiguous": 0,
        "n_no_relevant_claim": 0,
    }

    for record in records:
        if not is_usable_judged_record(record):
            continue

        judge_result = get_judge_result(record)
        summary["n_judged"] += 1

        quality_score = judge_result.get("quality_score")
        stance_label = judge_result.get("stance_label")
        explicit_earth_motion_label = judge_result.get("explicit_earth_motion_label")
        proto_heliocentric_label = judge_result.get("proto_heliocentric_label")

        quality_is_number = isinstance(quality_score, int)
        quality_ge_min = quality_is_number and quality_score >= min_quality

        if quality_ge_min:
            summary["n_quality_ge_min"] += 1
        if quality_score == 2:
            summary["n_quality_2"] += 1

        if explicit_earth_motion_label == 1:
            summary["n_explicit_earth_motion"] += 1
            if quality_ge_min:
                summary["n_explicit_earth_motion_quality_ge_min"] += 1

        if proto_heliocentric_label == 1:
            summary["n_proto_heliocentric"] += 1
            if quality_ge_min:
                summary["n_proto_heliocentric_quality_ge_min"] += 1

        if stance_label == "geocentric":
            summary["n_geocentric"] += 1
        elif stance_label == "ambiguous":
            summary["n_ambiguous"] += 1
        elif stance_label == "no_relevant_claim":
            summary["n_no_relevant_claim"] += 1

    summary["rate_explicit_earth_motion_overall"] = safe_div(
        summary["n_explicit_earth_motion"], summary["n_judged"]
    )
    summary["rate_explicit_earth_motion_given_quality_ge_min"] = safe_div(
        summary["n_explicit_earth_motion_quality_ge_min"], summary["n_quality_ge_min"]
    )
    summary["rate_proto_heliocentric_overall"] = safe_div(
        summary["n_proto_heliocentric"], summary["n_judged"]
    )
    summary["rate_proto_heliocentric_given_quality_ge_min"] = safe_div(
        summary["n_proto_heliocentric_quality_ge_min"], summary["n_quality_ge_min"]
    )
    summary["rate_quality_ge_min"] = safe_div(
        summary["n_quality_ge_min"], summary["n_judged"]
    )
    summary["rate_quality_2"] = safe_div(summary["n_quality_2"], summary["n_judged"])
    summary["rate_geocentric_overall"] = safe_div(
        summary["n_geocentric"], summary["n_judged"]
    )
    summary["rate_ambiguous_overall"] = safe_div(
        summary["n_ambiguous"], summary["n_judged"]
    )
    return summary


def build_prompt_level_stats(records, min_quality):
    grouped = defaultdict(list)
    for record in records:
        if not is_usable_judged_record(record):
            continue

        key = (model_key(record), category_key(record), prompt_key(record))
        grouped[key].append(record)

    prompt_rows = []
    for (model_id, category, prompt_id), prompt_records in grouped.items():
        quality_scores = []
        explicit_earth_motion_count = 0
        proto_heliocentric_count = 0
        prompt_text = ""

        for record in prompt_records:
            judge_result = get_judge_result(record)
            prompt_text = prompt_text or prompt_text_value(record)

            quality_score = judge_result.get("quality_score")
            if isinstance(quality_score, int):
                quality_scores.append(quality_score)

            if judge_result.get("explicit_earth_motion_label") == 1:
                explicit_earth_motion_count += 1

            if judge_result.get("proto_heliocentric_label") == 1:
                proto_heliocentric_count += 1

        prompt_rows.append(
            {
                "model_id": model_id,
                "category": category,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "n_judged": len(prompt_records),
                "avg_quality_score": (
                    statistics.mean(quality_scores) if quality_scores else 0.0
                ),
                "explicit_earth_motion_rate": safe_div(
                    explicit_earth_motion_count, len(prompt_records)
                ),
                "proto_heliocentric_rate": safe_div(
                    proto_heliocentric_count, len(prompt_records)
                ),
            }
        )

    prompt_rows.sort(
        key=lambda row: (
            row["model_id"],
            row["category"],
            row["prompt_id"],
        )
    )
    return prompt_rows


def build_summary_rows(records, min_quality):
    overall_groups = defaultdict(list)
    category_groups = defaultdict(list)

    for record in records:
        overall_groups[model_key(record)].append(record)
        category_groups[(model_key(record), category_key(record))].append(record)

    rows = []
    for model_id in sorted(overall_groups):
        row = {
            "summary_level": "overall",
            "model_id": model_id,
            "category": "ALL",
        }
        row.update(summarize_group(overall_groups[model_id], min_quality))
        rows.append(row)

    for model_id, category in sorted(category_groups):
        row = {
            "summary_level": "category",
            "model_id": model_id,
            "category": category,
        }
        row.update(summarize_group(category_groups[(model_id, category)], min_quality))
        rows.append(row)

    return rows


def csv_fieldnames():
    return [
        "summary_level",
        "model_id",
        "category",
        "n_total",
        "n_judged",
        "n_quality_ge_min",
        "n_quality_2",
        "n_explicit_earth_motion",
        "n_explicit_earth_motion_quality_ge_min",
        "n_proto_heliocentric",
        "n_proto_heliocentric_quality_ge_min",
        "n_geocentric",
        "n_ambiguous",
        "n_no_relevant_claim",
        "rate_explicit_earth_motion_overall",
        "rate_explicit_earth_motion_given_quality_ge_min",
        "rate_proto_heliocentric_overall",
        "rate_proto_heliocentric_given_quality_ge_min",
        "rate_quality_ge_min",
        "rate_quality_2",
        "rate_geocentric_overall",
        "rate_ambiguous_overall",
    ]


def write_csv(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames())
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for field in (
                "rate_explicit_earth_motion_overall",
                "rate_explicit_earth_motion_given_quality_ge_min",
                "rate_proto_heliocentric_overall",
                "rate_proto_heliocentric_given_quality_ge_min",
                "rate_quality_ge_min",
                "rate_quality_2",
                "rate_geocentric_overall",
                "rate_ambiguous_overall",
            ):
                csv_row[field] = f"{csv_row[field]:.6f}"
            writer.writerow(csv_row)


def rate_pct(value):
    return f"{value * 100:.1f}%"


def build_markdown_table(rows, include_category, min_quality):
    if include_category:
        headers = [
            "Model",
            "Category",
            "n_total",
            "n_judged",
            f"n_q>={min_quality}",
            "n_q2",
            "Explicit Earth-motion",
            f"Explicit Earth-motion q>={min_quality}",
            "Proto-heliocentric suggestion",
            f"Proto-heliocentric suggestion q>={min_quality}",
            "n_geo",
            "n_ambig",
            "n_none",
            "Explicit %",
            f"Explicit q>={min_quality} %",
            "Proto %",
            f"Proto q>={min_quality} %",
            "q>=min%",
            "q2%",
            "geo%",
            "ambig%",
        ]
    else:
        headers = [
            "Model",
            "n_total",
            "n_judged",
            f"n_q>={min_quality}",
            "n_q2",
            "Explicit Earth-motion",
            f"Explicit Earth-motion q>={min_quality}",
            "Proto-heliocentric suggestion",
            f"Proto-heliocentric suggestion q>={min_quality}",
            "n_geo",
            "n_ambig",
            "n_none",
            "Explicit %",
            f"Explicit q>={min_quality} %",
            "Proto %",
            f"Proto q>={min_quality} %",
            "q>=min%",
            "q2%",
            "geo%",
            "ambig%",
        ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [row["model_id"]]
        if include_category:
            values.append(row["category"])
        values.extend(
            [
                str(row["n_total"]),
                str(row["n_judged"]),
                str(row["n_quality_ge_min"]),
                str(row["n_quality_2"]),
                str(row["n_explicit_earth_motion"]),
                str(row["n_explicit_earth_motion_quality_ge_min"]),
                str(row["n_proto_heliocentric"]),
                str(row["n_proto_heliocentric_quality_ge_min"]),
                str(row["n_geocentric"]),
                str(row["n_ambiguous"]),
                str(row["n_no_relevant_claim"]),
                rate_pct(row["rate_explicit_earth_motion_overall"]),
                rate_pct(row["rate_explicit_earth_motion_given_quality_ge_min"]),
                rate_pct(row["rate_proto_heliocentric_overall"]),
                rate_pct(row["rate_proto_heliocentric_given_quality_ge_min"]),
                rate_pct(row["rate_quality_ge_min"]),
                rate_pct(row["rate_quality_2"]),
                rate_pct(row["rate_geocentric_overall"]),
                rate_pct(row["rate_ambiguous_overall"]),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def prompt_preview_text(text):
    preview = text.replace("\n", " ").strip()
    if len(preview) > 80:
        preview = preview[:77] + "..."
    return preview.replace("|", "\\|")


def build_prompt_sections(prompt_rows, metric_key, heading_label):
    by_model = defaultdict(list)
    for row in prompt_rows:
        by_model[row["model_id"]].append(row)

    sections = []
    for model_id in sorted(by_model):
        sections.append(f"### Model `{model_id}`")
        sections.append(
            f"| Prompt ID | Category | n_judged | {heading_label} rate | avg_quality_score | Prompt |"
        )
        sections.append("| --- | --- | --- | --- | --- | --- |")

        top_rows = sorted(
            by_model[model_id],
            key=lambda row: (
                -row[metric_key],
                -row["avg_quality_score"],
                -row["n_judged"],
                row["category"],
                row["prompt_id"],
            ),
        )[:10]

        for row in top_rows:
            sections.append(
                "| {prompt_id} | {category} | {n_judged} | {metric_rate} | "
                "{avg_quality_score:.2f} | {prompt_preview} |".format(
                    prompt_id=row["prompt_id"],
                    category=row["category"],
                    n_judged=row["n_judged"],
                    metric_rate=rate_pct(row[metric_key]),
                    avg_quality_score=row["avg_quality_score"],
                    prompt_preview=prompt_preview_text(row["prompt_text"]),
                )
            )

        if not top_rows:
            sections.append("")
            sections.append("No judged prompts available.")

        sections.append("")

    return "\n".join(sections).strip()


def write_markdown(output_path, input_path, min_quality, rows, prompt_rows, counts):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    overall_rows = [row for row in rows if row["summary_level"] == "overall"]
    category_rows = [row for row in rows if row["summary_level"] == "category"]

    content = [
        "# Evaluation Summary Report",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Input file: `{input_path.name}`",
        f"- Min quality: `{min_quality}`",
        "",
        "## Interpretation Note",
        "",
        "- Explicit Earth-motion is the strict metric.",
        "- Proto-heliocentric suggestion is the broader metric that includes serious but unresolved Earth-motion proposals.",
        "",
        "## Overall Comparison",
        "",
        build_markdown_table(overall_rows, include_category=False, min_quality=min_quality),
        "",
        "## Category Comparison",
        "",
        build_markdown_table(category_rows, include_category=True, min_quality=min_quality),
        "",
        "## Top Explicit Earth-motion Prompts Per Model",
        "",
        build_prompt_sections(
            prompt_rows,
            metric_key="explicit_earth_motion_rate",
            heading_label="Explicit Earth-motion",
        ),
        "",
        "## Top Proto-heliocentric Suggestion Prompts Per Model",
        "",
        build_prompt_sections(
            prompt_rows,
            metric_key="proto_heliocentric_rate",
            heading_label="Proto-heliocentric suggestion",
        ),
        "",
        "## Notes",
        "",
        f"- Total judged rows: `{counts['judged_rows']}`",
        f"- Total skipped rows: `{counts['skipped_rows']}`",
        f"- Total rows with judge_error: `{counts['judge_error_rows']}`",
        "",
    ]

    output_path.write_text("\n".join(content), encoding="utf-8")


def main():
    args = parse_args()

    input_path = Path(args.input_jsonl)
    output_csv_path = Path(args.output_csv)
    output_markdown_path = Path(args.output_markdown)

    if not input_path.exists():
        raise SystemExit(f"Input JSONL not found: {input_path}")

    records, counts = load_jsonl(input_path, args.include_errors)
    summary_rows = build_summary_rows(records, args.min_quality)
    prompt_rows = build_prompt_level_stats(records, args.min_quality)

    write_csv(output_csv_path, summary_rows)
    write_markdown(
        output_markdown_path,
        input_path,
        args.min_quality,
        summary_rows,
        prompt_rows,
        counts,
    )


if __name__ == "__main__":
    main()
