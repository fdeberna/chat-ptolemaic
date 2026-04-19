import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


TEST_METRICS = [
    ("earth_motion_mention_rate", "earth_motion_mention_rate"),
    ("explicit_earth_motion_rate", "explicit_earth_motion_rate"),
    ("proto_heliocentric_rate", "proto_heliocentric_rate"),
    ("geocentric_rate", "geocentric_rate"),
    ("ambiguous_rate", "ambiguous_rate"),
]

HEDGE_PHRASES = [
    "some say",
    "it may be understood",
    "it seems",
    "according to",
    "if one grants",
    "it may be supposed",
    "some have held",
    "it would seem",
    "it may be asked",
    "one may suppose",
]


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
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of permutations for prompt-level paired permutation tests.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=12345,
        help="RNG seed for prompt-level paired permutation tests.",
    )
    parser.add_argument(
        "--phrase-output-csv",
        default=None,
        help="Optional path to write flat hedge phrase frequency rows as CSV.",
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


def output_text_value(record):
    value = record.get("output_text")
    return value if isinstance(value, str) else ""


def has_judge_error(record):
    return bool(record.get("judge_error"))


def get_judge_result(record):
    result = record.get("judge_result")
    return result if isinstance(result, dict) else None


def is_usable_judged_record(record):
    return not has_judge_error(record) and get_judge_result(record) is not None


def has_optional_label(records, label_name):
    for record in records:
        if not is_usable_judged_record(record):
            continue

        judge_result = get_judge_result(record)
        if label_name in judge_result:
            return True

    return False


def load_jsonl(input_path, include_errors):
    records = []
    parsed_records = []
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
            parsed_records.append(record)

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
    return records, counts, parsed_records


def summarize_group(records, min_quality):
    summary = {
        "n_total": len(records),
        "n_judged": 0,
        "n_quality_ge_min": 0,
        "n_quality_2": 0,
        "n_earth_motion_mention": 0,
        "n_earth_motion_mention_quality_ge_min": 0,
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
        earth_motion_mention_label = judge_result.get("earth_motion_mention")
        explicit_earth_motion_label = judge_result.get("explicit_earth_motion_label")
        proto_heliocentric_label = judge_result.get("proto_heliocentric_label")

        quality_is_number = isinstance(quality_score, int)
        quality_ge_min = quality_is_number and quality_score >= min_quality

        if quality_ge_min:
            summary["n_quality_ge_min"] += 1
        if quality_score == 2:
            summary["n_quality_2"] += 1

        if earth_motion_mention_label == 1:
            summary["n_earth_motion_mention"] += 1
            if quality_ge_min:
                summary["n_earth_motion_mention_quality_ge_min"] += 1

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

    summary["rate_earth_motion_mention_overall"] = safe_div(
        summary["n_earth_motion_mention"], summary["n_judged"]
    )
    summary["rate_earth_motion_mention_given_quality_ge_min"] = safe_div(
        summary["n_earth_motion_mention_quality_ge_min"], summary["n_quality_ge_min"]
    )
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
    del min_quality

    grouped = defaultdict(list)
    for record in records:
        if not is_usable_judged_record(record):
            continue

        key = (model_key(record), category_key(record), prompt_key(record))
        grouped[key].append(record)

    prompt_rows = []
    for (model_id, category, prompt_id), prompt_records in grouped.items():
        quality_scores = []
        earth_motion_mention_count = 0
        explicit_earth_motion_count = 0
        proto_heliocentric_count = 0
        geocentric_count = 0
        ambiguous_count = 0
        prompt_text = ""

        for record in prompt_records:
            judge_result = get_judge_result(record)
            prompt_text = prompt_text or prompt_text_value(record)

            quality_score = judge_result.get("quality_score")
            stance_label = judge_result.get("stance_label")

            if isinstance(quality_score, int):
                quality_scores.append(quality_score)

            if judge_result.get("earth_motion_mention") == 1:
                earth_motion_mention_count += 1

            if judge_result.get("explicit_earth_motion_label") == 1:
                explicit_earth_motion_count += 1

            if judge_result.get("proto_heliocentric_label") == 1:
                proto_heliocentric_count += 1

            if stance_label == "geocentric":
                geocentric_count += 1

            if stance_label == "ambiguous":
                ambiguous_count += 1

        n_prompt_judged = len(prompt_records)
        prompt_rows.append(
            {
                "model_id": model_id,
                "category": category,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "n_judged": n_prompt_judged,
                "avg_quality_score": (
                    statistics.mean(quality_scores) if quality_scores else 0.0
                ),
                "earth_motion_mention_rate": safe_div(
                    earth_motion_mention_count, n_prompt_judged
                ),
                "explicit_earth_motion_rate": safe_div(
                    explicit_earth_motion_count, n_prompt_judged
                ),
                "proto_heliocentric_rate": safe_div(
                    proto_heliocentric_count, n_prompt_judged
                ),
                "geocentric_rate": safe_div(geocentric_count, n_prompt_judged),
                "ambiguous_rate": safe_div(ambiguous_count, n_prompt_judged),
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


def csv_fieldnames(include_earth_motion_mention):
    fields = [
        "summary_level",
        "model_id",
        "category",
        "n_total",
        "n_judged",
        "n_quality_ge_min",
        "n_quality_2",
    ]

    if include_earth_motion_mention:
        fields.extend(
            [
                "n_earth_motion_mention",
                "n_earth_motion_mention_quality_ge_min",
                "rate_earth_motion_mention_overall",
                "rate_earth_motion_mention_given_quality_ge_min",
            ]
        )

    fields.extend(
        [
            "n_explicit_earth_motion",
            "n_explicit_earth_motion_quality_ge_min",
            "n_proto_heliocentric",
            "n_proto_heliocentric_quality_ge_min",
            "rate_explicit_earth_motion_overall",
            "rate_explicit_earth_motion_given_quality_ge_min",
            "rate_proto_heliocentric_overall",
            "rate_proto_heliocentric_given_quality_ge_min",
            "n_geocentric",
            "n_ambiguous",
            "n_no_relevant_claim",
            "rate_quality_ge_min",
            "rate_quality_2",
            "rate_geocentric_overall",
            "rate_ambiguous_overall",
        ]
    )
    return fields


def write_csv(output_path, rows, include_earth_motion_mention):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=csv_fieldnames(include_earth_motion_mention)
        )
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            rate_fields = [
                "rate_quality_ge_min",
                "rate_quality_2",
                "rate_geocentric_overall",
                "rate_ambiguous_overall",
                "rate_explicit_earth_motion_overall",
                "rate_explicit_earth_motion_given_quality_ge_min",
                "rate_proto_heliocentric_overall",
                "rate_proto_heliocentric_given_quality_ge_min",
            ]
            if include_earth_motion_mention:
                rate_fields.extend(
                    [
                        "rate_earth_motion_mention_overall",
                        "rate_earth_motion_mention_given_quality_ge_min",
                    ]
                )

            for field in rate_fields:
                csv_row[field] = f"{csv_row[field]:.6f}"
            writer.writerow(csv_row)


def rate_pct(value):
    return f"{value * 100:.1f}%"


def build_markdown_table(rows, include_category, min_quality, include_earth_motion_mention):
    if include_category:
        headers = [
            "Model",
            "Category",
            "n_total",
            "n_judged",
            f"n_q>={min_quality}",
            "n_q2",
        ]
    else:
        headers = [
            "Model",
            "n_total",
            "n_judged",
            f"n_q>={min_quality}",
            "n_q2",
        ]

    if include_earth_motion_mention:
        headers.extend(
            [
                "Earth-motion mention",
                f"Earth-motion mention q>={min_quality}",
                "Earth-motion mention %",
                f"Earth-motion mention q>={min_quality} %",
            ]
        )

    headers.extend(
        [
            "Explicit Earth-motion",
            f"Explicit Earth-motion q>={min_quality}",
            "Proto-heliocentric suggestion",
            f"Proto-heliocentric suggestion q>={min_quality}",
            "Explicit %",
            f"Explicit q>={min_quality} %",
            "Proto %",
            f"Proto q>={min_quality} %",
            "n_geo",
            "n_ambig",
            "n_none",
            "q>=min%",
            "q2%",
            "geo%",
            "ambig%",
        ]
    )

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
            ]
        )
        if include_earth_motion_mention:
            values.extend(
                [
                    str(row["n_earth_motion_mention"]),
                    str(row["n_earth_motion_mention_quality_ge_min"]),
                    rate_pct(row["rate_earth_motion_mention_overall"]),
                    rate_pct(row["rate_earth_motion_mention_given_quality_ge_min"]),
                ]
            )
        values.extend(
            [
                str(row["n_explicit_earth_motion"]),
                str(row["n_explicit_earth_motion_quality_ge_min"]),
                str(row["n_proto_heliocentric"]),
                str(row["n_proto_heliocentric_quality_ge_min"]),
                rate_pct(row["rate_explicit_earth_motion_overall"]),
                rate_pct(row["rate_explicit_earth_motion_given_quality_ge_min"]),
                rate_pct(row["rate_proto_heliocentric_overall"]),
                rate_pct(row["rate_proto_heliocentric_given_quality_ge_min"]),
                str(row["n_geocentric"]),
                str(row["n_ambiguous"]),
                str(row["n_no_relevant_claim"]),
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


def normalize_text(text):
    return " ".join(text.lower().split())


def phrase_flags(text, phrases):
    normalized = normalize_text(text)
    flags = {}
    any_hedge = 0

    for phrase in phrases:
        present = int(phrase in normalized)
        flags[phrase] = present
        if present:
            any_hedge = 1

    flags["any_hedge_phrase_present"] = any_hedge
    return flags


def summarize_phrase_rates(records, phrases):
    grouped = {
        "overall": defaultdict(list),
        "category": defaultdict(list),
    }

    for record in records:
        text = output_text_value(record).strip()
        if not text:
            continue

        model_id = model_key(record)
        category = category_key(record)
        flags = phrase_flags(text, phrases)

        grouped["overall"][model_id].append(flags)
        grouped["category"][(model_id, category)].append(flags)

    rows = []

    for model_id in sorted(grouped["overall"]):
        group_flags = grouped["overall"][model_id]
        n_total_texts = len(group_flags)
        for phrase in list(phrases) + ["any_hedge_phrase_present"]:
            n_phrase_present = sum(flag_row[phrase] for flag_row in group_flags)
            rows.append(
                {
                    "scope": "overall",
                    "model_id": model_id,
                    "category": "ALL",
                    "phrase": phrase,
                    "n_total_texts": n_total_texts,
                    "n_phrase_present": n_phrase_present,
                    "rate_phrase_present": safe_div(n_phrase_present, n_total_texts),
                    "n_any_hedge": sum(
                        flag_row["any_hedge_phrase_present"] for flag_row in group_flags
                    ),
                    "rate_any_hedge": safe_div(
                        sum(flag_row["any_hedge_phrase_present"] for flag_row in group_flags),
                        n_total_texts,
                    ),
                }
            )

    for model_id, category in sorted(grouped["category"]):
        group_flags = grouped["category"][(model_id, category)]
        n_total_texts = len(group_flags)
        for phrase in list(phrases) + ["any_hedge_phrase_present"]:
            n_phrase_present = sum(flag_row[phrase] for flag_row in group_flags)
            rows.append(
                {
                    "scope": "category",
                    "model_id": model_id,
                    "category": category,
                    "phrase": phrase,
                    "n_total_texts": n_total_texts,
                    "n_phrase_present": n_phrase_present,
                    "rate_phrase_present": safe_div(n_phrase_present, n_total_texts),
                    "n_any_hedge": sum(
                        flag_row["any_hedge_phrase_present"] for flag_row in group_flags
                    ),
                    "rate_any_hedge": safe_div(
                        sum(flag_row["any_hedge_phrase_present"] for flag_row in group_flags),
                        n_total_texts,
                    ),
                }
            )

    return rows


def paired_permutation_test(differences, n_permutations=10000, seed=12345):
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    n_pairs = len(differences)
    if n_pairs == 0:
        return {
            "observed_mean_diff": 0.0,
            "p_value": 1.0,
            "n_pairs": 0,
        }

    observed_mean_diff = statistics.mean(differences)
    abs_observed = abs(observed_mean_diff)
    rng = random.Random(seed)
    extreme_count = 0

    for _ in range(n_permutations):
        permuted_total = 0.0
        for diff in differences:
            permuted_total += diff if rng.random() < 0.5 else -diff
        permuted_mean = permuted_total / n_pairs
        if abs(permuted_mean) >= abs_observed:
            extreme_count += 1

    return {
        "observed_mean_diff": observed_mean_diff,
        "p_value": safe_div(extreme_count, n_permutations),
        "n_pairs": n_pairs,
    }


def choose_model_pair(prompt_rows):
    model_ids = sorted({row["model_id"] for row in prompt_rows})
    model_id_set = set(model_ids)

    if "A" in model_id_set and "B" in model_id_set:
        return {
            "model_a": "A",
            "model_b": "B",
            "note": None,
        }

    if len(model_ids) == 2:
        return {
            "model_a": model_ids[0],
            "model_b": model_ids[1],
            "note": (
                "Model ids `A` and `B` were not both present. "
                f"Using `{model_ids[0]}` as A and `{model_ids[1]}` as B."
            ),
        }

    if len(model_ids) < 2:
        return {
            "model_a": None,
            "model_b": None,
            "note": "Paired permutation tests require two models, but fewer than two were found.",
        }

    return {
        "model_a": None,
        "model_b": None,
        "note": (
            "Paired permutation tests require model ids `A` and `B`, or exactly two "
            f"models in the file. Found: {', '.join(model_ids)}"
        ),
    }


def build_prompt_row_index(prompt_rows, model_id):
    index = {}
    for row in prompt_rows:
        if row["model_id"] != model_id:
            continue
        index[(row["category"], row["prompt_id"])] = row
    return index


def build_paired_permutation_results(
    prompt_rows,
    include_earth_motion_mention,
    n_permutations,
    seed,
):
    pair_info = choose_model_pair(prompt_rows)
    model_a = pair_info["model_a"]
    model_b = pair_info["model_b"]

    if not model_a or not model_b:
        return [], pair_info

    metrics = list(TEST_METRICS)
    if not include_earth_motion_mention:
        metrics = [item for item in metrics if item[0] != "earth_motion_mention_rate"]

    index_a = build_prompt_row_index(prompt_rows, model_a)
    index_b = build_prompt_row_index(prompt_rows, model_b)
    paired_keys_all = sorted(set(index_a) & set(index_b))

    results = []
    scope_keys = [("overall", paired_keys_all)]

    categories = sorted({category for category, _prompt_id in paired_keys_all})
    for category in categories:
        category_keys = [key for key in paired_keys_all if key[0] == category]
        scope_keys.append((category, category_keys))

    for scope, paired_keys in scope_keys:
        if not paired_keys:
            continue

        for metric_key, metric_name in metrics:
            rates_a = [index_a[key][metric_key] for key in paired_keys]
            rates_b = [index_b[key][metric_key] for key in paired_keys]
            differences = [rate_b - rate_a for rate_a, rate_b in zip(rates_a, rates_b)]
            test_result = paired_permutation_test(
                differences,
                n_permutations=n_permutations,
                seed=seed,
            )

            results.append(
                {
                    "scope": scope,
                    "metric": metric_name,
                    "n_pairs": test_result["n_pairs"],
                    "mean_rate_A": statistics.mean(rates_a) if rates_a else 0.0,
                    "mean_rate_B": statistics.mean(rates_b) if rates_b else 0.0,
                    "mean_diff_B_minus_A": test_result["observed_mean_diff"],
                    "p_value_two_sided": test_result["p_value"],
                }
            )

    return results, pair_info


def format_decimal(value, digits=4):
    return f"{value:.{digits}f}"


def build_paired_permutation_table(test_rows):
    headers = [
        "scope",
        "metric",
        "n_pairs",
        "mean_rate_A",
        "mean_rate_B",
        "mean_diff_B_minus_A",
        "p_value_two_sided",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in test_rows:
        lines.append(
            "| {scope} | {metric} | {n_pairs} | {mean_rate_a} | {mean_rate_b} | {mean_diff} | {p_value} |".format(
                scope=row["scope"],
                metric=row["metric"],
                n_pairs=row["n_pairs"],
                mean_rate_a=format_decimal(row["mean_rate_A"]),
                mean_rate_b=format_decimal(row["mean_rate_B"]),
                mean_diff=format_decimal(row["mean_diff_B_minus_A"]),
                p_value=format_decimal(row["p_value_two_sided"], digits=6),
            )
        )

    return "\n".join(lines)


def phrase_csv_fieldnames():
    return [
        "scope",
        "model_id",
        "category",
        "phrase",
        "n_total_texts",
        "n_phrase_present",
        "rate_phrase_present",
        "n_any_hedge",
        "rate_any_hedge",
    ]


def write_phrase_csv(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=phrase_csv_fieldnames())
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["rate_phrase_present"] = f"{csv_row['rate_phrase_present']:.6f}"
            csv_row["rate_any_hedge"] = f"{csv_row['rate_any_hedge']:.6f}"
            writer.writerow(csv_row)


def build_phrase_markdown_table(rows):
    headers = [
        "scope",
        "model_id",
        "category",
        "phrase",
        "n_total_texts",
        "n_phrase_present",
        "rate_phrase_present",
        "n_any_hedge",
        "rate_any_hedge",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append(
            "| {scope} | {model_id} | {category} | {phrase} | {n_total_texts} | {n_phrase_present} | {rate_phrase_present} | {n_any_hedge} | {rate_any_hedge} |".format(
                scope=row["scope"],
                model_id=row["model_id"],
                category=row["category"],
                phrase=row["phrase"],
                n_total_texts=row["n_total_texts"],
                n_phrase_present=row["n_phrase_present"],
                rate_phrase_present=format_decimal(row["rate_phrase_present"], digits=4),
                n_any_hedge=row["n_any_hedge"],
                rate_any_hedge=format_decimal(row["rate_any_hedge"], digits=4),
            )
        )

    return "\n".join(lines)


def write_markdown(
    output_path,
    input_path,
    min_quality,
    rows,
    prompt_rows,
    counts,
    include_earth_motion_mention,
    paired_test_rows,
    pair_info,
    n_permutations,
    test_seed,
    phrase_rows,
):
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
    ]

    if include_earth_motion_mention:
        content.append(
            "- Earth-motion mention is the broadest mention-level metric and is only shown when that label is present in the judged data."
        )

    content.extend(
        [
            "- Explicit Earth-motion is the strict metric.",
            "- Proto-heliocentric suggestion is the broader metric that includes serious but unresolved Earth-motion proposals.",
            "",
            "## Overall Comparison",
            "",
            build_markdown_table(
                overall_rows,
                include_category=False,
                min_quality=min_quality,
                include_earth_motion_mention=include_earth_motion_mention,
            ),
            "",
            "## Category Comparison",
            "",
            build_markdown_table(
                category_rows,
                include_category=True,
                min_quality=min_quality,
                include_earth_motion_mention=include_earth_motion_mention,
            ),
            "",
        ]
    )

    if include_earth_motion_mention:
        content.extend(
            [
                "## Top Earth-motion Mention Prompts Per Model",
                "",
                build_prompt_sections(
                    prompt_rows,
                    metric_key="earth_motion_mention_rate",
                    heading_label="Earth-motion mention",
                ),
                "",
            ]
        )

    content.extend(
        [
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
            "## Prompt-level paired permutation tests",
            "",
        ]
    )

    if pair_info["model_a"] and pair_info["model_b"]:
        content.extend(
            [
                f"- Model A: `{pair_info['model_a']}`",
                f"- Model B: `{pair_info['model_b']}`",
                f"- Permutations: `{n_permutations}`",
                f"- RNG seed: `{test_seed}`",
            ]
        )
        if pair_info["note"]:
            content.append(f"- Note: {pair_info['note']}")
        content.extend(
            [
                "",
                build_paired_permutation_table(paired_test_rows),
                "",
            ]
        )
    else:
        content.extend(
            [
                f"- {pair_info['note']}",
                "",
            ]
        )

    overall_phrase_rows = [row for row in phrase_rows if row["scope"] == "overall"]
    category_phrase_rows = [row for row in phrase_rows if row["scope"] == "category"]

    content.extend(
        [
            "## Hedge / qualification phrase frequencies",
            "",
            "These phrase counts are exploratory and intended to test whether astronomy fine-tuning increases qualified / non-committal discourse patterns.",
            "",
            "### Overall by model",
            "",
            build_phrase_markdown_table(overall_phrase_rows)
            if overall_phrase_rows
            else "No non-empty `output_text` rows were available for phrase analysis.",
            "",
            "### By model and category",
            "",
            build_phrase_markdown_table(category_phrase_rows)
            if category_phrase_rows
            else "No non-empty `output_text` rows were available for phrase analysis.",
            "",
        ]
    )

    content.extend(
        [
            "## Notes",
            "",
            f"- Total judged rows: `{counts['judged_rows']}`",
            f"- Total skipped rows: `{counts['skipped_rows']}`",
            f"- Total rows with judge_error: `{counts['judge_error_rows']}`",
            "",
        ]
    )

    output_path.write_text("\n".join(content), encoding="utf-8")


def main():
    args = parse_args()

    if args.permutations <= 0:
        raise SystemExit("--permutations must be positive")

    input_path = Path(args.input_jsonl)
    output_csv_path = Path(args.output_csv)
    output_markdown_path = Path(args.output_markdown)

    if not input_path.exists():
        raise SystemExit(f"Input JSONL not found: {input_path}")

    records, counts, parsed_records = load_jsonl(input_path, args.include_errors)
    include_earth_motion_mention = has_optional_label(records, "earth_motion_mention")
    summary_rows = build_summary_rows(records, args.min_quality)
    prompt_rows = build_prompt_level_stats(records, args.min_quality)
    phrase_rows = summarize_phrase_rates(parsed_records, HEDGE_PHRASES)
    paired_test_rows, pair_info = build_paired_permutation_results(
        prompt_rows,
        include_earth_motion_mention=include_earth_motion_mention,
        n_permutations=args.permutations,
        seed=args.test_seed,
    )

    write_csv(output_csv_path, summary_rows, include_earth_motion_mention)
    if args.phrase_output_csv:
        write_phrase_csv(Path(args.phrase_output_csv), phrase_rows)
    write_markdown(
        output_markdown_path,
        input_path,
        args.min_quality,
        summary_rows,
        prompt_rows,
        counts,
        include_earth_motion_mention,
        paired_test_rows,
        pair_info,
        args.permutations,
        args.test_seed,
        phrase_rows,
    )


if __name__ == "__main__":
    main()
