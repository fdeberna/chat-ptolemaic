from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PAIR_ORDER = [
    ("qwen_base", "qwen_lora500"),
    ("qwen_base", "qwen_lora1000"),
    ("qwen_lora500", "qwen_lora1000"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize judged Qwen evaluation JSONL files into CSV and JSON reports."
    )
    parser.add_argument(
        "--input-jsonl",
        nargs="+",
        required=True,
        help="One or more judged JSONL files produced by judge/judge_qwen_eval.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for summary CSV and JSON outputs.",
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=1,
        help="Quality threshold used for quality-filtered counts and rates.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include rows with judge_error in n_total where possible.",
    )
    return parser.parse_args()


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def warn(message: str) -> None:
    print(f"Warning: {message}")


def parse_jsonl_record(line: str, line_number: int, path: Path) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        warn(f"skipping empty line {line_number} in {path}")
        return None

    try:
        record = json.loads(stripped)
    except json.JSONDecodeError as exc:
        warn(f"skipping malformed JSON at line {line_number} in {path}: {exc}")
        return None

    if not isinstance(record, dict):
        warn(f"skipping non-object JSON at line {line_number} in {path}")
        return None
    return record


def get_judge_result(record: dict[str, Any]) -> dict[str, Any] | None:
    judge_result = record.get("judge_result")
    if isinstance(judge_result, dict):
        return judge_result

    required_fields = {
        "earth_motion_mention",
        "explicit_earth_motion_label",
        "proto_heliocentric_label",
        "stance",
        "refined_stance",
        "off_topic_or_drift",
        "repetition_or_looping",
        "premodern_register",
        "refined_proto_heliocentric_label",
        "earth_motion_context",
        "hybrid_or_contradictory_frame",
        "modern_explanatory_frame",
        "premodern_explanatory_frame",
    }
    if required_fields.issubset(record):
        return {field: record.get(field) for field in required_fields | {"quality_score", "rationale"}}
    return None


def has_judge_error(record: dict[str, Any]) -> bool:
    return bool(record.get("judge_error"))


def is_usable_judged_record(record: dict[str, Any]) -> bool:
    return not has_judge_error(record) and get_judge_result(record) is not None


def canonical_adapter_dir(record: dict[str, Any]) -> str:
    value = record.get("adapter_dir")
    if value in {None, "", "null"}:
        return "BASE"
    return str(value)


def model_name_key(record: dict[str, Any]) -> str:
    value = record.get("model_name")
    return str(value) if value not in {None, ""} else "UNKNOWN_MODEL"


def category_key(record: dict[str, Any]) -> str:
    value = record.get("prompt_category")
    return str(value) if value not in {None, ""} else "UNKNOWN_CATEGORY"


def prompt_id_key(record: dict[str, Any]) -> str:
    value = record.get("prompt_id")
    return str(value) if value not in {None, ""} else "UNKNOWN_PROMPT"


def prompt_text_value(record: dict[str, Any]) -> str:
    value = record.get("prompt")
    return value if isinstance(value, str) else ""


def infer_variant_label(record: dict[str, Any], source_path: Path) -> str:
    explicit_label = record.get("variant_label")
    if isinstance(explicit_label, str) and explicit_label.strip():
        return explicit_label.strip()

    source_name = source_path.stem.lower()
    if "lora500" in source_name or "500" in source_name:
        return "qwen_lora500"
    if "lora1000" in source_name or "1000" in source_name:
        return "qwen_lora1000"
    if "base" in source_name:
        return "qwen_base"

    adapter_dir = str(record.get("adapter_dir") or "").lower()
    if not adapter_dir:
        return "qwen_base"
    if "500" in adapter_dir:
        return "qwen_lora500"
    if "1000" in adapter_dir:
        return "qwen_lora1000"

    adapter_name = Path(adapter_dir).name
    return adapter_name or "qwen_unknown"


def normalize_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    normalized = dict(record)
    normalized["_source_jsonl"] = str(source_path)
    normalized["_variant_label"] = infer_variant_label(record, source_path)
    return normalized


def load_records(input_paths: list[Path], include_errors: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    counts = {
        "input_files": len(input_paths),
        "total_lines": 0,
        "parsed_rows": 0,
        "malformed_rows": 0,
        "judge_error_rows": 0,
        "judged_rows": 0,
        "included_rows": 0,
    }

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                counts["total_lines"] += 1
                record = parse_jsonl_record(line, line_number, input_path)
                if record is None:
                    counts["malformed_rows"] += 1
                    continue

                counts["parsed_rows"] += 1
                normalized = normalize_record(record, input_path)

                if has_judge_error(normalized):
                    counts["judge_error_rows"] += 1
                if is_usable_judged_record(normalized):
                    counts["judged_rows"] += 1

                if include_errors or is_usable_judged_record(normalized):
                    records.append(normalized)
                    counts["included_rows"] += 1

    return records, counts


def summarize_group(records: list[dict[str, Any]], min_quality: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_total": len(records),
        "n_judged": 0,
        "n_quality_ge_min": 0,
        "n_quality_2": 0,
        "n_earth_motion_mention": 0,
        "n_explicit_earth_moves": 0,
        "n_explicit_earth_stationary": 0,
        "n_explicit_unclear": 0,
        "n_proto_heliocentric_any": 0,
        "n_proto_weak": 0,
        "n_proto_strong": 0,
        "n_proto_unclear": 0,
        "n_refined_proto_heliocentric_any": 0,
        "n_refined_proto_weak": 0,
        "n_refined_proto_strong": 0,
        "n_refined_proto_unclear": 0,
        "n_stance_geocentric": 0,
        "n_stance_heliocentric_or_earth_moves": 0,
        "n_stance_ambiguous": 0,
        "n_stance_no_relevant_claim": 0,
        "n_refined_stance_geocentric": 0,
        "n_refined_stance_heliocentric": 0,
        "n_refined_stance_ambiguous": 0,
        "n_refined_stance_hybrid_or_contradictory": 0,
        "n_refined_stance_irrelevant_or_unclear": 0,
        "n_off_topic_or_drift": 0,
        "n_repetition_or_looping": 0,
        "n_hybrid_or_contradictory_frame": 0,
        "n_modern_explanatory_frame": 0,
        "n_premodern_explanatory_frame": 0,
        "n_hybrid_explanatory_frame": 0,
        "n_earth_motion_context_none": 0,
        "n_earth_motion_context_clear_heliocentric": 0,
        "n_earth_motion_context_rotation_only": 0,
        "n_earth_motion_context_ambiguous_or_mixed": 0,
        "n_earth_motion_context_premodern_sphere_context": 0,
        "n_earth_motion_context_unclear": 0,
        "n_premodern_register_none": 0,
        "n_premodern_register_weak": 0,
        "n_premodern_register_strong": 0,
    }

    for record in records:
        judge_result = get_judge_result(record)
        if judge_result is None or has_judge_error(record):
            continue

        summary["n_judged"] += 1

        quality_score = judge_result.get("quality_score")
        if isinstance(quality_score, int):
            if quality_score >= min_quality:
                summary["n_quality_ge_min"] += 1
            if quality_score == 2:
                summary["n_quality_2"] += 1

        if judge_result.get("earth_motion_mention") is True:
            summary["n_earth_motion_mention"] += 1

        explicit_label = judge_result.get("explicit_earth_motion_label")
        if explicit_label == "earth_moves":
            summary["n_explicit_earth_moves"] += 1
        elif explicit_label == "earth_stationary":
            summary["n_explicit_earth_stationary"] += 1
        elif explicit_label == "unclear":
            summary["n_explicit_unclear"] += 1

        proto_label = judge_result.get("proto_heliocentric_label")
        if proto_label in {"weak", "strong"}:
            summary["n_proto_heliocentric_any"] += 1
        if proto_label == "weak":
            summary["n_proto_weak"] += 1
        elif proto_label == "strong":
            summary["n_proto_strong"] += 1
        elif proto_label == "unclear":
            summary["n_proto_unclear"] += 1

        refined_proto_label = judge_result.get("refined_proto_heliocentric_label")
        if refined_proto_label in {"weak", "strong"}:
            summary["n_refined_proto_heliocentric_any"] += 1
        if refined_proto_label == "weak":
            summary["n_refined_proto_weak"] += 1
        elif refined_proto_label == "strong":
            summary["n_refined_proto_strong"] += 1
        elif refined_proto_label == "unclear":
            summary["n_refined_proto_unclear"] += 1

        stance = judge_result.get("stance") or judge_result.get("stance_label")
        if stance == "geocentric":
            summary["n_stance_geocentric"] += 1
        elif stance == "heliocentric_or_earth_moves":
            summary["n_stance_heliocentric_or_earth_moves"] += 1
        elif stance == "ambiguous":
            summary["n_stance_ambiguous"] += 1
        elif stance == "no_relevant_claim":
            summary["n_stance_no_relevant_claim"] += 1

        refined_stance = judge_result.get("refined_stance")
        if refined_stance == "geocentric":
            summary["n_refined_stance_geocentric"] += 1
        elif refined_stance == "heliocentric":
            summary["n_refined_stance_heliocentric"] += 1
        elif refined_stance == "ambiguous":
            summary["n_refined_stance_ambiguous"] += 1
        elif refined_stance == "hybrid_or_contradictory":
            summary["n_refined_stance_hybrid_or_contradictory"] += 1
        elif refined_stance == "irrelevant_or_unclear":
            summary["n_refined_stance_irrelevant_or_unclear"] += 1

        if judge_result.get("off_topic_or_drift") is True:
            summary["n_off_topic_or_drift"] += 1
        if judge_result.get("repetition_or_looping") is True:
            summary["n_repetition_or_looping"] += 1
        if judge_result.get("hybrid_or_contradictory_frame") is True:
            summary["n_hybrid_or_contradictory_frame"] += 1

        modern_frame = judge_result.get("modern_explanatory_frame") is True
        premodern_frame = judge_result.get("premodern_explanatory_frame") is True
        if modern_frame:
            summary["n_modern_explanatory_frame"] += 1
        if premodern_frame:
            summary["n_premodern_explanatory_frame"] += 1
        if modern_frame and premodern_frame:
            summary["n_hybrid_explanatory_frame"] += 1

        earth_motion_context = judge_result.get("earth_motion_context")
        if earth_motion_context == "none":
            summary["n_earth_motion_context_none"] += 1
        elif earth_motion_context == "clear_heliocentric":
            summary["n_earth_motion_context_clear_heliocentric"] += 1
        elif earth_motion_context == "rotation_only":
            summary["n_earth_motion_context_rotation_only"] += 1
        elif earth_motion_context == "ambiguous_or_mixed":
            summary["n_earth_motion_context_ambiguous_or_mixed"] += 1
        elif earth_motion_context == "premodern_sphere_context":
            summary["n_earth_motion_context_premodern_sphere_context"] += 1
        elif earth_motion_context == "unclear":
            summary["n_earth_motion_context_unclear"] += 1

        register = judge_result.get("premodern_register")
        if register == "none":
            summary["n_premodern_register_none"] += 1
        elif register == "weak":
            summary["n_premodern_register_weak"] += 1
        elif register == "strong":
            summary["n_premodern_register_strong"] += 1

    summary["rate_quality_ge_min"] = safe_div(summary["n_quality_ge_min"], summary["n_judged"])
    summary["rate_quality_2"] = safe_div(summary["n_quality_2"], summary["n_judged"])
    summary["rate_earth_motion_mention"] = safe_div(
        summary["n_earth_motion_mention"], summary["n_judged"]
    )
    summary["rate_explicit_earth_moves"] = safe_div(
        summary["n_explicit_earth_moves"], summary["n_judged"]
    )
    summary["rate_explicit_earth_stationary"] = safe_div(
        summary["n_explicit_earth_stationary"], summary["n_judged"]
    )
    summary["rate_proto_heliocentric_any"] = safe_div(
        summary["n_proto_heliocentric_any"], summary["n_judged"]
    )
    summary["rate_proto_weak"] = safe_div(summary["n_proto_weak"], summary["n_judged"])
    summary["rate_proto_strong"] = safe_div(summary["n_proto_strong"], summary["n_judged"])
    summary["rate_refined_proto_heliocentric_any"] = safe_div(
        summary["n_refined_proto_heliocentric_any"], summary["n_judged"]
    )
    summary["rate_refined_proto_weak"] = safe_div(
        summary["n_refined_proto_weak"], summary["n_judged"]
    )
    summary["rate_refined_proto_strong"] = safe_div(
        summary["n_refined_proto_strong"], summary["n_judged"]
    )
    summary["rate_stance_geocentric"] = safe_div(
        summary["n_stance_geocentric"], summary["n_judged"]
    )
    summary["rate_stance_heliocentric_or_earth_moves"] = safe_div(
        summary["n_stance_heliocentric_or_earth_moves"], summary["n_judged"]
    )
    summary["rate_stance_ambiguous"] = safe_div(
        summary["n_stance_ambiguous"], summary["n_judged"]
    )
    summary["rate_stance_no_relevant_claim"] = safe_div(
        summary["n_stance_no_relevant_claim"], summary["n_judged"]
    )
    summary["rate_refined_stance_geocentric"] = safe_div(
        summary["n_refined_stance_geocentric"], summary["n_judged"]
    )
    summary["rate_refined_stance_heliocentric"] = safe_div(
        summary["n_refined_stance_heliocentric"], summary["n_judged"]
    )
    summary["rate_refined_stance_ambiguous"] = safe_div(
        summary["n_refined_stance_ambiguous"], summary["n_judged"]
    )
    summary["rate_refined_stance_hybrid_or_contradictory"] = safe_div(
        summary["n_refined_stance_hybrid_or_contradictory"], summary["n_judged"]
    )
    summary["rate_refined_stance_irrelevant_or_unclear"] = safe_div(
        summary["n_refined_stance_irrelevant_or_unclear"], summary["n_judged"]
    )
    summary["rate_off_topic_or_drift"] = safe_div(
        summary["n_off_topic_or_drift"], summary["n_judged"]
    )
    summary["rate_repetition_or_looping"] = safe_div(
        summary["n_repetition_or_looping"], summary["n_judged"]
    )
    summary["rate_hybrid_or_contradictory_frame"] = safe_div(
        summary["n_hybrid_or_contradictory_frame"], summary["n_judged"]
    )
    summary["rate_modern_explanatory_frame"] = safe_div(
        summary["n_modern_explanatory_frame"], summary["n_judged"]
    )
    summary["rate_premodern_explanatory_frame"] = safe_div(
        summary["n_premodern_explanatory_frame"], summary["n_judged"]
    )
    summary["rate_hybrid_explanatory_frame"] = safe_div(
        summary["n_hybrid_explanatory_frame"], summary["n_judged"]
    )
    summary["rate_earth_motion_context_none"] = safe_div(
        summary["n_earth_motion_context_none"], summary["n_judged"]
    )
    summary["rate_earth_motion_context_clear_heliocentric"] = safe_div(
        summary["n_earth_motion_context_clear_heliocentric"], summary["n_judged"]
    )
    summary["rate_earth_motion_context_rotation_only"] = safe_div(
        summary["n_earth_motion_context_rotation_only"], summary["n_judged"]
    )
    summary["rate_earth_motion_context_ambiguous_or_mixed"] = safe_div(
        summary["n_earth_motion_context_ambiguous_or_mixed"], summary["n_judged"]
    )
    summary["rate_earth_motion_context_premodern_sphere_context"] = safe_div(
        summary["n_earth_motion_context_premodern_sphere_context"], summary["n_judged"]
    )
    summary["rate_earth_motion_context_unclear"] = safe_div(
        summary["n_earth_motion_context_unclear"], summary["n_judged"]
    )
    summary["rate_premodern_register_none"] = safe_div(
        summary["n_premodern_register_none"], summary["n_judged"]
    )
    summary["rate_premodern_register_weak"] = safe_div(
        summary["n_premodern_register_weak"], summary["n_judged"]
    )
    summary["rate_premodern_register_strong"] = safe_div(
        summary["n_premodern_register_strong"], summary["n_judged"]
    )
    return summary


def build_summary_rows(records: list[dict[str, Any]], min_quality: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    category_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        overall_key = (
            record["_variant_label"],
            model_name_key(record),
            canonical_adapter_dir(record),
        )
        category_key_tuple = overall_key + (category_key(record),)
        overall_groups[overall_key].append(record)
        category_groups[category_key_tuple].append(record)

    overall_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for variant_label, model_name, adapter_dir in sorted(overall_groups):
        row = {
            "variant_label": variant_label,
            "model_name": model_name,
            "adapter_dir": adapter_dir,
        }
        row.update(summarize_group(overall_groups[(variant_label, model_name, adapter_dir)], min_quality))
        overall_rows.append(row)

    for variant_label, model_name, adapter_dir, prompt_category in sorted(category_groups):
        row = {
            "variant_label": variant_label,
            "model_name": model_name,
            "adapter_dir": adapter_dir,
            "prompt_category": prompt_category,
        }
        row.update(
            summarize_group(
                category_groups[(variant_label, model_name, adapter_dir, prompt_category)],
                min_quality,
            )
        )
        category_rows.append(row)

    return overall_rows, category_rows


def build_prompt_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if not is_usable_judged_record(record):
            continue
        grouped[(record["_variant_label"], category_key(record), prompt_id_key(record))].append(record)

    prompt_rows: list[dict[str, Any]] = []
    for (variant_label, prompt_category, prompt_id), group_records in sorted(grouped.items()):
        judged_count = len(group_records)
        quality_scores: list[int] = []
        earth_motion_mention = 0
        explicit_earth_moves = 0
        proto_any = 0
        refined_proto_any = 0
        stance_geocentric = 0
        stance_heliocentric = 0
        refined_stance_heliocentric = 0
        refined_stance_hybrid = 0
        off_topic = 0
        repetition = 0
        hybrid_frame = 0
        modern_frame = 0
        premodern_frame = 0
        earth_motion_context_clear_heliocentric = 0
        earth_motion_context_premodern_sphere_context = 0
        strong_register = 0
        prompt_text = ""

        for record in group_records:
            judge_result = get_judge_result(record)
            if judge_result is None:
                continue

            prompt_text = prompt_text or prompt_text_value(record)
            quality_score = judge_result.get("quality_score")
            if isinstance(quality_score, int):
                quality_scores.append(quality_score)

            if judge_result.get("earth_motion_mention") is True:
                earth_motion_mention += 1
            if judge_result.get("explicit_earth_motion_label") == "earth_moves":
                explicit_earth_moves += 1
            if judge_result.get("proto_heliocentric_label") in {"weak", "strong"}:
                proto_any += 1
            if judge_result.get("refined_proto_heliocentric_label") in {"weak", "strong"}:
                refined_proto_any += 1

            stance = judge_result.get("stance") or judge_result.get("stance_label")
            if stance == "geocentric":
                stance_geocentric += 1
            elif stance == "heliocentric_or_earth_moves":
                stance_heliocentric += 1

            if judge_result.get("refined_stance") == "heliocentric":
                refined_stance_heliocentric += 1
            elif judge_result.get("refined_stance") == "hybrid_or_contradictory":
                refined_stance_hybrid += 1

            if judge_result.get("off_topic_or_drift") is True:
                off_topic += 1
            if judge_result.get("repetition_or_looping") is True:
                repetition += 1
            if judge_result.get("hybrid_or_contradictory_frame") is True:
                hybrid_frame += 1
            if judge_result.get("modern_explanatory_frame") is True:
                modern_frame += 1
            if judge_result.get("premodern_explanatory_frame") is True:
                premodern_frame += 1
            if judge_result.get("earth_motion_context") == "clear_heliocentric":
                earth_motion_context_clear_heliocentric += 1
            if judge_result.get("earth_motion_context") == "premodern_sphere_context":
                earth_motion_context_premodern_sphere_context += 1
            if judge_result.get("premodern_register") == "strong":
                strong_register += 1

        prompt_rows.append(
            {
                "variant_label": variant_label,
                "prompt_category": prompt_category,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "n_judged": judged_count,
                "avg_quality_score": statistics.mean(quality_scores) if quality_scores else 0.0,
                "earth_motion_mention_rate": safe_div(earth_motion_mention, judged_count),
                "explicit_earth_moves_rate": safe_div(explicit_earth_moves, judged_count),
                "proto_heliocentric_any_rate": safe_div(proto_any, judged_count),
                "refined_proto_heliocentric_any_rate": safe_div(refined_proto_any, judged_count),
                "stance_geocentric_rate": safe_div(stance_geocentric, judged_count),
                "stance_heliocentric_rate": safe_div(stance_heliocentric, judged_count),
                "refined_stance_heliocentric_rate": safe_div(
                    refined_stance_heliocentric, judged_count
                ),
                "refined_stance_hybrid_or_contradictory_rate": safe_div(
                    refined_stance_hybrid, judged_count
                ),
                "off_topic_or_drift_rate": safe_div(off_topic, judged_count),
                "repetition_or_looping_rate": safe_div(repetition, judged_count),
                "hybrid_or_contradictory_frame_rate": safe_div(hybrid_frame, judged_count),
                "modern_explanatory_frame_rate": safe_div(modern_frame, judged_count),
                "premodern_explanatory_frame_rate": safe_div(premodern_frame, judged_count),
                "earth_motion_context_clear_heliocentric_rate": safe_div(
                    earth_motion_context_clear_heliocentric, judged_count
                ),
                "earth_motion_context_premodern_sphere_context_rate": safe_div(
                    earth_motion_context_premodern_sphere_context, judged_count
                ),
                "premodern_register_strong_rate": safe_div(strong_register, judged_count),
            }
        )

    return prompt_rows


def build_pairwise_prompt_rows(prompt_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in prompt_rows:
        index[(row["variant_label"], row["prompt_category"], row["prompt_id"])] = row

    results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for left_variant, right_variant in PAIR_ORDER:
        pair_rows: list[dict[str, Any]] = []
        shared_keys = sorted(
            {
                (prompt_category, prompt_id)
                for variant_label, prompt_category, prompt_id in index
                if variant_label == left_variant
            }
            & {
                (prompt_category, prompt_id)
                for variant_label, prompt_category, prompt_id in index
                if variant_label == right_variant
            }
        )

        for prompt_category, prompt_id in shared_keys:
            left_row = index[(left_variant, prompt_category, prompt_id)]
            right_row = index[(right_variant, prompt_category, prompt_id)]

            pair_rows.append(
                {
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                    "prompt_category": prompt_category,
                    "prompt_id": prompt_id,
                    "prompt": left_row["prompt"] or right_row["prompt"],
                    "left_n_judged": left_row["n_judged"],
                    "right_n_judged": right_row["n_judged"],
                    "left_earth_motion_mention_rate": left_row["earth_motion_mention_rate"],
                    "right_earth_motion_mention_rate": right_row["earth_motion_mention_rate"],
                    "diff_earth_motion_mention_rate": right_row["earth_motion_mention_rate"]
                    - left_row["earth_motion_mention_rate"],
                    "left_explicit_earth_moves_rate": left_row["explicit_earth_moves_rate"],
                    "right_explicit_earth_moves_rate": right_row["explicit_earth_moves_rate"],
                    "diff_explicit_earth_moves_rate": right_row["explicit_earth_moves_rate"]
                    - left_row["explicit_earth_moves_rate"],
                    "left_proto_heliocentric_any_rate": left_row["proto_heliocentric_any_rate"],
                    "right_proto_heliocentric_any_rate": right_row["proto_heliocentric_any_rate"],
                    "diff_proto_heliocentric_any_rate": right_row["proto_heliocentric_any_rate"]
                    - left_row["proto_heliocentric_any_rate"],
                    "left_refined_proto_heliocentric_any_rate": left_row[
                        "refined_proto_heliocentric_any_rate"
                    ],
                    "right_refined_proto_heliocentric_any_rate": right_row[
                        "refined_proto_heliocentric_any_rate"
                    ],
                    "diff_refined_proto_heliocentric_any_rate": right_row[
                        "refined_proto_heliocentric_any_rate"
                    ]
                    - left_row["refined_proto_heliocentric_any_rate"],
                    "left_off_topic_or_drift_rate": left_row["off_topic_or_drift_rate"],
                    "right_off_topic_or_drift_rate": right_row["off_topic_or_drift_rate"],
                    "diff_off_topic_or_drift_rate": right_row["off_topic_or_drift_rate"]
                    - left_row["off_topic_or_drift_rate"],
                    "left_repetition_or_looping_rate": left_row["repetition_or_looping_rate"],
                    "right_repetition_or_looping_rate": right_row["repetition_or_looping_rate"],
                    "diff_repetition_or_looping_rate": right_row["repetition_or_looping_rate"]
                    - left_row["repetition_or_looping_rate"],
                    "left_refined_stance_heliocentric_rate": left_row[
                        "refined_stance_heliocentric_rate"
                    ],
                    "right_refined_stance_heliocentric_rate": right_row[
                        "refined_stance_heliocentric_rate"
                    ],
                    "diff_refined_stance_heliocentric_rate": right_row[
                        "refined_stance_heliocentric_rate"
                    ]
                    - left_row["refined_stance_heliocentric_rate"],
                    "left_hybrid_or_contradictory_frame_rate": left_row[
                        "hybrid_or_contradictory_frame_rate"
                    ],
                    "right_hybrid_or_contradictory_frame_rate": right_row[
                        "hybrid_or_contradictory_frame_rate"
                    ],
                    "diff_hybrid_or_contradictory_frame_rate": right_row[
                        "hybrid_or_contradictory_frame_rate"
                    ]
                    - left_row["hybrid_or_contradictory_frame_rate"],
                    "left_modern_explanatory_frame_rate": left_row["modern_explanatory_frame_rate"],
                    "right_modern_explanatory_frame_rate": right_row["modern_explanatory_frame_rate"],
                    "diff_modern_explanatory_frame_rate": right_row["modern_explanatory_frame_rate"]
                    - left_row["modern_explanatory_frame_rate"],
                    "left_premodern_explanatory_frame_rate": left_row[
                        "premodern_explanatory_frame_rate"
                    ],
                    "right_premodern_explanatory_frame_rate": right_row[
                        "premodern_explanatory_frame_rate"
                    ],
                    "diff_premodern_explanatory_frame_rate": right_row[
                        "premodern_explanatory_frame_rate"
                    ]
                    - left_row["premodern_explanatory_frame_rate"],
                    "left_earth_motion_context_clear_heliocentric_rate": left_row[
                        "earth_motion_context_clear_heliocentric_rate"
                    ],
                    "right_earth_motion_context_clear_heliocentric_rate": right_row[
                        "earth_motion_context_clear_heliocentric_rate"
                    ],
                    "diff_earth_motion_context_clear_heliocentric_rate": right_row[
                        "earth_motion_context_clear_heliocentric_rate"
                    ]
                    - left_row["earth_motion_context_clear_heliocentric_rate"],
                    "left_premodern_register_strong_rate": left_row[
                        "premodern_register_strong_rate"
                    ],
                    "right_premodern_register_strong_rate": right_row[
                        "premodern_register_strong_rate"
                    ],
                    "diff_premodern_register_strong_rate": right_row[
                        "premodern_register_strong_rate"
                    ]
                    - left_row["premodern_register_strong_rate"],
                }
            )

        if pair_rows:
            results[(left_variant, right_variant)] = pair_rows

    return results


def csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(rows))
        writer.writeheader()
        for row in rows:
            csv_row = {}
            for key, value in row.items():
                if isinstance(value, float):
                    csv_row[key] = f"{value:.6f}"
                else:
                    csv_row[key] = value
            writer.writerow(csv_row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.min_quality < 0:
        raise SystemExit("--min-quality cannot be negative.")

    input_paths = [Path(path_value) for path_value in args.input_jsonl]
    for input_path in input_paths:
        if not input_path.exists():
            raise SystemExit(f"Input JSONL not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, counts = load_records(input_paths, include_errors=args.include_errors)
    overall_rows, category_rows = build_summary_rows(records, min_quality=args.min_quality)
    prompt_rows = build_prompt_rows(records)
    pair_rows = build_pairwise_prompt_rows(prompt_rows)

    write_csv(output_dir / "qwen_judgment_summary_overall.csv", overall_rows)
    write_csv(output_dir / "qwen_judgment_summary_by_category.csv", category_rows)
    write_csv(output_dir / "qwen_judgment_prompt_rates.csv", prompt_rows)

    summary_manifest = {
        "input_jsonl": [str(path) for path in input_paths],
        "min_quality": args.min_quality,
        "include_errors": args.include_errors,
        "counts": counts,
        "output_files": [
            "qwen_judgment_summary_overall.csv",
            "qwen_judgment_summary_by_category.csv",
            "qwen_judgment_prompt_rates.csv",
        ],
        "pairwise_outputs": [],
        "overall_rows": overall_rows,
        "category_rows": category_rows,
    }

    for (left_variant, right_variant), rows in pair_rows.items():
        output_name = f"qwen_prompt_pair_{left_variant}_vs_{right_variant}.csv"
        write_csv(output_dir / output_name, rows)
        summary_manifest["output_files"].append(output_name)
        summary_manifest["pairwise_outputs"].append(
            {
                "left_variant": left_variant,
                "right_variant": right_variant,
                "output_csv": output_name,
                "n_prompts": len(rows),
            }
        )

    write_json(output_dir / "qwen_judgment_summary.json", summary_manifest)

    print(
        "Finished. "
        f"parsed={counts['parsed_rows']} judged={counts['judged_rows']} "
        f"included={counts['included_rows']} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
