from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Any

try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None


PAIR_ORDER = [
    ("qwen_base", "qwen_lora500"),
    ("qwen_base", "qwen_lora1000"),
    ("qwen_lora500", "qwen_lora1000"),
]

JUDGE_FIELD_NAMES = {
    "quality_score",
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
    "stance_label",
    "stance_normalized",
    "explicit_earth_motion_binary",
    "proto_heliocentric_binary",
    "refined_proto_heliocentric_binary",
    "rationale",
    "reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize judged Qwen evaluation JSONL files into CSV and JSON reports."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Repeatable input spec in the form alias=path/to/file.jsonl. "
            "If alias is omitted, the variant label is inferred from the file."
        ),
    )
    parser.add_argument(
        "--input-jsonl",
        nargs="+",
        default=[],
        help="Backward-compatible list of judged JSONL files produced by judge/judge_qwen_eval.py.",
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


def safe_rate_or_none(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return center - margin, center + margin


def mcnemar_test(b: int, c: int) -> tuple[float | None, float | None]:
    if chi2 is None or b + c == 0:
        return None, None
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


def warn(message: str) -> None:
    print(f"Warning: {message}")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def parse_input_spec(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)

    alias, path_text = value.split("=", 1)
    alias = alias.strip()
    path_text = path_text.strip()
    if not alias:
        raise SystemExit(f"Invalid --input spec with empty alias: {value}")
    if not path_text:
        raise SystemExit(f"Invalid --input spec with empty path: {value}")
    return alias, Path(path_text)


def collect_input_specs(args: argparse.Namespace) -> list[tuple[str | None, Path]]:
    specs: list[tuple[str | None, Path]] = []
    for value in args.input:
        specs.append(parse_input_spec(value))
    for value in args.input_jsonl:
        specs.append((None, Path(value)))
    if not specs:
        raise SystemExit("Provide at least one input via --input or --input-jsonl.")
    return specs


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


def flatten_judge_result_fields(record: dict[str, Any]) -> dict[str, Any] | None:
    flattened: dict[str, Any] = {}

    nested = record.get("judge_result")
    if isinstance(nested, dict):
        flattened.update(nested)

    for field_name in JUDGE_FIELD_NAMES:
        if field_name not in flattened and field_name in record:
            flattened[field_name] = record.get(field_name)

    return flattened or None


def get_judge_result(record: dict[str, Any]) -> dict[str, Any] | None:
    cached = record.get("_judge_result_flat")
    if isinstance(cached, dict):
        return cached
    return flatten_judge_result_fields(record)


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


def sample_id_key(record: dict[str, Any]) -> str:
    value = record.get("sample_id")
    return str(value) if value not in {None, ""} else "UNKNOWN_SAMPLE"


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


def normalize_record(record: dict[str, Any], source_path: Path, model_alias: str | None) -> dict[str, Any]:
    normalized = dict(record)
    normalized["_source_jsonl"] = str(source_path)
    normalized["_variant_label"] = infer_variant_label(record, source_path)
    normalized["_model_label"] = model_alias.strip() if model_alias else normalized["_variant_label"]
    normalized["_judge_result_flat"] = flatten_judge_result_fields(record)
    return normalized


def load_records(
    input_specs: list[tuple[str | None, Path]], include_errors: bool
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    counts = {
        "input_files": len(input_specs),
        "total_lines": 0,
        "parsed_rows": 0,
        "malformed_rows": 0,
        "judge_error_rows": 0,
        "judged_rows": 0,
        "included_rows": 0,
    }

    for model_alias, input_path in input_specs:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                counts["total_lines"] += 1
                record = parse_jsonl_record(line, line_number, input_path)
                if record is None:
                    counts["malformed_rows"] += 1
                    continue

                counts["parsed_rows"] += 1
                normalized = normalize_record(record, input_path, model_alias=model_alias)

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


def build_summary_rows(
    records: list[dict[str, Any]], min_quality: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


def build_pairwise_prompt_rows(
    prompt_rows: list[dict[str, Any]]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
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


def bool_is_true(judge_result: dict[str, Any], field_name: str) -> bool:
    return judge_result.get(field_name) is True


def has_neither_modern_nor_premodern_frame(judge_result: dict[str, Any]) -> bool:
    return not bool_is_true(judge_result, "modern_explanatory_frame") and not bool_is_true(
        judge_result, "premodern_explanatory_frame"
    )


def has_both_modern_and_premodern_frame(judge_result: dict[str, Any]) -> bool:
    return bool_is_true(judge_result, "modern_explanatory_frame") and bool_is_true(
        judge_result, "premodern_explanatory_frame"
    )


def conditional_metric_specs() -> list[dict[str, Any]]:
    return [
        {
            "metric_id": "refined_stance_geocentric_given_model",
            "definition": "P(refined_stance = geocentric | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": lambda judge_result: judge_result.get("refined_stance")
            == "geocentric",
        },
        {
            "metric_id": "refined_stance_heliocentric_given_model",
            "definition": "P(refined_stance = heliocentric | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": lambda judge_result: judge_result.get("refined_stance")
            == "heliocentric",
        },
        {
            "metric_id": "refined_stance_hybrid_or_contradictory_given_model",
            "definition": "P(refined_stance = hybrid_or_contradictory | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": lambda judge_result: judge_result.get("refined_stance")
            == "hybrid_or_contradictory",
        },
        {
            "metric_id": "premodern_explanatory_frame_true_given_model",
            "definition": "P(premodern_explanatory_frame = true | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "premodern_explanatory_frame"
            ),
        },
        {
            "metric_id": "modern_explanatory_frame_true_given_model",
            "definition": "P(modern_explanatory_frame = true | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "modern_explanatory_frame"
            ),
        },
        {
            "metric_id": "hybrid_or_contradictory_frame_true_given_model",
            "definition": "P(hybrid_or_contradictory_frame = true | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "hybrid_or_contradictory_frame"
            ),
        },
        {
            "metric_id": "anything_given_model",
            "definition": "P(neither premodern_explanatory_frame nor modern_explanatory_frame is true | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": has_neither_modern_nor_premodern_frame,
        },
        {
            "metric_id": "both_modern_and_premodern_given_model",
            "definition": "P(modern_explanatory_frame = true and premodern_explanatory_frame = true | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": has_both_modern_and_premodern_frame,
        },
        {
            "metric_id": "refined_stance_geocentric_given_premodern_explanatory_frame_true",
            "definition": "P(refined_stance = geocentric | premodern_explanatory_frame = true, model)",
            "denominator_predicate": lambda judge_result: bool_is_true(
                judge_result, "premodern_explanatory_frame"
            ),
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "premodern_explanatory_frame"
            )
            and judge_result.get("refined_stance") == "geocentric",
        },
        {
            "metric_id": "refined_stance_heliocentric_given_premodern_explanatory_frame_true",
            "definition": "P(refined_stance = heliocentric | premodern_explanatory_frame = true, model)",
            "denominator_predicate": lambda judge_result: bool_is_true(
                judge_result, "premodern_explanatory_frame"
            ),
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "premodern_explanatory_frame"
            )
            and judge_result.get("refined_stance") == "heliocentric",
        },
        {
            "metric_id": "refined_stance_everything_else_given_premodern_explanatory_frame_true",
            "definition": "P(refined_stance not in {heliocentric, geocentric} | premodern_explanatory_frame = true, model)",
            "denominator_predicate": lambda judge_result: bool_is_true(
                judge_result, "premodern_explanatory_frame"
            ),
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "premodern_explanatory_frame"
            )
            and judge_result.get("refined_stance") not in {"heliocentric", "geocentric"},
        },
        {
            "metric_id": "refined_stance_heliocentric_given_modern_explanatory_frame_true",
            "definition": "P(refined_stance = heliocentric | modern_explanatory_frame = true, model)",
            "denominator_predicate": lambda judge_result: bool_is_true(
                judge_result, "modern_explanatory_frame"
            ),
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "modern_explanatory_frame"
            )
            and judge_result.get("refined_stance") == "heliocentric",
        },
        {
            "metric_id": "refined_stance_geocentric_given_modern_explanatory_frame_true",
            "definition": "P(refined_stance = geocentric | modern_explanatory_frame = true, model)",
            "denominator_predicate": lambda judge_result: bool_is_true(
                judge_result, "modern_explanatory_frame"
            ),
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "modern_explanatory_frame"
            )
            and judge_result.get("refined_stance") == "geocentric",
        },
        {
            "metric_id": "refined_stance_everything_else_given_modern_explanatory_frame_true",
            "definition": "P(refined_stance not in {heliocentric, geocentric} | modern_explanatory_frame = true, model)",
            "denominator_predicate": lambda judge_result: bool_is_true(
                judge_result, "modern_explanatory_frame"
            ),
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "modern_explanatory_frame"
            )
            and judge_result.get("refined_stance") not in {"heliocentric", "geocentric"},
        },
        {
            "metric_id": "hybrid_or_contradictory_frame_true_given_earth_motion_mention_true",
            "definition": "P(hybrid_or_contradictory_frame = true | earth_motion_mention = true, model)",
            "denominator_predicate": lambda judge_result: bool_is_true(
                judge_result, "earth_motion_mention"
            ),
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "earth_motion_mention"
            )
            and bool_is_true(judge_result, "hybrid_or_contradictory_frame"),
        },
        {
            "metric_id": "refined_stance_geocentric_given_earth_motion_mention_true",
            "definition": "P(refined_stance = geocentric | earth_motion_mention = true, model)",
            "denominator_predicate": lambda judge_result: bool_is_true(
                judge_result, "earth_motion_mention"
            ),
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "earth_motion_mention"
            )
            and judge_result.get("refined_stance") == "geocentric",
        },
        {
            "metric_id": "premodern_explanatory_frame_true_given_premodern_register_strong",
            "definition": "P(premodern_explanatory_frame = true | premodern_register = strong, model)",
            "denominator_predicate": lambda judge_result: judge_result.get("premodern_register")
            == "strong",
            "numerator_predicate": lambda judge_result: judge_result.get("premodern_register")
            == "strong"
            and bool_is_true(judge_result, "premodern_explanatory_frame"),
        },
        {
            "metric_id": "modern_explanatory_frame_true_given_premodern_register_strong",
            "definition": "P(modern_explanatory_frame = true | premodern_register = strong, model)",
            "denominator_predicate": lambda judge_result: judge_result.get("premodern_register")
            == "strong",
            "numerator_predicate": lambda judge_result: judge_result.get("premodern_register")
            == "strong"
            and bool_is_true(judge_result, "modern_explanatory_frame"),
        },
        {
            "metric_id": "off_topic_or_drift_true_given_model",
            "definition": "P(off_topic_or_drift = true | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "off_topic_or_drift"
            ),
        },
        {
            "metric_id": "repetition_or_looping_true_given_model",
            "definition": "P(repetition_or_looping = true | model)",
            "denominator_predicate": lambda _: True,
            "numerator_predicate": lambda judge_result: bool_is_true(
                judge_result, "repetition_or_looping"
            ),
        },
    ]


def build_conditional_rows(
    records: list[dict[str, Any]], include_prompt_category: bool
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str | None], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        if not is_usable_judged_record(record):
            continue
        group_key = (
            record["_model_label"],
            record["_variant_label"],
            model_name_key(record),
            canonical_adapter_dir(record),
            category_key(record) if include_prompt_category else None,
        )
        grouped[group_key].append(record)

    rows: list[dict[str, Any]] = []
    for (
        model_label,
        variant_label,
        model_name,
        adapter_dir,
        prompt_category,
    ), group_records in sorted(grouped.items()):
        judge_results = [get_judge_result(record) for record in group_records]
        judge_results = [judge_result for judge_result in judge_results if judge_result is not None]
        for spec in conditional_metric_specs():
            denominator = sum(
                1 for judge_result in judge_results if spec["denominator_predicate"](judge_result)
            )
            numerator = sum(
                1 for judge_result in judge_results if spec["numerator_predicate"](judge_result)
            )
            row = {
                "model_label": model_label,
                "variant_label": variant_label,
                "model_name": model_name,
                "adapter_dir": adapter_dir,
                "metric_id": spec["metric_id"],
                "definition": spec["definition"],
                "numerator": numerator,
                "denominator": denominator,
                "rate": safe_rate_or_none(numerator, denominator),
            }
            row["ci_lower"], row["ci_upper"] = wilson_ci(numerator, denominator)
            if include_prompt_category:
                row["prompt_category"] = prompt_category
            rows.append(row)

    return rows


def available_pair_configs(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    labels_by_variant: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not is_usable_judged_record(record):
            continue
        labels_by_variant[record["_variant_label"]].add(record["_model_label"])

    configs: list[dict[str, str]] = []
    for reference_variant, comparison_variant in PAIR_ORDER:
        reference_labels = labels_by_variant.get(reference_variant, set())
        comparison_labels = labels_by_variant.get(comparison_variant, set())
        if not reference_labels or not comparison_labels:
            continue

        if len(reference_labels) > 1:
            warn(
                "multiple model labels found for "
                f"{reference_variant}: {sorted(reference_labels)}; using {sorted(reference_labels)[0]}"
            )
        if len(comparison_labels) > 1:
            warn(
                "multiple model labels found for "
                f"{comparison_variant}: {sorted(comparison_labels)}; using {sorted(comparison_labels)[0]}"
            )

        configs.append(
            {
                "reference_variant_label": reference_variant,
                "comparison_variant_label": comparison_variant,
                "reference_model_label": sorted(reference_labels)[0],
                "comparison_model_label": sorted(comparison_labels)[0],
            }
        )

    return configs


def build_paired_sample_rows(
    records: list[dict[str, Any]]
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for record in records:
        if not is_usable_judged_record(record):
            continue
        key = (
            record["_model_label"],
            category_key(record),
            prompt_id_key(record),
            sample_id_key(record),
        )
        if key in index:
            warn(
                "duplicate usable row for "
                f"model={key[0]} category={key[1]} prompt_id={key[2]} sample_id={key[3]}; "
                "keeping the first row"
            )
            continue
        index[key] = record

    results: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for config in available_pair_configs(records):
        reference_model_label = config["reference_model_label"]
        comparison_model_label = config["comparison_model_label"]
        reference_variant_label = config["reference_variant_label"]
        comparison_variant_label = config["comparison_variant_label"]

        reference_keys = {
            (prompt_category, prompt_id, sample_id)
            for model_label, prompt_category, prompt_id, sample_id in index
            if model_label == reference_model_label
        }
        comparison_keys = {
            (prompt_category, prompt_id, sample_id)
            for model_label, prompt_category, prompt_id, sample_id in index
            if model_label == comparison_model_label
        }
        shared_keys = sorted(reference_keys & comparison_keys)

        pair_rows: list[dict[str, Any]] = []
        for prompt_category, prompt_id, sample_id in shared_keys:
            reference_record = index[
                (reference_model_label, prompt_category, prompt_id, sample_id)
            ]
            comparison_record = index[
                (comparison_model_label, prompt_category, prompt_id, sample_id)
            ]
            reference_judge = get_judge_result(reference_record)
            comparison_judge = get_judge_result(comparison_record)
            if reference_judge is None or comparison_judge is None:
                continue

            reference_modern = bool_is_true(reference_judge, "modern_explanatory_frame")
            comparison_modern = bool_is_true(comparison_judge, "modern_explanatory_frame")
            reference_premodern = bool_is_true(reference_judge, "premodern_explanatory_frame")
            comparison_premodern = bool_is_true(comparison_judge, "premodern_explanatory_frame")
            reference_hybrid_frame = bool_is_true(reference_judge, "hybrid_or_contradictory_frame")
            comparison_hybrid_frame = bool_is_true(
                comparison_judge, "hybrid_or_contradictory_frame"
            )
            reference_earth_motion = bool_is_true(reference_judge, "earth_motion_mention")
            comparison_earth_motion = bool_is_true(comparison_judge, "earth_motion_mention")
            reference_refined_stance = reference_judge.get("refined_stance")
            comparison_refined_stance = comparison_judge.get("refined_stance")

            pair_rows.append(
                {
                    "reference_model_label": reference_model_label,
                    "comparison_model_label": comparison_model_label,
                    "reference_variant_label": reference_variant_label,
                    "comparison_variant_label": comparison_variant_label,
                    "prompt_category": prompt_category,
                    "prompt_id": prompt_id,
                    "sample_id": sample_id,
                    "prompt": prompt_text_value(reference_record)
                    or prompt_text_value(comparison_record),
                    "reference_refined_stance": reference_refined_stance,
                    "comparison_refined_stance": comparison_refined_stance,
                    "reference_modern_explanatory_frame": reference_modern,
                    "comparison_modern_explanatory_frame": comparison_modern,
                    "reference_premodern_explanatory_frame": reference_premodern,
                    "comparison_premodern_explanatory_frame": comparison_premodern,
                    "reference_hybrid_or_contradictory_frame": reference_hybrid_frame,
                    "comparison_hybrid_or_contradictory_frame": comparison_hybrid_frame,
                    "reference_earth_motion_mention": reference_earth_motion,
                    "comparison_earth_motion_mention": comparison_earth_motion,
                    "modern_to_premodern_frame_flip": reference_modern and comparison_premodern,
                    "premodern_to_modern_frame_flip": reference_premodern and comparison_modern,
                    "heliocentric_to_geocentric_refined_stance_flip": (
                        reference_refined_stance == "heliocentric"
                        and comparison_refined_stance == "geocentric"
                    ),
                    "geocentric_to_heliocentric_refined_stance_flip": (
                        reference_refined_stance == "geocentric"
                        and comparison_refined_stance == "heliocentric"
                    ),
                    "modern_suppression": reference_modern and not comparison_modern,
                    "premodern_activation": (not reference_premodern) and comparison_premodern,
                    "hybridization_activation": (not reference_hybrid_frame)
                    and comparison_hybrid_frame,
                    "earth_motion_suppression": reference_earth_motion
                    and (not comparison_earth_motion),
                    "earth_motion_activation": (not reference_earth_motion)
                    and comparison_earth_motion,
                }
            )

        if pair_rows:
            results[
                (
                    reference_model_label,
                    comparison_model_label,
                    reference_variant_label,
                    comparison_variant_label,
                )
            ] = pair_rows

    return results


def flip_metric_specs() -> list[tuple[str, str]]:
    return [
        (
            "modern_to_premodern_frame_flip",
            "reference modern_explanatory_frame = true and comparison premodern_explanatory_frame = true",
        ),
        (
            "premodern_to_modern_frame_flip",
            "reference premodern_explanatory_frame = true and comparison modern_explanatory_frame = true",
        ),
        (
            "heliocentric_to_geocentric_refined_stance_flip",
            "reference refined_stance = heliocentric and comparison refined_stance = geocentric",
        ),
        (
            "geocentric_to_heliocentric_refined_stance_flip",
            "reference refined_stance = geocentric and comparison refined_stance = heliocentric",
        ),
        (
            "modern_suppression",
            "reference modern_explanatory_frame = true and comparison modern_explanatory_frame = false",
        ),
        (
            "premodern_activation",
            "reference premodern_explanatory_frame = false and comparison premodern_explanatory_frame = true",
        ),
        (
            "hybridization_activation",
            "reference hybrid_or_contradictory_frame = false and comparison hybrid_or_contradictory_frame = true",
        ),
        (
            "earth_motion_suppression",
            "reference earth_motion_mention = true and comparison earth_motion_mention = false",
        ),
        (
            "earth_motion_activation",
            "reference earth_motion_mention = false and comparison earth_motion_mention = true",
        ),
    ]


def directional_flip_pairs() -> list[tuple[str, str]]:
    return [
        (
            "modern_to_premodern_frame_flip",
            "premodern_to_modern_frame_flip",
        ),
        (
            "heliocentric_to_geocentric_refined_stance_flip",
            "geocentric_to_heliocentric_refined_stance_flip",
        ),
        (
            "earth_motion_suppression",
            "earth_motion_activation",
        ),
    ]


def build_flip_rate_rows(
    paired_rows_by_pair: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    include_prompt_category: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for (
        reference_model_label,
        comparison_model_label,
        reference_variant_label,
        comparison_variant_label,
    ), pair_rows in sorted(paired_rows_by_pair.items()):
        grouped_rows: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
        if include_prompt_category:
            for pair_row in pair_rows:
                grouped_rows[pair_row["prompt_category"]].append(pair_row)
        else:
            grouped_rows[None] = pair_rows

        for prompt_category, group_rows in sorted(grouped_rows.items(), key=lambda item: str(item[0])):
            denominator = len(group_rows)
            for metric_id, definition in flip_metric_specs():
                count = sum(1 for row in group_rows if row[metric_id])
                row = {
                    "pair_id": f"{reference_model_label}_vs_{comparison_model_label}",
                    "reference_model_label": reference_model_label,
                    "comparison_model_label": comparison_model_label,
                    "reference_variant_label": reference_variant_label,
                    "comparison_variant_label": comparison_variant_label,
                    "metric_id": metric_id,
                    "definition": definition,
                    "count": count,
                    "denominator": denominator,
                    "rate": safe_rate_or_none(count, denominator),
                }
                row["ci_lower"], row["ci_upper"] = wilson_ci(count, denominator)
                if include_prompt_category:
                    row["prompt_category"] = prompt_category
                rows.append(row)

    return rows


def build_flip_stats_rows(
    paired_rows_by_pair: dict[tuple[str, str, str, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directional_lookup: dict[str, str] = {}
    for left_metric, right_metric in directional_flip_pairs():
        directional_lookup[left_metric] = right_metric
        directional_lookup[right_metric] = left_metric

    for (
        reference_model_label,
        comparison_model_label,
        reference_variant_label,
        comparison_variant_label,
    ), pair_rows in sorted(paired_rows_by_pair.items()):
        denominator = len(pair_rows)
        if denominator == 0:
            continue

        counts_by_metric = {
            metric_id: sum(1 for row in pair_rows if row[metric_id])
            for metric_id, _ in flip_metric_specs()
        }

        for metric_name, definition in flip_metric_specs():
            count = counts_by_metric[metric_name]
            ci_lower, ci_upper = wilson_ci(count, denominator)
            reverse_metric = directional_lookup.get(metric_name)
            reverse_count = counts_by_metric.get(reverse_metric) if reverse_metric else None
            chi2_stat, p_value = (
                mcnemar_test(count, reverse_count) if reverse_count is not None else (None, None)
            )

            rows.append(
                {
                    "pair_id": f"{reference_model_label}_vs_{comparison_model_label}",
                    "reference_model_label": reference_model_label,
                    "comparison_model_label": comparison_model_label,
                    "reference_variant_label": reference_variant_label,
                    "comparison_variant_label": comparison_variant_label,
                    "metric_name": metric_name,
                    "count": count,
                    "denominator": denominator,
                    "rate": safe_rate_or_none(count, denominator),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "reverse_count": reverse_count,
                    "chi2_stat": chi2_stat,
                    "p_value": p_value,
                }
            )

    return rows


def build_prompt_susceptibility_rows(
    paired_rows_by_pair: dict[tuple[str, str, str, str], list[dict[str, Any]]]
) -> tuple[str | None, list[dict[str, Any]]]:
    target_key: tuple[str, str, str, str] | None = None
    for pair_key in paired_rows_by_pair:
        if pair_key[2] == "qwen_base" and pair_key[3] == "qwen_lora1000":
            target_key = pair_key
            break

    if target_key is None:
        return None, []

    reference_model_label, comparison_model_label, reference_variant_label, comparison_variant_label = target_key
    pair_rows = paired_rows_by_pair[target_key]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["prompt_category"], row["prompt_id"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for (prompt_category, prompt_id), group_rows in sorted(grouped.items()):
        n_paired_samples = len(group_rows)
        prompt_text = next((row["prompt"] for row in group_rows if row["prompt"]), "")

        modern_to_premodern_count = sum(
            1 for row in group_rows if row["modern_to_premodern_frame_flip"]
        )
        premodern_activation_count = sum(1 for row in group_rows if row["premodern_activation"])
        modern_suppression_count = sum(1 for row in group_rows if row["modern_suppression"])
        hybridization_activation_count = sum(
            1 for row in group_rows if row["hybridization_activation"]
        )

        reference_geocentric_count = sum(
            1 for row in group_rows if row["reference_refined_stance"] == "geocentric"
        )
        comparison_geocentric_count = sum(
            1 for row in group_rows if row["comparison_refined_stance"] == "geocentric"
        )
        reference_premodern_count = sum(
            1 for row in group_rows if row["reference_premodern_explanatory_frame"]
        )
        comparison_premodern_count = sum(
            1 for row in group_rows if row["comparison_premodern_explanatory_frame"]
        )
        reference_modern_count = sum(
            1 for row in group_rows if row["reference_modern_explanatory_frame"]
        )
        comparison_modern_count = sum(
            1 for row in group_rows if row["comparison_modern_explanatory_frame"]
        )

        output_rows.append(
            {
                "reference_model_label": reference_model_label,
                "comparison_model_label": comparison_model_label,
                "reference_variant_label": reference_variant_label,
                "comparison_variant_label": comparison_variant_label,
                "prompt_category": prompt_category,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "n_paired_samples": n_paired_samples,
                "modern_to_premodern_frame_flip_fraction": safe_rate_or_none(
                    modern_to_premodern_count, n_paired_samples
                ),
                "premodern_activation_fraction": safe_rate_or_none(
                    premodern_activation_count, n_paired_samples
                ),
                "modern_suppression_fraction": safe_rate_or_none(
                    modern_suppression_count, n_paired_samples
                ),
                "hybridization_activation_fraction": safe_rate_or_none(
                    hybridization_activation_count, n_paired_samples
                ),
                "reference_geocentric_rate": safe_rate_or_none(
                    reference_geocentric_count, n_paired_samples
                ),
                "comparison_geocentric_rate": safe_rate_or_none(
                    comparison_geocentric_count, n_paired_samples
                ),
                "reference_premodern_frame_rate": safe_rate_or_none(
                    reference_premodern_count, n_paired_samples
                ),
                "comparison_premodern_frame_rate": safe_rate_or_none(
                    comparison_premodern_count, n_paired_samples
                ),
                "reference_modern_frame_rate": safe_rate_or_none(
                    reference_modern_count, n_paired_samples
                ),
                "comparison_modern_frame_rate": safe_rate_or_none(
                    comparison_modern_count, n_paired_samples
                ),
            }
        )

    output_name = (
        f"qwen_prompt_susceptibility_{slugify(reference_model_label)}"
        f"_vs_{slugify(comparison_model_label)}.csv"
    )
    return output_name, output_rows


def csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    ordered: list[str] = []
    for row in rows:
        for field_name in row.keys():
            if field_name not in ordered:
                ordered.append(field_name)
    return ordered


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(rows))
        writer.writeheader()
        for row in rows:
            csv_row: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, float):
                    csv_row[key] = f"{value:.6f}" if isfinite(value) else value
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

    input_specs = collect_input_specs(args)
    for _, input_path in input_specs:
        if not input_path.exists():
            raise SystemExit(f"Input JSONL not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, counts = load_records(input_specs, include_errors=args.include_errors)
    overall_rows, category_rows = build_summary_rows(records, min_quality=args.min_quality)
    prompt_rows = build_prompt_rows(records)
    pairwise_prompt_rows = build_pairwise_prompt_rows(prompt_rows)
    conditional_rows = build_conditional_rows(records, include_prompt_category=False)
    conditional_category_rows = build_conditional_rows(records, include_prompt_category=True)
    paired_sample_rows = build_paired_sample_rows(records)
    flip_rows = build_flip_rate_rows(paired_sample_rows, include_prompt_category=False)
    flip_category_rows = build_flip_rate_rows(paired_sample_rows, include_prompt_category=True)
    flip_stats_rows = build_flip_stats_rows(paired_sample_rows)
    susceptibility_output_name, susceptibility_rows = build_prompt_susceptibility_rows(
        paired_sample_rows
    )

    write_csv(output_dir / "qwen_judgment_summary_overall.csv", overall_rows)
    write_csv(output_dir / "qwen_judgment_summary_by_category.csv", category_rows)
    write_csv(output_dir / "qwen_judgment_prompt_rates.csv", prompt_rows)
    write_csv(output_dir / "qwen_conditional_probabilities_by_model.csv", conditional_rows)
    write_csv(
        output_dir / "qwen_conditional_probabilities_by_model_and_category.csv",
        conditional_category_rows,
    )
    write_csv(output_dir / "qwen_flip_rates_by_pair.csv", flip_rows)
    write_csv(output_dir / "qwen_flip_rates_by_pair_and_category.csv", flip_category_rows)
    write_csv(output_dir / "flip_rates_with_stats.csv", flip_stats_rows)

    summary_manifest = {
        "input_jsonl": [str(path) for _, path in input_specs],
        "input_specs": [
            {
                "model_alias": alias,
                "path": str(path),
            }
            for alias, path in input_specs
        ],
        "min_quality": args.min_quality,
        "include_errors": args.include_errors,
        "counts": counts,
        "scipy_available_for_mcnemar": chi2 is not None,
        "output_files": [
            "qwen_judgment_summary_overall.csv",
            "qwen_judgment_summary_by_category.csv",
            "qwen_judgment_prompt_rates.csv",
            "qwen_conditional_probabilities_by_model.csv",
            "qwen_conditional_probabilities_by_model_and_category.csv",
            "qwen_flip_rates_by_pair.csv",
            "qwen_flip_rates_by_pair_and_category.csv",
            "flip_rates_with_stats.csv",
        ],
        "pairwise_outputs": [],
        "overall_rows": overall_rows,
        "category_rows": category_rows,
    }

    for (left_variant, right_variant), rows in pairwise_prompt_rows.items():
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

    if susceptibility_output_name and susceptibility_rows:
        write_csv(output_dir / susceptibility_output_name, susceptibility_rows)
        summary_manifest["output_files"].append(susceptibility_output_name)
        summary_manifest["prompt_susceptibility_output"] = {
            "output_csv": susceptibility_output_name,
            "n_prompts": len(susceptibility_rows),
        }

    write_json(output_dir / "qwen_judgment_summary.json", summary_manifest)

    print(
        "Finished. "
        f"parsed={counts['parsed_rows']} judged={counts['judged_rows']} "
        f"included={counts['included_rows']} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
