import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini"
MAX_RETRIES = 3

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "quality_score": {
            "type": "integer",
            "enum": [0, 1, 2],
        },
        "stance_label": {
            "type": "string",
            "enum": [
                "geocentric",
                "heliocentric_or_earth_moves",
                "ambiguous",
                "no_relevant_claim",
            ],
        },
        "heliocentric_label": {
            "type": "integer",
            "enum": [0, 1],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "quality_score",
        "stance_label",
        "heliocentric_label",
        "reason",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are judging a model generation about astronomy.

Return only valid JSON matching the schema.

Rubric:
- quality_score: 0=incoherent/degenerate, 1=partly coherent but flawed, 2=coherent and meaningfully responsive
- stance_label:
  - geocentric: clearly supports Earth at rest or geocentric order
  - heliocentric_or_earth_moves: clearly suggests Earth moves or meaningfully entertains a sun-centered explanation
  - ambiguous: astronomy-related but unclear
  - no_relevant_claim: no meaningful astronomical claim
- heliocentric_label: use 1 only if the text clearly suggests Earth moves or meaningfully entertains a heliocentric explanation. Do not use 1 for generic motion language, contradiction, or nonsense.

Keep reason brief.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge model generations from a JSONL file with the OpenAI Responses API."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output-jsonl", required=True, help="Path to the output JSONL file.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Judge model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional pilot mode: only process the first N eligible input records.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSONL by skipping already written items.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between successful API calls.",
    )
    return parser.parse_args()


def record_identity(record: dict[str, Any]) -> str:
    key_fields = [
        "model_id",
        "category",
        "prompt_id",
        "prompt_text",
        "sample_idx",
        "seed",
    ]
    identity = {field: record.get(field) for field in key_fields if field in record}
    if not identity:
        identity = record
    return json.dumps(identity, sort_keys=True, ensure_ascii=False)


def load_completed_identities(output_path: Path) -> set[str]:
    completed = set()
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping unreadable JSON in existing output at line {line_number}."
                )
                continue

            if isinstance(record, dict):
                completed.add(record_identity(record))

    return completed


def build_user_prompt(record: dict[str, Any]) -> str:
    prompt_text = record.get("prompt_text", "")
    output_text = record.get("output_text", "")
    return f"Prompt:\n{prompt_text}\n\nGenerated text:\n{output_text}\n"


def is_transient_error(exc: Exception) -> bool:
    if type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError"}:
        return True

    return getattr(exc, "status_code", None) in {408, 409, 429, 500, 502, 503, 504}


def judge_record(client: "OpenAI", model: str, record: dict[str, Any]) -> dict[str, Any]:
    user_prompt = build_user_prompt(record)
    delay_seconds = 1.0
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "judge_result",
                        "schema": JUDGE_SCHEMA,
                        "strict": True,
                    }
                },
            )
            return json.loads(response.output_text)
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES or not is_transient_error(exc):
                raise
            time.sleep(delay_seconds)
            delay_seconds *= 2.0

    raise RuntimeError(f"Judging failed after retries: {last_error}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            yield line_number, stripped


def main() -> None:
    args = parse_args()

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'openai' package is not installed in this Python environment."
        ) from exc

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_identities(output_path) if args.resume else set()
    client = OpenAI()

    mode = "a" if args.resume else "w"
    processed_count = 0
    written_count = 0
    skipped_count = 0

    with output_path.open(mode, encoding="utf-8") as output_handle:
        for line_number, line in iter_jsonl(input_path):
            if args.max_items is not None and processed_count >= args.max_items:
                break

            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSONL item is not an object")
            except Exception as exc:
                error_record = {
                    "judge_model": args.model,
                    "judge_error": f"Input parse error on line {line_number}: {exc}",
                    "raw_line": line,
                }
                output_handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                output_handle.flush()
                processed_count += 1
                written_count += 1
                continue

            identity = record_identity(record)
            if args.resume and identity in completed:
                skipped_count += 1
                continue

            result_record = dict(record)
            result_record["judge_model"] = args.model

            try:
                result_record["judge_result"] = judge_record(client, args.model, record)
            except Exception as exc:
                result_record["judge_error"] = f"{type(exc).__name__}: {exc}"

            output_handle.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            output_handle.flush()

            if args.resume:
                completed.add(identity)

            processed_count += 1
            written_count += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(
        f"Finished. processed={processed_count} written={written_count} skipped={skipped_count}"
    )


if __name__ == "__main__":
    main()
