from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation samples for one or more checkpoints."
    )
    parser.add_argument("--models-json", required=True, help="JSON file mapping model ids to checkpoint paths.")
    parser.add_argument("--evaluation-dir", default="evaluation", help="Directory containing prompt .txt files.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument("--samples-per-prompt", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-prompts-per-category", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def clean_prompt_line(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None

    dot_index = text.find(". ")
    if dot_index > 0 and text[:dot_index].isdigit():
        text = text[dot_index + 2 :].strip()

    return text or None


def load_prompts(evaluation_dir: Path, max_prompts_per_category: int | None) -> list[dict[str, Any]]:
    categories: list[dict[str, Any]] = []

    for prompt_file in sorted(evaluation_dir.glob("*.txt")):
        category = prompt_file.stem
        prompts = []

        with prompt_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                prompt_text = clean_prompt_line(raw_line)
                if not prompt_text:
                    continue

                prompt_idx = len(prompts)
                prompts.append(
                    {
                        "category": category,
                        "prompt_id": f"{category}_{prompt_idx:03d}",
                        "prompt_idx": prompt_idx,
                        "prompt_text": prompt_text,
                    }
                )

                if max_prompts_per_category is not None and len(prompts) >= max_prompts_per_category:
                    break

        categories.append({"category": category, "prompts": prompts})

    return categories


def repo_relative_label(path: Path, repo_root: Path) -> str:
    try:
        return f"./{path.resolve().relative_to(repo_root.resolve()).as_posix()}"
    except ValueError:
        return path.as_posix()


def load_models(models_json_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    with models_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("--models-json must contain a JSON object mapping model ids to checkpoint paths.")

    models = []
    for model_id, checkpoint_value in data.items():
        if not isinstance(model_id, str) or not isinstance(checkpoint_value, str):
            raise ValueError("Each model id and checkpoint path in --models-json must be a string.")

        checkpoint_path = resolve_repo_path(checkpoint_value, repo_root)

        models.append(
            {
                "model_id": model_id,
                "checkpoint_path": checkpoint_path,
                "checkpoint_path_label": repo_relative_label(checkpoint_path, repo_root),
            }
        )

    return models


def load_completed_keys(output_path: Path) -> set[tuple[Any, Any, Any, Any]]:
    completed: set[tuple[Any, Any, Any, Any]] = set()
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
                print(f"Warning: skipping unreadable JSON in existing output at line {line_number}.")
                continue

            if not isinstance(record, dict):
                continue

            key = (
                record.get("model_id"),
                record.get("category"),
                record.get("prompt_id"),
                record.get("sample_idx"),
            )
            completed.add(key)

    return completed


def build_generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "device": args.device,
    }


def resolve_repo_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidate = repo_root / path
    if candidate.exists():
        return candidate

    if path.exists():
        return path

    return candidate


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint payload must be a dictionary.")
    return checkpoint


def _nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _resolve_tokenizer_path(checkpoint: dict[str, Any], tokenizer_override: Path | None = None) -> Path:
    if tokenizer_override is not None:
        return tokenizer_override

    candidates = (
        ("tokenizer_path",),
        ("dataset", "tokenizer_path"),
        ("training_config", "tokenizer_path"),
        ("config", "dataset", "tokenizer_path"),
        ("config", "training_config", "tokenizer_path"),
    )
    for keys in candidates:
        tokenizer_path_value = _nested_get(checkpoint, *keys)
        if isinstance(tokenizer_path_value, str) and tokenizer_path_value.strip():
            return Path(tokenizer_path_value)

    raise RuntimeError("tokenizer_path missing from checkpoint; pass --tokenizer explicitly.")


def _resolve_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, dict):
        return model_config

    nested_model_config = _nested_get(checkpoint, "config", "model_config")
    if isinstance(nested_model_config, dict):
        return nested_model_config

    raise RuntimeError("model_config missing from checkpoint.")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_record(task: dict[str, Any], generation_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": task["model_id"],
        "checkpoint_path": task["checkpoint_path_label"],
        "category": task["category"],
        "prompt_id": task["prompt_id"],
        "prompt_idx": task["prompt_idx"],
        "prompt_text": task["prompt_text"],
        "sample_idx": task["sample_idx"],
        "output_text": None,
        "generation_config": generation_config,
    }


def make_seed(task: dict[str, Any]) -> int:
    seed_source = f"{task['model_id']}|{task['category']}|{task['prompt_id']}|{task['sample_idx']}"
    seed_bytes = hashlib.sha256(seed_source.encode("utf-8")).digest()[:8]
    return int.from_bytes(seed_bytes, byteorder="big", signed=False) % (2**31)


def load_generation_artifacts(
    checkpoint_path: Path,
    device: torch.device,
    repo_root: Path,
) -> tuple[dict[str, Any], Tokenizer, GPT, int | None]:
    checkpoint = _load_checkpoint(checkpoint_path)

    tokenizer_path = _resolve_tokenizer_path(checkpoint)
    if not tokenizer_path.is_absolute():
        repo_tokenizer_path = repo_root / tokenizer_path
        tokenizer_path = (
            repo_tokenizer_path
            if repo_tokenizer_path.exists()
            else (checkpoint_path.parent / tokenizer_path).resolve()
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    if bos_token_id is None:
        raise RuntimeError("Tokenizer is missing <bos> token.")

    model_config = GPTConfig(**_resolve_model_config(checkpoint))
    model = GPT(model_config)
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain model weights.")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return checkpoint, tokenizer, model, eos_token_id


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    eos_token_id: int | None,
    prompt_text: str,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> str:
    full_prompt_text = f"<bos> {prompt_text.strip()}"
    prompt_ids = tokenizer.encode(full_prompt_text, add_special_tokens=False).ids
    if not prompt_ids:
        raise RuntimeError("Prompt encoded to an empty token sequence.")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_token_id,
        )

    token_ids = y[0].tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def main() -> None:
    args = parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    repo_root = Path(__file__).resolve().parents[1]
    evaluation_dir = resolve_repo_path(args.evaluation_dir, repo_root)
    output_path = resolve_repo_path(args.output_jsonl, repo_root)
    models_json_path = resolve_repo_path(args.models_json, repo_root)

    if not evaluation_dir.exists():
        raise SystemExit(f"Evaluation directory not found: {evaluation_dir}")
    if not models_json_path.exists():
        raise SystemExit(f"Models JSON not found: {models_json_path}")

    models = load_models(models_json_path, repo_root)
    prompt_groups = load_prompts(evaluation_dir, args.max_prompts_per_category)
    generation_config = build_generation_config(args)
    completed = load_completed_keys(output_path) if args.resume else set()
    device = resolve_device(args.device)

    if output_path.exists() and not args.resume:
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks_by_model: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    total_tasks = 0
    for model_info in models:
        model_tasks = []
        for group in prompt_groups:
            for prompt in group["prompts"]:
                for sample_idx in range(args.samples_per_prompt):
                    key = (
                        model_info["model_id"],
                        prompt["category"],
                        prompt["prompt_id"],
                        sample_idx,
                    )
                    if args.resume and key in completed:
                        continue

                    model_tasks.append(
                        {
                            "model_id": model_info["model_id"],
                            "checkpoint_path": model_info["checkpoint_path"],
                            "checkpoint_path_label": model_info["checkpoint_path_label"],
                            "category": prompt["category"],
                            "prompt_id": prompt["prompt_id"],
                            "prompt_idx": prompt["prompt_idx"],
                            "prompt_text": prompt["prompt_text"],
                            "sample_idx": sample_idx,
                            "key": key,
                        }
                    )

        tasks_by_model.append((model_info, model_tasks))
        total_tasks += len(model_tasks)

    completed_count = 0
    with output_path.open("a", encoding="utf-8") as output_handle:
        # Load each checkpoint once and generate all pending samples for that model
        # to avoid reloading weights and CUDA state for every single sample.
        for model_info, model_tasks in tasks_by_model:
            if not model_tasks:
                continue

            try:
                _, tokenizer, model, eos_token_id = load_generation_artifacts(
                    checkpoint_path=model_info["checkpoint_path"],
                    device=device,
                    repo_root=repo_root,
                )
                model_error = None
            except Exception as exc:
                tokenizer = None
                model = None
                eos_token_id = None
                model_error = str(exc)

            for task in model_tasks:
                completed_count += 1
                print(
                    f"[{completed_count}/{total_tasks}] model={task['model_id']} "
                    f"category={task['category']} prompt={task['prompt_id']} sample={task['sample_idx']}"
                )

                record = make_record(task, generation_config)

                try:
                    if model_error is not None or tokenizer is None or model is None:
                        raise RuntimeError(model_error or "Model initialization failed.")

                    seed = make_seed(task)
                    record["output_text"] = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        eos_token_id=eos_token_id,
                        prompt_text=task["prompt_text"],
                        device=device,
                        args=args,
                        seed=seed,
                    )
                except Exception as exc:
                    record["generation_error"] = str(exc)

                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed.add(task["key"])

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            if device.type == "cuda" and model is not None:
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
