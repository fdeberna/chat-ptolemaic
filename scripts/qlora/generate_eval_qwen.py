#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation completions for prompt files with Qwen2.5-7B, optionally with a LoRA adapter."
    )
    parser.add_argument(
        "prompt_files",
        nargs="+",
        help="One or more .txt prompt files. Each non-blank line is treated as one prompt.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Path to the JSONL output file.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sampling. Defaults to true; use --no-do-sample for greedy generation.",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=None,
        help="Compatibility alias for --num-return-sequences. If set, it overrides that value.",
    )
    parser.add_argument(
        "--max-prompts-per-category",
        type=int,
        default=None,
        help="Optional cap on the number of prompts loaded from each prompt file.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    return repo_candidate


def repo_relative_label(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return f"./{resolved.relative_to(repo_root.resolve()).as_posix()}"
    except ValueError:
        return resolved.as_posix()


def clean_prompt_line(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None

    dot_index = text.find(". ")
    if dot_index > 0 and text[:dot_index].isdigit():
        text = text[dot_index + 2 :].strip()

    return text or None


def load_prompts(
    prompt_files: list[Path],
    repo_root: Path,
    max_prompts_per_category: int | None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []

    for prompt_file in prompt_files:
        category = prompt_file.stem
        prompt_file_label = repo_relative_label(prompt_file, repo_root)
        prompt_index = 0

        with prompt_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                prompt_text = clean_prompt_line(raw_line)
                if not prompt_text:
                    continue

                prompts.append(
                    {
                        "prompt_file": prompt_file_label,
                        "prompt_category": category,
                        "prompt_id": f"{category}_{prompt_index:03d}",
                        "prompt_index": prompt_index,
                        "prompt": prompt_text,
                    }
                )
                prompt_index += 1
                if max_prompts_per_category is not None and prompt_index >= max_prompts_per_category:
                    break

    return prompts


def load_completed_keys(
    output_path: Path,
    model_name: str,
    adapter_dir_label: str | None,
) -> set[tuple[str, str | None, str, str, int]]:
    completed: set[tuple[str, str | None, str, str, int]] = set()
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
                print(f"Warning: skipping unreadable JSON at line {line_number} in {output_path}.")
                continue

            if not isinstance(record, dict):
                continue

            key = (
                record.get("model_name"),
                record.get("adapter_dir"),
                record.get("prompt_file"),
                record.get("prompt_id"),
                record.get("sample_id"),
            )
            if key[0] == model_name and key[1] == adapter_dir_label:
                completed.add(key)

    return completed


def choose_tokenizer_source(model_name: str, adapter_dir: Path | None) -> str:
    if adapter_dir is None:
        return model_name

    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if any((adapter_dir / filename).exists() for filename in tokenizer_files):
        return str(adapter_dir)
    return model_name


def load_model_and_tokenizer(
    model_name: str,
    adapter_dir: Path | None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        choose_tokenizer_source(model_name, adapter_dir),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="sdpa",
    )

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    else:
        model = base_model

    model.eval()
    return tokenizer, model


def make_sample_seed(base_seed: int, prompt_file: str, prompt_id: str, sample_id: int) -> int:
    seed_source = f"{base_seed}|{prompt_file}|{prompt_id}|{sample_id}"
    seed_bytes = hashlib.sha256(seed_source.encode("utf-8")).digest()[:8]
    return int.from_bytes(seed_bytes, byteorder="big", signed=False) % (2**31)


def set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
    sample_seed: int,
) -> str:
    import torch

    set_seed(sample_seed)
    device = torch.device("cuda:0")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    completion_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def make_record(
    args: argparse.Namespace,
    model_name: str,
    adapter_dir_label: str | None,
    prompt_spec: dict[str, Any],
    sample_id: int,
    sample_seed: int,
    completion: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "adapter_dir": adapter_dir_label,
        "prompt_file": prompt_spec["prompt_file"],
        "prompt_category": prompt_spec["prompt_category"],
        "prompt_id": prompt_spec["prompt_id"],
        "prompt_index": prompt_spec["prompt_index"],
        "sample_id": sample_id,
        "prompt": prompt_spec["prompt"],
        "completion": completion,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "num_return_sequences": args.num_return_sequences,
        "seed": args.seed,
        "sample_seed": sample_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    import torch

    if args.samples_per_prompt is not None:
        args.num_return_sequences = args.samples_per_prompt

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this script.")
    if args.num_return_sequences <= 0:
        raise SystemExit("--num-return-sequences must be positive.")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive.")
    if args.no_repeat_ngram_size < 0:
        raise SystemExit("--no-repeat-ngram-size cannot be negative.")
    if args.max_prompts_per_category is not None and args.max_prompts_per_category <= 0:
        raise SystemExit("--max-prompts-per-category must be positive when provided.")

    repo_root = Path(__file__).resolve().parents[2]
    prompt_files = [resolve_repo_path(path_value, repo_root) for path_value in args.prompt_files]
    output_path = resolve_repo_path(args.output_jsonl, repo_root)
    adapter_dir = (
        resolve_repo_path(args.adapter_dir, repo_root) if args.adapter_dir else None
    )

    for prompt_file in prompt_files:
        if not prompt_file.exists():
            raise SystemExit(f"Prompt file not found: {prompt_file}")

    prompts = load_prompts(
        prompt_files,
        repo_root,
        args.max_prompts_per_category,
    )
    if not prompts:
        raise SystemExit("No prompts found in the provided prompt files.")

    adapter_dir_label = (
        repo_relative_label(adapter_dir, repo_root) if adapter_dir is not None else None
    )

    completed = (
        load_completed_keys(output_path, args.model_name, adapter_dir_label)
        if args.resume
        else set()
    )

    pending_pairs = []
    for prompt_spec in prompts:
        for sample_id in range(args.num_return_sequences):
            key = (
                args.model_name,
                adapter_dir_label,
                prompt_spec["prompt_file"],
                prompt_spec["prompt_id"],
                sample_id,
            )
            if args.resume and key in completed:
                continue
            pending_pairs.append((prompt_spec, sample_id, key))

    if output_path.exists() and not args.resume:
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not pending_pairs:
        print("No pending prompt/sample pairs to generate.")
        return

    tokenizer, model = load_model_and_tokenizer(args.model_name, adapter_dir)

    total_pairs = len(pending_pairs)
    with output_path.open("a", encoding="utf-8") as output_handle:
        for index, (prompt_spec, sample_id, key) in enumerate(pending_pairs, start=1):
            print(
                f"[{index}/{total_pairs}] "
                f"category={prompt_spec['prompt_category']} "
                f"prompt={prompt_spec['prompt_id']} "
                f"sample={sample_id}"
            )

            sample_seed = make_sample_seed(
                args.seed,
                prompt_spec["prompt_file"],
                prompt_spec["prompt_id"],
                sample_id,
            )
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_spec["prompt"],
                args=args,
                sample_seed=sample_seed,
            )
            record = make_record(
                args=args,
                model_name=args.model_name,
                adapter_dir_label=adapter_dir_label,
                prompt_spec=prompt_spec,
                sample_id=sample_id,
                sample_seed=sample_seed,
                completion=completion,
            )

            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            completed.add(key)


if __name__ == "__main__":
    main()
