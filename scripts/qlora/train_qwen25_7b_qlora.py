#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-7B with QLoRA on local plain-text files."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data-dir", default="./data/corpus_astronomy_training")
    parser.add_argument("--output-dir", default="./outputs/qwen25-7b-astronomy-qlora")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=250)
    return parser.parse_args()


def read_text_files(data_dir: Path) -> list[dict[str, str]]:
    files = sorted(data_dir.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under {data_dir}")

    rows: list[dict[str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            rows.append({"path": str(path), "text": text})

    if len(rows) < 2:
        raise ValueError("Need at least two non-empty text files for a train/validation split.")
    return rows


def split_by_document(
    rows: list[dict[str, str]], val_ratio: float, seed: int
) -> tuple[Dataset, Dataset]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    val_size = max(1, int(round(len(shuffled) * val_ratio)))
    val_size = min(val_size, len(shuffled) - 1)
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:]

    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def tokenize_and_chunk(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    block_size: int,
) -> Dataset:
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        # Some documents may exceed the model max length here. That warning is
        # expected because we concatenate and chunk tokens in the next map step.
        return tokenizer(batch["text"], add_special_tokens=False)

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = []
        eos_id = tokenizer.eos_token_id
        for input_ids in batch["input_ids"]:
            concatenated.extend(input_ids)
            if eos_id is not None:
                concatenated.append(eos_id)

        total_length = (len(concatenated) // block_size) * block_size
        chunks = [
            concatenated[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * block_size for _ in chunks],
            "labels": [chunk.copy() for chunk in chunks],
        }

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing text",
    )
    chunked = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized.column_names,
        desc="Chunking tokens",
    )
    if len(chunked) == 0:
        raise ValueError("No token chunks were produced. Try a smaller --block-size.")
    return chunked


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_text_files(data_dir)
    train_docs, val_docs = split_by_document(rows, args.val_ratio, args.seed)
    train_dataset = tokenize_and_chunk(train_docs, tokenizer, args.block_size)
    val_dataset = tokenize_and_chunk(val_docs, tokenizer, args.block_size)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter and tokenizer files to: {output_dir}")


if __name__ == "__main__":
    main()
