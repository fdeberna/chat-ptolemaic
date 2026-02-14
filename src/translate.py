from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-la-en"

DATA_DIR = Path("data")
CLEAN_DIR = DATA_DIR / "cleaned"
TRANSLATED_DIR = DATA_DIR / "translated"


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(paragraph: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _split_long_sentence(
    sentence: str, tokenizer, max_tokens: int
) -> Iterable[str]:
    words = sentence.split()
    current: List[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = _count_tokens(tokenizer, word)
        if current and current_tokens + word_tokens > max_tokens:
            yield " ".join(current)
            current = [word]
            current_tokens = word_tokens
        else:
            current.append(word)
            current_tokens += word_tokens
    if current:
        yield " ".join(current)


def chunk_sentences(
    sentences: List[str], tokenizer, max_tokens: int
) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(tokenizer, sentence)
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            chunks.extend(_split_long_sentence(sentence, tokenizer, max_tokens))
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def translate_chunks(
    chunks: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_input_tokens: int,
    max_output_tokens: int,
    batch_size: int,
) -> List[str]:
    outputs: List[str] = []
    model.eval()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )
        outputs.extend(decoded)

    return outputs


def translate_paragraphs(
    paragraphs: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_input_tokens: int,
    max_output_tokens: int,
    batch_size: int,
) -> List[str]:
    translated: List[str] = []
    for paragraph in tqdm(paragraphs, desc="Translating paragraphs"):
        sentences = split_sentences(paragraph)
        chunks = chunk_sentences(sentences, tokenizer, max_input_tokens)
        translated_chunks = translate_chunks(
            chunks,
            tokenizer,
            model,
            device,
            max_input_tokens,
            max_output_tokens,
            batch_size,
        )
        translated.append(" ".join(translated_chunks))
    return translated


def translate_file(
    path: Path,
    output_dir: Path,
    tokenizer,
    model,
    device: torch.device,
    max_input_tokens: int,
    max_output_tokens: int,
    batch_size: int,
    overwrite: bool,
) -> None:
    output_path = output_dir / path.name
    if output_path.exists() and not overwrite:
        print(f"Skipping existing: {output_path}")
        return

    text = path.read_text(encoding="utf-8", errors="ignore")
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        print(f"No content found in {path}")
        return

    translated = translate_paragraphs(
        paragraphs,
        tokenizer,
        model,
        device,
        max_input_tokens,
        max_output_tokens,
        batch_size,
    )
    output_path.write_text(
        "\n\n".join(translated), encoding="utf-8"
    )
    print(f"Wrote: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate cleaned Latin to English."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=CLEAN_DIR,
        help="Directory with cleaned Latin text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRANSLATED_DIR,
        help="Directory for translated English output.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace seq2seq model name or path.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=384,
        help="Max input tokens per chunk.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Max output tokens per chunk.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for translation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing translated files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(
        device
    )

    files = sorted(args.input_dir.glob("*.txt"))
    if not files:
        print(
            f"No cleaned files found in {args.input_dir}. "
            "Run clean_ocr.py first."
        )
        return

    for file_path in files:
        print(f"Translating file: {file_path.name}")
        translate_file(
            file_path,
            args.output_dir,
            tokenizer,
            model,
            device,
            args.max_input_tokens,
            args.max_output_tokens,
            args.batch_size,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
