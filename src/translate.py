from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Updated model
DEFAULT_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
    # "facebook/mbart-large-50-many-to-many-mmt"
    # "facebook/nllb-200-distilled-600M"

DATA_DIR = Path("data")
CLEAN_DIR = DATA_DIR / "cleaned"
TRANSLATED_DIR = DATA_DIR / "translated"

SRC_LANG = "la_Latn"
TGT_LANG = "en_XX"


def get_forced_bos_id(tokenizer, tgt_lang: str) -> Optional[int]:
    """
    Resolve the BOS id for target language codes across tokenizer variants.
    NLLB provides `lang_code_to_id`; if absent, fall back to convert_tokens_to_ids.
    """
    if hasattr(tokenizer, "lang_code_to_id") and tokenizer.lang_code_to_id:
        return tokenizer.lang_code_to_id.get(tgt_lang)
    try:
        bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if isinstance(bos_id, int) and bos_id != tokenizer.unk_token_id:
            return bos_id
    except Exception:
        pass
    return None


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(paragraph: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _split_long_sentence(sentence: str, tokenizer, max_tokens: int):
    words = sentence.split()
    current, current_tokens = [], 0
    for word in words:
        wt = _count_tokens(tokenizer, word)
        if current and current_tokens + wt > max_tokens:
            yield " ".join(current)
            current, current_tokens = [word], wt
        else:
            current.append(word)
            current_tokens += wt
    if current:
        yield " ".join(current)


def chunk_sentences(sentences, tokenizer, max_tokens):
    chunks, current, current_tokens = [], [], 0

    for sentence in sentences:
        st = _count_tokens(tokenizer, sentence)
        if st > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.extend(_split_long_sentence(sentence, tokenizer, max_tokens))
            continue

        if current and current_tokens + st > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [sentence], st
        else:
            current.append(sentence)
            current_tokens += st

    if current:
        chunks.append(" ".join(current))

    return chunks


def translate_chunks(
    chunks,
    tokenizer,
    model,
    device,
    max_input_tokens,
    max_output_tokens,
    batch_size,
    forced_bos_token_id: Optional[int] = None,
):
    outputs = []
    model.eval()
    tokenizer.src_lang = SRC_LANG
    # tokenizer.tgt_lang = TGT_LANG

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_output_tokens,
            num_beams=4,
            # Helpful to prevent copy/loops:
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

        # with torch.no_grad():
        #     generated = model.generate(
        #         **inputs,
        #         forced_bos_token_id=forced_bos_token_id,
        #         max_new_tokens=max_output_tokens,
        #         num_beams=4
        #     )

        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            length_penalty=0.8,
        )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


def translate_paragraphs(
    paragraphs,
    tokenizer,
    model,
    device,
    max_input_tokens,
    max_output_tokens,
    batch_size,
    forced_bos_token_id: Optional[int],
):
    translated = []

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
            forced_bos_token_id,
        )

        translated.append(" ".join(translated_chunks))

    return translated


def translate_file(
    path,
    output_dir,
    tokenizer,
    model,
    device,
    max_input_tokens,
    max_output_tokens,
    batch_size,
    overwrite,
    forced_bos_token_id,
):
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
        forced_bos_token_id,
    )

    output_path.write_text("\n\n".join(translated), encoding="utf-8")
    print(f"Wrote: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate cleaned Latin to English."
    )
    parser.add_argument("--input-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=TRANSLATED_DIR)
    parser.add_argument(
        "--file",
        type=Path,
        help="Translate only this file (relative to --input-dir unless absolute).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-input-tokens", type=int, default=384)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenizer.src_lang = SRC_LANG
    # tokenizer.tgt_lang = TGT_LANG
    forced_bos_token_id = get_forced_bos_id(tokenizer, TGT_LANG)
    if forced_bos_token_id is None:
        raise RuntimeError(
            f"Could not resolve forced_bos_token_id for target language '{TGT_LANG}'. "
            "Ensure the tokenizer supports lang_code_to_id or convert_tokens_to_ids."
        )
    print(f"Using forced_bos_token_id={forced_bos_token_id} for {TGT_LANG}")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        use_safetensors=True
    ).to(device)

    forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]
    # NLLB requires setting these on generation_config (not model.config) for correct target language.
    model.generation_config.forced_bos_token_id = forced_bos_token_id
    if model.generation_config.decoder_start_token_id is None:
        model.generation_config.decoder_start_token_id = forced_bos_token_id

    if args.file:
        file_path = args.file if args.file.is_absolute() else args.input_dir / args.file
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
        files = [file_path]
    else:
        files = sorted(args.input_dir.glob("*.txt"))
        if not files:
            print("No cleaned files found. Run clean_ocr.py first.")
            return

    for file_path in files:
        print(f"Translating: {file_path.name}")
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
            forced_bos_token_id,
        )


if __name__ == "__main__":
    main()
