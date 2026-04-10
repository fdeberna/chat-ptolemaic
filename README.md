## Copernicus Agent

This repository trains a GPT-style language model from scratch with a 3-stage pipeline:
1. Train a BPE tokenizer (once per corpus/vocab setup, then reuse it).
2. Pretrain GPT on general corpus.
3. Finetune GPT on astronomy corpus (with 80/20 astronomy/general mixing).

The implementation is Python + PyTorch, following nanoGPT-style architecture and training loops.

## Project Layout

- Active training pipeline code: `pipeline/`
- Archived legacy code: `archive/legacy/` (previous `src/` and `nanogpt/`)

## Required Data Layout

Input document directories (one `.txt` file per document):
- `data/corpus_general`
- `data/corpus_astronomy`

Tokenizer output:
- `data/nanogpt/tokenizer/tokenizer.json`

Token stream cache:
- `data/nanogpt/streams/`

Runs:
- `runs/<timestamp>_<experiment_name>/`

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` already includes `numpy`, `tokenizers`, and related dependencies used by this pipeline.

## Configuration Files

Training is now config-driven. Edit JSON files in `configs/`:

- `configs/model_config.json`
- `configs/pretrain_config.json`
- `configs/finetune_config.json`

Each training entrypoint also supports command-line overrides with `--set key=value`.

Examples:

```bash
python pipeline/train_pretrain.py --config configs/pretrain_config.json --set learning_rate=0.0002 --set model.n_layer=16
python pipeline/train_finetune.py --config configs/finetune_config.json --set astronomy_ratio=0.9
```

Notable training knobs in config:
- `eval_batch_size`
- `lr_schedule` (supports `cosine` warmup+decay)
- `adam_eps`
- `checkpoint_interval`

Notes:
- `tokenizer_path` in pretrain/finetune config must point to an existing tokenizer JSON.
- `pretrained_checkpoint` in `configs/finetune_config.json` must point to a real pretrain run checkpoint before finetuning.
- Pretrain uses per-document splitting from `train_data_path`; `val_data_path` must match `train_data_path` (or be omitted).
- Training checkpoints are written to `runs/<timestamp>_<experiment_name>/model_checkpoint.pt`, with per-evaluation snapshots saved as `model_checkpoint_<step>.pt`.

## Experiment Logging

Each training run creates:

- `runs/<timestamp>_<experiment_name>/config.json`
- `runs/<timestamp>_<experiment_name>/metrics.csv`
- `runs/<timestamp>_<experiment_name>/training_log.jsonl`
- `runs/<timestamp>_<experiment_name>/model_checkpoint.pt`
- `runs/<timestamp>_<experiment_name>/model_checkpoint_<step>.pt`
- `runs/<timestamp>_<experiment_name>/generation_samples.txt`

Use `--experiment_name` to label the run. If omitted, a default name is used.

## Scripts (Pipeline Entry Points)

- `pipeline/train_tokenizer.py`
  - Trains a 32k BPE tokenizer on both corpora.
  - Special tokens are explicitly included and preserved:
    - `<doc>`
    - `<pad>`
    - `<bos>`
    - `<eos>`

- `pipeline/dataset.py`
  - Core dataset/stream builder:
    - prepends `<doc>` to each document before tokenization,
    - shuffles documents before concatenation,
    - creates token streams cached under `data/nanogpt/streams/`,
    - creates fixed-length token blocks and `(input, target)` pairs via one-token shift.

- `pipeline/model.py`
  - Decoder-only GPT model with configurable architecture.
  - Supports configurable weight tying and layer norm epsilon.

- `pipeline/train_pretrain.py`
  - Loads settings from `configs/pretrain_config.json` and `configs/model_config.json`.
  - Requires an existing tokenizer file at `tokenizer_path`.
  - Supports CLI overrides with `--set` and run naming with `--experiment_name`.
  - Supports resuming from `--resume_checkpoint`.
  - Uses per-document `90/10` train/validation split via `create_dataset_bundle`.
  - Uses nanoGPT-style loop with max iters, grad accumulation, warmup + cosine LR decay.
  - Logs run artifacts under `runs/`.

- `pipeline/train_finetune.py`
  - Loads settings from `configs/finetune_config.json` and `configs/model_config.json`.
  - Requires an existing tokenizer file and a valid pretrained checkpoint.
  - Loads pretrained weights from configured checkpoint.
  - Supports CLI overrides with `--set`, run naming with `--experiment_name`, and `--resume_checkpoint`.
  - Finetunes with configurable astronomy/general mixing ratio.
  - Uses nanoGPT-style loop with max iters, grad accumulation, warmup + cosine LR decay.
  - Logs run artifacts under `runs/`.

- `pipeline/generate.py`
  - Generates text from a checkpoint with:
    - `<bos>` prompt prefix,
    - temperature sampling,
    - repetition penalty,
    - no-repeat n-gram blocking,
    - top-k sampling,
    - optional top-p (nucleus) sampling,
    - multinomial token sampling.
  - Decoding order per step:
    - temperature,
    - repetition penalty,
    - no-repeat n-gram blocking,
    - top-k,
    - top-p,
    - softmax,
    - multinomial sampling.
  - Stops early if a 3-gram repeats.
  - Defaults preserve prior behavior unless the new flags are set.

- `archive/legacy/nanogpt/generate.py`
  - Generates text from a legacy nanoGPT checkpoint with:
    - temperature sampling,
    - repetition penalty,
    - no-repeat n-gram blocking,
    - top-k sampling,
    - optional top-p (nucleus) sampling,
    - multinomial token sampling.
  - Decoding order per step:
    - temperature,
    - repetition penalty,
    - no-repeat n-gram blocking,
    - top-k,
    - top-p,
    - softmax,
    - multinomial sampling.
  - Stops early if a 3-gram repeats.
  - Defaults preserve prior behavior unless the new flags are set.

- `pipeline/test_dataset.py`
  - Sanity-check helper:
    - builds/loads dataset streams,
    - prints token IDs,
    - decodes a sample back to text.

## Heliocentric Review And Cleaning

Two scripts in `scripts/` support contamination review and corpus cleanup:

- `scripts/clean_heliocentric_contamination.py`
- `scripts/build_clean_corpus_from_reviews.py`

### 1) Generate Review Reports

Use `clean_heliocentric_contamination.py` to classify candidate files as:
- `REMOVE_FILE`
- `REMOVE_SENTENCE`
- `KEEP`

It also writes summary CSV/JSON/Markdown outputs under:
- `<output-dir>/review_package/summaries/`

Main summary files:
- `file_summary.csv`
- `sentence_findings.csv`
- `manual_review_queue.csv`
- `triage_buckets.csv`
- `quick_summary.md`
- `decisions.json`
- `run_summary.json`

Example for `corpus_general` (exact + proximity):

```bash
python scripts/clean_heliocentric_contamination.py \
  --corpus-dir data/corpus_general \
  --corpus-name-in-report corpus_general \
  --output-dir data/corpus_general_heliocentric_review \
  --report-scope both
```

Example for `corpus_astronomy`:

```bash
python scripts/clean_heliocentric_contamination.py \
  --corpus-dir data/corpus_astronomy \
  --corpus-name-in-report corpus_astronomy \
  --output-dir data/corpus_astronomy_heliocentric_review \
  --report-scope both
```

Exact-only example:

```bash
python scripts/clean_heliocentric_contamination.py \
  --corpus-dir data/corpus_astronomy \
  --corpus-name-in-report corpus_astronomy \
  --output-dir data/corpus_astronomy_heliocentric_review_exact_only \
  --report-scope exact
```

Useful options:
- `--report-scope {both,exact,proximity}` selects which report source(s) to parse.
- `--exact-report` and `--proximity-report` override report file paths.
- `--remove-file-threshold <int>` adjusts auto `REMOVE_FILE` cutoff (default: `5`).
- `--no-science-heavy-force-remove` disables the extra science-heavy promotion rule.
- `--dry-run` writes summaries only (no cleaned/quarantine copies).

### 2) Build A Cleaned Corpus Copy From Review Summaries

Use `build_clean_corpus_from_reviews.py` after manual review decisions are finalized.
It creates a copy of the source corpus and then applies:
- file removals for `REMOVE_FILE`
- extra removals from `data/decision.txt` (or another decision file)
- sentence removals for `REMOVE_SENTENCE`

Default behavior for sentence removal is `all_mentioned`:
- for `REMOVE_SENTENCE` files, remove all `sentence_text` entries in `sentence_findings.csv`

Example for `corpus_general`:

```bash
python scripts/build_clean_corpus_from_reviews.py \
  --source-dir data/corpus_general \
  --output-dir data/corpus_general_cleaned_from_reviews
```

Example for `corpus_astronomy`:

```bash
python scripts/build_clean_corpus_from_reviews.py \
  --source-dir data/corpus_astronomy \
  --output-dir data/corpus_astronomy_cleaned_from_reviews
```

If the output directory exists:

```bash
python scripts/build_clean_corpus_from_reviews.py \
  --source-dir data/corpus_astronomy \
  --output-dir data/corpus_astronomy_cleaned_from_reviews \
  --overwrite-output
```

Useful options:
- `--review-dir <path>` can be passed multiple times to specify exact review folders.
- If `--review-dir` is omitted, review folders are auto-discovered as:
  `<source-dir-name>_heliocentric_review*` under the same parent directory.
- `--decision-file <path>` overrides the additional removal list (default: `data/decision.txt`).
- `--sentence-selection {all_mentioned,remove_only}` controls sentence-removal strictness.
  - `all_mentioned` removes all sentences listed for `REMOVE_SENTENCE` files.
  - `remove_only` removes only rows with `sentence_action=REMOVE`.

The script writes an execution report to:
- `<output-dir>/_cleanup_report.json`

## End-to-End Run Instructions

### 1) Train tokenizer (required once, then reuse)

Run this if `data/nanogpt/tokenizer/tokenizer.json` does not exist, or if you changed corpus content and want a new vocabulary.

```bash
python pipeline/train_tokenizer.py
```

Custom example (explicit corpora, vocab size, and output path):

```bash
python pipeline/train_tokenizer.py --corpus-dirs data/corpus_general data/corpus_astronomy --vocab-size 32000 --out data/nanogpt/tokenizer/tokenizer.json
```

Output:
- `data/nanogpt/tokenizer/tokenizer.json`

### 2) Validate dataset build

```bash
python pipeline/test_dataset.py --mode all
```

Optional modes:
- `--mode general`
- `--mode astronomy`

### 3) Pretrain on general corpus

```bash
python pipeline/train_pretrain.py --config configs/pretrain_config.json --experiment_name pretrain_v1
```

Pretrain writes checkpoint here:
- `runs/<timestamp>_pretrain_v1/model_checkpoint.pt`

Override example:

```bash
python pipeline/train_pretrain.py --config configs/pretrain_config.json --set learning_rate=0.0002 --set model.n_layer=16
```

### 4) Finetune on astronomy corpus

Before running finetune, set `pretrained_checkpoint` in `configs/finetune_config.json` to the actual pretrain checkpoint from step 3, or override it on the command line.

```bash
python pipeline/train_finetune.py --config configs/finetune_config.json --experiment_name astro110m_ft_v1 --set pretrained_checkpoint=runs/<timestamp>_pretrain_v1/model_checkpoint.pt
```

Override example:

```bash
python pipeline/train_finetune.py --config configs/finetune_config.json --set astronomy_ratio=0.9 --set max_iters=20000
```

Resume example:

```bash
python pipeline/train_finetune.py --config configs/finetune_config.json --experiment_name astro110m_ft_resume --resume_checkpoint runs/2026-03-12_astro110m_ft_v1/model_checkpoint.pt
```

### 5) Generate text

```bash
python pipeline/generate.py --checkpoint runs/<timestamp>_astro110m_ft_v1/model_checkpoint.pt --prompt "Explain why the planets move backwards in the sky"
```

Useful generation args:
- `--max-new-tokens 200`
- `--temperature 0.8`
- `--top-k 50`
- `--top-p 0.9`
- `--repetition-penalty 1.1`
- `--no-repeat-ngram-size 3`

Notes:
- `--top-p 1.0`, `--repetition-penalty 1.0`, and `--no-repeat-ngram-size 0` disable the new decoding controls and match the previous default behavior.
- Generation also stops early if the same 3-gram appears twice.

### 6) Legacy nanoGPT generation

Use this only with checkpoints and metadata produced by the archived legacy nanoGPT flow under `archive/legacy/nanogpt/`.

```bash
python archive/legacy/nanogpt/generate.py --ckpt runs/<timestamp>_legacy_run/ckpt.pt --meta data/nanogpt/meta.json --prompt "Explain why the planets move backwards in the sky"
```

Useful generation args:
- `--max-new-tokens 200`
- `--temperature 0.8`
- `--top-k 50`
- `--top-p 0.9`
- `--repetition-penalty 1.1`
- `--no-repeat-ngram-size 3`

Notes:
- `--top-p 1.0`, `--repetition-penalty 1.0`, and `--no-repeat-ngram-size 0` disable the new decoding controls and match the previous default behavior.
- Generation also stops early if the same 3-gram appears twice.

### 7) Generate evaluation samples

Use `evaluation/generate_eval.py` to run multiple prompts against one or more checkpoints and save the raw generations to JSONL for later judging.

Requirements:
- Prompt files live under `evaluation/` as `.txt` files.
- Each non-empty line in a prompt file is one prompt.
- Leading numeric prefixes such as `9. ` are removed before generation.
- A models JSON file must map model ids to checkpoint paths, for example:

```json
{
  "A": "runs/model_a/model_checkpoint.pt",
  "B": "runs/model_b/model_checkpoint.pt"
}
```

Basic example:

```bash
python evaluation/generate_eval.py \
  --models-json evaluation/models.json \
  --output-jsonl runs/generations.jsonl
```

Pilot run example:

```bash
python evaluation/generate_eval.py \
  --models-json evaluation/models.json \
  --output-jsonl runs/generations.jsonl \
  --samples-per-prompt 2 \
  --max-prompts-per-category 3
```

Resume example:

```bash
python evaluation/generate_eval.py \
  --models-json evaluation/models.json \
  --output-jsonl runs/generations.jsonl \
  --resume
```

Useful options:
- `--evaluation-dir <path>` overrides the default prompt directory (`evaluation`).
- `--samples-per-prompt <N>` controls how many samples are generated per prompt.
- `--max-prompts-per-category <N>` limits each prompt file for small pilot runs.
- `--python-executable <path>` lets you choose the interpreter used for `pipeline/generate.py`.
- `--device {cpu,cuda}` passes the generation device through to `pipeline/generate.py`.
- `--sleep-seconds <float>` adds a pause between subprocess calls.

Output behavior:
- The script writes one JSON object per generation to the output JSONL.
- Each record includes:
  - `model_id`
  - `checkpoint_path`
  - `category`
  - `prompt_id`
  - `prompt_idx`
  - `prompt_text`
  - `sample_idx`
  - `output_text`
  - `generation_config`
- If generation fails, the record is still written with:
  - `output_text: null`
  - `generation_error`

Operational notes:
- `evaluation/generate_eval.py` does not reimplement loading or decoding; it calls `pipeline/generate.py` as a subprocess and stores its stdout.
- Output is flushed after every line, so interrupted runs can be continued with `--resume`.
- Resume skips completed `(model_id, category, prompt_id, sample_idx)` combinations already present in the output file.

### 8) Judge generations with OpenAI

Use `judge/judge_eval.py` to score generated outputs from a JSONL file with the OpenAI Responses API.

Requirements:
- `OPENAI_API_KEY` must be set in the shell environment.
- The input file must be JSONL, one object per line.
- Each input record should include fields such as:
  - `model_id`
  - `category`
  - `prompt_id`
  - `prompt_text`
  - `sample_idx`
  - `output_text`

Basic example:

```bash
python judge/judge_eval.py \
  --input-jsonl runs/generations.jsonl \
  --output-jsonl runs/generations_judged.jsonl
```

Pilot run example:

```bash
python judge/judge_eval.py \
  --input-jsonl runs/generations.jsonl \
  --output-jsonl runs/generations_judged.jsonl \
  --max-items 10
```

Resume example:

```bash
python judge/judge_eval.py \
  --input-jsonl runs/generations.jsonl \
  --output-jsonl runs/generations_judged.jsonl \
  --resume
```

Useful options:
- `--model <name>` overrides the default judge model (`gpt-5.4-mini`).
- `--max-items <N>` runs a small pilot on the first `N` eligible items.
- `--resume` skips items already present in the output JSONL.
- `--sleep-seconds <float>` adds a pause between successful API calls.

Output behavior:
- The script writes one JSON object per line to the output JSONL.
- Successful items keep the original input fields and add:
  - `judge_model`
  - `judge_result`
- `judge_result` contains:
  - `quality_score`
  - `stance_label`
  - `heliocentric_label`
  - `reason`
- Failed items are still written and include:
  - `judge_model`
  - `judge_error`

Operational notes:
- The script flushes after every write so partial progress is preserved.
- It retries transient API failures up to 3 times with exponential backoff.
- `judge/judge_smoke_test.py` is a small single-example reference for the same structured-output pattern.

## Logging and Metrics

Metrics fields:
- `step`
- `epoch`
- `train_loss`
- `val_loss`
- `learning_rate`
- `tokens_processed`
- `time_elapsed`

Metrics are written per run in both:
- `metrics.csv`
- `training_log.jsonl`

`training_log.jsonl` writes one JSON object per training step.

### Gutenberg Non-Science Corpus Builder

Script: `scripts/build_expanded_safe_corpus.py`

What it does:
- Scans local Gutenberg RDF metadata in `data/rdf-files`.
- Builds/saves filtered candidates to `data/gutenberg/candidates_safe.csv` with columns:
  `book_id,title,url,extent,charset`
- Keeps English books with a plain-text format and excludes books whose title/subjects/bookshelves match science keywords.
- Downloads only plain-text files, strips Gutenberg header/footer markers, and saves each book as:
  `data/gutenberg/books/<book_id>.txt`

Resume behavior:
- If `candidates_safe.csv` exists, it loads candidates from CSV (no full RDF rescan).
- If interrupted, rerunning skips already saved books in `data/gutenberg/books/`.
- Already-downloaded raw files are reused from `data/gutenberg/downloads/`.

Common commands:

```bash
# Build or refresh candidate list only (no downloads)
python scripts/build_expanded_safe_corpus.py --refresh-candidates --target-bytes 0

# Normal run toward ~1 GB target
python scripts/build_expanded_safe_corpus.py

# Quick smoke test: save 2 books only
python scripts/build_expanded_safe_corpus.py --max-books 2 --progress-every 1
```
