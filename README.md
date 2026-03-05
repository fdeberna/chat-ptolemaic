## Copernicus Agent

This repository trains a GPT-style language model from scratch with a 3-stage pipeline:
1. Train a BPE tokenizer.
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

Checkpoints:
- `checkpoints/pretrain.pt`
- `checkpoints/astro_model.pt`

## Environment Setup

Install dependencies (example):

```bash
pip install -r requirements.txt
pip install torch
```

`requirements.txt` already includes `numpy`, `tokenizers`, and related dependencies used by this pipeline.

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
    - splits each document into first 90% train and last 10% val before concatenation,
    - shuffles documents before concatenation,
    - creates 1024-token blocks and `(input, target)` pairs via one-token shift,
    - supports padding utilities (`<pad>`) for variable-length collation when needed.

- `pipeline/model.py`
  - Decoder-only GPT model (12 layers, 12 heads, 768 embedding, context 1024, dropout 0.1).
  - Uses causal self-attention and tied token/output embeddings.

- `pipeline/train_pretrain.py`
  - Pretrains on `data/corpus_general`.
  - AdamW, lr `3e-4`, gradient clipping `1.0`, AMP on CUDA if available.
  - Logs step/loss/lr and runs validation each epoch.
  - Saves `checkpoints/pretrain.pt`.

- `pipeline/train_finetune.py`
  - Loads `checkpoints/pretrain.pt`.
  - Finetunes on astronomy with mixed batches:
    - ~80% astronomy
    - ~20% general
  - Default lr `5e-5`, default 4 epochs.
  - Logs step/loss/lr and runs validation each epoch.
  - Saves `checkpoints/astro_model.pt`.

- `pipeline/generate.py`
  - Generates text from a checkpoint with:
    - `<bos>` prompt prefix,
    - temperature sampling,
    - top-k sampling.

- `pipeline/test_dataset.py`
  - Sanity-check helper:
    - builds/loads dataset streams,
    - prints token IDs,
    - decodes a sample back to text.

## End-to-End Run Instructions

### 1) Train tokenizer

```bash
python pipeline/train_tokenizer.py
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
python pipeline/train_pretrain.py
```

Key defaults:
- context length: `1024`
- batch size: `8`
- learning rate: `3e-4`
- checkpoint: `checkpoints/pretrain.pt`
- log file: `checkpoints/pretrain_log.csv`

### 4) Finetune on astronomy corpus

```bash
python pipeline/train_finetune.py
```

Key defaults:
- checkpoint input: `checkpoints/pretrain.pt`
- checkpoint output: `checkpoints/astro_model.pt`
- astronomy/general mix: `80/20`
- learning rate: `5e-5`
- epochs: `4`
- log file: `checkpoints/finetune_log.csv`

### 5) Generate text

```bash
python pipeline/generate.py --checkpoint checkpoints/astro_model.pt --prompt "Explain why the planets move backwards in the sky"
```

Useful generation args:
- `--max-new-tokens 200`
- `--temperature 0.8`
- `--top-k 50`

## Logging and Metrics

Training scripts log:
- `step`
- `training loss`
- `validation loss`
- `learning rate`

CSV logs are written under `checkpoints/`:
- `pretrain_log.csv`
- `finetune_log.csv`
