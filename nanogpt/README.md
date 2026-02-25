# nanoGPT (local minimal setup)

This directory contains a pared-down nanoGPT setup to train a small GPT-style model from scratch on the pre-Copernican corpus.

## Data preparation
1) Ensure `data/corpus` holds the cleaned text files (one work per file).
2) BPE is now default. Train tokenizer and build tokenized splits:
```
python nanogpt/prepare_corpus.py --input data/corpus --out data/nanogpt/pre_copernican_bpe --tokenizer bpe --vocab-size 8000 --write-text
```
Outputs: `train.bin`, `val.bin`, `meta.json` (notes tokenizer + dtype), and `tokenizer.json` in the same folder.

Legacy char-level (optional):
```
python nanogpt/prepare_corpus.py --tokenizer char --input data/corpus --out data/nanogpt/pre_copernican
```

## Training
Train from scratch (no pretrained weights):
```
python nanogpt/train.py --config nanogpt.config.pre_copernican_bpe --data-dir data/nanogpt/pre_copernican_bpe --out-dir out/pre_copernican_bpe
```
Adjust hyperparameters in `nanogpt/config/pre_copernican_bpe.py`.

Char-level (legacy):
```
python nanogpt/train.py --config nanogpt.config.pre_copernican_char --data-dir data/nanogpt/pre_copernican --out-dir out/pre_copernican
```

### 25M-param option
BPE 25M:
```
python nanogpt/train.py --config nanogpt.config.pre_copernican_bpe_25m --data-dir data/nanogpt/pre_copernican_bpe --out-dir out/pre_copernican_bpe_25m
```

Char-level 25M (legacy):
```
python nanogpt/train.py --config nanogpt.config.pre_copernican_char_25m --data-dir data/nanogpt/pre_copernican --out-dir out/pre_copernican_25m
```

## Inference / probing
After training, generate continuations from a checkpoint:
```
python -m nanogpt.generate --ckpt out/pre_copernican_bpe/ckpt.pt --meta data/nanogpt/pre_copernican_bpe/meta.json --prompt "Aristotle said" --max-new-tokens 200 --temperature 0.8
```
Supports `--top-k` and `--device cpu|cuda`.

## Notes
- Byte-level BPE tokenizer by default; legacy char-level still supported.
- Single-device training loop, no distributed or logging frameworks.
- Training logs to `out/<run>/train_log.csv` (iter, train_loss, val_loss).
- Checkpoints: `ckpt.pt` (latest) and `ckpt_best.pt` (best val loss) under `out/<run>/`.
