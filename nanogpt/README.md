# nanoGPT (local minimal setup)

This directory contains a pared‑down nanoGPT setup to train a small GPT-style model from scratch on the pre-Copernican corpus.

## Data preparation
1) Ensure `data/corpus` holds the cleaned text files (one work per file).
2) Run the prep script to build a character-level vocabulary and binary splits:
```
python nanogpt/prepare_corpus.py --input data/corpus --out data/nanogpt/pre_copernican
```
This writes `train.bin`, `val.bin`, and `meta.json` into the output directory.
Add `--write-text` to also emit the concatenated `corpus.txt` for inspection.

## Training
Train from scratch (no pretrained weights):
```
python nanogpt/train.py --config nanogpt.config.pre_copernican_char --data-dir data/nanogpt/pre_copernican --out-dir out/pre_copernican
```
Defaults are modest (for a laptop GPU/CPU). Adjust hyperparameters in `nanogpt/config/pre_copernican_char.py`.

### 25M-param option
Use `nanogpt/config/pre_copernican_char_25m.py` (n_layer=8, n_head=8, n_embd=512, block_size=512, ~25M params):
```
python nanogpt/train.py --config nanogpt.config.pre_copernican_char_25m --data-dir data/nanogpt/pre_copernican --out-dir out/pre_copernican_25m
```
Rough VRAM budget: ~1–2 GB single-precision for batch_size=32, block_size=512.

## Inference / probing
After training, generate continuations from a checkpoint:
```
python nanogpt/generate.py --ckpt out/pre_copernican/ckpt.pt --meta data/nanogpt/pre_copernican/meta.json --prompt "Aristotle said" --max-new-tokens 200 --temperature 0.8
```
Supports `--top-k` and `--device cpu|cuda`.

## Notes
- Character-level tokenizer to avoid external downloads.
- Single-device training loop, no distributed or logging frameworks.
- Checkpoints are saved under `out/<run>/ckpt.pt`.
