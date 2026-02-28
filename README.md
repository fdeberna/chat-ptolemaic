## Copernicus Agent

This project builds a pre-Copernican astronomy corpus and trains a small language model to explore whether heliocentric-adjacent ideas can emerge from historical discourse.

### Project Layout
- `src/`: reusable pipeline scripts and core processing code.
- `scripts/extract/`: one-off extraction and scraping helpers.
- `scripts/inspect/`: OCR/PDF inspection and debugging probes.
- `scripts/stats/`: quick corpus/file counting scripts.
- `scripts/translate/`: ad-hoc translation helpers and API probes.
- `data/`: raw/intermediate/final corpora and downloaded sources.
- `nanogpt/`: training and model configuration code.

### Pipeline
1. Download OCR texts
2. Clean Renaissance Latin OCR
3. Translate to English (attempted for Sacrobosco - defaulted to codex translation)
4. Build training corpus
5. Train transformer model
6. Evaluate conceptual emergence

### Data Sources

Astro specific:

- Sacrobosco — Sphaera Mundi
- Aristotle — On the Heavens
- Peuerbach — Theoricae Novae Planetarum
- Ptolemy - Almagestus
- Pliny The Elder - Naturalis Historia

Additional classic authors from 
https://classics.mit.edu/Browse/index.html 
