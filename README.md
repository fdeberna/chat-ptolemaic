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
