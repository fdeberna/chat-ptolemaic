import pathlib
root = pathlib.Path("data/corpus")
rows = []
for path in sorted(p for p in root.glob("**/*") if p.is_file()):
    size = path.stat().st_size
    rel = path.relative_to(root)
    rows.append(f"{rel}\t{size}")
out = root / "corpus_sizes.tsv"
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"wrote {out}")
