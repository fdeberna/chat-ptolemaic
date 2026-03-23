import pathlib
root = pathlib.Path("data/corpus")
if not root.exists():
    print('corpus dir missing')
    raise SystemExit
sizes = [p.stat().st_size for p in root.rglob('*') if p.is_file()]
print(sum(sizes))
