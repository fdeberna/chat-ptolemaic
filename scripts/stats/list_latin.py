from pathlib import Path

path = Path("data/packhum/latin")
files = list(path.glob("*.txt"))
print(len(files), "latin files")
print([p.name for p in files[:5]])
