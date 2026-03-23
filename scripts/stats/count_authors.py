from pathlib import Path

def char_count(paths):
    total = 0
    for p in paths:
        total += len(p.read_text(encoding="utf-8", errors="ignore"))
    return total

base = Path("data/corpus")
files = list(base.glob("*"))

aristotle = [p for p in files if p.name.lower().startswith("aristotle_") or p.name.lower().startswith("aristotele-")]
ptolemy = [p for p in files if p.name.lower().startswith("ptolemy_") or p.name.lower().startswith("ptolomy-")]
sacrobosco = [base / "sacrobosco_sphaera_mundi_1501.txt"]
cleomedes = [base / "cleomedes_theheavens.txt"]
peuerbach = [p for p in files if p.name.lower().startswith("peuerbach_")]
pliny = [base / "Pliny-historianaturalis.txt"]

report = {
    "Aristotle": char_count(aristotle),
    "Ptolemy": char_count(ptolemy),
    "Sacrobosco": char_count(sacrobosco),
    "Cleomedes": char_count(cleomedes),
    "Peuerbach": char_count(peuerbach),
    "Pliny": char_count(pliny),
}
for k,v in report.items():
    print(f"{k}\t{v}")
