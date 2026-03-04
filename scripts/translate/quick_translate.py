from pathlib import Path
import json, requests, time

lat_path = Path("data/packhum/latin")
eng_path = Path("data/packhum/english")
eng_path.mkdir(parents=True, exist_ok=True)
files = list(lat_path.glob("*.txt"))
url = "https://translate.googleapis.com/translate_a/single"

for f in files:
    lines = f.read_text(encoding="utf-8").splitlines()
    out_lines = []
    for line in lines:
        if not line.strip():
            out_lines.append("")
            continue
        params = {"client": "gtx", "sl": "la", "tl": "en", "dt": "t", "q": line}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        out_lines.append(" ".join(part[0] for part in data[0] if part[0]))
        time.sleep(0.1)
    out_path = eng_path / f.name
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("translated", f.name)
