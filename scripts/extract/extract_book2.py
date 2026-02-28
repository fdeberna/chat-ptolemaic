import pdfplumber, math, re
from collections import defaultdict
pdf = pdfplumber.open("data/raw/ptolemy_almagest_english.pdf")
start, end = 73, 129
lines_out = []
for pno in range(start, end+1):
    page = pdf.pages[pno]
    words = page.extract_words(extra_attrs=['size','top','x0'])
    words = [w for w in words if w['size'] >= 8.0]
    buckets = defaultdict(list)
    for w in words:
        y = round(w['top'], 0)
        buckets[y].append(w)
    for y in sorted(buckets):
        ws = sorted(buckets[y], key=lambda w:w['x0'])
        line = ' '.join(w['text'] for w in ws)
        # filters
        if not line.strip():
            continue
        if re.match(r'^\d+\s*/', line):
            continue
        if re.match(r'^(Figure|TABLE|Table|Fig\.|FIG\.)\b', line):
            continue
        if re.match(r'^[A-Z]{1,2}\s*$', line):
            continue
        if re.match(r'^[A-Z]{1,2}\d+$', line):
            line = re.sub(r'^[A-Z]{1,2}\d+\s+', '', line)
        # drop footnote-like tokens at start
        line = re.sub(r'^[A-Z]{1,2}\d+\s+', '', line)
        line = re.sub(r'^[0-9]+\s+', '', line)
        lines_out.append(line)
output_path = "data/corpus/ptolemy_almagest_book2.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    for line in lines_out:
        f.write(line.rstrip() + '\n')
print(f"Wrote {len(lines_out)} lines to {output_path}")
