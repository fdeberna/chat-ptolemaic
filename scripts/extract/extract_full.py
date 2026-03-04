import pdfplumber, re
from pathlib import Path
pdf = pdfplumber.open("data/raw/ptolemy_almagest_english.pdf")
lines_out = []
skip_prefix = re.compile(r'^(Figure|Fig\.|Table|TAB|TABLE)\b', re.IGNORECASE)
skip_pagehdr = re.compile(r'^\s*\d+\s*/')
skip_section_num = re.compile(r'^\d+\s+\d+\.\s+')
skip_single_cap = re.compile(r'^[A-Z]$')
leading_marker = re.compile(r'^[A-Z]{0,2}\d{2,4}\s+')
for pno, page in enumerate(pdf.pages):
    # filter out small text to drop footnotes/tables
    page_filt = page.filter(lambda obj: obj.get("object_type") == "char" and obj.get("size",0) >= 8.5)
    text = page_filt.extract_text()
    if not text:
        continue
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if skip_pagehdr.match(line):
            continue
        if skip_prefix.match(line):
            continue
        if skip_single_cap.match(line):
            continue
        if skip_section_num.match(line):
            continue
        line = leading_marker.sub('', line)
        if line.isdigit():
            continue
        # drop lone running headers like 'B'
        if len(line)==1:
            continue
        lines_out.append(line)
out_path = Path('data/corpus/ptolemy_almagest_full.txt')
out_path.write_text('\n'.join(lines_out), encoding='utf-8')
print(f"Pages processed: {len(pdf.pages)}; lines kept: {len(lines_out)}; output: {out_path}")
