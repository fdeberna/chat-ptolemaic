import pdfplumber, re
start, end = 73, 129
pdf = pdfplumber.open("data/raw/ptolemy_almagest_english.pdf")
lines_out = []
skip_prefix = re.compile(r'^(Figure|Fig\.|Table|TAB)\b', re.IGNORECASE)
skip_page = re.compile(r'^\s*\d+\s*/')
skip_short_alpha = re.compile(r'^[A-Z]$')
leading_marker = re.compile(r'^[A-Z]?\d{2,3}\s+')
for pno in range(start, end+1):
    text = pdf.pages[pno].extract_text()
    if not text:
        continue
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if skip_page.match(line):
            continue
        if skip_prefix.match(line):
            continue
        if skip_short_alpha.match(line):
            continue
        # remove leading footnote markers like H72, 127 etc
        line = leading_marker.sub('', line)
        # remove lone running page numbers at end lines
        if len(line) <= 3 and line.isdigit():
            continue
        lines_out.append(line)
output_path = "data/corpus/ptolemy_almagest_book2.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    for line in lines_out:
        f.write(line + '\n')
print(f"Wrote {len(lines_out)} lines")
