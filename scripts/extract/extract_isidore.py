# coding: utf-8
import re
import pdfplumber
from pathlib import Path

PDF_PATH = Path("data/raw/isidore_etymologies.pdf")
OUT_PATH = Path("")

HEADING_PATTERNS = [
    r"^Book\s+[IVXLCDM]+",
    r"^The\s+Etymologies",
    r"^Grammar\b",
    r"^Mathematics\b",
    r"^I\.",
    r"^[IVXLCDM]+\.\S*\s*\d+$",
    r"^[ivxlcdmIVXLCDM]+\.",
    r"^[A-Z][A-Za-z\s\-]+\(De ",
    r"^Page\s+\d+",
]


def should_skip(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    for pat in HEADING_PATTERNS:
        if re.match(pat, s):
            return True
    if re.match(r"^\d+$", s):
        return True
    if len(s) <= 6 and "book" in s.lower():
        return True
    # running heads like "40 I.iii.6.–iv.4"
    if "–" in s and re.search(r"\d+\s*–\s*\d+$", s):
        return True
    if "-" in s and re.search(r"\d+\s*-\s*\d+$", s):
        return True
    if "(" in s and s.count("(") == 1 and s.count(")") == 0 and len(s) < 40:
        return True
    return False


def fix_spacing(line: str) -> str:
    # Insert spaces that are often missing in the OCR/text.
    line = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", line)
    line = re.sub(r"(?<=[A-Za-z])(?=[‘’“”'\"(\[])", " ", line)
    line = re.sub(r"(?<=[A-Za-z0-9])\.(?=[A-Za-z])", ". ", line)
    line = re.sub(r"(?<=[A-Za-z0-9]),(?=[A-Za-z])", ", ", line)
    line = re.sub(r"(?<=[A-Za-z0-9])\?(?=[A-Za-z])", "? ", line)
    line = re.sub(r"(?<=[A-Za-z0-9])\!(?=[A-Za-z])", "! ", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def chars_to_lines(chars, y_tol=2.0, gap_factor=0.6):
    lines = []
    # group by y within tolerance
    current_y = None
    current_line = []
    for ch in sorted(chars, key=lambda c: (c["top"], c["x0"])):
        if current_y is None or abs(ch["top"] - current_y) > y_tol:
            if current_line:
                lines.append(current_line)
            current_line = [ch]
            current_y = ch["top"]
        else:
            current_line.append(ch)
    if current_line:
        lines.append(current_line)

    out_lines = []
    for line_chars in lines:
        line_chars = sorted(line_chars, key=lambda c: c["x0"])
        text_parts = []
        prev = None
        for ch in line_chars:
            if prev is not None:
                gap = ch["x0"] - prev["x1"]
                avg_width = max(1e-3, (prev["x1"] - prev["x0"]))
                if gap > avg_width * gap_factor:
                    text_parts.append(" ")
            text_parts.append(ch["text"])
            prev = ch
        out_lines.append("".join(text_parts))
    return out_lines


def extract_two_columns(pdf_path: Path):
    lines_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # drop small-font footnotes by size threshold
            chars = [c for c in page.chars if c.get("size", 0) >= 9.5]
            if not chars:
                continue
            mid = page.width / 2
            left_chars = [c for c in chars if c["x0"] < mid]
            right_chars = [c for c in chars if c["x0"] >= mid]

            for col_chars in (left_chars, right_chars):
                for line in chars_to_lines(col_chars):
                    if should_skip(line):
                        continue
                    lines_out.append(fix_spacing(line))
            lines_out.append("")  # page break spacer
    # collapse multiple blanks
    cleaned = []
    prev_blank = False
    for line in lines_out:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = blank
    return cleaned


def main():
    lines = extract_two_columns(PDF_PATH)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
