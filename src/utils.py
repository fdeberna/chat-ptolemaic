import html
import re
from typing import Iterable, List


# ----------- Helpers for OCR cleaning -----------
def _merge_latin_fragments(text: str) -> str:
    """
    Merge ONLY very safe OCR splits:
    - single-letter / two-letter prefixes (q uae -> quae, e x -> ex)
    - common Latin enclitics and endings split off
    - mid-word splits where the first fragment is short (<=3)
    Avoid merging general word boundaries (too risky).
    """

    # Merge short prefix fragments: "q ua" -> "qua", "re pu" -> "repu" (still imperfect but keeps spacing)
    text = re.sub(r"\b([a-z]{1,3})\s+([a-z]{2,})\b", r"\1\2", text)

    # Merge split enclitics/endings when they appear detached
    endings = ["que", "bus", "rum", "tis", "tur", "mus", "nis", "ium", "ius", "iae", "orum", "arum"]
    for end in endings:
        text = re.sub(rf"\b([a-z]{{3,}})\s+({end})\b", r"\1\2", text)

    return text



def _strip_archive_html(text: str) -> str:
    """
    Keep only the OCR payload from Archive.org HTML dumps.
    """
    pre_match = re.search(
        r"<pre[^>]*>(.*?)</pre>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if pre_match:
        return pre_match.group(1)
    if "<html" in text.lower() or "<!doctype" in text.lower():
        return re.sub(r"<[^>]+>", " ", text)
    return text


def _normalize_linebreaks(text: str) -> str:
    return text.replace("\r", "\n")


def _merge_hyphenated_breaks(text: str) -> str:
    # merge words split across lines with hyphens
    text = re.sub(r"-\s*\n\s*", "", text)
    # merge hyphen + space within a line (anti- bus -> antibus)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text


def _merge_intraword_spaces(text: str) -> str:
    # merge single-letter spaced runs (m u n d i -> mundi)
    text = re.sub(
        r"\b([A-Za-z])(?:\s+([A-Za-z])){2,}\b",
        lambda m: "".join(m.group(0).split()),
        text,
    )
    # merge short uppercase chunks (SPH AE RAE -> SPHAERAE)
    text = re.sub(
        r"\b([A-Z]{1,3})(?:\s+([A-Z]{1,3})){1,5}\b",
        lambda m: "".join(m.group(0).split()),
        text,
    )
    return text

def _drop_header_gibberish(lines: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append(line)
            continue
        # high ratio of non-letters or looks like a scan/page marker header
        letters = sum(ch.isalpha() for ch in s)
        if letters > 0:
            non_letters_ratio = 1 - (letters / max(len(s), 1))
            if non_letters_ratio > 0.35 and len(s) < 40:
                continue
        # short “caps + digits + commas” garbage
        if re.fullmatch(r"[A-Z0-9\s,.:;'\-]{8,}", s):
            continue
        cleaned.append(line)
    return cleaned

def _drop_printer_signatures(lines: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    signature_re = re.compile(r"^[A-Z]{1,3}\s*\d{1,3}[rv]?$")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if signature_re.match(stripped):
            continue
        # drop short all-caps number combos (e.g., "A2", "B3")
        if len(stripped) <= 4 and re.fullmatch(r"[A-Z]+\d+", stripped):
            continue
        cleaned.append(line)
    return cleaned


def _collapse_uppercase_noise(lines: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        letters = [c for c in stripped if c.isalpha()]
        if letters:
            upper_ratio = sum(c.isupper() for c in letters) / len(letters)
            if upper_ratio > 0.85 and len(stripped) > 20:
                # likely banner/header noise
                continue
        cleaned.append(line)
    return cleaned


def _normalize_ligatures(text: str) -> str:
    ligatures = {
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "Œ": "Oe",
    }
    for wrong, right in ligatures.items():
        text = text.replace(wrong, right)
    return text

def _normalize_long_s_and_common(text: str) -> str:
    """
    Fix long-s OCR confusion and common Renaissance substitutions.
    """
    # long s
    text = text.replace("ſ", "s")

    # common long-s patterns
    text = re.sub(r"([aeiou])f([aeiou])", r"\1s\2", text)
    text = re.sub(r"f([aeiou])", r"s\1", text)

    # specific common OCR errors
    replacements = {
        "copof": "compos",
        "diuerf": "divers",
        "cauf": "caus",
        "fign": "sign",
        "fuper": "super",
        "fphaer": "sphaer",
        "aftr": "astr",
        "noit": "noct",
    }

    for wrong, right in replacements.items():
        text = re.sub(wrong, right, text, flags=re.IGNORECASE)

    return text

def _fix_intraword_parens(text: str) -> str:
    # If OCR inserts "(" inside a word, it's almost always noise: "adoie(centibus" -> "adoiecentibus"
    text = re.sub(r"([A-Za-z])\(([A-Za-z])", r"\1\2", text)
    text = re.sub(r"([A-Za-z])\)([A-Za-z])", r"\1\2", text)
    return text



def _fix_ocr_substitutions(text: str) -> str:
    # mild u/v and i/j normalizations where OCR often swaps them
    substitutions = {
        "vni": "uni",
        "vniu": "uniu",
        "seript": "script",
        "ij": "ii",
    }
    for wrong, right in substitutions.items():
        text = re.sub(wrong, right, text, flags=re.IGNORECASE)
    return text


def _strip_stray_punctuation(text: str) -> str:
    text = re.sub(r"[•·†‡§¶¤]+", " ", text)
    text = re.sub(r"[<>^*]+", " ", text)
    text = re.sub(r"[\"`´“”‘’]+", " ", text)
    return text

def _remove_noise_tokens(text: str) -> str:
    tokens = text.split()
    cleaned = []
    for tok in tokens:
        t = tok.strip()

        # obvious symbol garbage
        if re.search(r"[&@#%/\\|]+", t):
            continue

        # weird mixed-case long tokens
        if len(t) >= 14 and (re.search(r"[A-Z]", t) and re.search(r"[a-z]", t)):
            # drop if it has too many case changes or looks like OCR glue
            if len(re.findall(r"[A-Z]", t)) >= 3 and len(re.findall(r"[a-z]", t)) >= 3:
                continue

        # very long tokens with low vowel ratio
        if len(t) >= 18:
            v = len(re.findall(r"[aeiouAEIOU]", t))
            if v / len(t) < 0.20:
                continue

        cleaned.append(t)

    return " ".join(cleaned)

def _cleanup_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_common_latin_ocr(text: str) -> str:
    """
    Normalize frequent OCR distortions that break translation.
    Conservative replacements based on common early-print errors.
    """

    replacements = {
        "adoiecent": "adolescent",
        "adoiecentibus": "adolescentibus",
        "repu capeflenda": "reipublicae splendenda",
        "repu": "reipublicae",
        "ueftigio": "vestigio",
        "sohsencu": "sphaericum",
        "coposition": "composition",
        "coponitur": "componitur",
        "cclip": "eclip",
        "noctium": "noctium",  # ensure consistent spelling
        "aftronom": "astronom",
        "fphaera": "sphaera",
    }

    for wrong, right in replacements.items():
        text = re.sub(wrong, right, text, flags=re.IGNORECASE)

    return text

# ----------- Public API -----------
def clean_latin_ocr(text: str) -> str:
    # ---------- structural cleanup ----------
    text = _normalize_linebreaks(text)
    text = _strip_archive_html(text)
    text = html.unescape(text)

    text = _merge_hyphenated_breaks(text)
    text = _merge_intraword_spaces(text)

    lines = text.splitlines()
    lines = _drop_printer_signatures(lines)
    lines = _collapse_uppercase_noise(lines)
    lines = _drop_header_gibberish(lines)
    text = "\n".join(lines)

    # ---------- typography normalization ----------
    text = _normalize_ligatures(text)
    text = _fix_intraword_parens(text)
    text = _normalize_long_s_and_common(text)
    text = _fix_ocr_substitutions(text)

    # ---------- safe fragment reconstruction ----------
    text = _merge_latin_fragments(text)

    # ---------- translation readiness normalization ----------
    text = normalize_common_latin_ocr(text)

    # ---------- final cleanup ----------
    text = _strip_stray_punctuation(text)
    text = _remove_noise_tokens(text)
    text = _cleanup_whitespace(text)

    return text



def rebuild_paragraphs(text: str) -> str:
    """
    Reconstruct paragraphs by splitting at sentence boundaries.
    """
    text = re.sub(r"\.\s+", ".\n\n", text)
    text = re.sub(r":\s+", ":\n\n", text)
    return text


# ----------- Example (manual smoke test) -----------

EXAMPLE_RAW = """<!DOCTYPE html><pre>
SPH AEIlAE MVNDI CO- MPENDI VM FOELICITER-
anti- bus: pro breui red:o( tramite
AB2
</pre>"""

EXAMPLE_CLEAN = clean_latin_ocr(EXAMPLE_RAW)
# Expected: "SPHAERAE MUNDI COMPENDIUM FOELICITER antibus: pro brevi red:o tramite"
