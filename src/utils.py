import re

def clean_latin_ocr(text: str) -> str:
    """
    Clean Renaissance Latin OCR artifacts.
    """

    # normalize line breaks
    text = text.replace("\r", "\n")

    # remove bracket expansions: q[uam] → quam
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # remove page numbers & signatures (A2, B3 etc)
    text = re.sub(r"\b[A-Z]\d\b", " ", text)

    # fix long s (ſ) and common OCR confusions
    replacements = {
        "ſ": "s",
        "fp": "sp",
        "fph": "sph",
        "ff": "ss",
        "conuex": "convex",
        "concaua": "concava",
        "rcuol": "revol",
        "cufpid": "cuspid",
        "econuer": "econver",
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # remove stray symbols
    text = re.sub(r"[«»^*]", "", text)

    # merge hyphenated line breaks
    text = re.sub(r"-\n", "", text)

    # remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def rebuild_paragraphs(text: str) -> str:
    """
    Reconstruct paragraphs by splitting at sentence boundaries.
    """
    text = re.sub(r"\.\s+", ".\n\n", text)
    text = re.sub(r":\s+", ":\n\n", text)
    return text

