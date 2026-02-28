import pdfplumber
from collections import Counter
pdf = pdfplumber.open("data/raw/ptolemy_almagest_english.pdf")
page = pdf.pages[73]
words = page.extract_words(extra_attrs=['size'])
c = Counter(round(w['size'],1) for w in words)
print('font sizes top counts', c.most_common(10))
print('page text sample:\n', page.extract_text()[:800])
