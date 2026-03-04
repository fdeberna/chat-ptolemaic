import pdfplumber
pdf = pdfplumber.open("data/raw/ptolemy_almagest_english.pdf")
print(pdf.pages[73].extract_text()[:400])
