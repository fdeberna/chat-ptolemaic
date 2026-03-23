import pdfplumber
pdf = pdfplumber.open("data/raw/ptolemy_almagest_english.pdf")
page = pdf.pages[73].filter(lambda obj: obj["object_type"]=="char" and obj["size"]>=8.5)
print(page.extract_text()[:600])
