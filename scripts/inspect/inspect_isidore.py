import pdfplumber
pdf=pdfplumber.open("data/raw/isidore_etymologies.pdf")
print(len(pdf.pages))
page0=pdf.pages[0]
print(page0.extract_text()[:1000])
