import pdfplumber
pdf=pdfplumber.open("data/raw/isidore_etymologies.pdf")
for i in [1,2,10,50,100,200,300]:
    text=pdf.pages[i].extract_text()[:400]
    print("-- page",i+1,"--")
    print(text.replace("\n"," ")[:400])
