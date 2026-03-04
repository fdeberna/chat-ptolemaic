import pdfplumber
from collections import Counter
pdf=pdfplumber.open("data/raw/isidore_etymologies.pdf")
ctr=Counter()
for page in pdf.pages[:10]:
    for ch in page.chars:
        ctr[round(ch.get("size",0),1)] +=1
print(ctr.most_common())
