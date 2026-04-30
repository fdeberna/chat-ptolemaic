MAIN=main
OUTDIR=build

pdf:
	mkdir -p $(OUTDIR)
	latexmk -pdf -outdir=$(OUTDIR) $(MAIN).tex

clean:
	latexmk -C -outdir=$(OUTDIR) $(MAIN).tex
	rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk