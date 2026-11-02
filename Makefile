# The write-up as a PDF, from WRITEUP.md (also read as is on GitHub).
writeup: build/WRITEUP.pdf

build/WRITEUP.pdf: WRITEUP.md writeup/metadata.yaml writeup/header.tex
	mkdir -p build
	pandoc WRITEUP.md --metadata-file=writeup/metadata.yaml --pdf-engine=xelatex \
		--include-in-header=writeup/header.tex \
		--shift-heading-level-by=-1 -o $@

.PHONY: writeup
