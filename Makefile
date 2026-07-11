# Reproducibility shortcuts for the journal revision.
# Override tools when needed, for example: make test PY=python3
PY ?= python
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex

.PHONY: help test figures validate paper bundle

help:
	@echo "make test      - run the complete current test suite"
	@echo "make figures   - regenerate manuscript figures from result JSON"
	@echo "make validate  - validate the completed systems result artifact"
	@echo "make paper     - build the Springer Nature manuscript PDF"
	@echo "make bundle    - build arXiv, journal-source, and Online Resource archives"

test:
	$(PY) -m pytest tests -q

figures:
	$(PY) system/make_revision_figures.py
	$(PY) system/make_expansion_figures.py

validate:
	$(PY) -m system.validate_systems_expansion \
		--input results/system_revision/systems_expansion/windows_full.json \
		--output results/system_revision/systems_expansion/windows_full_validation.json

paper: figures
	cd paper_revised && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	cd paper_revised && $(BIBTEX) main
	cd paper_revised && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	cd paper_revised && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex

bundle: paper
	$(PY) -m system.build_release_bundles
