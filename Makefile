# SHARD — convenience targets. Override the interpreter with `make PY=python3`.
PY ?= python

.PHONY: help smoke test figures shard baseline paper clean

help:
	@echo "make smoke     - synthetic correctness tests (no embeddings needed)"
	@echo "make figures   - regenerate the SHARD figures into paper/figs/"
	@echo "make shard     - run all SHARD experiments        (needs SHARD_DATA)"
	@echo "make baseline  - run the lightweight baseline runs (needs SHARD_DATA)"
	@echo "make paper     - build paper/paper_en.pdf"
	@echo "make clean     - remove __pycache__ and LaTeX aux files"

smoke test:
	$(PY) shard/test_shard.py

figures:
	cd shard && $(PY) make_fig_shard.py

shard:
	cd shard && for e in exp12_shard_utility exp13_shard_alignment exp14_shard_leakage \
	    exp15_shard_reference exp17_beir_shard exp18_shard_cost exp19_shard_targeted \
	    exp20_shard_microkey exp21_shard_vs_dp exp22_shard_learned_attack; do \
	  echo "== $$e ==" && $(PY) $$e.py || exit 1; done

baseline:
	cd baseline && for e in exp07_alignment_pq_leakage exp08_tradeoff_noise_sweep \
	    exp09_reference_corpus_attack exp10_denoiser_significance exp11_beir_denoiser; do \
	  echo "== $$e ==" && $(PY) $$e.py || exit 1; done

paper:
	cd paper && pdflatex -interaction=nonstopmode paper_en.tex && \
	            pdflatex -interaction=nonstopmode paper_en.tex

clean:
	rm -rf shard/__pycache__ baseline/__pycache__ \
	       paper/*.aux paper/*.log paper/*.out paper/*.toc
