# SHARD convenience targets. Override the interpreter with `make PY=python3`.
PY ?= python

.PHONY: help smoke test figures audit shard systems privacy inversion maximal legacy baseline paper jisa clean

help:
	@echo "make smoke     - synthetic correctness tests (no embeddings needed)"
	@echo "make figures   - regenerate manuscript figures from exp23--28 summaries"
	@echo "make audit     - run corrective experiments 23--25 (needs SHARD_DATA)"
	@echo "make systems   - run measured TenSEAL CKKS/block-SIMD experiment 26"
	@echo "make privacy   - run formal-DP and release-churn experiments 27--28"
	@echo "make inversion - run GPU Vec2Text outcome experiment 29"
	@echo "make maximal   - run experiments 23--29 and regenerate figures"
	@echo "make shard     - alias for make audit"
	@echo "make legacy    - run superseded SHARD experiments 12--22 for provenance"
	@echo "make baseline  - run the lightweight baseline runs (needs SHARD_DATA)"
	@echo "make paper     - build paper/paper_en.pdf"
	@echo "make jisa      - synchronize and build the Elsevier review source"
	@echo "make clean     - remove __pycache__ and LaTeX aux files"

smoke test:
	$(PY) shard/test_shard.py

figures:
	$(PY) shard/make_fig_corrected_audit.py
	$(PY) shard/make_fig_maximal_program.py

audit shard:
	$(PY) shard/exp23_corrected_score.py
	$(PY) shard/exp24_partial_alignment.py
	$(PY) shard/exp25_cross_release_linkage.py

systems:
	$(PY) shard/exp26_ckks_blocksimd.py

privacy:
	$(PY) shard/exp27_formal_dp_baseline.py
	$(PY) shard/exp28_cross_release_churn.py

inversion:
	$(PY) shard/exp29_shard_vec2text.py

maximal: audit systems privacy inversion figures

legacy:
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

jisa:
	cd jisa && $(PY) sync_from_canonical.py && \
	            pdflatex -interaction=nonstopmode paper_jisa.tex && \
	            pdflatex -interaction=nonstopmode paper_jisa.tex

clean:
	rm -rf shard/__pycache__ baseline/__pycache__ \
	       paper/*.aux paper/*.log paper/*.out paper/*.toc
