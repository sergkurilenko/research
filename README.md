# Hybrid privacy-aware semantic search

Reproducibility code and manuscript for:

> **Hybrid privacy-aware semantic search: SVD-truncated document geometry
> and CKKS-encrypted query reranking under a restricted threat model**
> Sergey Kurilenko, arXiv:2606.26373.

This is the `article1` branch of
<https://github.com/sergkurilenko/research>. The branch is isolated from the
other papers in that repository.

## Layout

- `paper_revised/` — current Springer Nature LaTeX source and figures.
- `system/` — public-PQ, block-packed CKKS, graded-IR, security, and systems
  experiment harnesses used by the current manuscript.
- `tests/` — unit and protocol tests for the current harnesses.
- `results/system_revision/` — curated machine-readable experiment outputs.
- `docs/` — reproduction notes, pinned environment, and evidence manifest.
- `paper/` and `baseline/` — superseded preliminary material retained for
  provenance; these files do not support the current manuscript's claims.

Large embeddings and indexes are regenerated rather than committed. The
million-vector corpus is a Russian Wikipedia paragraph slice; graded utility
uses pinned BEIR collections. Exact commands and artifact boundaries are in
`docs/reproduce_system_revision.md`.

## Local verification

```powershell
$py = 'python'
& $py -m pytest tests -q
& $py system\make_revision_figures.py
& $py system\make_expansion_figures.py
cd paper_revised
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The LWE-estimator transcript requires SageMath; its pinned input, raw output,
runtime description, and parsed costs are kept under
`results/system_revision/security_expansion/lwe_security/`. Large experiments
checkpoint atomically, so the IR/PQ sweeps can be resumed without accepting a
partially written index.
