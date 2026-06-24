# Hybrid privacy-aware semantic search (Alpha) — code & paper

Self-contained reproduction code and paper for:

> **Hybrid privacy-aware semantic search: SVD-truncated document geometry
> and CKKS-encrypted query reranking under a restricted threat model**
> S. M. Kurilenko, arXiv:2606.XXXXX.

This is the `article1` branch of
<https://github.com/sergkurilenko/research> (orphan branch; isolated from
the other papers in the repository).

## Layout
- `paper/`    — LaTeX source (`paper.tex`) and compiled PDF, with all figures.
- `baseline/` — experiment scripts (CKKS ct-pt timing, Vec2Text inversion,
  integral 1M-document evaluation, end-to-end latency, the known-plaintext
  Procrustes / PQ-leakage / reference-corpus attacks, and the BEIR check)
  plus the figure generator `make_paper_figures.py`.
- `results/`  — curated JSON/CSV outputs of the experiments.
- `docs/`     — reproduction notes.

Large embedding and index caches (`notebooks/`, `*.npy`, `*.index`) are
regenerated rather than committed; see `.gitignore` and `docs/`.

## Reproduce
Install `requirements.txt`, then regenerate figures with
`python baseline/make_paper_figures.py`. The corpus is a one-million
Russian-Wikipedia paragraph slice (date and checksum recorded in `docs/`).

## Companion work
The storage-side defence that removes the single global alignment axis and
resists the known-plaintext alignment attack reported here is **SHARD**
(`article5` branch; arXiv:2606.YYYYY).
