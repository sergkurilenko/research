# SHARD — alignment-resistant private dense retrieval — code & paper

Self-contained reproduction code and paper for:

> **SHARD: Cell-keyed residual splitting for alignment-resistant private
> dense retrieval**
> S. M. Kurilenko, arXiv:2606.YYYYY.

This is the `article5` branch of
<https://github.com/sergkurilenko/research> (orphan branch; isolated from
the other papers in the repository).

## Layout
- `paper/`    — LaTeX source (`paper_en.tex`) and compiled PDF; figures in `paper/figs/`.
- `shard/`    — the SHARD construction (`shard_lib.py`), correctness test
  (`test_shard.py`), and SHARD experiments (utility, alignment cost,
  leakage, reference attack, BEIR, cost, targeted, micro-key, DP
  comparison, learned attack) plus `make_fig_shard.py`.
- `baseline/` — shared global-linear baseline scripts and figure generator.
- `results/`  — curated JSON/CSV outputs of the experiments.
- `docs/`     — reproduction notes.

Large embedding and index caches (`notebooks/`, `*.npy`, `*.index`) are
regenerated rather than committed; see `.gitignore` and `docs/`.

## Reproduce
Install `requirements.txt`, then run the SHARD experiments in `shard/` and
regenerate figures with `python shard/make_fig_shard.py`.

## Companion work
The hybrid SVD + secret-rotation + CKKS scheme that SHARD takes as its
global-linear baseline — and which is shown there to fall to known-plaintext
alignment — is **Alpha** (`article1` branch; arXiv:2606.XXXXX).
