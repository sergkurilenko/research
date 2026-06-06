# Reproducing the results

This repo ships the manuscript, all code, and every experiment's JSON/CSV
output. The only thing not included is the ~17 GB of cached embeddings (see
[data](#1-data)). All paths are resolved by `shard/paths.py` /
`baseline/paths.py` and can be overridden with the `SHARD_DATA`,
`SHARD_RESULTS`, and `SHARD_FIGS` environment variables; by default results
go to `results/` and figures to `paper/figs/`.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The geometry / privacy / utility experiments need only
`numpy scipy scikit-learn matplotlib`. The BEIR experiments (`exp11`,
`exp17`) additionally need a CPU build of `torch` plus `transformers` and
`datasets`:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install transformers datasets
```

`baseline/exp_integral.py` also uses `faiss-cpu` and `tenseal` (CKKS); the
optional Vec2Text stress test needs a GPU (see
[heavy experiment](#5-optional-heavy-gpu-experiment)).

## 1. Data

The experiments read L2-normalised embeddings of a 1M-paragraph
Russian-Wikipedia slice for five encoders, plus 500 self-retrieval queries
per encoder. Point `SHARD_DATA` at a directory containing:

```
E_docs_e5small_1000000.npy      E_queries_e5small_self_q500.npy
E_docs_e5base_1000000.npy       E_queries_e5base_self_q500.npy
E_docs_mpnet_1000000.npy        E_queries_mpnet_self_q500.npy
E_docs_e5large_1000000.npy      E_queries_e5large_self_q500.npy
E_docs_bgem3_1000000.npy        E_queries_bgem3_self_q500.npy
corpus_wiki_ru_1000000.pkl      # paragraph text, for the reference-lookup attack
```

```bash
export SHARD_DATA=/path/to/corpus_cache
```

Encoders, shapes, seeds, and the slicing recipe are in
[`data_manifest.md`](data_manifest.md). If `SHARD_DATA` is unset, the code
looks for `data/corpus_cache/` (then a legacy `notebooks/_corpus_cache/`)
under the repo root.

## 2. SHARD experiments (the contribution)

Numpy / scikit-learn only (the per-cell keys are orthogonal, so the
residual reranking score is exact and is computed directly — no FHE library
needed to measure it). All write to `results/`.

```bash
cd shard
python exp12_shard_utility.py        # self-retrieval utility (all 5 encoders)
python exp17_beir_shard.py           # BEIR utility (downloads SciFact/NFCorpus)
python exp18_shard_cost.py           # active cells/query -> upload bandwidth (C trade-off)
python exp13_shard_alignment.py      # diffuse anchor-complexity m50 vs C  (e5-small)
python exp19_shard_targeted.py       # targeted attacker: m50 ~ d_priv regardless of C
python exp22_shard_learned_attack.py # ridge (ALGEN) / MLP / unsupervised (vec2vec) cores
python exp14_shard_leakage.py        # public-prefix NN-overlap + within-cell leak
python exp20_shard_microkey.py       # micro-key: residual leak -> 0, unlinkability AUC
python exp21_shard_vs_dp.py          # vs. distortion-aware DP-noise at matched utility
python exp15_shard_reference.py      # overlap reference-lookup limitation
python make_fig_shard.py             # regenerate fig_shard_* -> ../paper/figs/
```

Second-encoder validation (e5-base) for the privacy experiments:

```bash
E13_ENC=e5-base E13_DPUB=192 python exp13_shard_alignment.py
E14_ENC=e5-base               python exp14_shard_leakage.py
E19_ENC=e5-base E19_DPUB=192  python exp19_shard_targeted.py
```

Expected headline numbers: diffuse alignment `m50` 200 → 25,600 (C=64) →
102,400 (C=256); targeted `m50 ≈ d_priv` (320 / 576); online cost ≈ 7–30
encrypted residual queries per search; public prefix NN-overlap 0.20–0.55
vs. 0.76 baseline; micro-key residual leak 0.00 and unlinkability AUC ≈ 0.5;
SHARD recovers raw nDCG@10 on BEIR where SVD-`k/2` loses 2–8 points.

## 3. Baseline experiments (the foil)

```bash
cd baseline
python exp07_alignment_pq_leakage.py   # Procrustes recovers global R; public-PQ leakage
python exp08_tradeoff_noise_sweep.py   # sigma_rec is not a privacy metric
python exp09_reference_corpus_attack.py# overlap reference lookup (99.8% top-1)
python exp10_denoiser_significance.py  # paired McNemar/bootstrap on the SVD effect
python exp11_beir_denoiser.py          # the denoiser does NOT transfer to BEIR
python make_fig_significance.py
python make_paper_figures.py
```

`exp_integral.py` (the 10⁶-document integral retrieval + CKKS reranking,
paper §8.4) additionally needs `faiss-cpu` and `tenseal`.

## 4. Build the paper

```bash
cd paper
pdflatex paper_en.tex && pdflatex paper_en.tex
```

## 5. Optional: heavy GPU experiment

The aligned Vec2Text stress test (paper §8.9) needs a GPU and is **not**
required for any SHARD result. Dependencies are pinned in
[`requirements_rtx4090.txt`](requirements_rtx4090.txt).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r docs/requirements_rtx4090.txt
python baseline/heavy_adaptive_inversion_rtx4090.py \
  --profile rtx4090_11gb --output-dir results/adaptive_inversion_outputs
```

It writes per-case `case_*.json` checkpoints (resumable with `--resume`)
and rebuilds `adaptive_inversion_summary.json` / `_samples.csv`. If Hub
dataset downloads are blocked, add `--dataset synthetic_news
--disable-ssl-verification` to use the built-in offline corpus.
