# Reproducing the paper artifacts

This repository contains the LaTeX manuscript, cached experiment outputs,
and scripts used to regenerate the additional revision artifacts.

## Local lightweight artifacts

Use the bundled or project Python with `numpy`, `pandas`, `matplotlib`,
`scikit-learn`, and `scipy`.

```bash
python notebooks/exp7_alignment_pq_leakage.py
python notebooks/exp8_tradeoff_noise_sweep.py
python notebooks/exp9_reference_corpus_attack.py
python notebooks/exp10_denoiser_significance.py     # NEW: paired McNemar/bootstrap (numpy+scipy+sklearn)
python notebooks/exp11_beir_denoiser.py             # NEW: BEIR generality (CPU torch+transformers+datasets)
python notebooks/make_fig_significance.py           # NEW: forest plot of the denoiser deltas
python notebooks/make_paper_figures.py
pdflatex paper_en.tex
pdflatex paper_en.tex
```

## New measurement experiments (this revision)

`exp10_denoiser_significance.py` reads the cached 1M-doc embeddings and the
500 self-retrieval queries for each encoder, reproduces the raw/SVD Acc
numbers of Table 8 (top-10 inner-product search, identical SVD/qrels to the
integral script), and reports paired McNemar exact tests and 10k-sample
paired bootstrap CIs for the SVD-vs-raw deltas at k=d/2. Needs only
numpy/scipy/scikit-learn (no faiss/tenseal/GPU). Writes
`notebooks/exp10_outputs/exp10_significance.json` and per-encoder
`hits_*.npz`.

`exp11_beir_denoiser.py` downloads the public BEIR SciFact/NFCorpus
datasets, encodes them with the e5 family (mean pooling, query:/passage:
prefixes), fits V_k on the corpus at k=d/2, and compares raw vs SVD nDCG@10,
Recall@10 and Acc@1 with paired bootstrap CIs and McNemar. CPU-only.
Configurable via env: `E11_ENCODERS`, `E11_DATASETS`, `E11_MAXLEN`,
`E11_BATCH`. Writes `notebooks/exp11_outputs/exp11_beir.json`.

## SHARD (the new transform)

`shard_lib.py` implements the SHARD construction (PCA rotation, public
prefix / private residual split, k-means cells, per-cell Householder keys).
All SHARD experiments are numpy-only on the cached embeddings (except exp17,
which CPU-encodes BEIR), and are deterministic given the master-key seeds.

```bash
python notebooks/exp12_shard_utility.py     # self-retrieval utility: SHARD vs raw vs SVD-k/2
E13_ENC=e5-small E13_DPUB=96 python notebooks/exp13_shard_alignment.py   # anchor-complexity (m50/m90) vs C
E14_ENC=e5-small python notebooks/exp14_shard_leakage.py                 # public-prefix NN-overlap + within-cell leak
E15_ENC=e5-small E15_DPUB=96 python notebooks/exp15_shard_reference.py   # overlap reference-lookup limitation
python notebooks/exp17_beir_shard.py        # SHARD utility on BEIR (recovers raw nDCG)
python notebooks/exp18_shard_cost.py        # active cells/query -> upload bandwidth, C trade-off
E19_ENC=e5-small E19_DPUB=96 python notebooks/exp19_shard_targeted.py    # targeted attacker: m50 ~ d_priv regardless of C
E20_ENC=e5-small E20_DPUB=96 python notebooks/exp20_shard_microkey.py    # micro-key: residual leak -> 0, unlinkability AUC ~0.5
python notebooks/make_fig_shard.py          # fig_shard_alignment / _leakage / _beir
```
Privacy experiments (exp13/14/19) take `E1x_ENC=e5-base E1x_DPUB=192` for
the second-encoder validation.

Outputs land in `notebooks/exp12_outputs/` ... `exp20_outputs/`. Key
results: diffuse alignment m50 grows 200 -> 25,600 (C=64) -> 102,400
(C=256), reproduced on e5-base; targeted m50 ~ d_priv (320/576) regardless
of C; online cost ~7-30 encrypted residual queries/search; public prefix
NN-overlap 0.20-0.55 vs 0.76 baseline; micro-key residual leak 0.00 and
unlinkability AUC ~0.5; SHARD recovers raw nDCG@10 on BEIR where SVD-k/2
loses 2-8 points.

`exp7_alignment_pq_leakage.py` reads cached e5-small embeddings from
`notebooks/_corpus_cache/`, runs the known-plaintext Procrustes and PQ
leakage experiments, and writes:

- `notebooks/exp7_outputs/exp7_summary.json`
- `notebooks/exp7_outputs/exp7_procrustes_summary.csv`
- `notebooks/exp7_outputs/exp7_pq_leakage.csv`

`make_paper_figures.py` rebuilds every file under `figs/` used by
`paper_en.tex`.

`exp8_tradeoff_noise_sweep.py` reads cached e5-small/e5-base embeddings,
runs the matched-distortion SVD/noise utility-leakage diagnostic, and
writes:

- `notebooks/exp8_outputs/exp8_tradeoff_summary.csv`
- `notebooks/exp8_outputs/exp8_tradeoff_summary.json`

`exp9_reference_corpus_attack.py` reads cached e5-small embeddings and
the cached Russian-Wikipedia paragraph text, runs the reference-corpus
lookup stress test after known-plaintext alignment, and writes:

- `notebooks/exp9_outputs/exp9_reference_attack_by_seed.csv`
- `notebooks/exp9_outputs/exp9_reference_attack_summary.csv`
- `notebooks/exp9_outputs/exp9_reference_attack_summary.json`

## Heavy RTX 4090 adaptive inversion experiment

The adaptive/few-shot inversion experiment is intentionally not run on
the local 8 GB RTX 5060. The primary portable profile is sized for an
RTX 4090 with roughly 11 GiB of free VRAM:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r notebooks/requirements_rtx4090.txt
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
python notebooks/heavy_adaptive_inversion_rtx4090.py \
  --profile rtx4090_11gb \
  --output-dir notebooks/exp8_adaptive_outputs_rtx4090
```

If an earlier install selected a CUDA 13 PyTorch wheel on a driver 550.x host,
discard that virtual environment and recreate it before reinstalling:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r notebooks/requirements_rtx4090.txt
```

For a standalone remote directory such as `~/train2`, copy
the root-level `heavy_adaptive_inversion_rtx4090.py`,
`requirements_rtx4090.txt`, and `setup_rtx4090_env.sh`, then recreate the
environment. If a proxy is needed:

```bash
FORCE_RECREATE=1 PIP_PROXY=http://172.17.130.124:3128 bash setup_rtx4090_env.sh
```

If Hugging Face dataset downloads fail because of SSL/proxy policy, bypass the
Hub dataset with the built-in corpus:

```bash
export HTTP_PROXY=http://172.17.130.124:3128
export HTTPS_PROXY=http://172.17.130.124:3128
python heavy_adaptive_inversion_rtx4090.py \
  --profile rtx4090_11gb_smoke \
  --dataset synthetic_news \
  --disable-ssl-verification \
  --output-dir exp8_adaptive_outputs_smoke
```

For a real local corpus, provide a JSONL file with a `text` field:

```bash
python heavy_adaptive_inversion_rtx4090.py \
  --profile rtx4090_11gb \
  --texts-jsonl corpus.jsonl \
  --disable-ssl-verification \
  --output-dir exp8_adaptive_outputs_rtx4090
```

For a fast dependency and CUDA smoke test:

```bash
python notebooks/heavy_adaptive_inversion_rtx4090.py \
  --profile rtx4090_11gb_smoke \
  --output-dir notebooks/exp8_adaptive_outputs_smoke
```

The script writes per-case `case_*.json` checkpoints, supports `--resume`,
and rebuilds `adaptive_inversion_summary.json` and
`adaptive_inversion_samples.csv` from completed cases. On a larger GPU, the
same entry point accepts `--profile a100`; the legacy
`notebooks/heavy_adaptive_inversion_a100.py` file remains a compatibility
wrapper.

If all `case_*.json` files were written but the final JSON/CSV aggregation
failed, rebuild only the final outputs:

```bash
python heavy_adaptive_inversion_rtx4090.py \
  --finalize-only \
  --output-dir exp8_adaptive_outputs_rtx4090
```
