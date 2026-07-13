# Reproducing the corrected SHARD audit

The repository includes the manuscript, scripts, configurations, summaries,
per-seed measurements, and run logs. Large embedding caches are not committed.
The current manuscript relies on corrective and maximal-audit Experiments
23--29; Experiments 12--22 remain available for provenance and baseline
characterization.

## 1. Environment

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The geometry, alignment, linkage, DP and plotting code uses NumPy, SciPy,
scikit-learn, pandas, psutil and Matplotlib. Experiment 26 additionally uses
`tenseal==0.3.16`; Experiment 29 uses the frozen PyTorch/Transformers/Vec2Text
environment recorded in its result directory and an NVIDIA GPU. Encoding new
BEIR/MIRACL caches also requires PyTorch, Transformers, and Datasets.

## 2. Data

Set `SHARD_DATA` to the embedding-cache directory described in
[`data_manifest.md`](../data_manifest.md):

```bash
export SHARD_DATA=/path/to/corpus_cache
```

On Windows PowerShell:

```powershell
$env:SHARD_DATA = 'D:\path\to\corpus_cache'
```

The code also recognizes the legacy cache locations documented in
`shard/paths.py`. Cached BEIR and MIRACL arrays are intentionally excluded from
the public repository because of their size.

## 3. Corrective experiments

Run from the repository root:

```bash
python shard/test_shard.py
python shard/exp23_corrected_score.py
python shard/exp24_partial_alignment.py
python shard/exp25_cross_release_linkage.py
python shard/exp26_ckks_blocksimd.py
python shard/exp27_formal_dp_baseline.py
python shard/exp28_cross_release_churn.py
python shard/exp29_shard_vec2text.py
python shard/make_fig_corrected_audit.py
python shard/make_fig_maximal_program.py
```

Outputs:

| Experiment | Directory | Purpose |
|---|---|---|
| 23 | `results/exp23_corrected_score/` | Corrected query/document score, BEIR/MIRACL utility, paired-bootstrap intervals |
| 24 primary | `results/exp24_partial_alignment_main_v2/` | Rank-deficient Procrustes, minimum-norm OLS, ridge, and polar estimators |
| 24 low-anchor | `results/exp24_partial_alignment_lowm/` | Extension down to 4--16 global pairs |
| 25 | `results/exp25_cross_release_linkage/` | Norm, Gram, prefix, quantization, and perturbation linkage controls |
| 26 | `results/exp26_ckks_blocksimd/` | Actual TenSEAL/SEAL phase timings, traffic, errors and block packing |
| 27 | `results/exp27_formal_dp_baseline/` | Analytic Gaussian calibration, utility and native-gallery linkage |
| 28 | `results/exp28_cross_release_churn/` | Partial-overlap, insert/delete-churn and quantization linkage |
| 29 | `results/exp29_shard_vec2text/` | GPU text-inversion outcomes for raw and SHARD alignment views |
| figures | `results/maximal_program_figures/` | Deterministic CKKS, DP and churn PDF/PNG plots |

The main checks to expect are:

- corrected full scoring reproduces the raw rank in float64; centering the
  scoring query caused up to 0.080 nDCG loss;
- minimum-norm OLS first exceeds residual-gallery R@1 = 0.9 at global budgets
  32, 512, 2,048, and 8,192 for `C = 1, 16, 64, 256`, corresponding to about
  32--36 anchors in the average target cell and zero full-key-rank coverage;
- at `N = 10,000`, clean norm-rank linkage is 0.9957 for cell keys and 0.9943
  for micro-keys; cell Gram linkage and full-view prefix linkage are 0.9993.
- block-SIMD cuts median query upload by 74--87% at `K=128`, but the measured
  p50 latency rises by 14--26% because per-candidate encrypted responses remain;
- at `epsilon=1`, the formal Gaussian release reaches at most 0.011 nDCG@10;
  only 3/8 strict SHARD-utility matches occur on the finite grid, all at
  `epsilon=32768` with linkage R@1 at least 0.995;
- under 25--100% release overlap, clean prefix and norm still link persistent
  rows almost perfectly, while cell-Gram linkage degrades with churn.
- in the strengthened GTR outcome audit, largest-cell mean token-F1 is 0.665
  for raw geometry, 0.242 for an unknown key, 0.433 for prefix-only and 0.450
  for eight-pair OLS; raw itself recovers no exact unique email or phone, so
  the PII diagnostic is explicitly inconclusive.

These findings supersede the interpretations previously attached to
Experiments 13, 19, 20, 21, and 22. In particular, the old full-rank gate is
not a de-anonymization threshold, the cosine-only cross-key test is not an
unlinkability test, and the uncalibrated Gaussian perturbation is not a DP
mechanism.

## 4. Figures and manuscripts

Regenerate the corrected vector/PDF and PNG figures:

```bash
python shard/make_fig_corrected_audit.py
python shard/make_fig_maximal_program.py
```

Build the canonical manuscript:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error paper_en.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_en.tex
```

Generate and build the JISA review wrapper:

```bash
cd jisa
python sync_from_canonical.py
pdflatex -interaction=nonstopmode -halt-on-error paper_jisa.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_jisa.tex
```

`jisa/sync_from_canonical.py` copies the scientific body and bibliography from
`paper/paper_en.tex` and applies only review-layout formatting.

## 5. Scope of the maximal audit

Experiment 26 is a measured local CKKS implementation: its online latency
includes transform, packing, encryption, serialization, ct--pt evaluation,
response serialization, decryption and ranking checks. It excludes network
RTT/TLS, ANN latency and concurrent clients. The server emits one ciphertext
per candidate, so packed multi-score output is not claimed.

Experiment 27 is central DP for document-vector content under fixed-size
replacement adjacency and public participation. It is not an add/remove
membership guarantee, does not protect queries, and assumes public or
independently trained preprocessing parameters. Repeated releases compose.

Experiment 28 inserts and deletes rows but does not change the embedding of a
persistent document. Experiment 29 is tied to its named GTR/Vec2Text
checkpoints and intentionally grants the observer the exact PCA basis and
corpus mean. Its native GTR vectors are unnormalised to match the public
corrector. Neither result is a universal claim about all embedding drift or
all learned reconstruction models; the evaluator-selected PII cohort and
repeated-name recall are kept separate from exact unique email/phone recovery.
