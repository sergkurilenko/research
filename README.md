# SHARD: cell-keyed residual splitting for alignment-resistant private dense retrieval

This repository contains the manuscript, code, and experimental artifacts for
SHARD. The current revision treats SHARD as an alignment-compartmentalization
mechanism with explicit leakage. It does **not** claim unlinkability,
cancellable templates, differential privacy, or cryptographic protection of
stored document embeddings.

- Manuscript: [`paper/paper_en.pdf`](paper/paper_en.pdf)
- LaTeX source: [`paper/paper_en.tex`](paper/paper_en.tex)
- Experiment code: [`shard/`](shard/)
- Curated outputs: [`results/`](results/)

## What changed in the corrected audit

### Rank-preserving scoring

PCA is fitted to centered documents, but the scoring query must not be
centered. With an orthogonal PCA basis `V`, documents use
`V^T(x - mu)` and scoring queries use `V^T q`. Their full-space inner product
is

```text
q^T (x - mu) = q^T x - q^T mu,
```

so the document order is exactly the raw dot-product order. A separate
centered query is retained for coarse routing. Across ten BEIR/MIRACL
encoder--dataset configurations, the corrected full score reproduces raw
nDCG and recall; the former centered-query implementation lost as much as
0.080 nDCG.

### Partial alignment has no hard threshold

The older analysis evaluated a full-rank-gated Procrustes attacker and
mistook complete key identifiability for a de-anonymization threshold.
Experiment 24 removes the gate and compares rank-deficient Procrustes,
minimum-norm OLS, ridge, and polar estimators. A known-pair set identifies the
secret map on its observed span even when the complete orthogonal key remains
underdetermined.

In the measured e5-small residual gallery, minimum-norm OLS exceeds residual
R@1 = 0.9 with about 32--36 well-spread pairs in the target cell. Increasing
the cell count spreads diffuse disclosures across cells: the global pair
budget at the same target moves from 32 pairs at `C=1` to 8,192 at `C=256`.
This is compartmentalization, not a formal security threshold.

### Re-keying is linkable

Experiment 25 independently re-keys and permutes two releases of the same
records. Raw cross-key cosine is near chance, but orthogonal invariants are
much stronger:

- norm-rank assignment links 99.6% of cell-keyed records and 99.4% of
  micro-keyed records at `N=10,000`;
- within-cell Gram signatures link 99.9% of cell-keyed records;
- the unchanged public prefix links 99.9% of either variant.

Consequently, re-keying changes coordinates but does not establish
unlinkability or cancellable-template renewal.

Experiment 28 repeats the game with 25--100% document overlap, insert/delete
churn, two encoders, fp16 and int8 observations. Churn weakens cell-Gram
matching at low overlap, but the unchanged prefix and clean residual norm link
persistent documents almost perfectly. Per-document micro-keys remove the
Gram channel without removing prefix or norm linkage.

### Measured CKKS/block-SIMD path

Experiment 26 performs 315 actual TenSEAL/SEAL CKKS trials. At 128 candidates,
block packing reduces median query upload by 86.7% for e5-small (width 8) and
74.2% for e5-base (width 4), with maximum score error `2.29e-6`, no top-1
flips, and perfect top-10 overlap. It is not a latency win in the current
layout: p50 rises by 26.3% and 14.5%, because the server still returns one
encrypted scalar per candidate (`30.13 MB` total).

### Formal Gaussian-release baseline

Experiment 27 uses fixed-size replacement adjacency, a fixed clipping bound,
global sensitivity `2.000002`, and exact analytic Gaussian calibration at
`delta=1e-6`. At `epsilon=1`, nDCG@10 is at most 0.011. Strict utility matches
to corrected SHARD appear in only 3/8 cases on the finite grid, all at
`epsilon=32768` with native-gallery linkage R@1 at least 0.995. This is a
different formal guarantee; SHARD itself is not DP.

### Checkpoint-compatible GTR text outcome

Experiment 29 feeds raw and reconstructed SHARD views to the official public
GTR Vec2Text corrector using its mask-aware mean-pooling path. It deliberately
grants the observer the exact PCA basis and corpus mean, leaving only the cell
key unknown or partially learned from 8--64 in-cell pairs. Native GTR vectors
remain unnormalised to stay compatible with the checkpoint, so this is a
strengthened-oracle, GTR-specific stress test rather than the exact
L2-normalised e5 deployment or a universal learned-decoder claim. The
evaluator-selected PII cohort is reported separately; unique email/phone
recovery is not pooled with repeated-name recall.

Across the two globally largest-cell designs, the descriptive mean token-F1
is 0.665 for raw/exact geometry, 0.433 for prefix-only, 0.242 when the keyed
residual is used without its key, and 0.450 after minimum-norm OLS with eight
in-cell pairs. Thus the unknown key has a real no-anchor effect, but prefix
leakage and targeted disclosure recover much of it.

## Construction

Let `mu` be the corpus centroid and split an orthogonal PCA basis into a
public prefix and private residual:

```text
u_i = V_pub^T  (x_i - mu)
r_i = V_priv^T (x_i - mu)
z_i = H_c r_i
```

The centered prefix `V_pub^T(q - mu)` is used only for routing. Reranking uses
the uncentered scoring coordinates

```text
u_q     = V_pub^T q
t_{q,c} = H_c V_priv^T q,
```

which give

```text
<u_q,u_i> + <t_{q,c},z_i> = q^T x_i - q^T mu.
```

The client holds the cell keys. The CKKS layer encrypts the cell-specific
scoring query for ciphertext--plaintext reranking. Experiment 26 implements
this path, including block-SIMD packing, serialization, server evaluation and
client decryption. Network RTT/TLS, concurrency and packed multi-score
responses remain outside the local benchmark.

## Main corrected experiments

| Script | Purpose | Output |
|---|---|---|
| `shard/exp23_corrected_score.py` | Corrected score identity, BEIR/MIRACL utility, bootstrap CIs | `results/exp23_corrected_score/` |
| `shard/exp24_partial_alignment.py` | Rank-deficient partial-alignment audit | `results/exp24_partial_alignment_main_v2/`, `results/exp24_partial_alignment_lowm/` |
| `shard/exp25_cross_release_linkage.py` | Cross-release matching from norm, Gram, and prefix invariants | `results/exp25_cross_release_linkage/` |
| `shard/exp26_ckks_blocksimd.py` | Measured TenSEAL CKKS and block-SIMD layouts | `results/exp26_ckks_blocksimd/` |
| `shard/exp27_formal_dp_baseline.py` | Analytically calibrated Gaussian-release utility/linkage audit | `results/exp27_formal_dp_baseline/` |
| `shard/exp28_cross_release_churn.py` | Cross-release linkage under partial overlap and churn | `results/exp28_cross_release_churn/` |
| `shard/exp29_shard_vec2text.py` | GPU Vec2Text outcome under SHARD alignment views | `results/exp29_shard_vec2text/` |
| `shard/make_fig_corrected_audit.py` | Regenerate corrected utility/alignment figures | `paper/figs/` |
| `shard/make_fig_maximal_program.py` | Regenerate CKKS, formal-DP and churn figures | `results/maximal_program_figures/` |
| `shard/test_shard.py` | Algebraic and implementation smoke tests | terminal output |

Experiments 21 and 21b are retained only as legacy distortion diagnostics.
Their Gaussian noise is not a DP mechanism because no clipping rule,
sensitivity, adjacency relation, or `(epsilon, delta)` accounting was defined.

## Reproduction

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Run the lightweight correctness tests:

```bash
python shard/test_shard.py
```

Run the corrected audit from the repository root:

```bash
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

The large embedding caches are not committed. Point `SHARD_DATA` at the
local cache described in [`docs/data_manifest.md`](docs/data_manifest.md),
or regenerate the embeddings. Each result directory records its configuration,
per-seed measurements, summaries, and run log.

Build the manuscript with two LaTeX passes:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error paper_en.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_en.tex
```

## Scope

The public prefix, cell labels, candidate identities, residual norms,
within-cell Gram geometry, active cells, and access patterns remain exposed in
the evaluated design. The churn experiment keeps each persistent embedding
fixed, so text edits and encoder-version drift remain untested. The CKKS run is
a local single-process benchmark, not a networked concurrent service. The
formal DP baseline assumes public participation and replacement adjacency.
Future work should pack multiple scores per response and test a non-orthogonal
score-preserving transform that breaks the measured invariants.

## Citation

```bibtex
@misc{kurilenko_shard,
  title  = {SHARD: cell-keyed residual splitting for alignment-resistant
            private dense retrieval},
  author = {Kurilenko, Sergey},
  year   = {2026}
}
```

Code and documentation are released under the [MIT License](LICENSE).
