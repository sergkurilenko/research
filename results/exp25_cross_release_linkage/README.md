# Experiment 25: cross-release linkage measurement

This directory contains a local defensive measurement of whether two fresh-key
releases of the same e5-small document set can be linked after all release row
handles are independently shuffled. Persistent document IDs are deliberately
withheld. The residual-only view contains keyed residuals and, for the cell-key
scheme, the stable cell label. The full-server view additionally contains the
public PCA prefix stored by the current protocol.

## Reproduction

Run from the repository root with the environment used for the experiment:

```powershell
D:\PHD\research\RES\experiments\.venv\Scripts\python.exe shard\exp25_cross_release_linkage.py
```

The full run used e5-small, `d_pub=96`, `d_priv=288`, `C=64`, nested sets of
500, 2,000 and 10,000 documents, and release seeds 11, 23 and 47. The storage
conditions were clean float32, float16, symmetric release-wide int8, and
independent Gaussian perturbation with coordinate RMS multiplier 0.01. The run
completed in 82.5 seconds. Exact arguments and provenance are in `config.json`
and `run_info.json`.

## Main result

At N=10,000 in the clean condition (mean over three independently keyed release
pairs; 1,000 evaluated query rows):

| Scheme/view | Linkage method | Top-1 | Top-10 | MRR | AUC |
|---|---|---:|---:|---:|---:|
| Cell C=64, residual | Cell-label chance control | 0.0064 | 0.0644 | 0.0359 | 0.5000 |
| Cell C=64, residual | Raw cross-key cosine control | 0.0057 | 0.0683 | 0.0367 | 0.5011 |
| Cell C=64, residual | Residual norm NN | 0.8247 | 1.0000 | 0.9078 | 0.9987 |
| Cell C=64, residual | Norm-rank one-to-one assignment | 0.9957 | n/a | n/a | n/a |
| Cell C=64, residual | Gram-row signature NN | 0.9993 | 1.0000 | 0.9997 | 1.0000 |
| Cell C=64, residual | Combined invariant signature | 0.9997 | 1.0000 | 0.9998 | 1.0000 |
| Cell C=64, residual | Known inner products, 1% anchors | 0.8102 | 0.8297 | 0.8192 | 0.8937 |
| Cell C=64, residual | Known inner products, 10% anchors | 0.9996 | 1.0000 | 0.9998 | 1.0000 |
| Microkey, residual | Raw cross-key cosine control | 0.0003 | 0.0030 | 0.0016 | 0.4967 |
| Microkey, residual | Norm-rank one-to-one assignment | 0.9943 | n/a | n/a | n/a |
| Either scheme, full server | Public-prefix NN | 0.9993 | 1.0000 | 0.9997 | 1.0000 |

The small deviation of public-prefix NN from 1.0 at N=10,000 is a float32
distance-ranking/tie effect: the prefix rows themselves are unchanged across
clean releases and can be byte-matched. It therefore does not support an
unlinkability claim.

Within-cell Gram/nearest-neighbour signatures remain at 0.9993--0.9997 Top-1
under float16, int8, and the 1% Gaussian diagnostic control. The public prefix
also remains at least 0.9993 Top-1 in every tested condition. By contrast,
int8/noise substantially reduce the microkey residual norm-rank result, but the
public prefix still links the full server view.

## Interpretation and limitations

The earlier mated-versus-non-mated cross-key cosine statistic is a valid
measurement of that one coordinate-sensitive comparator, but it is not an
unlinkability game. Orthogonal transforms preserve every residual norm and,
when a key is shared within a cell, the complete within-cell Gram graph. Those
invariants are sufficient to link almost all rows in these same-population
snapshot experiments.

This experiment is intentionally local and does not access any external
service. It measures one encoder and one split/cell configuration. Both
snapshots contain exactly the same documents; additions, deletions and partial
overlap should be measured separately. At N=10,000, exact global methods use
1,000 query rows, while signatures are constructed from all 10,000 rows.
Microkey outputs are sampled directly from the exact equal-norm Haar-image
distribution. The quantization and Gaussian conditions are diagnostic storage
controls and have not been calibrated for retrieval utility or formal privacy.

Machine-readable per-run results are in `metrics.csv`/`raw_metrics.json`; the
mean, standard deviation, minimum and maximum over release seeds are in
`summary.csv`/`summary.json`.
