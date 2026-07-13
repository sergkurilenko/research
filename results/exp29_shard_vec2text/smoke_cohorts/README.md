# Experiment 29: SHARD outcome-level Vec2Text audit

This directory contains a real text-reconstruction attack on SHARD's exposed public prefix and cell-keyed residual. It uses the exact mask-aware mean-pooled GTR embedding path bundled with the official `jxm/gtr__nq__32` / `jxm/gtr__nq__32__correct` Vec2Text corrector. The old CLS extraction is not used.

The corpus combines locally cached AG News with controlled synthetic PII. PCA is full-dimensional; SHARD only splits the coordinates. The primary cohort targets the globally largest public-prefix cell and uses nested in-cell known pairs. When that cell has no PII, a separately labelled secondary diagnostic cohort targets the largest PII-containing cell solely to make the controlled PII outcome measurable. The intervals are descriptive, conditional 95% hierarchical bootstrap intervals; only two independent geometry designs are present, so they are not significance claims.

## Globally largest-cell cohort

| Method | m | token-F1 | BLEU | exact PII-ID recall | input cosine | decoded-text cosine |
|---|---:|---:|---:|---:|---:|---:|
| raw | -- | 0.519 [0.429, 0.609] | 0.153 [0.096, 0.211] | n/a | 1.000 [1.000, 1.000] | 0.879 [0.844, 0.913] |
| prefix_only | -- | 0.529 [0.500, 0.558] | 0.120 [0.085, 0.155] | n/a | 0.990 [0.983, 0.997] | 0.848 [0.826, 0.869] |
| unknown_key | -- | 0.469 [0.450, 0.489] | 0.066 [0.056, 0.076] | n/a | 0.980 [0.966, 0.994] | 0.867 [0.850, 0.884] |
| ols_m8 | 8 | 0.503 [0.391, 0.615] | 0.110 [0.076, 0.144] | n/a | 0.991 [0.986, 0.997] | 0.902 [0.887, 0.918] |
| ols_m16 | 16 | 0.489 [0.489, 0.490] | 0.100 [0.048, 0.153] | n/a | 0.993 [0.988, 0.998] | 0.877 [0.866, 0.888] |
| oracle_residual | -- | 0.519 [0.429, 0.609] | 0.153 [0.096, 0.211] | n/a | 1.000 [1.000, 1.000] | 0.879 [0.844, 0.913] |
| procrustes_m16 | 16 | 0.500 [0.429, 0.571] | 0.082 [0.048, 0.116] | n/a | 0.985 [0.975, 0.995] | 0.844 [0.802, 0.885] |

## Secondary largest-PII-cell cohort

| Method | m | token-F1 | BLEU | exact PII-ID recall | input cosine | decoded-text cosine |
|---|---:|---:|---:|---:|---:|---:|
| raw | -- | 0.147 [0.133, 0.160] | 0.027 [0.025, 0.029] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.804 [0.766, 0.842] |
| prefix_only | -- | 0.071 [0.000, 0.143] | 0.017 [0.015, 0.019] | 0.000 [0.000, 0.000] | 0.989 [0.988, 0.990] | 0.652 [0.556, 0.748] |
| unknown_key | -- | 0.043 [0.000, 0.087] | 0.013 [0.000, 0.025] | 0.000 [0.000, 0.000] | 0.978 [0.974, 0.982] | 0.558 [0.367, 0.749] |
| ols_m8 | 8 | 0.114 [0.074, 0.154] | 0.024 [0.021, 0.027] | 0.000 [0.000, 0.000] | 0.991 [0.991, 0.991] | 0.715 [0.638, 0.792] |
| ols_m16 | 16 | 0.074 [0.074, 0.074] | 0.028 [0.025, 0.032] | 0.000 [0.000, 0.000] | 0.992 [0.991, 0.992] | 0.739 [0.731, 0.746] |
| oracle_residual | -- | 0.147 [0.133, 0.160] | 0.027 [0.025, 0.029] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.804 [0.766, 0.842] |
| procrustes_m16 | 16 | 0.235 [0.174, 0.296] | 0.046 [0.045, 0.047] | 0.000 [0.000, 0.000] | 0.984 [0.983, 0.984] | 0.727 [0.651, 0.802] |

Exact controlled-PII recall by visible item type:

| Method | all visible items | repeated-name | unique email | phone |
|---|---:|---:|---:|---:|
| raw | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| prefix_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| unknown_key | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| ols_m8 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| ols_m16 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| oracle_residual | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| procrustes_m16 | 0.200 [0.000, 0.500] | 0.500 [0.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |


## Files

- `raw_control.json`: mandatory positive-control sanity check for the official pooling path.
- `config.json`, `runtime.json`, `corpus_metadata.json`, `embeddings_metadata.json`: exact protocol and environment.
- `cases/`: checkpointed JSON for every geometry/key/method.
- `per_case.csv` / `.json`: every decoded sample and outcome metric.
- `summary_by_run.csv` / `.json`: within-run bootstrap summaries.
- `summary_by_cohort.csv` / `.json`: primary and PII cohorts kept separate.
- `paired_deltas_by_cohort.csv` / `.json`: paired descriptive deltas against raw and prefix-only controls.
- `summary_pooled.csv` / `.json`: exploratory pooled summaries; do not use them in place of the cohort panels.
- `run.log`: execution, OOM, and fallback log.

## Scope and interpretation

The reconstruction deliberately grants a strengthened observer the exact full PCA basis V and document mean mu; the only unknown in unknown-key/OLS/Procrustes is the target cell key H. The official corrector requires native unnormalized mean-pooled GTR vectors, so this is a checkpoint-compatible SHARD split stress test rather than the paper's literal unit-normalized e5 retrieval instance. Results are encoder- and attacker-specific. Prefix-only and unknown-key are baselines, not formal privacy guarantees. Raw and oracle-residual are exact-geometry reference controls: oracle must reconstruct the raw embedding numerically, but neither is an upper bound on token-F1 or BLEU because iterative neural decoding is non-monotone under embedding perturbations. Under an exact orthogonal key, minimum-norm OLS prediction depends on the anchor span and is algebraically invariant to the key seed; the clustered bootstrap therefore does not count the duplicated seed result as an independent target sample. Synthetic PII measures exact recovery of controlled visible items. Email usernames are unique; names repeat and are reported separately, some target names can also occur among anchors, and the card field has no visible denominator after 32-token truncation in this corpus. Headline PII-ID recall excludes names and includes only visible email/phone/card identifiers.

Profile recorded by the finalizer: `smoke`; requested correction steps: `2`.
