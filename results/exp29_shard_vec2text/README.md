# Experiment 29: SHARD outcome-level Vec2Text audit

This directory contains a real text-reconstruction attack on SHARD's exposed public prefix and cell-keyed residual. It uses the exact mask-aware mean-pooled GTR embedding path bundled with the official `jxm/gtr__nq__32` / `jxm/gtr__nq__32__correct` Vec2Text corrector. The old CLS extraction is not used.

The corpus combines locally cached AG News with controlled synthetic PII. PCA is full-dimensional; SHARD only splits the coordinates. The primary cohort targets the globally largest public-prefix cell and uses nested in-cell known pairs. When that cell has no PII, a separately labelled secondary diagnostic cohort targets the largest PII-containing cell solely to make the controlled PII outcome measurable. The intervals are descriptive, conditional 95% hierarchical bootstrap intervals; only two independent geometry designs are present, so they are not significance claims.

## Globally largest-cell cohort

| Method | m | token-F1 | BLEU | exact PII-ID recall | input cosine | decoded-text cosine |
|---|---:|---:|---:|---:|---:|---:|
| raw | -- | 0.665 [0.613, 0.725] | 0.210 [0.149, 0.277] | n/a | 1.000 [1.000, 1.000] | 0.905 [0.879, 0.929] |
| prefix_only | -- | 0.433 [0.294, 0.570] | 0.068 [0.035, 0.110] | n/a | 0.936 [0.878, 0.986] | 0.811 [0.724, 0.892] |
| unknown_key | -- | 0.242 [0.128, 0.374] | 0.034 [0.015, 0.058] | n/a | 0.880 [0.773, 0.972] | 0.643 [0.511, 0.782] |
| ols_m8 | 8 | 0.450 [0.311, 0.586] | 0.082 [0.040, 0.140] | n/a | 0.939 [0.882, 0.986] | 0.817 [0.725, 0.900] |
| ols_m16 | 16 | 0.442 [0.286, 0.593] | 0.078 [0.042, 0.121] | n/a | 0.941 [0.886, 0.987] | 0.812 [0.709, 0.895] |
| ols_m32 | 32 | 0.429 [0.304, 0.548] | 0.068 [0.040, 0.105] | n/a | 0.944 [0.894, 0.988] | 0.804 [0.682, 0.905] |
| ols_m64 | 64 | 0.444 [0.342, 0.551] | 0.078 [0.040, 0.129] | n/a | 0.953 [0.908, 0.990] | 0.830 [0.753, 0.898] |
| procrustes_m32 | 32 | 0.283 [0.158, 0.418] | 0.040 [0.020, 0.068] | n/a | 0.896 [0.803, 0.976] | 0.679 [0.543, 0.808] |
| procrustes_m64 | 64 | 0.301 [0.164, 0.455] | 0.042 [0.018, 0.087] | n/a | 0.909 [0.825, 0.980] | 0.707 [0.560, 0.829] |
| oracle_residual | -- | 0.665 [0.611, 0.725] | 0.210 [0.151, 0.277] | n/a | 1.000 [1.000, 1.000] | 0.905 [0.878, 0.931] |

## Secondary largest-PII-cell cohort

| Method | m | token-F1 | BLEU | exact PII-ID recall | input cosine | decoded-text cosine |
|---|---:|---:|---:|---:|---:|---:|
| raw | -- | 0.412 [0.342, 0.491] | 0.113 [0.080, 0.150] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.928 [0.899, 0.954] |
| prefix_only | -- | 0.403 [0.342, 0.460] | 0.103 [0.074, 0.135] | 0.000 [0.000, 0.000] | 0.991 [0.984, 0.997] | 0.909 [0.879, 0.935] |
| unknown_key | -- | 0.298 [0.186, 0.407] | 0.053 [0.032, 0.077] | 0.000 [0.000, 0.000] | 0.982 [0.969, 0.995] | 0.834 [0.747, 0.907] |
| ols_m8 | 8 | 0.400 [0.357, 0.444] | 0.086 [0.060, 0.126] | 0.000 [0.000, 0.000] | 0.992 [0.986, 0.998] | 0.901 [0.880, 0.920] |
| ols_m16 | 16 | 0.410 [0.364, 0.457] | 0.126 [0.086, 0.176] | 0.000 [0.000, 0.000] | 0.994 [0.989, 0.998] | 0.919 [0.900, 0.939] |
| ols_m32 | 32 | 0.419 [0.383, 0.456] | 0.083 [0.064, 0.105] | 0.000 [0.000, 0.000] | 0.996 [0.993, 0.998] | 0.919 [0.903, 0.935] |
| ols_m64 | 64 | 0.409 [0.312, 0.514] | 0.096 [0.073, 0.123] | 0.000 [0.000, 0.000] | 0.998 [0.996, 0.999] | 0.898 [0.840, 0.932] |
| procrustes_m32 | 32 | 0.340 [0.237, 0.458] | 0.064 [0.039, 0.102] | 0.000 [0.000, 0.000] | 0.992 [0.985, 0.997] | 0.885 [0.831, 0.928] |
| procrustes_m64 | 64 | 0.337 [0.276, 0.390] | 0.059 [0.045, 0.077] | 0.000 [0.000, 0.000] | 0.996 [0.993, 0.998] | 0.862 [0.823, 0.892] |
| oracle_residual | -- | 0.412 [0.343, 0.486] | 0.113 [0.080, 0.153] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.928 [0.898, 0.953] |

Exact controlled-PII recall by visible item type:

| Method | all visible items | repeated-name | unique email | phone |
|---|---:|---:|---:|---:|
| raw | 0.292 [0.231, 0.349] | 0.792 [0.624, 0.958] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| prefix_only | 0.292 [0.206, 0.354] | 0.792 [0.542, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| unknown_key | 0.169 [0.065, 0.268] | 0.458 [0.167, 0.750] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| ols_m8 | 0.262 [0.190, 0.324] | 0.708 [0.500, 0.875] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| ols_m16 | 0.331 [0.257, 0.400] | 0.896 [0.708, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| ols_m32 | 0.323 [0.274, 0.369] | 0.875 [0.708, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| ols_m64 | 0.308 [0.243, 0.367] | 0.833 [0.667, 0.958] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| procrustes_m32 | 0.246 [0.140, 0.329] | 0.667 [0.375, 0.917] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| procrustes_m64 | 0.254 [0.167, 0.328] | 0.688 [0.417, 0.875] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| oracle_residual | 0.292 [0.227, 0.349] | 0.792 [0.625, 0.958] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |


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

Profile recorded by the finalizer: `full`; requested correction steps: `8`.
