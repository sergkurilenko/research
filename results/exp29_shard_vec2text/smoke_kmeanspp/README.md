# Experiment 29: SHARD outcome-level Vec2Text audit

This directory contains a real text-reconstruction attack on SHARD's exposed public prefix and cell-keyed residual. It uses the exact mask-aware mean-pooled GTR embedding path bundled with the official `jxm/gtr__nq__32` / `jxm/gtr__nq__32__correct` Vec2Text corrector. The old CLS extraction is not used.

The corpus combines locally cached AG News with controlled synthetic PII. PCA is full-dimensional; SHARD only splits the coordinates. Each attack targets the largest public-prefix cell and uses nested in-cell known pairs. Confidence intervals are 95% hierarchical bootstrap intervals over geometry/key runs and target records.

## Pooled outcome results

| Method | m | token-F1 | BLEU | PII recall | input cosine | decoded-text cosine |
|---|---:|---:|---:|---:|---:|---:|
| raw | -- | 0.535 [0.458, 0.612] | 0.152 [0.055, 0.249] | n/a | 1.000 [1.000, 1.000] | 0.871 [0.855, 0.887] |
| prefix_only | -- | 0.444 [0.400, 0.489] | 0.119 [0.026, 0.213] | n/a | 0.979 [0.969, 0.988] | 0.841 [0.822, 0.860] |
| unknown_key | -- | 0.470 [0.390, 0.549] | 0.147 [0.040, 0.253] | n/a | 0.958 [0.939, 0.977] | 0.802 [0.702, 0.902] |
| ols_m8 | 8 | 0.489 [0.478, 0.500] | 0.070 [0.053, 0.086] | n/a | 0.983 [0.976, 0.990] | 0.833 [0.791, 0.875] |
| ols_m16 | 16 | 0.523 [0.468, 0.578] | 0.103 [0.074, 0.133] | n/a | 0.984 [0.978, 0.990] | 0.851 [0.847, 0.855] |
| oracle_residual | -- | 0.535 [0.458, 0.612] | 0.152 [0.055, 0.249] | n/a | 1.000 [1.000, 1.000] | 0.871 [0.855, 0.887] |
| procrustes_m16 | 16 | 0.383 [0.375, 0.391] | 0.037 [0.025, 0.048] | n/a | 0.967 [0.954, 0.981] | 0.808 [0.719, 0.896] |

## Files

- `raw_control.json`: mandatory positive-control sanity check for the official pooling path.
- `config.json`, `runtime.json`, `corpus_metadata.json`, `embeddings_metadata.json`: exact protocol and environment.
- `cases/`: checkpointed JSON for every geometry/key/method.
- `per_case.csv` / `.json`: every decoded sample and outcome metric.
- `summary_by_run.csv` / `.json`: within-run bootstrap summaries.
- `summary_pooled.csv` / `.json`: hierarchical-bootstrap pooled summaries.
- `run.log`: execution, OOM, and fallback log.

## Scope and interpretation

The pretrained corrector is specialized for GTR-base and Natural Questions-style 32-token inputs. Results are therefore an encoder- and attacker-specific outcome audit, not a claim about e5 or arbitrary embedding models. Prefix-only and unknown-key are attacker baselines, not formal privacy guarantees. Oracle residual is a numerical upper control. Synthetic PII measures exact recovery of controlled visible items; an item truncated before embedding is excluded from its denominator.

Profile recorded by the finalizer: `smoke`; requested correction steps: `2`.
