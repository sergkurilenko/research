# Experiment 29: SHARD outcome-level Vec2Text audit

This directory contains a real text-reconstruction attack on SHARD's exposed public prefix and cell-keyed residual. It uses the exact mask-aware mean-pooled GTR embedding path bundled with the official `jxm/gtr__nq__32` / `jxm/gtr__nq__32__correct` Vec2Text corrector. The old CLS extraction is not used.

The corpus combines locally cached AG News with controlled synthetic PII. PCA is full-dimensional; SHARD only splits the coordinates. Each attack targets the largest public-prefix cell and uses nested in-cell known pairs. Confidence intervals are 95% hierarchical bootstrap intervals over geometry/key runs and target records.

## Pooled outcome results

| Method | m | token-F1 | BLEU | PII recall | input cosine | decoded-text cosine |
|---|---:|---:|---:|---:|---:|---:|
| raw | -- | 0.391 [0.174, 0.609] | 0.032 [0.002, 0.062] | n/a | 1.000 [1.000, 1.000] | 0.758 [0.722, 0.793] |
| prefix_only | -- | 0.494 [0.489, 0.500] | 0.080 [0.049, 0.111] | n/a | 0.986 [0.986, 0.986] | 0.851 [0.825, 0.878] |
| unknown_key | -- | 0.363 [0.261, 0.465] | 0.026 [0.003, 0.048] | n/a | 0.972 [0.972, 0.973] | 0.701 [0.645, 0.757] |
| ols_m8 | 8 | 0.568 [0.511, 0.625] | 0.090 [0.084, 0.096] | n/a | 0.987 [0.986, 0.988] | 0.880 [0.824, 0.936] |
| ols_m16 | 16 | 0.607 [0.565, 0.649] | 0.151 [0.061, 0.242] | n/a | 0.989 [0.988, 0.989] | 0.907 [0.877, 0.938] |
| oracle_residual | -- | 0.391 [0.174, 0.609] | 0.032 [0.002, 0.062] | n/a | 1.000 [1.000, 1.000] | 0.758 [0.722, 0.793] |
| procrustes_m16 | 16 | 0.424 [0.370, 0.478] | 0.110 [0.049, 0.171] | n/a | 0.978 [0.977, 0.979] | 0.789 [0.750, 0.829] |

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
