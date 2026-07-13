# Experiment 23: corrected SHARD score

This experiment corrects the query-centering mismatch in the SHARD scoring
rule.  The PCA basis is fitted on centred documents, but the score uses

\[
d_i=(x_i-\mu)V,\qquad q'=qV,
\]

so that `q' @ d_i = q @ x_i - q @ mu`.  The last term is constant over all
documents for a fixed query, hence the corrected full-space ranking is the raw
inner-product ranking.

## Reproduction

The final full run used all cached judged queries, a PCA fit of at most 200,000
documents, and 10,000 paired bootstrap replicates:

```powershell
& 'D:\PHD\research\RES\experiments\.venv\Scripts\python.exe' -u `
  shard\exp23_corrected_score.py --suite all --query-batch 8 `
  --full-order-queries 8 --bootstrap 10000 `
  --output results\exp23_corrected_score --force
```

No corpus was downloaded or re-encoded.  The run consumed the existing exp17
BEIR and exp17b MIRACL embedding caches.

## Full-space and half-PCA results

All values below are nDCG@10.  `Old full` centres both document and query;
`corrected full` centres only the document.  The two half-PCA columns use the
same respective query conventions and therefore separate the centering error
from the actual loss due to truncation.

| Suite | Encoder | Data | Raw | Old full | Corrected full | Old half | Corrected half |
|---|---|---:|---:|---:|---:|---:|---:|
| BEIR | e5-small | SciFact | 0.598215 | 0.558840 | 0.598215 | 0.522189 | 0.549367 |
| BEIR | e5-small | NFCorpus | 0.302097 | 0.294780 | 0.302097 | 0.280718 | 0.290279 |
| BEIR | e5-small | ArguAna | 0.357659 | 0.319188 | 0.357690 | 0.297942 | 0.327941 |
| BEIR | e5-base | SciFact | 0.637036 | 0.586298 | 0.637036 | 0.585246 | 0.622432 |
| BEIR | e5-base | NFCorpus | 0.326852 | 0.305318 | 0.326852 | 0.304379 | 0.326834 |
| BEIR | e5-base | ArguAna | 0.345765 | 0.265997 | 0.345730 | 0.263874 | 0.337861 |
| MIRACL | e5-small | Swahili | 0.679508 | 0.643561 | 0.679508 | 0.614728 | 0.626869 |
| MIRACL | e5-small | Bengali | 0.688274 | 0.655624 | 0.688274 | 0.639694 | 0.649110 |
| MIRACL | e5-base | Swahili | 0.707414 | 0.688364 | 0.707414 | 0.686243 | 0.695544 |
| MIRACL | e5-base | Bengali | 0.726616 | 0.695429 | 0.726616 | 0.696383 | 0.710987 |

The old half-PCA loss cannot be attributed solely to truncation.  Correcting
the query convention recovers 0.0093--0.0740 nDCG, depending on the case.  For
e5-base/NFCorpus, the corrected half-PCA score is only 0.000018 below raw;
almost the entire previously reported gap was caused by the centering mismatch.

## Numerical rank check

For eight deterministic queries from each of the ten cases, raw and transformed
scores were independently recomputed in float64.  The worst score-identity
error over these checks was `7.71e-16`; top-10, top-100, and complete rankings
were exactly identical in every check.

The main evaluation uses float32 embeddings, matching the cached artifact.  Its
corrected top-100 set overlap with raw is 0.999938--1.000000.  The only metric
differences are near-tie swaps on ArguAna (absolute nDCG difference below
`3.6e-5`).

## Two-stage routing

The corrected scoring prefix `q V_pub` can also produce the former centred
routing vector by subtracting the public constant `mu V_pub`.  Thus a centred
shortlist and corrected rerank require neither an additional secret nor extra
query disclosure.  At `d_pub=d/4, K_c=200`, this choice retains the previous
shortlist results while using the corrected score explicitly:

| Suite / encoder / data | Raw | Corrected, centred router | Corrected, uncentred router |
|---|---:|---:|---:|
| BEIR / small / SciFact | 0.598215 | 0.589733 | 0.593668 |
| BEIR / small / NFCorpus | 0.302097 | 0.300414 | 0.300808 |
| BEIR / small / ArguAna | 0.357659 | 0.355779 | 0.356267 |
| BEIR / base / SciFact | 0.637036 | 0.638088 | 0.637076 |
| BEIR / base / NFCorpus | 0.326852 | 0.326222 | 0.326850 |
| BEIR / base / ArguAna | 0.345765 | 0.344971 | 0.346129 |
| MIRACL / small / Swahili | 0.679508 | 0.673118 | 0.663541 |
| MIRACL / small / Bengali | 0.688274 | 0.679421 | 0.678619 |
| MIRACL / base / Swahili | 0.707414 | 0.707704 | 0.707144 |
| MIRACL / base / Bengali | 0.726616 | 0.726616 | 0.726616 |

The uncentred router is retained as an ablation, not as a recommended default:
it is notably weaker on MIRACL/e5-small.

## Artifact files

- `config.json`: command, environment, revision, and score definitions.
- `summary.json`: complete results, paired bootstrap intervals, score errors,
  and per-query rank diagnostics.
- `summary.csv`: flat metric table for manuscript table generation.
- `beir_*.json` and `miracl_*.json`: one self-contained result per cache.
- `run.log`: timestamped progress log; the final full run starts at
  `2026-07-12T10:28:47+00:00`.
