# Experiment 28: cross-release linkage under insert/delete churn

## Question and threat game

This experiment asks whether independently re-keyed SHARD releases can be
linked when their row handles are independently shuffled and only part of the
underlying corpus persists.  Each release contains `N` rows.  For overlap
fraction `rho`, `round(rho*N)` documents are common, while each release also
contains `N-round(rho*N)` private rows representing deletions and insertions.
Persistent document IDs are hidden from every attack and are used only by the
evaluator after predictions have been produced.

The evaluated full-server view contains the public prefix, its deterministic
cell label, and the keyed residual.  Releases use independent keys.  We test
both one key per public cell (`cell_C64`) and one independent Haar key per
document (`microkey_perdoc`).  The latter is sampled by drawing the exact
equal-norm spherical image of each residual rather than by materialising one
dense QR factorisation per row.

No ground-truth pair, overlap label, or persistent ID is used to construct or
standardise a feature.  Scaling pools only the marginal rows observed in the
two releases.  Candidate matching is restricted to the same disclosed public
cell.  FP16 and release-local symmetric INT8 are diagnostic storage controls,
not privacy mechanisms.

## Experimental grid

- encoders: `e5-small` (`d=384`, `d_pub=96`) and `e5-base`
  (`d=768`, `d_pub=192`);
- release sizes: `N=2,000` and `N=10,000`;
- overlap: `25%, 50%, 75%, 90%, 100%`;
- independent release-pair seeds: `11, 23, 47`;
- observations: clean float32, FP16, and symmetric INT8;
- public cells: `C=64`;
- linkage features: public prefix, residual norm, within-cell Gram quantiles,
  robust strongest-neighbour signatures, their residual combination, and a
  prefix+residual combination.  Raw cross-key cosine is retained as a clean
  negative control.

The completed run contains **2,640 raw records** and **880 three-seed summary
rows**.  Wall-clock time was **311.1 s** on the recorded local environment.

## Metric definitions

`top1_recall` is the fraction of truly shared A rows whose forced nearest
neighbour in B is the mate.  `top1_precision` divides the same correct-link
count by all A rows; because forced matching emits one prediction for every A
row, A-only rows count as false links.  MRR and top-5/top-10 recall are computed
only on shared rows.  ROC AUC compares every shared mate score with ten sampled
non-mates from the same public cell; the negative sampling is used only for
evaluation.

The open-set MNN rule accepts a link only if the two rows are mutual nearest
neighbours.  It has no fitted threshold and therefore requires neither an
overlap estimate nor labelled calibration data.  `mnn_false_matches` includes
accepted A-only links and incorrect accepted links for shared rows;
`mnn_unmatched_a` is the number of A rows for which MNN abstains.  MNN is a
transparent threshold-free diagnostic, not an optimal open-set linker.

Tied distance ranks use competition ranking (`1 + number strictly closer`).
The concrete forced top-1/MNN choice uses the independently random release row
order as a tie breaker.  Exact embedding duplicates therefore do not receive a
persistent-ID tie break.

## Headline results (`N=10,000`, clean, three-seed means)

The unchanged public prefix remains an almost deterministic linkage channel
for the common subset.  The table reports common-item top-1 recall followed by
MNN precision/recall.  The values use `cell_C64`; the microkey results are the
same to rounding because microkeys do not change the prefix.

| Encoder | Overlap | Prefix R@1 | MNN precision | MNN recall | Mean MNN false links |
|---|---:|---:|---:|---:|---:|
| e5-small | 25% | 0.9999 | 0.5829 | 0.9996 | 1788.7 |
| e5-small | 50% | 0.9999 | 0.8599 | 0.9998 | 814.3 |
| e5-small | 75% | 0.9998 | 0.9733 | 0.9997 | 206.0 |
| e5-small | 90% | 0.9999 | 0.9961 | 0.9999 | 35.7 |
| e5-small | 100% | 0.9999 | 1.0000 | 0.9999 | 0.3 |
| e5-base | 25% | 0.9997 | 0.6157 | 0.9995 | 1560.0 |
| e5-base | 50% | 0.9999 | 0.8736 | 0.9997 | 723.0 |
| e5-base | 75% | 0.9999 | 0.9757 | 0.9999 | 187.0 |
| e5-base | 90% | 0.9997 | 0.9967 | 0.9996 | 30.0 |
| e5-base | 100% | 0.9998 | 0.9999 | 0.9997 | 1.0 |

The lower MNN precision at 25--50% overlap is not protection of the common
documents: their recall remains essentially one.  It reflects false reciprocal
matches among churn-only rows when no distance threshold is allowed.  In clean
and FP16 observations, an attacker can additionally exploit the zero/near-zero
prefix distance; the present MNN figures are therefore conservative for that
channel.

Residual norm is also a strong key-invariant identifier.  In clean float32 at
`N=10,000`, its common-item R@1 is at least 0.9996 for every overlap, encoder,
and key scheme.  At 50% overlap, FP16 lowers R@1 to 0.9632 (`e5-small`) and
0.9782 (`e5-base`); INT8 lowers it to 0.4505 and 0.5522, respectively, while
pairwise AUC remains 0.9930 and 0.9951.  Quantisation weakens exact scalar
nearest-neighbour recovery but does not remove the channel.

Cell-key Gram geometry is highly sensitive to corpus churn yet becomes strong
again as overlap rises.  The following table gives clean R@1 for the combined
norm/Gram/neighbour residual signature.  Parentheses give the Gram-only R@1.

| Encoder | 25% | 50% | 75% | 90% | 100% |
|---|---:|---:|---:|---:|---:|
| e5-small, cell keys | 0.1483 (0.1335) | 0.3153 (0.2850) | 0.6503 (0.6040) | 0.9125 (0.8840) | 0.9998 (0.9998) |
| e5-base, cell keys | 0.1404 (0.1337) | 0.2998 (0.2731) | 0.6380 (0.5853) | 0.9134 (0.8754) | 0.9998 (0.9998) |
| e5-small, microkeys | 0.0163 (0.0145) | 0.0199 (0.0175) | 0.0196 (0.0150) | 0.0172 (0.0150) | 0.0205 (0.0169) |
| e5-base, microkeys | 0.0192 (0.0172) | 0.0209 (0.0187) | 0.0189 (0.0170) | 0.0199 (0.0175) | 0.0207 (0.0183) |

Thus per-document microkeys do destroy cross-release *Gram* invariance, but
they do not conceal the unchanged public prefix or the preserved residual
norm.  Cell keys preserve Gram geometry exactly only for an unchanged cell
membership; with 25--50% overlap the signature is degraded, but at 90% overlap
it already recovers more than 0.87 Gram-only R@1.  These results rule out a
general unlinkability interpretation for either scheme.

## Reproduction

Run from the repository root with the Python environment recorded in
`config.json`:

```powershell
& 'D:\PHD\research\RES\experiments\.venv\Scripts\python.exe' `
  'D:\PHD\research\RES1\shard\exp28_cross_release_churn.py'
```

The exact expanded command used by the script is stored in `reproduce.txt`.
The embedding arrays are external cache files and are not duplicated in this
result directory; their absolute paths and sizes are recorded in `config.json`.

## Files

- `config.json`: full grid, environment, threat-game definition, input paths,
  and exact reproduction command;
- `metrics.csv` / `raw_metrics.json`: all per-seed measurements;
- `summary.csv` / `summary.json`: mean, sample standard deviation, minimum and
  maximum over the three seeds;
- `run.log`: complete timestamped execution trace;
- `run_info.json`: completion state, counts, runtime, Git state and script hash;
- `reproduce.txt`: one-line expanded command;
- `../../shard/exp28_cross_release_churn.py`: executable experiment.

## Scope and limitations

This is a geometric linkage audit, not a learned text-inversion attack.  Churn
changes which cached corpus embeddings appear in a release; it does not mutate
the text or embedding of a persistent document.  A deployment that re-embeds
documents with stochastic or version-changing encoders requires a separate
study.  No claim of differential privacy follows from the FP16/INT8 controls.
