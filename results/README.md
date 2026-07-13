# Experiment outputs

Each directory contains the configuration, logs and machine-readable results
produced by the scripts in `../shard/` or `../baseline/`. Large embedding
caches are intentionally excluded.

## Corrective SHARD experiments used by the revised manuscript

| Directory | Script | Main result |
|---|---|---|
| `exp23_corrected_score/` | `exp23_corrected_score.py` | Corrected split scoring preserves raw ranking on 10 BEIR/MIRACL cells; legacy query centring lost up to 0.080 nDCG. |
| `exp24_partial_alignment_main_v2/` | `exp24_partial_alignment.py` | Partial OLS reaches high residual-gallery R@1 far below full key rank; increasing `C` compartmentalises diffuse pairs but creates no hard threshold. |
| `exp24_partial_alignment_lowm/` | same | Low-anchor extension (`m=4..8192`) confirming the transition. |
| `exp25_cross_release_linkage/` | `exp25_cross_release_linkage.py` | Norm, Gram and prefix invariants link independently re-keyed snapshots almost perfectly. |
| `exp26_ckks_blocksimd/` | `exp26_ckks_blocksimd.py` | 315 real CKKS trials: block packing reduces query upload but exposes the per-candidate response/evaluation bottleneck. |
| `exp27_formal_dp_baseline/` | `exp27_formal_dp_baseline.py` | Exact analytic Gaussian calibration under explicit replacement adjacency; high-utility evaluated points remain natively linkable. |
| `exp28_cross_release_churn/` | `exp28_cross_release_churn.py` | Prefix/norm linkage persists under partial overlap and churn; Gram linkage depends strongly on overlap. |
| `exp29_shard_vec2text/` | `exp29_shard_vec2text.py` | GPU text-level outcomes for raw, prefix-only, unknown-key, oracle and learned alignment views. |
| `maximal_program_figures/` | `make_fig_maximal_program.py` | Deterministic publication figures for experiments 26--28, with source hashes and filter definitions. |

These outputs supersede the security interpretations attached to the older
`exp13`, `exp20`, `exp21` and `exp22` results. The old files remain for
provenance, not as current evidence:

- `exp13` and `exp19` applied Procrustes only after full in-cell rank;
- `exp20` tested row-wise cross-key cosine but omitted norm, Gram and prefix;
- `exp21` used uncalibrated Gaussian perturbation, not differential privacy,
  and reused the full-rank gate; `exp27` is the formal replacement;
- `exp22` used a fixed biased ridge setting.

The remaining baseline directories (`exp1` through `exp11` and
`adaptive_inversion_outputs`) reproduce the earlier global-linear foil.
