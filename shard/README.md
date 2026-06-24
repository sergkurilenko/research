# `shard/` — the SHARD construction and experiments

Numpy / scikit-learn only (the per-cell keys are orthogonal, so the residual
reranking score is exact and needs no FHE library to be measured). Outputs go
to `../results/`, figures to `../paper/figs/`. Set `SHARD_DATA` to the cached
embeddings (see `../docs/reproduce.md`).

| File | Paper § | Produces |
|---|---|---|
| `shard_lib.py` | §7 | the construction (PCA split, cells, Householder keys, two-stage retrieval) |
| `paths.py` | — | portable path resolution (`SHARD_DATA`, `SHARD_RESULTS`, `SHARD_FIGS`) |
| `exp12_shard_utility.py` | §8.11 | self-retrieval utility vs raw vs SVD-baseline |
| `exp17_beir_shard.py` | §8.11 | BEIR utility — SHARD recovers raw nDCG@10 |
| `exp18_shard_cost.py` | §8.12 | active cells/query → upload bandwidth (the `C` trade-off) |
| `exp13_shard_alignment.py` | §8.13 | diffuse anchor-complexity `m₅₀` vs `C` |
| `exp19_shard_targeted.py` | §8.13 | targeted attacker — `m₅₀ ≈ d_priv` regardless of `C` |
| `exp22_shard_learned_attack.py` | §8.16 | ridge (ALGEN core) / MLP / unsupervised (vec2vec core) |
| `exp14_shard_leakage.py` | §8.14 | public-prefix NN-overlap + within-cell leak |
| `exp20_shard_microkey.py` | §8.15 | micro-key: residual leak → 0, unlinkability AUC |
| `exp21_shard_vs_dp.py` | §8.17 | vs. distortion-aware DP-noise at matched utility |
| `exp15_shard_reference.py` | §8.18 | the overlap reference-lookup limitation |
| `make_fig_shard.py` | — | regenerates `fig_shard_*` figures |

Most scripts take `E1x_ENC` / `E1x_DPUB` env vars for the second encoder, e.g.
`E13_ENC=e5-base E13_DPUB=192 python exp13_shard_alignment.py`.
