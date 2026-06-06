# `results/` — experiment outputs

One sub-directory per experiment, holding the JSON/CSV produced by the
scripts in `../shard/` and `../baseline/`. These are the exact numbers cited
in the paper; large intermediate arrays (`.npy`/`.npz`/embeddings) are not
included (they are reproducible from the scripts).

**SHARD (contribution)**

| Directory | Script | Headline |
|---|---|---|
| `exp12_outputs/` | `exp12_shard_utility.py` | SHARD ≈ raw on self-retrieval |
| `exp17_outputs/` | `exp17_beir_shard.py` | recovers raw nDCG@10 on BEIR |
| `exp18_outputs/` | `exp18_shard_cost.py` | 7–30 enc. queries/search |
| `exp13_outputs/` | `exp13_shard_alignment.py` | diffuse `m₅₀` 200 → 25.6k → 102.4k (e5-small, e5-base) |
| `exp19_outputs/` | `exp19_shard_targeted.py` | targeted `m₅₀ ≈ d_priv` (320 / 576) |
| `exp22_outputs/` | `exp22_shard_learned_attack.py` | no learned/unsupervised attacker beats `d_priv` |
| `exp14_outputs/` | `exp14_shard_leakage.py` | prefix NN-overlap 0.20–0.55 vs 0.76 |
| `exp20_outputs/` | `exp20_shard_microkey.py` | micro-key residual leak 0.00, AUC 0.50 |
| `exp21_outputs/` | `exp21_shard_vs_dp.py` | DP de-anon 1.0 vs SHARD 0.0 |
| `exp15_outputs/` | `exp15_shard_reference.py` | overlap reference-lookup limitation |

**Baseline (foil)**

| Directory | Headline |
|---|---|
| `exp1_outputs/`, `exp2_outputs/` … `exp6_outputs/` | CKKS modes, integral retrieval, projection/noise baselines |
| `exp5_outputs/` | integral `v3_multi_results*.json` (the 10⁶-doc experiment) |
| `exp7_outputs/` | Procrustes + PQ leakage |
| `exp8_outputs/` | SVD-vs-noise diagnostic |
| `exp9_outputs/` | reference-corpus lookup |
| `exp10_outputs/` | denoiser significance |
| `exp11_outputs/` | BEIR denoiser non-transfer |
| `adaptive_inversion_outputs/` | aligned Vec2Text stress test |
