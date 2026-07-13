# SHARD construction and experiments

The PCA basis is fitted on centred documents. Documents store
`(x-mu) @ V`; a centred query prefix is used for routing, while final scoring
uses `q @ V`. The full split score is therefore
`q @ (x-mu) = q @ x - q @ mu`, which preserves the raw document ranking.

Orthogonal cell keys preserve query-document products in plaintext algebra,
but they also preserve residual norms and within-cell Gram geometry. SHARD is
therefore evaluated as an alignment-compartmentalisation mechanism, not as an
unlinkable or cryptographically private document template.

| File | Purpose |
|---|---|
| `shard_lib.py` | PCA coordinates, cells, orthogonal keys and retrieval helpers |
| `test_shard.py` | synthetic invariants, including rank-correct score and partial-span recovery |
| `exp23_corrected_score.py` | corrected score on cached BEIR and MIRACL embeddings |
| `exp24_partial_alignment.py` | rank-deficient Procrustes, OLS, ridge and polar alignment without a full-rank gate |
| `exp25_cross_release_linkage.py` | shuffled-ID cross-release linkage from prefix, norm and Gram invariants |
| `exp26_ckks_blocksimd.py` | actual TenSEAL CKKS phase timings, serialized traffic and block-SIMD packing |
| `exp27_formal_dp_baseline.py` | analytic Gaussian mechanism under explicit replacement adjacency |
| `exp28_cross_release_churn.py` | linkage under partial overlap, insert/delete churn and quantization |
| `exp29_shard_vec2text.py` | GPU text inversion after raw/unknown/oracle/learned SHARD views |
| `make_fig_corrected_audit.py` | vector manuscript figures from exp23/exp24 summaries |
| `make_fig_maximal_program.py` | vector manuscript figures from exp26/exp27/exp28 summaries |

Older `exp12`--`exp22` scripts are retained for provenance. In particular,
the `exp13`/`exp19` full-rank gate, `exp20` cosine-only unlinkability metric,
and `exp21` pseudo-DP comparison are superseded by experiments 23--29.

Run the current smoke tests with:

```powershell
python shard/test_shard.py
```

Set `SHARD_DATA` to the local embedding cache; outputs default to `results/`.
The CKKS script additionally needs TenSEAL, while the Vec2Text script needs
the frozen GPU environment recorded in `results/exp29_shard_vec2text/`.
