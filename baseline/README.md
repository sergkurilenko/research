# `baseline/` — the global-linear stack (the foil)

These scripts characterise the popular **SVD + global-rotation + PQ + CKKS**
baseline and the three failure modes that motivate SHARD: known-plaintext
Procrustes alignment, public-PQ leakage, and exact reference-corpus lookup —
plus the measurement findings that the SVD "denoiser" does not transfer to
real IR and that `σ_rec` is not a privacy metric.

| File | Paper § | Produces |
|---|---|---|
| `exp_integral.py` | §8.4 | 10⁶-document integral retrieval + CKKS reranking (uses `faiss`, `tenseal`) |
| `exp07_alignment_pq_leakage.py` | §8.7 | Procrustes recovers the global rotation; public-PQ NN leakage |
| `exp08_tradeoff_noise_sweep.py` | §8.4 | matched-distortion SVD vs noise — `σ_rec` is not privacy |
| `exp09_reference_corpus_attack.py` | §8.8 | overlap reference lookup (99.8% top-1 after alignment) |
| `exp10_denoiser_significance.py` | §8.5 | paired McNemar/bootstrap on the SVD effect |
| `exp11_beir_denoiser.py` | §8.6 | the denoiser does **not** transfer to BEIR |
| `heavy_adaptive_inversion_rtx4090.py` | §8.9 | aligned Vec2Text stress test (**GPU**; deps in `../docs/`) |
| `make_fig_significance.py` | — | the significance forest plot |
| `make_paper_figures.py` | — | baseline figures |

`exp_integral.py` and `heavy_adaptive_inversion_rtx4090.py` need extra
dependencies (`faiss-cpu`, `tenseal`; and a GPU PyTorch stack respectively);
the other scripts are numpy/scikit-learn/CPU-torch only.
