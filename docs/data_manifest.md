# Data manifest

The cached embeddings are **not** in the repo (~17 GB). Place them in the
directory pointed to by `SHARD_DATA` (default search: `data/corpus_cache/`,
then a legacy `notebooks/_corpus_cache/`, under the repo root). Experiment
outputs go to `results/<exp>_outputs/`.

## Cached embeddings (`$SHARD_DATA/`)

L2-normalised document embeddings of the same 1M-paragraph
Russian-Wikipedia slice for five encoders:

- `E_docs_e5small_1000000.npy`  — `intfloat/multilingual-e5-small`, `(1000000, 384)`
- `E_docs_e5base_1000000.npy`   — `intfloat/multilingual-e5-base`, `(1000000, 768)`
- `E_docs_mpnet_1000000.npy`    — `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, `(1000000, 768)`
- `E_docs_e5large_1000000.npy`  — `intfloat/multilingual-e5-large`, `(1000000, 1024)`
- `E_docs_bgem3_1000000.npy`    — `BAAI/bge-m3`, `(1000000, 1024)`

The `E_queries_<enc>_self_q500.npy` files hold 500 deterministic
self-retrieval query embeddings per encoder. The ground-truth document ids
are reconstructed in code as `numpy.random.default_rng(42).choice(N, 500)`
(matching the query order). `corpus_wiki_ru_1000000.pkl` holds the paragraph
text, used only by the reference-corpus lookup attacks.

The SHARD experiments recompute the PCA basis on the fly from the cached
`.npy` arrays (a 200k-row sample, `RandomState(0)`), so no precomputed SVD /
index artifacts are required. A legacy `index_cache/` (SVD bases, rotated
arrays, HNSW/PQ indexes used by the baseline pipeline) can be rebuilt from
the embeddings.

## Seeds and outputs per experiment

SHARD (`shard/`, outputs in `results/`):

- query / ground-truth seed `42`; bootstrap seed `2026`; rotation/master-key
  seeds `{11, 23, 31}`; gallery seed `7`.
- `exp12_shard_utility.py` → `results/exp12_outputs/exp12_shard_utility.json`
- `exp13_shard_alignment.py` → `results/exp13_outputs/exp13_alignment_<enc>.json`
- `exp14_shard_leakage.py` → `results/exp14_outputs/exp14_leakage_<enc>.json`
- `exp15_shard_reference.py` → `results/exp15_outputs/exp15_reference_<enc>.json`
- `exp17_beir_shard.py` → `results/exp17_outputs/exp17_beir_shard.json` (BEIR data from the Hub)
- `exp18_shard_cost.py` → `results/exp18_outputs/exp18_cost_<enc>.json`
- `exp19_shard_targeted.py` → `results/exp19_outputs/exp19_targeted_<enc>.json`
- `exp20_shard_microkey.py` → `results/exp20_outputs/exp20_microkey_<enc>.json`
- `exp21_shard_vs_dp.py` → `results/exp21_outputs/exp21_vs_dp_<enc>.json`
- `exp22_shard_learned_attack.py` → `results/exp22_outputs/exp22_learned_<enc>.json`

Baseline (`baseline/`, outputs in `results/`):

- `exp07_alignment_pq_leakage.py`: e5-small + on-the-fly SVD; sample seed
  `20260530`, rotation seeds `11,23,31,47,53` → `results/exp7_outputs/`.
- `exp08_tradeoff_noise_sweep.py`: e5-small/e5-base; query seed `42`, noise
  seed `2026` → `results/exp8_outputs/`.
- `exp09_reference_corpus_attack.py`: e5-small + `corpus_wiki_ru_1000000.pkl`;
  sample seed `20260531`, rotation seeds `11,23,31,47,53` → `results/exp9_outputs/`.
- `exp10_denoiser_significance.py`: per encoder; query/gt seed `42`, bootstrap
  `2026` → `results/exp10_outputs/`.
- `exp11_beir_denoiser.py`: HF `BeIR/{scifact,nfcorpus}(-qrels)`; bootstrap
  `2026`, `E11_MAXLEN=128`, `E11_BATCH=64` → `results/exp11_outputs/`.
- `exp_integral.py`: the 10⁶-doc integral retrieval (faiss + tenseal),
  rotation seeds `{11,23,47,31,53}` → `results/exp5_outputs/`.
