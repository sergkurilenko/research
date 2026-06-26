"""Experiment 14b: a PQ-only hardened store as a defense baseline vs SHARD.

A "PQ-only" defense publishes product-quantization codes of the (secretly
rotated) store, instead of SHARD's short public prefix plus a cell-keyed,
*exact* residual. We compare both on the same data and the same leakage
metric:

  * leakage   -- NN-overlap@10 of the public/accessible channel with the
                 full-space top-10 neighbours (SHARD short prefix d/4 vs the
                 PQ-reconstructed full vector); lower = less geometry leaked.
  * distortion-- reconstruction cosine of the channel used for reranking.
                 SHARD's per-cell orthogonal keys cancel, so its residual
                 rerank is EXACT (cosine 1.0); PQ-only is lossy.

Pure numpy + scikit-learn PQ codebooks (same method as the public PQ-leakage
experiment), reading the cached embeddings. Deterministic.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import RESULTS
import json, os, time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import shard_lib as S

OUT = (RESULTS / "exp14_outputs"); OUT.mkdir(exist_ok=True)
ENCODERS = os.environ.get("E14B_ENCS", "e5-small,e5-base").split(",")
N_POOL = int(os.environ.get("E14B_POOL", 100_000))
N_PROBE = 1000
N_TRAIN = 20_000


def top10(D, Q):
    sims = Q @ D.T
    n = len(Q)
    sims[np.arange(n), np.arange(n)] = -np.inf          # probes are the first rows: drop self
    return np.argpartition(-sims, 10, axis=1)[:, :10]


def overlap(a, b):
    return float(np.mean([len(set(x) & set(y)) / 10.0 for x, y in zip(a, b)]))


def rown(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pq_reconstruct(Y, M, nbits=8, seed=2026):
    n, k = Y.shape
    sub = k // M
    Yrec = np.empty_like(Y)
    for j in range(M):
        sl = slice(j * sub, (j + 1) * sub)
        km = MiniBatchKMeans(n_clusters=2 ** nbits, random_state=seed + j, batch_size=4096,
                             n_init=1, max_iter=80, reassignment_ratio=0.0)
        km.fit(Y[:N_TRAIN, sl])
        Yrec[:, sl] = km.cluster_centers_[km.predict(Y[:, sl])]
    return Yrec.astype(np.float32)


def run(enc):
    print(f"=== exp14b PQ-only vs SHARD: {enc} pool={N_POOL} ===", flush=True)
    X, _, d = S.load(enc, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    full_nn = top10(Xrot, Xrot[:N_PROBE])
    res = {"encoder": enc, "d": int(d), "n_pool": N_POOL, "n_probe": N_PROBE,
           "shard_prefix": {}, "pq_only": []}

    dp = d // 4
    U = np.ascontiguousarray(Xrot[:, :dp])
    res["shard_prefix"] = {"d_pub": dp, "nn_overlap10": overlap(full_nn, top10(U, U[:N_PROBE])),
                           "recon_cosine": 1.0, "rerank": "exact (cell-keyed residual)"}
    print(f"  SHARD prefix d/4={dp}: NN-overlap@10={res['shard_prefix']['nn_overlap10']:.3f} "
          f"(rerank exact, cos=1.000)", flush=True)

    Y = (Xrot @ S.random_orthogonal(d, 11)).astype(np.float32)
    Yn = rown(Y)
    for M in [d // 16, d // 8]:
        t0 = time.time()
        Yrec = pq_reconstruct(Y, M)
        Yrecn = rown(Yrec)
        cos = float(np.mean(np.sum(Yn * Yrecn, axis=1)))
        nn = overlap(full_nn, top10(Yrecn, Yrecn[:N_PROBE]))
        res["pq_only"].append({"pq_m": M, "nbits": 8, "bytes_per_vec": M,
                               "recon_cosine": cos, "nn_overlap10": nn})
        print(f"  PQ-only M={M} ({M} B): NN-overlap@10={nn:.3f} recon_cos={cos:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    json.dump(res, open(OUT / f"exp14b_pq_only_{enc}.json", "w"), indent=2)
    print("saved", OUT / f"exp14b_pq_only_{enc}.json", flush=True)


def main():
    for enc in ENCODERS:
        run(enc)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
