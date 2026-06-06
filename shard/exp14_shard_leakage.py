"""Experiment 14: public-index leakage of SHARD vs the baseline geometry.

The SHARD public ANN index is built on the SHORT prefix u only. We quantify
how much true (full-space) neighbour structure each public channel reveals:

  - prefix u (top d_pub PCA), swept over d_pub      -> SHARD public index
  - SVD k=d/2 exact projection                       -> baseline public geometry
We report NN-overlap@10 with the full-space top-10 neighbours (lower = less
leakage).

We also account, honestly, for the residual store z: within a cell the key
cancels (<z_i,z_j>=<r_i,r_j>), so an honest-but-curious server can compute
EXACT within-cell similarities. We therefore report the fraction of each
document's true top-10 neighbours that lie in its own cell (recoverable
within-cell), and contrast it with the baseline, where a single global key
cancels for ALL pairs (the whole neighbour graph is recoverable).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os
from pathlib import Path
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp14_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E14_ENC", "e5-small")
N_POOL = int(os.environ.get("E14_POOL", 100_000))
N_PROBE = 1000
DPUBS = None  # set per-d below
CELLS = [64, 256]


def top10(D, Q, exclude_self=True):
    sims = Q @ D.T
    if exclude_self:
        # probes are the first N_PROBE rows of D; blank the self column
        for i in range(len(Q)):
            sims[i, i] = -np.inf
    return np.argpartition(-sims, 10, axis=1)[:, :10]


def overlap(a, b):
    return float(np.mean([len(set(x) & set(y)) / 10.0 for x, y in zip(a, b)]))


def main():
    print(f"=== exp14 leakage: enc={ENC} pool={N_POOL} ===", flush=True)
    X, _, d = S.load(ENC, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    probes = Xrot[:N_PROBE]
    full_nn = top10(Xrot, probes)                          # full-space neighbours
    dpubs = [max(8, d // f) for f in (16, 8, 4, 2)]
    res = {"encoder": ENC, "d": int(d), "n_pool": N_POOL, "n_probe": N_PROBE,
           "prefix_leakage": {}, "baseline_svd_k2": None, "within_cell": {}}

    # public prefix leakage, swept over d_pub
    for dp in dpubs:
        U = np.ascontiguousarray(Xrot[:, :dp])
        nn = top10(U, U[:N_PROBE])
        res["prefix_leakage"][dp] = overlap(full_nn, nn)
        print(f"  prefix d_pub={dp:4d}: NN-overlap@10={res['prefix_leakage'][dp]:.3f}", flush=True)

    # baseline public geometry: SVD k=d/2 exact projection
    k = d // 2
    Pk = np.ascontiguousarray(Xrot[:, :k])
    nnk = top10(Pk, Pk[:N_PROBE])
    res["baseline_svd_k2"] = overlap(full_nn, nnk)
    print(f"  baseline SVD k=d/2 exact: NN-overlap@10={res['baseline_svd_k2']:.3f}", flush=True)

    # within-cell recoverable fraction (honest accounting of the z store)
    for dp in [d // 4]:                                    # operating-point prefix
        U = np.ascontiguousarray(Xrot[:, :dp])
        for C in CELLS:
            labels, _ = S.kmeans_cells(U, C, seed=0)
            same = []
            for i in range(N_PROBE):
                nbrs = full_nn[i]
                same.append(np.mean(labels[nbrs] == labels[i]))
            res["within_cell"][f"dpub{dp}_C{C}"] = {
                "frac_neighbours_same_cell": float(np.mean(same)), "cells": C, "d_pub": dp}
            print(f"  within-cell d_pub={dp} C={C}: "
                  f"frac true-top10 nbrs in own cell = {np.mean(same):.3f} "
                  f"(baseline global key recovers 1.000)", flush=True)

    json.dump(res, open(OUT / f"exp14_leakage_{ENC}.json", "w"), indent=2)
    print("saved", OUT / f"exp14_leakage_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
