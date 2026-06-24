"""Experiment 15: reference-corpus lookup / de-anonymisation, SHARD vs global.

This directly counters the draft's strongest negative result (the global
SVD+R baseline allowed near-exact paragraph recovery once an overlapping
reference corpus and ~k known pairs were available). SHARD keys BOTH the
short prefix (one global key K, client-side stage-1) and the residual (cell
keys H_c). Exact de-anonymisation of a target requires recovering its cell's
residual key, i.e. ~C x more anchors; the recovered global prefix key K only
yields COARSE (prefix-level) matches.

We report, vs the number of known plaintext anchors m:
  - baseline-global : exact de-anon R@1 (single key over top-k coords)
  - SHARD-C         : exact de-anon R@1 (full native match) and the coarse
                      prefix-only R@10 floor.
Overlap case: the reference corpus contains the targets' native embeddings.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os
from pathlib import Path
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp15_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E15_ENC", "e5-small")
N_POOL = int(os.environ.get("E15_POOL", 100_000))
D_PUB = int(os.environ.get("E15_DPUB", 96))
N_TARGET = 500
CELLS = [64, 256]
M_GRID = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
SEEDS = [11, 23]


def procrustes(R, Z):
    A, _, Bt = np.linalg.svd(R.T @ Z); return (A @ Bt).astype(np.float32)


def r_at(query, gallery, true_pos, ks=(1, 10)):
    sims = query @ gallery.T
    ranks = (sims >= sims[np.arange(len(true_pos)), true_pos][:, None]).sum(1)
    return {k: float((ranks <= k).mean()) for k in ks}


def main():
    print(f"=== exp15 reference lookup: enc={ENC} pool={N_POOL} d_pub={D_PUB} ===", flush=True)
    X, _, d = S.load(ENC, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0); idx = rng.choice(len(X), min(200000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    k = d // 2
    U = np.ascontiguousarray(Xrot[:, :D_PUB]); Rr = np.ascontiguousarray(Xrot[:, D_PUB:])
    nat_full = Xrot                                   # native reference [u | r]
    grng = np.random.default_rng(7)
    targets = np.sort(grng.choice(len(Xrot), N_TARGET, replace=False))
    anchor_pool = np.setdiff1d(np.arange(len(Xrot)), targets)
    res = {"encoder": ENC, "d": int(d), "d_pub": D_PUB, "n_pool": N_POOL,
           "m_grid": M_GRID, "schemes": {}}

    # baseline-global: single key over top-k coords (no prefix split)
    Pk = np.ascontiguousarray(Xrot[:, :k])
    gcurve = {m: [] for m in M_GRID}
    for sd in SEEDS:
        G = S.random_orthogonal(k, sd); Z = Pk @ G.T
        r = np.random.default_rng(sd)
        for m in M_GRID:
            a = r.choice(anchor_pool, min(m, len(anchor_pool)), replace=False)
            q = Z[targets] @ procrustes(Pk[a], Z[a]).T if len(a) >= k else Z[targets]
            gcurve[m].append(r_at(q, Pk, targets)[1])
    res["schemes"]["global"] = {m: float(np.mean(v)) for m, v in gcurve.items()}
    print("  global R@1:", {m: round(res['schemes']['global'][m], 3) for m in [200, 800, 6400]}, flush=True)

    # SHARD: global prefix key K + per-cell residual keys
    for C in CELLS:
        labels, _ = S.kmeans_cells(U, C, seed=0)
        d_priv = Rr.shape[1]
        exact = {m: [] for m in M_GRID}; coarse = {m: [] for m in M_GRID}
        for sd in SEEDS:
            K = S.random_orthogonal(D_PUB, 500 + sd); Uk = U @ K.T
            Z = S.apply_keys(Rr, labels, master_seed=1000 + sd)
            r = np.random.default_rng(sd); tgt_cell = labels[targets]
            for m in M_GRID:
                a = r.choice(anchor_pool, min(m, len(anchor_pool)), replace=False)
                # recover prefix key K (global, needs d_pub anchors)
                if len(a) >= D_PUB:
                    OK = procrustes(U[a], Uk[a]); uhat = Uk[targets] @ OK.T
                else:
                    uhat = Uk[targets]
                rhat = Z[targets].copy()
                a_cell = labels[a]
                for c in np.unique(tgt_cell):
                    ac = a[a_cell == c]
                    if len(ac) >= d_priv:
                        sel = tgt_cell == c
                        rhat[sel] = Z[targets[sel]] @ procrustes(Rr[ac], Z[ac]).T
                full_q = np.concatenate([uhat, rhat], 1)
                exact[m].append(r_at(full_q, nat_full, targets)[1])     # exact R@1
                # coarse prefix-only match (R@10) against native prefix
                coarse_q = uhat
                coarse[m].append(r_at(coarse_q, U, targets)[10])
        res["schemes"][f"shard_C{C}_exactR1"] = {m: float(np.mean(v)) for m, v in exact.items()}
        res["schemes"][f"shard_C{C}_coarseR10"] = {m: float(np.mean(v)) for m, v in coarse.items()}
        print(f"  shard C={C} exact R@1:",
              {m: round(res['schemes'][f'shard_C{C}_exactR1'][m], 3) for m in [200, 6400, 51200]},
              "| coarse R@10:",
              {m: round(res['schemes'][f'shard_C{C}_coarseR10'][m], 3) for m in [200, 6400]}, flush=True)

    json.dump(res, open(OUT / f"exp15_reference_{ENC}.json", "w"), indent=2)
    print("saved", OUT / f"exp15_reference_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
