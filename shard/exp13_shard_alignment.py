"""Experiment 13: alignment anchor-complexity, SHARD vs a single global key.

Threat: known-plaintext attacker holds m pairs (native residual r_i, stored
shard z_i) and wants to re-identify held-out target documents by recovering
the secret orthogonal key(s) via orthogonal Procrustes, then matching the
de-keyed vector against a native gallery.

GLOBAL (draft-style foil): one global key over the top-(d/2) PCA coords; one
Procrustes with >= d/2 anchors recovers everything.

SHARD: a short public prefix u (unkeyed) + a private residual r split into C
cells, each with its own key H_c. The attacker must recover the key of the
TARGET's cell, which needs ~d_priv anchors *inside that cell*; with anchors
spread over C cells this needs ~C x more known pairs. We report the
re-identification rate vs m and the anchor complexities m_50, m_90.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os, time
from pathlib import Path
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp13_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E13_ENC", "e5-small")
N_POOL = int(os.environ.get("E13_POOL", 200_000))
D_PUB = int(os.environ.get("E13_DPUB", 48))
N_GALLERY = 10_000
N_TARGET = 500
CELLS = [int(x) for x in os.environ.get("E13_CELLS", "1,64,256").split(",")]
M_GRID = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
SEEDS = [11, 23, 31]


def procrustes(Rk, Zk):
    """Given Z = R @ Omega (Omega orthogonal), recover Omega."""
    M = Rk.T @ Zk
    A, _, Bt = np.linalg.svd(M)
    return (A @ Bt).astype(np.float32)


def reident_rate(query_vecs, gallery, target_gallery_idx):
    """R@1: nearest gallery row (by inner product) == the true target row."""
    sims = query_vecs @ gallery.T
    nn = sims.argmax(1)
    return float((nn == target_gallery_idx).mean())


def run_global(Xrot, gt_pool, gallery_ids, target_ids, d, seeds):
    """One global key over top-(d/2) coords (no public channel)."""
    k = d // 2
    P = np.ascontiguousarray(Xrot[:, :k])                  # native top-k coords
    gal = P[gallery_ids]
    tgt_gallery_pos = np.searchsorted(gallery_ids, target_ids)  # targets in gallery
    curve = {m: [] for m in M_GRID}
    anchor_pool = np.setdiff1d(np.arange(len(P)), gallery_ids, assume_unique=False)
    for seed in seeds:
        G = S.random_orthogonal(k, seed)
        Z = P @ G.T                                        # stored (global key)
        rng = np.random.default_rng(seed)
        for m in M_GRID:
            a = rng.choice(anchor_pool, size=min(m, len(anchor_pool)), replace=False)
            if len(a) >= k:
                Om = procrustes(P[a], Z[a])                # recover G^T
                rhat = Z[target_ids] @ Om.T                # de-key targets
            else:
                rhat = Z[target_ids]                        # cannot align
            curve[m].append(reident_rate(rhat, gal, tgt_gallery_pos))
    return {m: float(np.mean(v)) for m, v in curve.items()}


def run_shard(U, Rr, labels, gallery_ids, target_ids, C, seeds):
    """Re-identification in the PRIVATE RESIDUAL space only (the public prefix
    is excluded, since u_t trivially self-matches and would mask the residual
    protection we are measuring)."""
    d_priv = Rr.shape[1]
    gal = Rr[gallery_ids]                                   # native residual gallery
    tgt_gallery_pos = np.searchsorted(gallery_ids, target_ids)
    anchor_pool = np.setdiff1d(np.arange(len(Rr)), gallery_ids, assume_unique=False)
    curve = {m: [] for m in M_GRID}
    for seed in seeds:
        Z = S.apply_keys(Rr, labels, master_seed=1000 + seed)  # stored shards
        rng = np.random.default_rng(seed)
        tgt_cell = labels[target_ids]
        for m in M_GRID:
            a = rng.choice(anchor_pool, size=min(m, len(anchor_pool)), replace=False)
            a_cell = labels[a]
            rhat = Z[target_ids].copy()
            # recover keys per cell that has enough anchors
            for c in np.unique(tgt_cell):
                ac = a[a_cell == c]
                if len(ac) >= d_priv:
                    Om = procrustes(Rr[ac], Z[ac])
                    tt = target_ids[tgt_cell == c]
                    rhat[tgt_cell == c] = Z[tt] @ Om.T         # de-keyed residual
                # else: residual stays keyed (garbage for matching)
            curve[m].append(reident_rate(rhat, gal, tgt_gallery_pos))
    return {m: float(np.mean(v)) for m, v in curve.items()}


def anchor_complexity(curve, level):
    xs = sorted(curve)
    for m in xs:
        if curve[m] >= level:
            return m
    return None


def main():
    print(f"=== exp13 alignment: enc={ENC} pool={N_POOL} d_pub={D_PUB} ===", flush=True)
    X, _, d = S.load(ENC, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    U = np.ascontiguousarray(Xrot[:, :D_PUB])
    Rr = np.ascontiguousarray(Xrot[:, D_PUB:])
    print(f"  d={d} d_pub={D_PUB} d_priv={Rr.shape[1]}", flush=True)
    grng = np.random.default_rng(7)
    gallery_ids = np.sort(grng.choice(len(Xrot), size=N_GALLERY, replace=False))
    target_ids = np.sort(grng.choice(gallery_ids, size=N_TARGET, replace=False))

    res = {"encoder": ENC, "d": int(d), "d_pub": D_PUB, "d_priv": int(Rr.shape[1]),
           "n_gallery": N_GALLERY, "n_target": N_TARGET, "m_grid": M_GRID,
           "schemes": {}}

    t0 = time.time()
    g = run_global(Xrot, None, gallery_ids, target_ids, d, SEEDS)
    res["schemes"]["global_key"] = {"curve": g,
        "m50": anchor_complexity(g, 0.5), "m90": anchor_complexity(g, 0.9)}
    print(f"  global: m50={res['schemes']['global_key']['m50']} "
          f"m90={res['schemes']['global_key']['m90']} ({time.time()-t0:.0f}s)", flush=True)

    for C in CELLS:
        t0 = time.time()
        labels, _ = S.kmeans_cells(U, C, seed=0)
        sh = run_shard(U, Rr, labels, gallery_ids, target_ids, C, SEEDS)
        key = f"shard_C{C}"
        res["schemes"][key] = {"curve": sh, "cells": C,
            "m50": anchor_complexity(sh, 0.5), "m90": anchor_complexity(sh, 0.9)}
        sizes = np.bincount(labels)
        print(f"  shard C={C}: m50={res['schemes'][key]['m50']} "
              f"m90={res['schemes'][key]['m90']} "
              f"cellsize[min/med/max]={sizes.min()}/{int(np.median(sizes))}/{sizes.max()} "
              f"({time.time()-t0:.0f}s)", flush=True)

    json.dump(res, open(OUT / f"exp13_alignment_{ENC}.json", "w"), indent=2)
    print("saved", OUT / f"exp13_alignment_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
