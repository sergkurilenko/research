"""Experiment 13-CI: alignment anchor-complexity with bootstrap confidence
bands and m25/m50/m75 thresholds, plus a random-cell baseline.

Extends exp13_shard_alignment.py for the journal revision:
  * 5 rotation seeds (was 3);
  * per-target R@1 outcomes preserved (not collapsed to a mean), so we can
    bootstrap over (seed x target) units;
  * paired bootstrap CI bands on the recovery curve and on the anchor
    thresholds m25 / m50 / m75 / m90;
  * an extra baseline `shard_randcell_C*` whose cells are assigned at random
    (NOT from the public prefix), to test whether the alignment resistance
    comes from cell-keying per se rather than from k-means structure.

Pure numpy, deterministic, reads the cached embeddings. Mirrors the exact
attack of exp13 (residual-only re-identification via orthogonal Procrustes).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import RESULTS
import json, os, time
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp13_outputs"); OUT.mkdir(exist_ok=True)
ENCODERS = os.environ.get("E13CI_ENCS", "e5-small,e5-base").split(",")
N_POOL = int(os.environ.get("E13CI_POOL", 200_000))
N_GALLERY = 10_000
N_TARGET = 500
CELLS = [int(x) for x in os.environ.get("E13CI_CELLS", "64,256").split(",")]
M_GRID = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
SEEDS = [11, 23, 31, 47, 53]
NBOOT = int(os.environ.get("E13CI_NBOOT", 2000))
BOOT_SEED = 2026
LEVELS = [0.25, 0.50, 0.75, 0.90]
# d_pub per encoder (top PCA prefix width); matches the paper's operating points.
DPUB = {"e5-small": 96, "e5-base": 192, "e5-large": 256, "mpnet": 192, "bge-m3": 256}


def procrustes(Rk, Zk):
    A, _, Bt = np.linalg.svd(Rk.T @ Zk)
    return (A @ Bt).astype(np.float32)


def hits_global(P, gallery_ids, target_ids, tgt_pos, anchor_pool, seed):
    k = P.shape[1]
    G = S.random_orthogonal(k, seed)
    Z = P @ G.T
    gal = P[gallery_ids]
    rng = np.random.default_rng(seed)
    out = {}
    for m in M_GRID:
        a = rng.choice(anchor_pool, size=min(m, len(anchor_pool)), replace=False)
        rhat = (Z[target_ids] @ procrustes(P[a], Z[a]).T) if len(a) >= k else Z[target_ids]
        out[m] = ((rhat @ gal.T).argmax(1) == tgt_pos)
    return out


def hits_shard(Rr, labels, gallery_ids, target_ids, tgt_pos, anchor_pool, seed):
    d_priv = Rr.shape[1]
    Z = S.apply_keys(Rr, labels, master_seed=1000 + seed)
    gal = Rr[gallery_ids]
    rng = np.random.default_rng(seed)
    tgt_cell = labels[target_ids]
    out = {}
    for m in M_GRID:
        a = rng.choice(anchor_pool, size=min(m, len(anchor_pool)), replace=False)
        a_cell = labels[a]
        rhat = Z[target_ids].copy()
        for c in np.unique(tgt_cell):
            ac = a[a_cell == c]
            if len(ac) >= d_priv:
                tt = target_ids[tgt_cell == c]
                rhat[tgt_cell == c] = Z[tt] @ procrustes(Rr[ac], Z[ac]).T
        out[m] = ((rhat @ gal.T).argmax(1) == tgt_pos)
    return out


def summarise(hits_by_seed):
    """hits_by_seed: dict m -> list (over seeds) of bool array (N_TARGET).
    Returns curve {m:{rate,lo,hi}} and thresholds {mLL:{m,lo,hi,frac_reached}}."""
    ms = sorted(hits_by_seed)
    HM = np.stack([np.array(hits_by_seed[m]).ravel().astype(np.float64) for m in ms], 1)  # (U, M)
    U = HM.shape[0]
    rate = HM.mean(0)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty((NBOOT, len(ms)))
    for b in range(NBOOT):
        boot[b] = HM[rng.integers(0, U, U)].mean(0)
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    curve = {int(ms[j]): {"rate": float(rate[j]), "lo": float(lo[j]), "hi": float(hi[j])}
             for j in range(len(ms))}

    def first_at(vec, lv):
        idx = np.argmax(vec >= lv) if (vec >= lv).any() else -1
        return ms[idx] if idx >= 0 else None
    thr = {}
    for lv in LEVELS:
        obs = first_at(rate, lv)
        bt = [first_at(boot[b], lv) for b in range(NBOOT)]
        reached = [x for x in bt if x is not None]
        thr[f"m{int(lv*100)}"] = {
            "m": obs,
            "lo": float(np.percentile(reached, 2.5)) if reached else None,
            "hi": float(np.percentile(reached, 97.5)) if reached else None,
            "frac_reached": float(len(reached) / NBOOT)}
    return curve, thr


def run_encoder(enc):
    d_pub = DPUB[enc]
    print(f"=== exp13-CI {enc} pool={N_POOL} d_pub={d_pub} seeds={SEEDS} nboot={NBOOT} ===", flush=True)
    X, _, d = S.load(enc, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    U = np.ascontiguousarray(Xrot[:, :d_pub])
    Rr = np.ascontiguousarray(Xrot[:, d_pub:])
    P = np.ascontiguousarray(Xrot[:, :d // 2])           # global-key foil space
    print(f"  d={d} d_pub={d_pub} d_priv={Rr.shape[1]}", flush=True)
    grng = np.random.default_rng(7)
    gallery_ids = np.sort(grng.choice(len(Xrot), size=N_GALLERY, replace=False))
    target_ids = np.sort(grng.choice(gallery_ids, size=N_TARGET, replace=False))
    tgt_pos = np.searchsorted(gallery_ids, target_ids)
    anchor_pool = np.setdiff1d(np.arange(len(Xrot)), gallery_ids)

    res = {"encoder": enc, "d": int(d), "d_pub": d_pub, "d_priv": int(Rr.shape[1]),
           "n_gallery": N_GALLERY, "n_target": N_TARGET, "seeds": SEEDS,
           "m_grid": M_GRID, "nboot": NBOOT, "boot_seed": BOOT_SEED, "schemes": {}}

    def collect(fn):
        h = {m: [] for m in M_GRID}
        for seed in SEEDS:
            hs = fn(seed)
            for m in M_GRID:
                h[m].append(hs[m])
        return h

    t0 = time.time()
    g = collect(lambda s: hits_global(P, gallery_ids, target_ids, tgt_pos, anchor_pool, s))
    c, thr = summarise(g)
    res["schemes"]["global_key"] = {"curve": c, "thresholds": thr}
    print(f"  global: m50={thr['m50']['m']} ({time.time()-t0:.0f}s)", flush=True)

    for C in CELLS:
        # k-means cells (SHARD)
        t0 = time.time()
        km_labels, _ = S.kmeans_cells(U, C, seed=0)
        h = collect(lambda s, L=km_labels: hits_shard(Rr, L, gallery_ids, target_ids, tgt_pos, anchor_pool, s))
        c, thr = summarise(h)
        res["schemes"][f"shard_C{C}"] = {"curve": c, "thresholds": thr, "cells": C, "cell_kind": "kmeans"}
        print(f"  shard C={C} (kmeans): m50={thr['m50']['m']} m90={thr['m90']['m']} ({time.time()-t0:.0f}s)", flush=True)
        # random cells (baseline: celling without public-prefix structure)
        t0 = time.time()
        rc_labels = np.random.default_rng(0).integers(0, C, len(U)).astype(np.int32)
        h = collect(lambda s, L=rc_labels: hits_shard(Rr, L, gallery_ids, target_ids, tgt_pos, anchor_pool, s))
        c, thr = summarise(h)
        res["schemes"][f"shard_randcell_C{C}"] = {"curve": c, "thresholds": thr, "cells": C, "cell_kind": "random"}
        print(f"  shard C={C} (random): m50={thr['m50']['m']} m90={thr['m90']['m']} ({time.time()-t0:.0f}s)", flush=True)

    path = OUT / f"exp13ci_alignment_{enc}.json"
    json.dump(res, open(path, "w"), indent=2)
    print("saved", path, flush=True)


def main():
    t = time.time()
    for enc in ENCODERS:
        run_encoder(enc)
    print(f"ALL DONE in {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
