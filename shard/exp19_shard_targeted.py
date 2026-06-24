"""Experiment 19: targeted attacker vs the C-times anchor-complexity claim.

The C-times factor (exp13) assumes the known-plaintext leak is DIFFUSE
(anchors spread uniformly over cells). Cells are public (defined by the
public prefix), so a TARGETED attacker can instead concentrate anchors in
the victim's own cell and recover only that cell's key. This experiment
sweeps the number of in-cell anchors a and shows that targeted
re-identification succeeds at a ~ d_priv REGARDLESS of C -- i.e. the
per-victim worst case is ~d_priv (about the baseline's d/2), and the C x
factor is an aggregate/diffuse-leak property. The micro-key limit defeats
even this: a cell of size 1 cannot supply d_priv in-cell anchors.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os
from pathlib import Path
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp19_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E19_ENC", "e5-small")
N_POOL = int(os.environ.get("E19_POOL", 200_000))
D_PUB = int(os.environ.get("E19_DPUB", 96))
N_GALLERY = 10_000
N_TARGET = 300
CELLS = [64, 256]
A_GRID = [32, 64, 128, 192, 256, 320, 448, 576]
SEEDS = [11, 23]


def procrustes(R, Z):
    A, _, Bt = np.linalg.svd(R.T @ Z); return (A @ Bt).astype(np.float32)


def reident(q, gal, true_pos):
    return float((q @ gal.T).argmax(1).__eq__(true_pos).mean())


def main():
    print(f"=== exp19 targeted: enc={ENC} d_pub={D_PUB} ===", flush=True)
    X, _, d = S.load(ENC, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0); idx = rng.choice(len(X), min(200000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    Rr = np.ascontiguousarray(Xrot[:, D_PUB:]); d_priv = Rr.shape[1]
    U = np.ascontiguousarray(Xrot[:, :D_PUB])
    grng = np.random.default_rng(7)
    gallery = np.sort(grng.choice(len(Rr), N_GALLERY, replace=False))
    targets = np.sort(grng.choice(gallery, N_TARGET, replace=False))
    tgt_pos = np.searchsorted(gallery, targets)
    galR = Rr[gallery]
    res = {"encoder": ENC, "d": int(d), "d_pub": D_PUB, "d_priv": int(d_priv),
           "a_grid": A_GRID, "schemes": {}}
    for C in CELLS:
        labels, _ = S.kmeans_cells(U, C, seed=0)
        sizes = np.bincount(labels, minlength=C)
        curve = {a: [] for a in A_GRID}
        for sd in SEEDS:
            Z = S.apply_keys(Rr, labels, master_seed=2000 + sd)
            rr = np.random.default_rng(sd)
            tcell = labels[targets]
            # candidate anchors per cell = cell members excluding targets
            for a in A_GRID:
                rhat = Z[targets].copy()
                for c in np.unique(tcell):
                    members = np.where(labels == c)[0]
                    members = np.setdiff1d(members, targets, assume_unique=False)
                    if len(members) >= a >= d_priv:                 # enough in-cell anchors AND a>=d_priv
                        anc = rr.choice(members, a, replace=False)
                        Om = procrustes(Rr[anc], Z[anc])
                        sel = tcell == c
                        rhat[sel] = Z[targets[sel]] @ Om.T
                    # else: cannot recover this cell's key with a in-cell anchors
                curve[a].append(reident(rhat, galR, tgt_pos))
        m = {a: float(np.mean(v)) for a, v in curve.items()}
        m50 = next((a for a in A_GRID if m[a] >= 0.5), None)
        res["schemes"][f"targeted_C{C}"] = {"curve": m, "m50_targeted": m50,
            "cellsize_med": int(np.median(sizes))}
        print(f"  C={C}: targeted m50={m50} (d_priv={d_priv}, cellsize_med={int(np.median(sizes))}) "
              f"curve={ {a: round(m[a],2) for a in [192,256,320]} }", flush=True)
    json.dump(res, open(OUT / f"exp19_targeted_{ENC}.json", "w"), indent=2)
    print("saved", OUT / f"exp19_targeted_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
