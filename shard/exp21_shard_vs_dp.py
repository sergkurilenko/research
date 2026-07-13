"""Legacy Experiment 21: SHARD vs uncalibrated Gaussian perturbation.

This script is retained for provenance but is NOT evidence of differential
privacy and is no longer used for the manuscript's security comparison.  It
has no clipping/sensitivity/accounting, and its original alignment branch
contains the full-rank gate corrected by exp24_partial_alignment.py.

All defenses protect the SAME private residual r (the public prefix u is a
common, coarse channel). We place each on the utility-vs-de-anonymisation
plane:
  - utility: self-retrieval Acc@1 of the two-stage pipeline (prefix
    short-list -> residual rerank). Keyed schemes rerank exactly; DP-noise
    reranks against the stored noised residual, so utility drops with sigma.
  - de-anonymisation: a known-plaintext attacker with a DIFFUSE budget of m
    anchors recovers the native residual and matches it to a native gallery
    (R@1). Keyed schemes need ~d_priv in-cell anchors per cell; DP has NO
    transform to invert, so the stored vector already matches the native one
    (R@1 high) until sigma is large enough to destroy utility.

The point: SHARD reaches high-utility + low-de-anon at a fixed budget;
DP-noise can only lower de-anon by lowering utility, and provides essentially
no resistance to known-plaintext alignment (there is no key to recover).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os
from pathlib import Path
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp21_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E21_ENC", "e5-small")
N_POOL = int(os.environ.get("E21_POOL", 200_000))
D_PUB = 96
KCANDS = 100
N_Q = 500
N_GALLERY = 10_000
M_BUDGET = [400, 6400]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
SEEDS = [11, 23]


def procrustes(R, Z):
    A, _, Bt = np.linalg.svd(R.T @ Z); return (A @ Bt).astype(np.float32)


def two_stage_acc1(U, Uq_route, Uq_score, prot_R, q_R, gt):
    """Acc@1: prefix short-list then rerank by <u_q,u_i>+<q_r, prot_r_i>."""
    short = S.topk_search(U, Uq_route, KCANDS)
    hit = 0
    for i in range(len(Uq_route)):
        cand = short[i]
        sc = (Uq_score[i] @ U[cand].T) + (q_R[i] @ prot_R[cand].T)
        if cand[np.argmax(sc)] == gt[i]:
            hit += 1
    return hit / len(Uq_route)


def deanon_r1(est, galR, tgt_pos):
    return float((est @ galR.T).argmax(1).__eq__(tgt_pos).mean())


def main():
    print(f"=== exp21 SHARD vs DP: enc={ENC} ===", flush=True)
    X, Q, d = S.load(ENC)                                   # full corpus (queries match 1M gt)
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0); idx = rng.choice(len(X), min(200000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = S.document_pca_coordinates(X, mu, V)
    Qrot = S.query_pca_coordinates(Q[:N_Q], V)
    Qroute = ((Q[:N_Q] - mu) @ V).astype(np.float32); del X
    U = np.ascontiguousarray(Xrot[:, :D_PUB])
    Uq = np.ascontiguousarray(Qrot[:, :D_PUB])
    Uq_route = np.ascontiguousarray(Qroute[:, :D_PUB])
    Rr = np.ascontiguousarray(Xrot[:, D_PUB:]); qR = np.ascontiguousarray(Qrot[:, D_PUB:])
    d_priv = Rr.shape[1]
    gt = S.qrels(len(Xrot), N_Q)                            # matches cached self-retrieval queries
    grng = np.random.default_rng(7)
    gallery = np.sort(grng.choice(len(Rr), N_GALLERY, replace=False))
    targets = np.sort(grng.choice(gallery, 500, replace=False))
    tgt_pos = np.searchsorted(gallery, targets)
    galR = Rr[gallery]; anchor_pool = np.setdiff1d(np.arange(len(Rr)), gallery)
    res = {"encoder": ENC, "d_priv": int(d_priv), "m_budget": M_BUDGET, "defenses": [],
           "warning": "legacy diagnostic: not DP; full-rank-gated alignment is superseded by exp24"}

    def add(name, util, deanon):
        res["defenses"].append({"name": name, "acc1": util, "deanon_r1": deanon})
        print(f"  {name:22s} util Acc@1={util:.3f}  de-anon R@1={deanon}", flush=True)

    # DP-Gaussian noise on the residual (distortion-aware)
    for sig in SIGMAS:
        u_acc, da = [], {m: [] for m in M_BUDGET}
        for sd in SEEDS:
            r = np.random.default_rng(100 + sd)
            noise = r.standard_normal(Rr.shape).astype(np.float32)
            scale = sig * Rr.std()
            Y = Rr + scale * noise                       # stored noised residual
            u_acc.append(two_stage_acc1(U, Uq_route, Uq, Y, qR, gt))
            est = Y[targets]                              # no key to invert
            for m in M_BUDGET:
                da[m].append(deanon_r1(est, galR, tgt_pos))
        add(f"DP-noise sigma={sig}", float(np.mean(u_acc)),
            {m: round(float(np.mean(da[m])), 3) for m in M_BUDGET})

    # keyed utility is identical for all C (client de-keys exactly): compute once.
    keyed_util = two_stage_acc1(U, Uq_route, Uq, Rr, qR, gt)

    # keyed schemes: global key (C=1) and SHARD cells
    for C in [1, 64, 256]:
        labels = np.zeros(len(Rr), np.int32) if C == 1 else S.kmeans_cells(U, C, seed=0)[0]
        da = {m: [] for m in M_BUDGET}
        for sd in SEEDS:
            Z = S.apply_keys(Rr, labels, master_seed=300 + sd)
            r = np.random.default_rng(sd); tc = labels[targets]
            for m in M_BUDGET:
                a = r.choice(anchor_pool, min(m, len(anchor_pool)), replace=False)
                ac = labels[a]; rhat = Z[targets].copy()
                for c in np.unique(tc):
                    sub = a[ac == c]
                    if len(sub) >= d_priv:
                        Om = procrustes(Rr[sub], Z[sub]); sel = tc == c
                        rhat[sel] = Z[targets[sel]] @ Om.T
                da[m].append(deanon_r1(rhat, galR, tgt_pos))
        name = "global key (C=1)" if C == 1 else f"SHARD C={C}"
        add(name, keyed_util,
            {m: round(float(np.mean(da[m])), 3) for m in M_BUDGET})

    json.dump(res, open(OUT / f"exp21_vs_dp_{ENC}.json", "w"), indent=2)
    print("saved", OUT / f"exp21_vs_dp_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
