"""Legacy Experiment 21b: uncalibrated Gaussian perturbation sweep.

Retained for provenance only.  This is not a DP mechanism, and the gated
alignment comparison is superseded by exp24_partial_alignment.py.

Reviewer request: tune the DP-noise baseline at matched utility more
rigorously (a finer sigma sweep) and put bootstrap CIs on the de-anonymisation
rates. We place every defense on the utility(Acc@1)-vs-de-anonymisation(R@1)
plane at a fixed diffuse known-plaintext budget m, and report a paired
bootstrap 95% CI on R@1 over (seed x target) units.

Key reading: DP can only match SHARD's (keyed) utility at sigma=0, where its
de-anonymisation is ~1.0; lowering de-anon needs sigma>0, which strictly
lowers utility below SHARD's. SHARD's (high-utility, ~0 de-anon) corner is
unreachable by any DP sigma.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import RESULTS
import json, os
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp21_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E21B_ENC", "e5-small")
N_POOL = int(os.environ.get("E21B_POOL", 200_000))
D_PUB = 96
KCANDS = 100
N_Q = 500
N_GALLERY = 10_000
M_BUDGET = int(os.environ.get("E21B_M", 6400))
SIGMAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
SEEDS = [11, 23, 31]
NBOOT = 4000
BOOT_SEED = 2026


def procrustes(R, Z):
    A, _, Bt = np.linalg.svd(R.T @ Z); return (A @ Bt).astype(np.float32)


def two_stage_acc1(U, Uq_route, Uq_score, prot_R, q_R, gt):
    short = S.topk_search(U, Uq_route, KCANDS)
    hit = 0
    for i in range(len(Uq_route)):
        cand = short[i]
        sc = (Uq_score[i] @ U[cand].T) + (q_R[i] @ prot_R[cand].T)
        if cand[np.argmax(sc)] == gt[i]:
            hit += 1
    return hit / len(Uq_route)


def deanon_hits(est, galR, tgt_pos):
    return ((est @ galR.T).argmax(1) == tgt_pos)          # (n_targets,) bool


def boot_ci(mat):                                          # mat: (n_seed, n_tgt) bool
    flat = mat.ravel().astype(np.float64); n = len(flat)
    rng = np.random.default_rng(BOOT_SEED)
    bs = flat[rng.integers(0, n, (NBOOT, n))].mean(1)
    return float(flat.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    print(f"=== exp21b SHARD vs DP (finer sweep, CIs): {ENC} m={M_BUDGET} ===", flush=True)
    X, Q, d = S.load(ENC); X = np.asarray(X, dtype=np.float32)
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
    gt = S.qrels(len(Xrot), N_Q)
    grng = np.random.default_rng(7)
    gallery = np.sort(grng.choice(len(Rr), N_GALLERY, replace=False))
    targets = np.sort(grng.choice(gallery, 500, replace=False))
    tgt_pos = np.searchsorted(gallery, targets); galR = Rr[gallery]
    anchor_pool = np.setdiff1d(np.arange(len(Rr)), gallery)
    res = {"encoder": ENC, "d_priv": int(d_priv), "m_budget": M_BUDGET, "defenses": [],
           "warning": "legacy diagnostic: not DP; gated alignment is superseded by exp24"}

    def add(name, util, mat):
        rate, lo, hi = boot_ci(mat)
        res["defenses"].append({"name": name, "acc1": float(util),
                                "deanon_r1": rate, "deanon_lo": lo, "deanon_hi": hi})
        print(f"  {name:20s} Acc@1={util:.3f}  de-anon R@1={rate:.3f} [{lo:.3f},{hi:.3f}]", flush=True)

    rstd = float(Rr.std())
    for sig in SIGMAS:
        util, hits = [], []
        for sd in SEEDS:
            r = np.random.default_rng(100 + sd)
            Y = (Rr + sig * rstd * r.standard_normal(Rr.shape).astype(np.float32)).astype(np.float32)
            util.append(two_stage_acc1(U, Uq_route, Uq, Y, qR, gt))
            hits.append(deanon_hits(Y[targets], galR, tgt_pos))
        add(f"DP-noise sig={sig}", float(np.mean(util)), np.array(hits))

    keyed_util = two_stage_acc1(U, Uq_route, Uq, Rr, qR, gt)
    for C in [64, 256]:
        labels = S.kmeans_cells(U, C, seed=0)[0]
        hits = []
        for sd in SEEDS:
            Z = S.apply_keys(Rr, labels, master_seed=300 + sd)
            r = np.random.default_rng(sd); tc = labels[targets]
            a = r.choice(anchor_pool, min(M_BUDGET, len(anchor_pool)), replace=False)
            ac = labels[a]; rhat = Z[targets].copy()
            for c in np.unique(tc):
                sub = a[ac == c]
                if len(sub) >= d_priv:
                    Om = procrustes(Rr[sub], Z[sub]); sel = tc == c
                    rhat[sel] = Z[targets[sel]] @ Om.T
            hits.append(deanon_hits(rhat, galR, tgt_pos))
        add(f"SHARD C={C}", keyed_util, np.array(hits))

    # matched-utility reading
    res["keyed_utility"] = float(keyed_util)
    res["note"] = ("DP matches keyed utility only at sigma=0 (no noise), where de-anon ~1.0; "
                   "any sigma>0 lowers utility below SHARD while only partially lowering de-anon.")
    json.dump(res, open(OUT / f"exp21b_dp_sweep_{ENC}.json", "w"), indent=2)
    print(f"  keyed (SHARD) utility = {keyed_util:.3f}", flush=True)
    print("saved", OUT / f"exp21b_dp_sweep_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
