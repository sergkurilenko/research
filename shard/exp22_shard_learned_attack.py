"""Experiment 22 (review #8): stronger/learned attackers on SHARD's residual.

The alignment result of exp13 used orthogonal Procrustes. A reviewer will
ask whether the d_priv-anchor barrier is an artefact of assuming an
orthogonal map. We therefore attack the cell-keyed residual with the
*cores* of the modern attacks:
  - Procrustes (orthogonal)            -- the exp13 attacker;
  - Ridge least-squares (linear)       -- the learned-linear core of ALGEN;
  - MLP (nonlinear)                    -- a small learned non-linear map;
  - unsupervised covariance matching   -- the no-pairs core of vec2vec-style
                                          cross-space translation.
Metric: cosine of the recovered residual to the true native residual on
held-out documents in one cell, vs the number of in-cell anchors. The
supervised maps cannot beat the d_priv barrier (the problem is genuinely
d_priv-dimensional); the unsupervised map fails outright (the residual
covariance is non-isotropic, so eigenvector matching recovers the key only
up to an unresolvable per-axis sign, i.e. 2^{d_priv} ambiguity).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import shard_lib as S

OUT = (RESULTS / "exp22_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E22_ENC", "e5-small")
N_POOL = 200_000
D_PUB = 96
C = 64
M_GRID = [64, 128, 192, 256, 288, 320, 384, 512]
N_TGT = 200


def procrustes(R, Z):
    A, _, Bt = np.linalg.svd(R.T @ Z); return (A @ Bt).astype(np.float32)


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(A * B, axis=1)))


def main():
    print(f"=== exp22 learned attack: enc={ENC} ===", flush=True)
    X, _, d = S.load(ENC, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0); idx = rng.choice(len(X), min(200000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    Rr = np.ascontiguousarray(Xrot[:, D_PUB:]); d_priv = Rr.shape[1]
    U = np.ascontiguousarray(Xrot[:, :D_PUB])
    labels, _ = S.kmeans_cells(U, C, seed=0)
    # pick the largest cell
    sizes = np.bincount(labels); cell = int(np.argmax(sizes))
    members = np.where(labels == cell)[0]
    print(f"  d_priv={d_priv}, target cell size={len(members)}", flush=True)
    H = S.cell_key(cell, d_priv, master_seed=777)
    Zc = Rr[members] @ H.T                                  # keyed shards in this cell
    Rc = Rr[members]
    rng2 = np.random.default_rng(0)
    perm = rng2.permutation(len(members))
    tgt = perm[:N_TGT]; pool = perm[N_TGT:]                 # held-out targets vs anchor pool
    # reference native-residual distribution for the unsupervised attacker
    ref = Rr[rng2.choice(len(Rr), 50_000, replace=False)]
    cov_r = (ref.T @ ref) / len(ref)
    Wr = np.linalg.eigh(cov_r.astype(np.float64))[1]

    res = {"encoder": ENC, "d_priv": int(d_priv), "cell_size": int(len(members)),
           "m_grid": M_GRID, "supervised": {"procrustes": {}, "ridge": {}, "mlp": {}},
           "unsupervised_covmatch_cos": None}

    for m in M_GRID:
        if m > len(pool):
            break
        anc = pool[:m]
        Za, Ra = Zc[anc], Rc[anc]
        Zt, Rt = Zc[tgt], Rc[tgt]
        # Procrustes
        Om = procrustes(Ra, Za); res["supervised"]["procrustes"][m] = cos_rows(Zt @ Om.T, Rt)
        # Ridge (learned linear, ALGEN core)
        rg = Ridge(alpha=1.0).fit(Za, Ra); res["supervised"]["ridge"][m] = cos_rows(rg.predict(Zt), Rt)
        # MLP (nonlinear) -- only when enough anchors to be meaningful and not too slow
        if m <= 384:
            ml = MLPRegressor(hidden_layer_sizes=(256,), max_iter=300, alpha=1e-3,
                              random_state=0).fit(Za, Ra)
            res["supervised"]["mlp"][m] = cos_rows(ml.predict(Zt), Rt)
        print(f"  m={m:4d}: procrustes={res['supervised']['procrustes'][m]:+.3f} "
              f"ridge={res['supervised']['ridge'][m]:+.3f} "
              f"mlp={res['supervised']['mlp'].get(m, float('nan')):+.3f}", flush=True)

    # unsupervised covariance/eigenvector matching (vec2vec core, no pairs)
    cov_z = (Zc.T @ Zc) / len(Zc)
    Wz = np.linalg.eigh(cov_z.astype(np.float64))[1]
    Hhat = (Wz @ Wr.T).astype(np.float32)                  # align eigenvectors (sign-ambiguous)
    res["unsupervised_covmatch_cos"] = cos_rows(Zc[tgt] @ Hhat, Rc[tgt])
    print(f"  unsupervised cov-match (no pairs): cos={res['unsupervised_covmatch_cos']:+.3f}", flush=True)

    json.dump(res, open(OUT / f"exp22_learned_{ENC}.json", "w"), indent=2)
    print("saved", OUT / f"exp22_learned_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
