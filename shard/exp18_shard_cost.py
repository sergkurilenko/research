"""Experiment 18: online cost of SHARD (active cells per query -> bandwidth).

SHARD sends one encrypted residual query per ACTIVE cell (a cell touched by
the stage-1 short-list). This experiment measures how many cells the
short-list spans, which sets the per-query upload bandwidth and client-side
encryption count relative to the single-query baseline. It exposes the core
trade-off: larger C raises alignment resistance but also the number of
active cells per query.

One fresh CKKS ciphertext at N_poly=8192 is ~0.21 MB; the residual fits in
one ciphertext per cell. Server compute stays ~K_cands ct-pt ops (one per
candidate); the extra cost is upload (active-cells ciphertexts) and client
encryption.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os
from pathlib import Path
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp18_outputs"); OUT.mkdir(exist_ok=True)
ENCS = os.environ.get("E18_ENCS", "e5-small,e5-base").split(",")
N_POOL = int(os.environ.get("E18_POOL", 200_000))
N_Q = 500
CELLS = [1, 64, 256]
KCANDS = [40, 100, 200]
CT_MB = 0.21                                   # one CKKS ciphertext at N=8192


def run(enc):
    print(f"\n=== {enc} ===", flush=True)
    X, Q, d = S.load(enc, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0); idx = rng.choice(len(X), min(200000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    Qrot = ((Q[:N_Q] - mu) @ V).astype(np.float32)
    d_pub = max(8, d // 4)
    U = np.ascontiguousarray(Xrot[:, :d_pub]); Uq = np.ascontiguousarray(Qrot[:, :d_pub])
    res = {"encoder": enc, "d": int(d), "d_pub": d_pub, "rows": []}
    for C in CELLS:
        labels = np.zeros(len(U), np.int32) if C == 1 else S.kmeans_cells(U, C, seed=0)[0]
        for kc in KCANDS:
            short = S.topk_search(U, Uq, kc)
            ac = np.array([len(np.unique(labels[short[i]])) for i in range(len(Uq))])
            row = {"C": C, "kc": kc, "active_mean": float(ac.mean()),
                   "active_p95": float(np.percentile(ac, 95)),
                   "active_max": int(ac.max()),
                   "upload_MB_mean": float(ac.mean() * CT_MB),
                   "upload_ratio_vs_baseline": float(ac.mean())}  # baseline=1 ciphertext
            res["rows"].append(row)
            print(f"  C={C:3d} Kc={kc:3d}: active cells mean={row['active_mean']:.1f} "
                  f"p95={row['active_p95']:.0f}  upload~{row['upload_MB_mean']:.1f} MB "
                  f"({row['upload_ratio_vs_baseline']:.1f}x baseline)", flush=True)
    json.dump(res, open(OUT / f"exp18_cost_{enc}.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    for e in ENCS:
        run(e.strip())
    print("\nsaved exp18_outputs/")
