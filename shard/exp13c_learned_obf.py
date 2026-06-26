"""Experiment 13c: a learned-obfuscation defense baseline vs SHARD.

Reviewer baseline: instead of SHARD's per-cell secret orthogonal key, use a
single global *learned* (nonlinear) transform f as the obfuscation, trained
to preserve retrieval (inner products). We then run the same known-plaintext
alignment attack: the attacker holds m pairs (native residual r, stored
z=f(r)) and trains an inverse network g to recover r, then re-identifies
held-out targets in a native residual gallery.

Hypothesis: a retrieval-preserving global transform is necessarily
near-isometric, hence invertible from ~d_priv anchors spread anywhere -- so
a learned global obfuscation gives NO C x resistance (its m50 is comparable
to the global key, far below SHARD's C*d_priv). This isolates the source of
SHARD's resistance as cell-local secret keying, not transform complexity.

Lightweight MLPs, CPU torch, cached embeddings. Deterministic per seed.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import RESULTS
import json, os, time
import numpy as np
import torch, torch.nn as nn
import shard_lib as S

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
OUT = (RESULTS / "exp13_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E13C_ENC", "e5-small")
D_PUB = int(os.environ.get("E13C_DPUB", 96))
N_POOL = int(os.environ.get("E13C_POOL", 40_000))
N_GAL = 5_000
N_TGT = 300
SEEDS = [11, 23, 31]
M_GRID = [int(x) for x in os.environ.get("E13C_MGRID", "72,144,288,576,1152,2304").split(",")]
F_STEPS = int(os.environ.get("E13C_FSTEPS", 400))
G_STEPS = int(os.environ.get("E13C_GSTEPS", 400))


def mlp(d_in, d_h, d_out):
    return nn.Sequential(nn.Linear(d_in, d_h), nn.Tanh(), nn.Linear(d_h, d_out))


def train_f(R, d, seed):
    """Learned obfuscation f: trained to preserve inner products (retrieval)."""
    torch.manual_seed(seed)
    f = mlp(d, d, d)
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    Rt = torch.from_numpy(R)
    n = len(Rt)
    g = torch.Generator().manual_seed(seed)
    for _ in range(F_STEPS):
        ia = torch.randint(0, n, (1024,), generator=g)
        ib = torch.randint(0, n, (1024,), generator=g)
        a, b = Rt[ia], Rt[ib]
        fa, fb = f(a), f(b)
        ip_true = (a * b).sum(1)
        ip_obf = (fa * fb).sum(1)
        loss = ((ip_obf - ip_true) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        # utility proxy: how well f preserves inner products on fresh pairs
        ia = torch.randint(0, n, (4000,), generator=g); ib = torch.randint(0, n, (4000,), generator=g)
        a, b = Rt[ia], Rt[ib]
        ip_true = (a * b).sum(1); ip_obf = (f(a) * f(b)).sum(1)
        ip_corr = float(np.corrcoef(ip_true.numpy(), ip_obf.numpy())[0, 1])
    return f, ip_corr


def attack(f, R, gallery_ids, target_ids, tgt_pos, anchor_pool, d, seed):
    Rt = torch.from_numpy(R)
    with torch.no_grad():
        Z = f(Rt)                                   # stored obfuscated residuals
    gal = R[gallery_ids]
    rng = np.random.default_rng(seed)
    curve = {}
    for m in M_GRID:
        a = rng.choice(anchor_pool, size=min(m, len(anchor_pool)), replace=False)
        za, ra = Z[a], Rt[a]
        torch.manual_seed(seed + m)
        g = mlp(d, 2 * d, d)                        # attacker's inverse network
        opt = torch.optim.Adam(g.parameters(), lr=1e-3)
        for _ in range(G_STEPS):
            opt.zero_grad()
            loss = ((g(za) - ra) ** 2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            rhat = g(Z[target_ids]).numpy()
        nn_idx = (rhat @ gal.T).argmax(1)
        curve[m] = float((nn_idx == tgt_pos).mean())
    return curve


def m50(curve):
    for m in sorted(curve):
        if curve[m] >= 0.5:
            return m
    return None


def main():
    print(f"=== exp13c learned-obf: {ENC} d_pub={D_PUB} pool={N_POOL} seeds={SEEDS} ===", flush=True)
    X, _, d = S.load(ENC, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    R = np.ascontiguousarray(Xrot[:, D_PUB:])          # native residual
    d_priv = R.shape[1]
    print(f"  d={d} d_pub={D_PUB} d_priv={d_priv}", flush=True)
    grng = np.random.default_rng(7)
    gallery_ids = np.sort(grng.choice(len(R), size=N_GAL, replace=False))
    target_ids = np.sort(grng.choice(gallery_ids, size=N_TGT, replace=False))
    tgt_pos = np.searchsorted(gallery_ids, target_ids)
    anchor_pool = np.setdiff1d(np.arange(len(R)), gallery_ids)

    curves, m50s, utils = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        f, ip_corr = train_f(R, d_priv, seed)
        cur = attack(f, R, gallery_ids, target_ids, tgt_pos, anchor_pool, d_priv, seed)
        curves.append(cur); m50s.append(m50(cur)); utils.append(ip_corr)
        print(f"  seed {seed}: util(ip-corr)={ip_corr:.3f} m50={m50(cur)} "
              f"curve={{ {', '.join(f'{m}:{cur[m]:.2f}' for m in M_GRID)} }} ({time.time()-t0:.0f}s)", flush=True)

    mean_curve = {m: float(np.mean([c[m] for c in curves])) for m in M_GRID}
    res = {"encoder": ENC, "d": int(d), "d_pub": D_PUB, "d_priv": int(d_priv),
           "n_gallery": N_GAL, "n_target": N_TGT, "seeds": SEEDS, "m_grid": M_GRID,
           "mean_curve": mean_curve, "m50_per_seed": m50s,
           "m50_mean": (None if any(x is None for x in m50s) else float(np.mean(m50s))),
           "utility_ip_corr_mean": float(np.mean(utils)),
           "note": "learned global nonlinear obfuscation; attacker trains an inverse MLP."}
    json.dump(res, open(OUT / f"exp13c_learned_obf_{ENC}.json", "w"), indent=2)
    print(f"  MEAN m50={m50(mean_curve)} util={np.mean(utils):.3f}", flush=True)
    print("saved", OUT / f"exp13c_learned_obf_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
