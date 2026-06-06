"""SHARD: Split-prefix Hierarchical Anchored Residual Defence.

A retrieval-preserving protective transform for dense embeddings designed
against alignment- and index-leakage attacks. Replaces the single global
SVD+rotation baseline with:

  1. an orthogonal PCA rotation  W  of the centred embedding  x-mu;
  2. a split  Wx = [u | r]  into a short PUBLIC prefix u (top d_pub PCA
     directions, used for stage-1 ANN) and a PRIVATE residual r (the
     remaining d_priv = d - d_pub directions);
  3. coarse CELLS defined by clustering the public prefix u into C cells;
  4. a per-cell secret orthogonal key H_c (cell-local keying), so the
     stored private shard is  z_i = H_{c(i)} r_i.

Because each H_c is orthogonal, <H_c r_q, z_i> = <r_q, r_i> exactly, so the
two-stage score  S = alpha <u_q,u_i> + beta <r_q,r_i>  reranks in the FULL
space with no truncation loss; the public index sees only the short prefix.

Everything here is numpy-only (no GPU/faiss/tenseal) and deterministic
given the seeds, so the privacy/utility geometry experiments are exactly
reproducible from the cached embeddings.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import numpy as np

CACHE = (str(DATA) + "/")

ENC = {
    "e5-small": ("E_docs_e5small_1000000.npy", "E_queries_e5small_self_q500.npy", 384),
    "e5-base":  ("E_docs_e5base_1000000.npy",  "E_queries_e5base_self_q500.npy",  768),
    "mpnet":    ("E_docs_mpnet_1000000.npy",   "E_queries_mpnet_self_q500.npy",   768),
    "e5-large": ("E_docs_e5large_1000000.npy", "E_queries_e5large_self_q500.npy", 1024),
    "bge-m3":   ("E_docs_bgem3_1000000.npy",   "E_queries_bgem3_self_q500.npy",   1024),
}

SEED_BASE = 42


def load(enc, n=None):
    d_docs, d_q, dim = ENC[enc]
    X = np.load(CACHE + d_docs, mmap_mode="r")
    Q = np.load(CACHE + d_q)
    if n is not None:
        X = np.asarray(X[:n], dtype=np.float32)
    return X, np.asarray(Q, dtype=np.float32), dim


def qrels(n_docs, n_q=500, seed=SEED_BASE):
    return np.random.default_rng(seed).choice(n_docs, size=n_q, replace=False)


def pca_basis(Xc_sample):
    """Full PCA basis V (d x d, columns sorted by descending variance) from a
    centred sample, via eigendecomposition of the dxd covariance (cheap)."""
    C = (Xc_sample.T @ Xc_sample) / max(1, len(Xc_sample))
    evals, evecs = np.linalg.eigh(C.astype(np.float64))
    order = np.argsort(evals)[::-1]
    return evecs[:, order].astype(np.float32), evals[order].astype(np.float64)


def fit_transform(X, mu, V, sample_for_cells=None, d_pub=None, C=64,
                  cell_seed=0, n_sample_cov=200_000):
    """Rotate X into PCA coords and split into prefix/residual; assign cells.

    Returns dict with rotated coords (Xrot), prefix U, residual R, cell ids,
    cell centroids (in prefix space) and the d_pub/d_priv split.
    """
    d = X.shape[1]
    if d_pub is None:
        d_pub = d // 2
    Xrot = ((X - mu) @ V).astype(np.float32)      # centred PCA coordinates
    U = Xrot[:, :d_pub]
    R = Xrot[:, d_pub:]
    return {"Xrot": Xrot, "U": U, "R": R, "d_pub": d_pub, "d_priv": d - d_pub}


def kmeans_cells(U, C, seed=0, iters=15, n_train=100_000):
    """Lightweight k-means on the public prefix to define C coarse cells.
    Returns (labels for all rows, centroids)."""
    rng = np.random.default_rng(seed)
    n = len(U)
    train_idx = rng.choice(n, size=min(n_train, n), replace=False)
    T = U[train_idx]
    cent = T[rng.choice(len(T), size=C, replace=False)].copy()
    for _ in range(iters):
        lab = _assign_chunked(T, cent)               # chunked: memory-safe for any C
        for c in range(C):
            m = lab == c
            if m.any():
                cent[c] = T[m].mean(0)
    labels = _assign_chunked(U, cent)
    return labels, cent


def _assign_chunked(U, cent, chunk=200_000):
    out = np.empty(len(U), dtype=np.int32)
    cn = (cent ** 2).sum(1)
    for s in range(0, len(U), chunk):
        u = U[s:s + chunk]
        d2 = cn[None, :] - 2.0 * (u @ cent.T)        # + |u|^2 (const per row)
        out[s:s + chunk] = d2.argmin(1)
    return out


def cell_key(cell_id, d_priv, master_seed=12345):
    """Deterministic per-cell secret orthogonal key via QR of a seeded
    Gaussian (Haar-uniform after sign correction). Stands in for a product
    of Householder reflections; only orthogonality matters for the geometry."""
    rng = np.random.default_rng(master_seed * 1_000_003 + int(cell_id))
    A = rng.standard_normal((d_priv, d_priv))
    Qm, Rm = np.linalg.qr(A)
    Qm *= np.sign(np.diag(Rm))                       # Haar correction
    return Qm.astype(np.float32)


def apply_keys(R, labels, master_seed=12345):
    """Return z_i = H_{c(i)} r_i for all rows (the stored private shards)."""
    d_priv = R.shape[1]
    Z = np.empty_like(R)
    for c in np.unique(labels):
        m = labels == c
        H = cell_key(c, d_priv, master_seed)
        Z[m] = R[m] @ H.T
    return Z


def random_orthogonal(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Qm, Rm = np.linalg.qr(A)
    Qm *= np.sign(np.diag(Rm))
    return Qm.astype(np.float32)


def topk_search(D, Q, k=10, chunk=100_000):
    """Chunked exact inner-product top-k. Returns (nq,k) doc ids (desc)."""
    n, nq = len(D), len(Q)
    bs = np.full((nq, k), -np.inf, np.float32)
    bi = np.full((nq, k), -1, np.int64)
    for s in range(0, n, chunk):
        sc = Q @ D[s:s + chunk].T
        kk = min(k, sc.shape[1])
        part = np.argpartition(-sc, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(sc, part, axis=1)
        ids = part + s
        AS = np.concatenate([bs, ps], 1); AI = np.concatenate([bi, ids], 1)
        sel = np.argsort(-AS, axis=1)[:, :k]
        bs = np.take_along_axis(AS, sel, 1); bi = np.take_along_axis(AI, sel, 1)
    return bi


def metrics_from_top(top, gt):
    h1 = (top[:, 0] == gt).astype(np.float64)
    h10 = np.any(top == gt[:, None], axis=1).astype(np.float64)
    rr = np.zeros(len(gt))
    for i in range(len(gt)):
        p = np.where(top[i] == gt[i])[0]
        rr[i] = 1.0 / (p[0] + 1) if p.size else 0.0
    return h1, h10, rr
