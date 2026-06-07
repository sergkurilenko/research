"""Self-contained correctness tests for the SHARD construction.

These exercise the core invariants on small SYNTHETIC arrays, so they need
none of the cached embeddings and run in a second. They double as CI smoke
tests. Run directly:

    python shard/test_shard.py

or under pytest:

    pytest shard/test_shard.py
"""
import numpy as np
import shard_lib as S

RNG = np.random.default_rng(0)


def _toy(n=2000, d=64):
    X = RNG.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X


def _cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(A * B, axis=1)))


def _procrustes(Rk, Zk):
    A, _, Bt = np.linalg.svd(Rk.T @ Zk)
    return (A @ Bt).astype(np.float32)


def test_cell_key_is_orthogonal():
    H = S.cell_key(0, 40, master_seed=2)
    assert np.allclose(H @ H.T, np.eye(40), atol=1e-4)


def test_key_cancellation_preserves_inner_product():
    """<H_c r_q, H_c r_i> == <r_q, r_i> exactly (the CKKS rerank identity)."""
    d_priv = 32
    H = S.cell_key(3, d_priv, master_seed=1)
    a = RNG.standard_normal((5, d_priv)).astype(np.float32)
    b = RNG.standard_normal((7, d_priv)).astype(np.float32)
    lhs = (a @ H.T) @ (b @ H.T).T
    rhs = a @ b.T
    assert np.allclose(lhs, rhs, atol=1e-4), "orthogonal key did not cancel"


def test_full_dim_score_equals_centered_raw():
    """alpha<u_q,u_i> + beta<r_q,r_i> == <x_q-mu, x_i-mu> (no truncation)."""
    X = _toy()
    mu = X.mean(0, keepdims=True)
    V, _ = S.pca_basis((X[:500] - mu).astype(np.float32))
    rot = ((X - mu) @ V).astype(np.float32)
    d_pub = 16
    u, r = rot[:, :d_pub], rot[:, d_pub:]
    full = (u @ u[:1].T + r @ r[:1].T).ravel()
    centered = ((X - mu) @ (X[:1] - mu).T).ravel()
    assert np.allclose(full, centered, atol=1e-3), "split score != centered raw"


def test_procrustes_needs_d_priv_anchors():
    """Exact recovery at d_priv anchors; partial below it (the alignment barrier)."""
    d_priv = 24
    R = RNG.standard_normal((300, d_priv)).astype(np.float32)
    H = S.random_orthogonal(d_priv, 7)
    Z = R @ H.T
    held = slice(d_priv, None)
    full = _procrustes(R[:d_priv], Z[:d_priv])
    half = _procrustes(R[: d_priv // 2], Z[: d_priv // 2])
    cos_full = _cos_rows(Z[held] @ full.T, R[held])
    cos_half = _cos_rows(Z[held] @ half.T, R[held])
    assert cos_full > 0.99, f"full-anchor recovery weak: {cos_full:.3f}"
    assert cos_half < 0.9, f"half-anchor recovery should be partial: {cos_half:.3f}"


def test_microkey_decorrelates():
    """Per-document keys: same key preserves the vector, independent keys decorrelate."""
    d_priv = 48
    cos_same, cos_diff = [], []
    for t in range(100):
        r = RNG.standard_normal((d_priv,)).astype(np.float32)
        Hi = S.random_orthogonal(d_priv, 1000 + t)
        Hj = S.random_orthogonal(d_priv, 5000 + t)
        zi, zi2, zj = r @ Hi.T, r @ Hi.T, r @ Hj.T
        n = (np.linalg.norm(r) ** 2) + 1e-9
        cos_same.append(float(zi @ zi2) / n)
        cos_diff.append(float(zi @ zj) / n)
    assert np.allclose(cos_same, 1.0, atol=1e-3), "same key must preserve the vector"
    assert abs(np.mean(cos_diff)) < 0.05, f"independent keys not decorrelated: {np.mean(cos_diff):.3f}"


def test_topk_search_matches_bruteforce():
    D = _toy(n=400, d=16)
    Q = D[:5].copy()
    top = S.topk_search(D, Q, 10)
    brute = np.argsort(-(Q @ D.T), axis=1)[:, :10]
    for i in range(len(Q)):
        assert set(top[i].tolist()) == set(brute[i].tolist()), "topk != brute force"


def test_cells_are_deterministic():
    U = _toy(n=3000, d=16)
    a, _ = S.kmeans_cells(U, 8, seed=0)
    b, _ = S.kmeans_cells(U, 8, seed=0)
    assert np.array_equal(a, b), "cell assignment not deterministic"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} SHARD correctness tests passed.")


if __name__ == "__main__":
    main()
