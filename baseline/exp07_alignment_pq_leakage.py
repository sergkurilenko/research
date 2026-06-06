"""
Experiment 7: known-plaintext alignment and PQ-artifact leakage.

The script is intentionally light-weight: it uses only NumPy and
scikit-learn, and operates on the cached e5-small embeddings already
present in notebooks/_corpus_cache. It produces JSON/CSV artifacts used
by the revised paper.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "_corpus_cache"
INDEX_CACHE = CACHE / "index_cache"
OUT = ROOT / "exp7_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--rotation-seeds", type=str, default="11,23,31,47,53")
    parser.add_argument("--n-gallery", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--n-pq", type=int, default=20000)
    parser.add_argument("--n-pq-train", type=int, default=12000)
    parser.add_argument("--n-pq-query", type=int, default=300)
    parser.add_argument("--pq-configs", type=str, default="24x8,48x8,48x6")
    return parser.parse_args()


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def make_random_orthogonal(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return W minimizing ||source W - target||_F over orthogonal W."""
    c = source.T @ target
    u, _, vt = np.linalg.svd(c, full_matrices=False)
    return (u @ vt).astype(np.float32)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def target_recall_at_k(
    queries: np.ndarray,
    gallery: np.ndarray,
    target_pos: np.ndarray,
    k: int,
    chunk: int = 128,
) -> float:
    hits = 0
    for start in range(0, len(queries), chunk):
        q = queries[start : start + chunk]
        scores = q @ gallery.T
        top = topk_indices(scores, k)
        tgt = target_pos[start : start + len(q), None]
        hits += int(np.any(top == tgt, axis=1).sum())
    return hits / len(queries)


def neighbor_overlap_metrics(
    true_vecs: np.ndarray,
    approx_vecs: np.ndarray,
    n_queries: int,
    top_exact: int = 10,
    top_approx: int = 40,
) -> dict[str, float]:
    q_true = true_vecs[:n_queries]
    q_approx = approx_vecs[:n_queries]
    exact_scores = q_true @ true_vecs.T
    approx_scores = q_approx @ approx_vecs.T
    rows = np.arange(n_queries)
    exact_scores[rows, rows] = -np.inf
    approx_scores[rows, rows] = -np.inf
    exact10 = topk_indices(exact_scores, top_exact)
    approx10 = topk_indices(approx_scores, top_exact)
    approx40 = topk_indices(approx_scores, top_approx)

    overlap10 = []
    recall10_40 = []
    for i in range(n_queries):
        e = set(exact10[i].tolist())
        overlap10.append(len(e.intersection(approx10[i].tolist())) / top_exact)
        recall10_40.append(len(e.intersection(approx40[i].tolist())) / top_exact)
    return {
        "neighbor_overlap_at_10": float(np.mean(overlap10)),
        "exact10_recall_in_approx40": float(np.mean(recall10_40)),
    }


def load_projected_sample(k: int, total_needed: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    docs_path = CACHE / "E_docs_e5small_1000000.npy"
    svd_path = INDEX_CACHE / "svd_e5small_proj336.npz"
    if not docs_path.exists():
        raise FileNotFoundError(docs_path)
    if not svd_path.exists():
        raise FileNotFoundError(svd_path)

    docs = np.load(docs_path, mmap_mode="r")
    svd = np.load(svd_path)
    mu = svd["mu"].astype(np.float32)
    vk = svd["Vk"][:, :k].astype(np.float32)

    rng = np.random.default_rng(seed)
    ids = rng.choice(docs.shape[0], size=total_needed, replace=False)
    x = np.asarray(docs[ids], dtype=np.float32)
    z = ((x - mu) @ vk).astype(np.float32)
    return ids.astype(np.int64), z


def run_procrustes(z: np.ndarray, args: argparse.Namespace) -> list[dict[str, float]]:
    n_test = args.n_test
    n_gallery = args.n_gallery
    if len(z) < n_test + n_gallery + 2500:
        raise ValueError("Projected sample is too small for the configured experiment")

    anchors_all = z[:2500]
    test = z[2500 : 2500 + n_test]
    gallery_extra = z[2500 + n_test : 2500 + n_test + n_gallery - n_test]
    gallery = np.vstack([test, gallery_extra]).astype(np.float32)

    test_n = row_normalize(test)
    gallery_n = row_normalize(gallery)
    target_pos = np.arange(n_test)
    anchor_sizes = [0, 10, 50, 100, args.k, 250, 500, 1000, 2000]
    rotation_seeds = [int(x) for x in args.rotation_seeds.split(",") if x.strip()]
    rows: list[dict[str, float]] = []

    for seed in rotation_seeds:
        r = make_random_orthogonal(seed, args.k)
        y_anchors_all = anchors_all @ r
        y_test = test @ r

        for m in anchor_sizes:
            if m == 0:
                w = np.eye(args.k, dtype=np.float32)
            else:
                w = procrustes(y_anchors_all[:m], anchors_all[:m])
            z_hat = y_test @ w
            z_hat_n = row_normalize(z_hat)
            rel_err = float(np.linalg.norm(z_hat - test) / np.linalg.norm(test))
            cos = float(np.mean(np.sum(z_hat_n * test_n, axis=1)))
            rec1 = target_recall_at_k(z_hat_n, gallery_n, target_pos, 1)
            rec10 = target_recall_at_k(z_hat_n, gallery_n, target_pos, 10)
            rot_err = float(np.linalg.norm(w - r.T) / math.sqrt(args.k))
            rows.append(
                {
                    "rotation_seed": seed,
                    "known_pairs": m,
                    "relative_l2_error": rel_err,
                    "mean_cosine_to_projected": cos,
                    "target_recall_at_1": rec1,
                    "target_recall_at_10": rec10,
                    "rotation_fro_error_per_dim": rot_err,
                }
            )

        w = r.T.astype(np.float32)
        z_hat = y_test @ w
        z_hat_n = row_normalize(z_hat)
        rows.append(
            {
                "rotation_seed": seed,
                "known_pairs": -1,
                "relative_l2_error": float(np.linalg.norm(z_hat - test) / np.linalg.norm(test)),
                "mean_cosine_to_projected": float(np.mean(np.sum(z_hat_n * test_n, axis=1))),
                "target_recall_at_1": target_recall_at_k(z_hat_n, gallery_n, target_pos, 1),
                "target_recall_at_10": target_recall_at_k(z_hat_n, gallery_n, target_pos, 10),
                "rotation_fro_error_per_dim": 0.0,
            }
        )
    return rows


def parse_pq_configs(spec: str) -> list[tuple[int, int]]:
    configs = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        m_s, b_s = item.split("x", 1)
        configs.append((int(m_s), int(b_s)))
    return configs


def fit_encode_pq(
    y: np.ndarray,
    m: int,
    nbits: int,
    n_train: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    n, k = y.shape
    if k % m != 0:
        raise ValueError(f"k={k} is not divisible by M={m}")
    subdim = k // m
    n_clusters = 2**nbits
    if n_train < n_clusters:
        raise ValueError("n_train must be at least the number of PQ centroids")

    y_rec = np.empty_like(y)
    t0 = time.time()
    for j in range(m):
        sl = slice(j * subdim, (j + 1) * subdim)
        train = y[:n_train, sl]
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed + j,
            batch_size=4096,
            n_init=1,
            max_iter=80,
            reassignment_ratio=0.0,
        )
        km.fit(train)
        labels = km.predict(y[:, sl])
        y_rec[:, sl] = km.cluster_centers_[labels]
    seconds = time.time() - t0
    return y_rec.astype(np.float32), {"train_encode_seconds": seconds}


def run_pq_leakage(z: np.ndarray, args: argparse.Namespace) -> list[dict[str, float]]:
    n_pq = args.n_pq
    y = z[:n_pq] @ make_random_orthogonal(11, args.k)
    y = y.astype(np.float32)
    y_norm = row_normalize(y)
    rows = []
    for m, nbits in parse_pq_configs(args.pq_configs):
        y_rec, timing = fit_encode_pq(y, m, nbits, args.n_pq_train, args.seed)
        y_rec_norm = row_normalize(y_rec)
        cos = float(np.mean(np.sum(y_norm * y_rec_norm, axis=1)))
        rel = float(np.linalg.norm(y_rec - y) / np.linalg.norm(y))
        nn = neighbor_overlap_metrics(y_norm, y_rec_norm, args.n_pq_query)
        rows.append(
            {
                "pq_m": m,
                "nbits": nbits,
                "bytes_per_vector": float(m * nbits / 8.0),
                "reconstruction_relative_l2": rel,
                "reconstruction_mean_cosine": cos,
                **nn,
                **timing,
            }
        )
    return rows


def aggregate_procrustes(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    keys = [
        "relative_l2_error",
        "mean_cosine_to_projected",
        "target_recall_at_1",
        "target_recall_at_10",
        "rotation_fro_error_per_dim",
    ]
    groups: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        groups.setdefault(int(row["known_pairs"]), []).append(row)
    out = []
    for known_pairs, vals in sorted(groups.items()):
        rec = {"known_pairs": known_pairs, "n_rotation_seeds": len(vals)}
        for key in keys:
            arr = np.array([v[key] for v in vals], dtype=np.float64)
            rec[f"{key}_mean"] = float(arr.mean())
            rec[f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total_needed = max(
        2500 + args.n_test + args.n_gallery,
        args.n_pq,
    )
    ids, z = load_projected_sample(args.k, total_needed, args.seed)

    pro_rows = run_procrustes(z, args)
    pro_agg = aggregate_procrustes(pro_rows)
    pq_rows = run_pq_leakage(z, args)

    write_csv(OUT / "exp7_procrustes_by_seed.csv", pro_rows)
    write_csv(OUT / "exp7_procrustes_summary.csv", pro_agg)
    write_csv(OUT / "exp7_pq_leakage.csv", pq_rows)
    with (OUT / "exp7_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "sample_doc_ids_sha256_note": "IDs are deterministic from seed; raw IDs are in exp7_doc_ids.npy.",
                "procrustes_summary": pro_agg,
                "pq_leakage": pq_rows,
            },
            f,
            indent=2,
        )
    np.save(OUT / "exp7_doc_ids.npy", ids)


if __name__ == "__main__":
    main()
