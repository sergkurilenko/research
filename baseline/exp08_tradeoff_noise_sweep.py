"""Diagnostic utility/leakage sweep for SVD truncation vs Gaussian noise.

The main paper already evaluates the complete PQ+CKKS wrapper on a
1M-document corpus. This script adds a local, reproducible diagnostic that
does not require FAISS, PyTorch, or GPU memory:

* build a PCA/SVD basis from a random embedding sample;
* evaluate self-retrieval on a 100k-document gallery that contains all
  500 query targets;
* measure how much raw nearest-neighbour structure survives the protection;
* compare SVD truncation to an independent Gaussian-noise baseline calibrated
  to the same mean relative distortion.

Outputs:
  results/exp8_outputs/exp8_tradeoff_summary.csv
  results/exp8_outputs/exp8_tradeoff_summary.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks" / "_corpus_cache"
OUT = ROOT / "notebooks" / "exp8_outputs"


@dataclass(frozen=True)
class EncoderCfg:
    key: str
    dim: int
    docs: str
    queries: str


ENCODERS = {
    "e5small": EncoderCfg(
        key="e5small",
        dim=384,
        docs="E_docs_e5small_1000000.npy",
        queries="E_queries_e5small_self_q500.npy",
    ),
    "e5base": EncoderCfg(
        key="e5base",
        dim=768,
        docs="E_docs_e5base_1000000.npy",
        queries="E_queries_e5base_self_q500.npy",
    ),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="e5small,e5base")
    ap.add_argument("--gallery-size", type=int, default=100_000)
    ap.add_argument("--svd-sample-size", type=int, default=80_000)
    ap.add_argument("--neighbor-probes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-seed", type=int, default=2026)
    return ap.parse_args()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def topk_global(
    docs: np.ndarray,
    queries: np.ndarray,
    global_ids: np.ndarray,
    k: int = 10,
    doc_block: int = 20_000,
) -> list[list[int]]:
    """Chunked exact inner-product top-k returning global document IDs."""

    nq = queries.shape[0]
    best_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    best_ids = np.full((nq, k), -1, dtype=np.int64)

    for start in range(0, docs.shape[0], doc_block):
        end = min(start + doc_block, docs.shape[0])
        scores = queries @ docs[start:end].T
        block_ids = global_ids[start:end]

        cand_scores = np.concatenate([best_scores, scores], axis=1)
        cand_ids = np.concatenate(
            [best_ids, np.broadcast_to(block_ids, (nq, end - start))],
            axis=1,
        )
        part = np.argpartition(-cand_scores, kth=k - 1, axis=1)[:, :k]
        row = np.arange(nq)[:, None]
        best_scores = cand_scores[row, part]
        best_ids = cand_ids[row, part]
        order = np.argsort(-best_scores, axis=1)
        best_scores = best_scores[row, order]
        best_ids = best_ids[row, order]

    return [[int(v) for v in row] for row in best_ids]


def acc_at(preds: list[list[int]], qids: np.ndarray, k: int) -> float:
    return float(np.mean([int(qid) in pred[:k] for pred, qid in zip(preds, qids)]))


def strip_self(row: Iterable[int], self_id: int, k: int = 10) -> list[int]:
    out: list[int] = []
    for v in row:
        if int(v) == int(self_id):
            continue
        out.append(int(v))
        if len(out) == k:
            break
    return out


def neighbor_overlap_at_10(
    raw_docs: np.ndarray,
    protected_docs: np.ndarray,
    global_ids: np.ndarray,
    probe_positions: np.ndarray,
) -> float:
    raw_q = raw_docs[probe_positions]
    prot_q = protected_docs[probe_positions]
    probe_ids = global_ids[probe_positions]
    raw_pred = topk_global(raw_docs, raw_q, global_ids, k=11)
    prot_pred = topk_global(protected_docs, prot_q, global_ids, k=11)
    overlaps = []
    for rid, rrow, prow in zip(probe_ids, raw_pred, prot_pred):
        rset = set(strip_self(rrow, int(rid), 10))
        pset = set(strip_self(prow, int(rid), 10))
        overlaps.append(len(rset & pset) / 10.0)
    return float(np.mean(overlaps))


def pca_basis(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = sample.mean(axis=0, keepdims=True).astype(np.float32)
    centered = (sample - mu).astype(np.float32)
    cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    return mu, eigvecs[:, order].astype(np.float32)


def sigma_rec(sample: np.ndarray, mu: np.ndarray, basis: np.ndarray, k: int) -> float:
    vk = basis[:, :k]
    proj = (sample - mu) @ vk
    recon = proj @ vk.T + mu
    return float(
        np.mean(
            np.linalg.norm(sample - recon, axis=1)
            / (np.linalg.norm(sample, axis=1) + 1e-12)
        )
    )


def calibrate_noise_std(sample: np.ndarray, target_sigma: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(sample.shape, dtype=np.float32)

    def distortion(std: float) -> float:
        noisy = l2_normalize(sample + std * z)
        return float(
            np.mean(
                np.linalg.norm(noisy - sample, axis=1)
                / (np.linalg.norm(sample, axis=1) + 1e-12)
            )
        )

    lo, hi = 0.0, 2.0
    while distortion(hi) < target_sigma:
        hi *= 2.0
    for _ in range(32):
        mid = (lo + hi) / 2.0
        if distortion(mid) < target_sigma:
            lo = mid
        else:
            hi = mid
    return float(hi)


def choose_gallery(n_docs: int, qids: np.ndarray, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    qset = set(int(x) for x in qids)
    need = max(0, size - len(qids))
    decoys = []
    while len(decoys) < need:
        cand = rng.choice(n_docs, size=min(need * 2 + 1000, n_docs), replace=False)
        decoys.extend([int(x) for x in cand if int(x) not in qset])
        decoys = decoys[:need]
    gallery = np.array(list(qids.astype(np.int64)) + decoys, dtype=np.int64)
    rng.shuffle(gallery)
    return gallery


def run_encoder(cfg: EncoderCfg, args: argparse.Namespace) -> list[dict[str, object]]:
    docs_path = CACHE / cfg.docs
    queries_path = CACHE / cfg.queries
    if not docs_path.exists() or not queries_path.exists():
        raise FileNotFoundError(f"Missing cache for {cfg.key}: {docs_path} / {queries_path}")

    print(f"\n=== {cfg.key} ===")
    docs_mm = np.load(docs_path, mmap_mode="r")
    queries = np.load(queries_path).astype(np.float32)
    n_docs = docs_mm.shape[0]
    rng = np.random.default_rng(args.seed)
    qids = rng.choice(n_docs, size=queries.shape[0], replace=False).astype(np.int64)

    gallery_ids = choose_gallery(n_docs, qids, args.gallery_size, args.seed + 17)
    gallery = np.asarray(docs_mm[gallery_ids], dtype=np.float32)
    queries = queries.astype(np.float32)
    print(f"gallery={gallery.shape}, queries={queries.shape}")

    sample_ids = rng.choice(n_docs, size=min(args.svd_sample_size, n_docs), replace=False)
    sample = np.asarray(docs_mm[sample_ids], dtype=np.float32)
    mu, basis = pca_basis(sample)
    print(f"basis={basis.shape}")

    probe_rng = np.random.default_rng(args.seed + 99)
    probe_positions = probe_rng.choice(
        gallery.shape[0],
        size=min(args.neighbor_probes, gallery.shape[0]),
        replace=False,
    )

    raw_pred = topk_global(gallery, queries, gallery_ids, k=10)
    raw_acc1 = acc_at(raw_pred, qids, 1)
    raw_acc10 = acc_at(raw_pred, qids, 10)
    print(f"raw: Acc@1={raw_acc1:.3f}, Acc@10={raw_acc10:.3f}")

    rows: list[dict[str, object]] = []
    ratios = [0.125, 0.25, 0.5, 0.875]
    for ratio in ratios:
        k = max(4, int(round(cfg.dim * ratio)))
        k = min(k, cfg.dim)
        vk = basis[:, :k]
        g_svd = ((gallery - mu) @ vk).astype(np.float32)
        q_svd = ((queries - mu) @ vk).astype(np.float32)
        svd_pred = topk_global(g_svd, q_svd, gallery_ids, k=10)
        svd_acc1 = acc_at(svd_pred, qids, 1)
        svd_acc10 = acc_at(svd_pred, qids, 10)
        svd_sigma = sigma_rec(sample[: min(10_000, len(sample))], mu, basis, k)
        svd_overlap = neighbor_overlap_at_10(gallery, g_svd, gallery_ids, probe_positions)

        noise_std = calibrate_noise_std(
            sample[: min(10_000, len(sample))],
            svd_sigma,
            seed=args.noise_seed + k,
        )
        noise_rng_docs = np.random.default_rng(args.noise_seed + 10_000 + k)
        noise_rng_q = np.random.default_rng(args.noise_seed + 20_000 + k)
        g_noise = l2_normalize(
            gallery
            + noise_std * noise_rng_docs.standard_normal(gallery.shape, dtype=np.float32)
        ).astype(np.float32)
        q_noise = l2_normalize(
            queries
            + noise_std * noise_rng_q.standard_normal(queries.shape, dtype=np.float32)
        ).astype(np.float32)
        noise_pred = topk_global(g_noise, q_noise, gallery_ids, k=10)
        noise_acc1 = acc_at(noise_pred, qids, 1)
        noise_acc10 = acc_at(noise_pred, qids, 10)
        noise_overlap = neighbor_overlap_at_10(gallery, g_noise, gallery_ids, probe_positions)

        row = {
            "encoder": cfg.key,
            "dim": cfg.dim,
            "k": k,
            "k_over_d": ratio,
            "gallery_size": int(gallery.shape[0]),
            "raw_acc1": raw_acc1,
            "raw_acc10": raw_acc10,
            "sigma_rec": svd_sigma,
            "svd_acc1": svd_acc1,
            "svd_acc10": svd_acc10,
            "svd_nn_overlap_at_10": svd_overlap,
            "noise_std": noise_std,
            "noise_acc1": noise_acc1,
            "noise_acc10": noise_acc10,
            "noise_nn_overlap_at_10": noise_overlap,
        }
        rows.append(row)
        print(
            f"k/d={ratio:.3f} k={k}: sigma={svd_sigma:.3f}; "
            f"SVD Acc@1={svd_acc1:.3f}, NNoverlap={svd_overlap:.3f}; "
            f"noise Acc@1={noise_acc1:.3f}, NNoverlap={noise_overlap:.3f}"
        )

    return rows


def main() -> None:
    args = parse_args()
    OUT.mkdir(exist_ok=True)
    selected = [x.strip() for x in args.encoders.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    for key in selected:
        if key not in ENCODERS:
            raise KeyError(f"Unknown encoder {key}; available: {sorted(ENCODERS)}")
        rows.extend(run_encoder(ENCODERS[key], args))

    df = pd.DataFrame(rows)
    csv_path = OUT / "exp8_tradeoff_summary.csv"
    json_path = OUT / "exp8_tradeoff_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "parameters": vars(args),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
