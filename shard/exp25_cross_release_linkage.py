"""Experiment 25: honest cross-release linkage of SHARD releases.

The game deliberately removes the easiest identifier channel: each release has
fresh, independently shuffled row handles and the evaluator is never given the
hidden document IDs.  The two releases contain the same documents under fresh
keys.  We evaluate two observer views:

  residual-only: (cell label, keyed residual z), but no public prefix;
  full-server:   (cell label, public prefix u, keyed residual z).

Cell labels are stable for the cell-key construction because the cells are
defined by the unchanged public prefix.  For microkeys, the key handles are
assumed fresh and therefore are not a stable identifier.

The linkage methods test invariants that a row-wise cosine AUC misses:

  * residual norm nearest-neighbour and sorted-rank assignment;
  * within-cell Gram-row and nearest-neighbour-graph signatures;
  * one-to-one Hungarian assignment of Gram signatures (where feasible);
  * known-anchor inner-product fingerprints;
  * public-prefix and combined full-view matching;
  * raw cross-key residual cosine as a negative/control measurement.

Float16, symmetric int8 and independent Gaussian-noise storage controls are
included.  The script writes config.json, metrics.csv, raw_metrics.json,
summary.json and run.log.  It does not modify any prior experiment or paper.

Default full run (e5-small, three release pairs, N=500/2k/10k):

  python shard/exp25_cross_release_linkage.py

Quick smoke test:

  python shard/exp25_cross_release_linkage.py --quick
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import shard_lib as S  # noqa: E402


LOG = logging.getLogger("exp25")


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(sh)


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def file_sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def as_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def condition_view(x: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Simulate stored/observed values independently in each release."""
    x = np.asarray(x, dtype=np.float32)
    if condition == "clean":
        return x.copy()
    if condition == "fp16":
        return x.astype(np.float16).astype(np.float32)
    if condition == "int8":
        # A release-wide symmetric scale is realistic metadata and avoids the
        # extra per-row scale itself becoming an unreported linkage channel.
        scale = float(np.max(np.abs(x))) / 127.0
        if scale == 0.0:
            return x.copy()
        return (np.clip(np.rint(x / scale), -127, 127) * scale).astype(np.float32)
    if condition.startswith("noise_"):
        frac = float(condition.split("_", 1)[1])
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
        noise = rng.standard_normal(x.shape, dtype=np.float32) * np.float32(frac * rms)
        return (x + noise).astype(np.float32)
    raise ValueError(f"unknown condition: {condition}")


def microkey_release(r: np.ndarray, seed: int, chunk: int = 4096) -> np.ndarray:
    """Exact marginal distribution of an independent Haar rotation per row.

    For fixed r_i and Haar H_i, H_i r_i is uniform on the sphere with radius
    ||r_i||.  Sampling that image directly is O(Nd), unlike N dense QR
    decompositions, and preserves precisely the norm channel under test.
    """
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(r, axis=1).astype(np.float32)
    z = np.empty_like(r, dtype=np.float32)
    for start in range(0, len(r), chunk):
        stop = min(start + chunk, len(r))
        g = rng.standard_normal((stop - start, r.shape[1]), dtype=np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + np.float32(1e-30)
        z[start:stop] = g * norms[start:stop, None]
    return z


def make_release(
    u: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(u)).astype(np.int64)
    return {
        "hidden_ids": ids,
        "u": np.ascontiguousarray(u[ids]),
        "z": np.ascontiguousarray(z[ids]),
        "labels": np.ascontiguousarray(labels[ids]),
    }


def truth_map(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    inv_b = np.empty(len(ids_b), dtype=np.int64)
    inv_b[ids_b] = np.arange(len(ids_b), dtype=np.int64)
    return inv_b[ids_a]


def query_rows(ids_a: np.ndarray, canonical_queries: np.ndarray) -> np.ndarray:
    inv_a = np.empty(len(ids_a), dtype=np.int64)
    inv_a[ids_a] = np.arange(len(ids_a), dtype=np.int64)
    return inv_a[canonical_queries]


def standardize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([a, b], axis=0).astype(np.float64, copy=False)
    mean = pooled.mean(axis=0, keepdims=True)
    sd = pooled.std(axis=0, keepdims=True)
    sd[sd < 1e-10] = 1.0
    return ((a - mean) / sd).astype(np.float32), ((b - mean) / sd).astype(np.float32)


def _distance_matrix(a: np.ndarray, b: np.ndarray, cosine: bool) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if cosine:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-30)
    an = np.sum(a * a, axis=1, dtype=np.float32)[:, None]
    bn = np.sum(b * b, axis=1, dtype=np.float32)[None, :]
    d = an + bn - np.float32(2.0) * (a @ b.T)
    np.maximum(d, 0.0, out=d)
    return d


def evaluate_feature_linkage(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    truth: np.ndarray,
    queries: np.ndarray,
    labels_a: np.ndarray | None,
    labels_b: np.ndarray | None,
    *,
    cosine: bool = False,
    auc_negatives: int = 10,
    rng_seed: int = 0,
    batch: int = 128,
) -> dict[str, float | int]:
    """Exact NN metrics, exact MRR and sampled-pair ROC AUC.

    If labels are supplied, candidates are restricted to the same disclosed
    cell.  Ranking is still exact within the observer's candidate set.
    """
    feat_a = np.asarray(feat_a, dtype=np.float32)
    feat_b = np.asarray(feat_b, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.int64)
    rng = np.random.default_rng(rng_seed)
    ranks: list[int] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    if labels_a is None:
        groups: list[tuple[np.ndarray, np.ndarray]] = [
            (queries, np.arange(len(feat_b), dtype=np.int64))
        ]
    else:
        groups = []
        for c in np.unique(labels_a[queries]):
            qa = queries[labels_a[queries] == c]
            cb = np.flatnonzero(labels_b == c)
            groups.append((qa, cb))

    for qa, cb in groups:
        if len(qa) == 0 or len(cb) == 0:
            continue
        # Translate release-B row id to local candidate-column index.
        local = {int(row): j for j, row in enumerate(cb)}
        for start in range(0, len(qa), batch):
            q = qa[start:start + batch]
            dmat = _distance_matrix(feat_a[q], feat_b[cb], cosine=cosine)
            true_cols = np.fromiter((local[int(truth[i])] for i in q), dtype=np.int64)
            true_d = dmat[np.arange(len(q)), true_cols]
            # Deterministic row order is a fresh random release handle, hence
            # it is an honest random tie breaker for quantized equal scores.
            order = np.argsort(dmat, axis=1, kind="stable")
            for row in range(len(q)):
                rank = int(np.flatnonzero(order[row] == true_cols[row])[0]) + 1
                ranks.append(rank)
                pos_scores.append(-float(true_d[row]))
                if len(cb) > 1:
                    choices = rng.integers(0, len(cb) - 1, size=auc_negatives)
                    choices += choices >= true_cols[row]
                    neg_scores.extend((-dmat[row, choices]).astype(float).tolist())

    r = np.asarray(ranks, dtype=np.int64)
    if len(r) == 0:
        raise RuntimeError("linkage method evaluated no queries")
    auc = float("nan")
    if neg_scores:
        y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        s = np.asarray(pos_scores + neg_scores, dtype=np.float64)
        auc = float(roc_auc_score(y, s))
    return {
        "n_queries": int(len(r)),
        "top1": float(np.mean(r <= 1)),
        "top5": float(np.mean(r <= 5)),
        "top10": float(np.mean(r <= 10)),
        "mrr": float(np.mean(1.0 / r)),
        "median_rank": float(np.median(r)),
        "mean_rank": float(np.mean(r)),
        "auc": auc,
    }


def rank_assignment_linkage(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    truth: np.ndarray,
    queries: np.ndarray,
) -> dict[str, float | int | None]:
    order_a = np.argsort(norm_a, kind="stable")
    order_b = np.argsort(norm_b, kind="stable")
    rank_a = np.empty(len(order_a), dtype=np.int64)
    rank_a[order_a] = np.arange(len(order_a))
    pred = order_b[rank_a]
    hit = pred[queries] == truth[queries]
    return {
        "n_queries": int(len(queries)),
        "top1": float(np.mean(hit)),
        "top5": None,
        "top10": None,
        "mrr": None,
        "median_rank": None,
        "mean_rank": None,
        "auc": None,
        "note": "one-to-one monotone assignment; only top-1 is defined",
    }


def analytic_label_control(
    labels_a: np.ndarray | None,
    labels_b: np.ndarray | None,
    queries: np.ndarray,
) -> dict[str, float | int]:
    if labels_a is None:
        sizes = np.full(len(queries), len(labels_b), dtype=np.int64)
    else:
        counts = {int(c): int(np.sum(labels_b == c)) for c in np.unique(labels_b)}
        sizes = np.asarray([counts[int(labels_a[q])] for q in queries], dtype=np.int64)
    top1 = np.mean(1.0 / sizes)
    top5 = np.mean(np.minimum(5, sizes) / sizes)
    top10 = np.mean(np.minimum(10, sizes) / sizes)
    # Expected reciprocal rank for a uniformly random ordering.
    mrr = np.mean([sum(1.0 / j for j in range(1, int(m) + 1)) / m for m in sizes])
    return {
        "n_queries": int(len(queries)),
        "top1": float(top1),
        "top5": float(top5),
        "top10": float(top10),
        "mrr": float(mrr),
        "median_rank": float(np.median((sizes + 1) / 2.0)),
        "mean_rank": float(np.mean((sizes + 1) / 2.0)),
        "auc": 0.5,
        "note": "analytic chance level given disclosed cell labels",
    }


def graph_features(
    z: np.ndarray,
    labels: np.ndarray,
    gram_width: int = 32,
    graph_width: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation-invariant row signatures of each disclosed cell graph."""
    n = len(z)
    gram = np.zeros((n, gram_width + 2), dtype=np.float32)
    graph = np.zeros((n, graph_width + 2), dtype=np.float32)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        g = z[idx] @ z[idx].T
        m = len(idx)
        norm2 = np.diag(g).copy()
        gram[idx, 0] = norm2
        gram[idx, 1] = float(m)
        graph[idx, 0] = norm2
        graph[idx, 1] = float(m)
        if m == 1:
            continue
        mask = ~np.eye(m, dtype=bool)
        off = g[mask].reshape(m, m - 1)
        sorted_asc = np.sort(off, axis=1)
        take = np.rint(np.linspace(0, m - 2, gram_width)).astype(np.int64)
        gram[idx, 2:] = sorted_asc[:, take]
        sorted_desc = sorted_asc[:, ::-1]
        k = min(graph_width, m - 1)
        graph[idx, 2:2 + k] = sorted_desc[:, :k]
    return gram, graph


def known_anchor_features(
    z_a: np.ndarray,
    z_b: np.ndarray,
    ids_a: np.ndarray,
    ids_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    anchor_ids: np.ndarray,
    stable_cells: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner products to disclosed cross-release anchor pairs.

    Under a shared cell key, each within-cell inner product is exactly
    invariant across rekeying.  Under per-document microkeys it is not.
    """
    inv_a = np.empty(len(ids_a), dtype=np.int64)
    inv_b = np.empty(len(ids_b), dtype=np.int64)
    inv_a[ids_a] = np.arange(len(ids_a))
    inv_b[ids_b] = np.arange(len(ids_b))
    aa = inv_a[anchor_ids]
    ab = inv_b[anchor_ids]
    fa = np.zeros((len(z_a), len(anchor_ids)), dtype=np.float32)
    fb = np.zeros((len(z_b), len(anchor_ids)), dtype=np.float32)
    if stable_cells:
        for c in np.unique(labels_a):
            rows_a = np.flatnonzero(labels_a == c)
            rows_b = np.flatnonzero(labels_b == c)
            cols = np.flatnonzero(labels_a[aa] == c)
            if len(cols):
                fa[np.ix_(rows_a, cols)] = z_a[rows_a] @ z_a[aa[cols]].T
                fb[np.ix_(rows_b, cols)] = z_b[rows_b] @ z_b[ab[cols]].T
    else:
        fa = z_a @ z_a[aa].T
        fb = z_b @ z_b[ab].T
    return fa, fb


def hungarian_linkage(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    truth: np.ndarray,
    queries: np.ndarray,
    labels_a: np.ndarray | None,
    labels_b: np.ndarray | None,
) -> dict[str, float | int | None]:
    pred = np.full(len(feat_a), -1, dtype=np.int64)
    if labels_a is None:
        groups = [(np.arange(len(feat_a)), np.arange(len(feat_b)))]
    else:
        groups = [
            (np.flatnonzero(labels_a == c), np.flatnonzero(labels_b == c))
            for c in np.unique(labels_a)
        ]
    for ia, ib in groups:
        d = _distance_matrix(feat_a[ia], feat_b[ib], cosine=False)
        ra, rb = linear_sum_assignment(d)
        pred[ia[ra]] = ib[rb]
    hit = pred[queries] == truth[queries]
    return {
        "n_queries": int(len(queries)),
        "top1": float(np.mean(hit)),
        "top5": None,
        "top10": None,
        "mrr": None,
        "median_rank": None,
        "mean_rank": None,
        "auc": None,
        "note": "global one-to-one assignment; only top-1 is defined",
    }


def add_record(
    records: list[dict],
    meta: dict,
    view: str,
    method: str,
    metrics: dict,
) -> None:
    rec = dict(meta)
    rec.update({"view": view, "linkage_method": method})
    rec.update(metrics)
    records.append(rec)
    LOG.info(
        "N=%s seed=%s %-9s %-11s %-32s top1=%s top10=%s mrr=%s auc=%s",
        meta["n"], meta["seed"], meta["condition"], view, method,
        f"{metrics.get('top1'):.4f}" if metrics.get("top1") is not None else "NA",
        f"{metrics.get('top10'):.4f}" if metrics.get("top10") is not None else "NA",
        f"{metrics.get('mrr'):.4f}" if metrics.get("mrr") is not None else "NA",
        f"{metrics.get('auc'):.4f}" if metrics.get("auc") is not None else "NA",
    )


def aggregate(records: list[dict]) -> list[dict]:
    keys = ("encoder", "scheme", "n", "condition", "view", "linkage_method")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    metrics = ("top1", "top5", "top10", "mrr", "median_rank", "mean_rank", "auc")
    for key, rows in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        item = dict(zip(keys, key))
        item["runs"] = len(rows)
        for metric in metrics:
            vals = [r.get(metric) for r in rows]
            vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
            item[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None
            item[f"{metric}_min"] = float(np.min(vals)) if vals else None
            item[f"{metric}_max"] = float(np.max(vals)) if vals else None
        out.append(item)
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    fields: list[str] = []
    for row in records:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def run(args: argparse.Namespace) -> None:
    start_time = time.time()
    out = Path(args.output).resolve()
    setup_logging(out)
    sizes = sorted(set(parse_csv_ints(args.sizes)))
    seeds = parse_csv_ints(args.seeds)
    conditions = parse_csv_strings(args.conditions)
    max_n = max(sizes)
    if args.quick:
        sizes = [min(500, max_n)]
        seeds = seeds[:1]
        conditions = ["clean", "noise_0.01"]
        max_n = max(sizes)
    if args.pool < max_n:
        raise ValueError("--pool must be at least max(--sizes)")

    source_name = S.ENC[args.encoder][0]
    source_path = Path(S.CACHE) / source_name
    config = {
        "experiment": "exp25_cross_release_linkage",
        "encoder": args.encoder,
        "pool": args.pool,
        "sizes": sizes,
        "d_pub": args.d_pub,
        "cells": args.cells,
        "seeds": seeds,
        "conditions": conditions,
        "large_query_limit": args.large_query_limit,
        "hungarian_max_n": args.hungarian_max_n,
        "anchor_fractions": args.anchor_fractions,
        "dataset_path": str(source_path),
        "dataset_sha256": file_sha256(source_path) if args.hash_input else None,
        "git_head_before_run": git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "threat_game": {
            "same_documents": True,
            "fresh_row_handles": True,
            "independent_row_permutation": True,
            "persistent_document_ids_visible": False,
            "residual_only_view": ["keyed_residual", "stable_cell_label_for_cell_keys"],
            "full_server_view": ["public_prefix", "keyed_residual", "cell_label"],
        },
        "limitations": [
            "N=10000 reports all scalar/graph invariants but caps exact global NN queries.",
            "Microkey Haar images are sampled directly on the equal-norm sphere; this is distributionally exact for each row.",
            "Gaussian noise and quantization are diagnostic controls, not a formal privacy mechanism.",
            "AUC uses ten sampled non-mates per evaluated query; ranking metrics are exact in the disclosed candidate set.",
        ],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=as_jsonable), encoding="utf-8")
    LOG.info("Configuration: %s", json.dumps(config, default=as_jsonable))

    LOG.info("Loading %s pool=%d", args.encoder, args.pool)
    x, _, d = S.load(args.encoder, n=args.pool)
    if not 0 < args.d_pub < d:
        raise ValueError("d_pub must be between zero and embedding dimension")
    select_rng = np.random.default_rng(args.selection_seed)
    mu = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    pca_n = min(args.pca_sample, len(x))
    pca_idx = select_rng.choice(len(x), size=pca_n, replace=False)
    LOG.info("Fitting PCA d=%d sample=%d", d, pca_n)
    v, eig = S.pca_basis((x[pca_idx] - mu).astype(np.float32))
    eval_ids = select_rng.choice(len(x), size=max_n, replace=False)
    rot = ((x[eval_ids] - mu) @ v).astype(np.float32)
    del x
    u_all = np.ascontiguousarray(rot[:, :args.d_pub])
    r_all = np.ascontiguousarray(rot[:, args.d_pub:])
    del rot
    LOG.info("Clustering public prefix into C=%d cells", args.cells)
    labels_all, centroids = S.kmeans_cells(u_all, args.cells, seed=args.cell_seed, n_train=max_n)
    LOG.info(
        "Cell occupancy min/median/max=%d/%g/%d; residual d=%d",
        int(np.bincount(labels_all, minlength=args.cells).min()),
        float(np.median(np.bincount(labels_all, minlength=args.cells))),
        int(np.bincount(labels_all, minlength=args.cells).max()),
        r_all.shape[1],
    )

    records: list[dict] = []
    anchor_fractions = [float(x) for x in args.anchor_fractions.split(",") if x]

    for seed in seeds:
        LOG.info("Generating fresh-key releases for seed=%d", seed)
        # Generate at max N once; the requested sizes are nested subsets.
        z_cell_a_all = S.apply_keys(r_all, labels_all, master_seed=100_000 + 2 * seed)
        z_cell_b_all = S.apply_keys(r_all, labels_all, master_seed=100_001 + 2 * seed)
        z_micro_a_all = microkey_release(r_all, seed=200_000 + 2 * seed)
        z_micro_b_all = microkey_release(r_all, seed=200_001 + 2 * seed)

        for n in sizes:
            u = u_all[:n]
            r = r_all[:n]
            labels_cell = labels_all[:n]
            # Microkey handles are fresh, not a stable candidate restriction.
            labels_micro = np.zeros(n, dtype=np.int32)
            q_rng = np.random.default_rng(300_000 + seed * 100 + n)
            q_count = n if n <= args.large_query_limit else args.large_query_limit
            canonical_queries = np.sort(q_rng.choice(n, size=q_count, replace=False))

            schemes = {
                f"cell_C{args.cells}": (
                    z_cell_a_all[:n], z_cell_b_all[:n], labels_cell, True,
                ),
                "microkey_perdoc": (
                    z_micro_a_all[:n], z_micro_b_all[:n], labels_micro, False,
                ),
            }
            for scheme, (z_a0, z_b0, canonical_labels, stable_cells) in schemes.items():
                rel_a = make_release(u, z_a0, canonical_labels, 400_000 + seed * 1000 + n)
                rel_b = make_release(u, z_b0, canonical_labels, 500_000 + seed * 1000 + n)
                truth = truth_map(rel_a["hidden_ids"], rel_b["hidden_ids"])
                queries = query_rows(rel_a["hidden_ids"], canonical_queries)
                la = rel_a["labels"] if stable_cells else None
                lb = rel_b["labels"] if stable_cells else None

                for condition in conditions:
                    cond_code = sum((i + 1) * ord(ch) for i, ch in enumerate(condition))
                    ca = np.random.default_rng(600_000 + seed * 1000 + n + cond_code)
                    cb = np.random.default_rng(700_000 + seed * 1000 + n + cond_code)
                    ua = condition_view(rel_a["u"], condition, ca)
                    ub = condition_view(rel_b["u"], condition, cb)
                    za = condition_view(rel_a["z"], condition, ca)
                    zb = condition_view(rel_b["z"], condition, cb)
                    norm_a = np.linalg.norm(za, axis=1).astype(np.float32)
                    norm_b = np.linalg.norm(zb, axis=1).astype(np.float32)
                    meta = {
                        "encoder": args.encoder,
                        "d": d,
                        "d_pub": args.d_pub,
                        "d_priv": d - args.d_pub,
                        "scheme": scheme,
                        "n": n,
                        "seed": seed,
                        "condition": condition,
                        "evaluated_queries": len(queries),
                    }

                    add_record(records, meta, "residual", "cell_label_chance_control",
                               analytic_label_control(la, lb if lb is not None else np.zeros(n), queries))
                    add_record(records, meta, "residual", "residual_norm_nn",
                               evaluate_feature_linkage(norm_a[:, None], norm_b[:, None], truth, queries,
                                                       la, lb, rng_seed=seed + n))
                    add_record(records, meta, "residual", "residual_norm_rank_assignment",
                               rank_assignment_linkage(norm_a, norm_b, truth, queries))

                    # Full server view: public u is explicitly stored unchanged
                    # by the current protocol.  Stable cells are used when known.
                    add_record(records, meta, "full_server", "public_prefix_nn",
                               evaluate_feature_linkage(ua, ub, truth, queries, la, lb,
                                                       rng_seed=seed + n + 1))

                    gram_a = gram_b = graph_a = graph_b = None
                    if stable_cells:
                        gram_a, graph_a = graph_features(za, rel_a["labels"])
                        gram_b, graph_b = graph_features(zb, rel_b["labels"])
                        ga, gb = standardize_pair(gram_a, gram_b)
                        na, nb = standardize_pair(graph_a, graph_b)
                        add_record(records, meta, "residual", "gram_signature_nn",
                                   evaluate_feature_linkage(ga, gb, truth, queries, la, lb,
                                                           rng_seed=seed + n + 2))
                        add_record(records, meta, "residual", "nn_graph_signature_nn",
                                   evaluate_feature_linkage(na, nb, truth, queries, la, lb,
                                                           rng_seed=seed + n + 3))
                        residual_a, residual_b = standardize_pair(
                            np.concatenate([norm_a[:, None], gram_a, graph_a], axis=1),
                            np.concatenate([norm_b[:, None], gram_b, graph_b], axis=1),
                        )
                        add_record(records, meta, "residual", "combined_invariants_nn",
                                   evaluate_feature_linkage(residual_a, residual_b, truth, queries,
                                                           la, lb, rng_seed=seed + n + 4))
                        full_a, full_b = standardize_pair(
                            np.concatenate([ua, norm_a[:, None], gram_a, graph_a], axis=1),
                            np.concatenate([ub, norm_b[:, None], gram_b, graph_b], axis=1),
                        )
                        add_record(records, meta, "full_server", "combined_prefix_invariants_nn",
                                   evaluate_feature_linkage(full_a, full_b, truth, queries,
                                                           la, lb, rng_seed=seed + n + 5))
                        if condition == "clean" and n <= args.hungarian_max_n:
                            add_record(records, meta, "residual", "gram_signature_hungarian",
                                       hungarian_linkage(ga, gb, truth, queries, la, lb))
                    else:
                        full_a, full_b = standardize_pair(
                            np.concatenate([ua, norm_a[:, None]], axis=1),
                            np.concatenate([ub, norm_b[:, None]], axis=1),
                        )
                        add_record(records, meta, "full_server", "combined_prefix_norm_nn",
                                   evaluate_feature_linkage(full_a, full_b, truth, queries,
                                                           None, None, rng_seed=seed + n + 5))

                    # Raw cross-key cosine is the weak metric used by the old
                    # experiment.  Run it on clean observations for contrast.
                    if condition == "clean":
                        add_record(records, meta, "residual", "raw_cross_key_cosine_control",
                                   evaluate_feature_linkage(za, zb, truth, queries, la, lb,
                                                           cosine=True, rng_seed=seed + n + 6))

                    # Anchor fingerprints are most relevant for cell keys.
                    # For microkeys we include a small-N control showing that
                    # document-specific keys do not transfer anchor geometry.
                    run_anchor = stable_cells or n <= 500
                    if run_anchor and condition in {"clean", "fp16", "int8", "noise_0.01"}:
                        for fraction in anchor_fractions:
                            count = max(1, int(round(fraction * n)))
                            arng = np.random.default_rng(800_000 + seed * 1000 + n + int(fraction * 10000))
                            anchor_ids = np.sort(arng.choice(n, size=count, replace=False))
                            fa, fb = known_anchor_features(
                                za, zb, rel_a["hidden_ids"], rel_b["hidden_ids"],
                                rel_a["labels"], rel_b["labels"], anchor_ids, stable_cells,
                            )
                            query_hidden = rel_a["hidden_ids"][queries]
                            unknown_queries = queries[~np.isin(query_hidden, anchor_ids)]
                            if len(unknown_queries) == 0:
                                continue
                            method = f"known_anchor_ip_{fraction:g}"
                            add_record(records, meta, "residual", method,
                                       evaluate_feature_linkage(fa, fb, truth, unknown_queries,
                                                               la, lb, rng_seed=seed + n + count))

                # Save an incremental checkpoint after each scheme/size.
                (out / "raw_metrics.json").write_text(
                    json.dumps(records, indent=2, default=as_jsonable), encoding="utf-8"
                )
                write_csv(out / "metrics.csv", records)

    summary = aggregate(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=as_jsonable), encoding="utf-8")
    write_csv(out / "summary.csv", summary)
    elapsed = time.time() - start_time
    run_info = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "records": len(records),
        "git_head_after_run": git_head(),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "result_files": ["README.md", "config.json", "metrics.csv", "raw_metrics.json", "summary.csv", "summary.json", "run.log"],
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    LOG.info("Complete: %d records in %.1f s; outputs=%s", len(records), elapsed, out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", default="e5-small", choices=sorted(S.ENC))
    p.add_argument("--pool", type=int, default=50_000)
    p.add_argument("--pca-sample", type=int, default=50_000)
    p.add_argument("--sizes", default="500,2000,10000")
    p.add_argument("--d-pub", type=int, default=96)
    p.add_argument("--cells", type=int, default=64)
    p.add_argument("--seeds", default="11,23,47")
    p.add_argument("--conditions", default="clean,fp16,int8,noise_0.01")
    p.add_argument("--anchor-fractions", default="0.01,0.1")
    p.add_argument("--large-query-limit", type=int, default=1000)
    p.add_argument("--hungarian-max-n", type=int, default=2000)
    p.add_argument("--selection-seed", type=int, default=2501)
    p.add_argument("--cell-seed", type=int, default=2502)
    p.add_argument("--output", default=str(ROOT / "results" / "exp25_cross_release_linkage"))
    p.add_argument("--hash-input", action="store_true", help="hash the multi-GB input embedding file")
    p.add_argument("--quick", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
