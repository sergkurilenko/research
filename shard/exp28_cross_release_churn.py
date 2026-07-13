"""Experiment 28: cross-release linkage under partial overlap and churn.

Two independently re-keyed releases contain ``N`` rows each.  Only a chosen
fraction of the underlying documents is shared; the remaining rows are
independent deletions/insertions.  Both releases receive fresh random row
handles, and persistent document identifiers are available only to the
evaluator, never to a linkage method.

The full-server attacker observes the public prefix, cell label and keyed
residual.  We test cell keys and independent per-document microkeys using:

* public-prefix nearest-neighbour linkage;
* residual norm;
* within-cell Gram-distribution and robust-neighbourhood signatures;
* residual and prefix+residual combinations;
* raw cross-key residual cosine as a clean-condition negative control.

All feature scaling is unsupervised and uses only the two observed releases.
No persistent IDs, matched pairs or overlap labels enter feature construction,
standardisation, threshold selection or hyper-parameter tuning.  Metrics use
the hidden correspondence only after predictions have been frozen.  In
addition to common-item R@1/MRR/AUC, mutual-nearest-neighbour (MNN) matching
reports open-set precision, recall, false matches and unmatched rows without a
ground-truth-tuned distance threshold.

Full reproduction command (from the repository root):

  python shard/exp28_cross_release_churn.py

Quick smoke test:

  python shard/exp28_cross_release_churn.py --quick
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import shard_lib as S  # noqa: E402


LOG = logging.getLogger("exp28")


def csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def csv_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def csv_strings(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)
    LOG.addHandler(stream_handler)


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def condition_view(x: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a storage control independently to one observed release."""
    x = np.asarray(x, dtype=np.float32)
    if condition == "clean":
        return x.copy()
    if condition == "fp16":
        return x.astype(np.float16).astype(np.float32)
    if condition == "int8":
        # Each release has its own public symmetric scale.  Under churn these
        # scales can differ, which makes this stricter than quantising a pooled
        # or correspondence-aligned pair of releases.
        scale = float(np.max(np.abs(x))) / 127.0
        if scale == 0.0:
            return x.copy()
        return (np.clip(np.rint(x / scale), -127, 127) * scale).astype(np.float32)
    raise ValueError(f"unknown condition: {condition}")


def microkey_release(r: np.ndarray, seed: int, chunk: int = 4096) -> np.ndarray:
    """Sample the exact image distribution of one independent Haar key/row."""
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(r, axis=1).astype(np.float32)
    out = np.empty_like(r, dtype=np.float32)
    for start in range(0, len(r), chunk):
        stop = min(start + chunk, len(r))
        directions = rng.standard_normal((stop - start, r.shape[1]), dtype=np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + np.float32(1e-30)
        out[start:stop] = directions * norms[start:stop, None]
    return out


def assign_cells(u: np.ndarray, centroids: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    labels = np.empty(len(u), dtype=np.int32)
    cn = np.sum(centroids * centroids, axis=1)
    for start in range(0, len(u), chunk):
        block = u[start:start + chunk]
        labels[start:start + len(block)] = np.argmin(
            cn[None, :] - 2.0 * block @ centroids.T, axis=1,
        )
    return labels


def make_membership(
    universe_size: int,
    n: int,
    overlap: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select hidden canonical IDs for two N-row releases."""
    shared_n = int(round(n * overlap))
    unique_n = n - shared_n
    required = shared_n + 2 * unique_n
    if required > universe_size:
        raise ValueError(f"universe {universe_size} is too small for {required} IDs")
    order = np.random.default_rng(seed).permutation(universe_size)[:required]
    shared = order[:shared_n]
    a_only = order[shared_n:shared_n + unique_n]
    b_only = order[shared_n + unique_n:]
    ids_a = np.concatenate([shared, a_only])
    ids_b = np.concatenate([shared, b_only])
    return ids_a, ids_b, shared_n


def shuffle_release(
    ids: np.ndarray,
    u: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled_ids = ids[order]
    return {
        "hidden_ids": shuffled_ids,
        "u": np.ascontiguousarray(u[shuffled_ids]),
        "z": np.ascontiguousarray(z[shuffled_ids]),
        "labels": np.ascontiguousarray(labels[shuffled_ids]),
    }


def truth_from_hidden(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    """Evaluator-only mapping; -1 denotes a deleted/A-only document."""
    lookup = {int(canonical): row for row, canonical in enumerate(ids_b)}
    return np.asarray([lookup.get(int(canonical), -1) for canonical in ids_a], dtype=np.int64)


def standardize_observed_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unsupervised pooled scaling; row correspondences are not used."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    pooled = np.concatenate([a, b], axis=0).astype(np.float64, copy=False)
    mean = pooled.mean(axis=0, keepdims=True)
    scale = pooled.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return ((a - mean) / scale).astype(np.float32), ((b - mean) / scale).astype(np.float32)


def signature_features(
    z: np.ndarray,
    labels: np.ndarray,
    quantile_count: int,
    neighbour_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation-invariant within-cell Gram signatures robust to churn.

    The first signature uses fixed empirical quantiles and low-order moments
    of a row of the within-cell Gram matrix.  The second uses its strongest
    positive and negative neighbours.  Neither signature uses cross-release
    anchors or persistent IDs.
    """
    n = len(z)
    quant = np.zeros((n, quantile_count + 6), dtype=np.float32)
    neigh = np.zeros((n, 2 * neighbour_width + 3), dtype=np.float32)
    q_levels = np.linspace(0.0, 1.0, quantile_count, dtype=np.float64)
    for cell in np.unique(labels):
        rows = np.flatnonzero(labels == cell)
        gram = z[rows] @ z[rows].T
        size = len(rows)
        norm2 = np.diag(gram).copy()
        quant[rows, 0] = norm2
        quant[rows, 1] = float(size)
        neigh[rows, 0] = norm2
        neigh[rows, 1] = float(size)
        if size == 1:
            continue
        off = gram[~np.eye(size, dtype=bool)].reshape(size, size - 1)
        quant[rows, 2] = np.mean(off, axis=1)
        quant[rows, 3] = np.std(off, axis=1)
        quant[rows, 4] = np.min(off, axis=1)
        quant[rows, 5] = np.max(off, axis=1)
        quant[rows, 6:] = np.quantile(off, q_levels, axis=1).T.astype(np.float32)
        ordered = np.sort(off, axis=1)
        width = min(neighbour_width, size - 1)
        neigh[rows, 2:2 + width] = ordered[:, -width:][:, ::-1]
        neigh[rows, 2 + neighbour_width:2 + neighbour_width + width] = ordered[:, :width]
        neigh[rows, -1] = np.median(off, axis=1)
    return quant, neigh


def squared_distances(a: np.ndarray, b: np.ndarray, cosine: bool) -> np.ndarray:
    # Float64 avoids catastrophic cancellation around zero.  This matters for
    # an unchanged public prefix: a genuine mate should have distance zero,
    # not tie spuriously with a merely close vector after float32 clipping.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if cosine:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-30)
    an = np.sum(a * a, axis=1)[:, None]
    bn = np.sum(b * b, axis=1)[None, :]
    distances = an + bn - 2.0 * (a @ b.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def evaluate_linkage(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    truth: np.ndarray,
    *,
    cosine: bool = False,
    auc_negatives: int = 10,
    rng_seed: int = 0,
) -> dict[str, float | int]:
    """Evaluate closed-rank and threshold-free open-set linkage.

    Candidate rows are restricted to the same public cell.  Top-1 precision
    treats every A row as a forced prediction; top-1 recall is measured over
    the shared rows.  MNN precision/recall instead accept only reciprocal
    nearest neighbours and expose false matches/unmatched rows.
    """
    n_a, n_b = len(feat_a), len(feat_b)
    pred_a = np.full(n_a, -1, dtype=np.int64)
    pred_b = np.full(n_b, -1, dtype=np.int64)
    shared_rows = np.flatnonzero(truth >= 0)
    reciprocal_ranks: list[float] = []
    top5_hits = 0
    top10_hits = 0
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    rng = np.random.default_rng(rng_seed)

    cells = np.union1d(np.unique(labels_a), np.unique(labels_b))
    for cell in cells:
        ia = np.flatnonzero(labels_a == cell)
        ib = np.flatnonzero(labels_b == cell)
        if len(ia) == 0 or len(ib) == 0:
            continue
        distances = squared_distances(feat_a[ia], feat_b[ib], cosine=cosine)
        nearest_b_local = np.argmin(distances, axis=1)
        nearest_a_local = np.argmin(distances, axis=0)
        pred_a[ia] = ib[nearest_b_local]
        pred_b[ib] = ia[nearest_a_local]

        local_b = {int(row): column for column, row in enumerate(ib)}
        for local_a, row_a in enumerate(ia):
            row_b = int(truth[row_a])
            if row_b < 0:
                continue
            true_column = local_b.get(row_b)
            if true_column is None:
                # This should not occur because a shared document has the same
                # deterministic public prefix/cell in both releases.
                continue
            row_distances = distances[local_a]
            true_distance = float(row_distances[true_column])
            rank = int(np.sum(row_distances < true_distance)) + 1
            reciprocal_ranks.append(1.0 / rank)
            top5_hits += int(rank <= 5)
            top10_hits += int(rank <= 10)
            positive_scores.append(-true_distance)
            if len(ib) > 1:
                choices = rng.integers(0, len(ib) - 1, size=auc_negatives)
                choices += choices >= true_column
                negative_scores.extend((-row_distances[choices]).astype(float).tolist())

    shared_n = len(shared_rows)
    correct = int(np.sum(pred_a[shared_rows] == truth[shared_rows]))
    forced_false = int(n_a - correct)
    reciprocal = np.zeros(n_a, dtype=bool)
    valid = pred_a >= 0
    valid_rows = np.flatnonzero(valid)
    reciprocal[valid_rows] = pred_b[pred_a[valid_rows]] == valid_rows
    accepted_rows = np.flatnonzero(reciprocal)
    accepted_correct = int(np.sum(
        (truth[accepted_rows] >= 0) & (pred_a[accepted_rows] == truth[accepted_rows])
    ))
    accepted = len(accepted_rows)
    mnn_false = accepted - accepted_correct
    auc = float("nan")
    if positive_scores and negative_scores:
        labels = np.concatenate([
            np.ones(len(positive_scores), dtype=np.int8),
            np.zeros(len(negative_scores), dtype=np.int8),
        ])
        scores = np.asarray(positive_scores + negative_scores, dtype=np.float64)
        auc = float(roc_auc_score(labels, scores))
    return {
        "n_a": n_a,
        "n_b": n_b,
        "shared_n": shared_n,
        "top1_precision": correct / n_a if n_a else float("nan"),
        "top1_recall": correct / shared_n if shared_n else float("nan"),
        "top5_recall": top5_hits / shared_n if shared_n else float("nan"),
        "top10_recall": top10_hits / shared_n if shared_n else float("nan"),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
        "auc": auc,
        "forced_false_matches": forced_false,
        "mnn_accepted": accepted,
        "mnn_precision": accepted_correct / accepted if accepted else float("nan"),
        "mnn_recall": accepted_correct / shared_n if shared_n else float("nan"),
        "mnn_false_matches": mnn_false,
        "mnn_unmatched_a": int(n_a - accepted),
    }


def chance_control(labels_a: np.ndarray, labels_b: np.ndarray, truth: np.ndarray) -> dict:
    shared_rows = np.flatnonzero(truth >= 0)
    counts = {int(c): int(np.sum(labels_b == c)) for c in np.unique(labels_b)}
    probabilities = np.asarray([1.0 / counts[int(labels_a[row])] for row in shared_rows])
    recalls5 = np.asarray([min(1.0, 5.0 / counts[int(labels_a[row])]) for row in shared_rows])
    recalls10 = np.asarray([min(1.0, 10.0 / counts[int(labels_a[row])]) for row in shared_rows])
    expected_correct = float(np.sum(probabilities))
    return {
        "n_a": len(labels_a), "n_b": len(labels_b), "shared_n": len(shared_rows),
        "top1_precision": expected_correct / len(labels_a),
        "top1_recall": float(np.mean(probabilities)),
        "top5_recall": float(np.mean(recalls5)),
        "top10_recall": float(np.mean(recalls10)),
        "mrr": None, "auc": 0.5,
        "forced_false_matches": float(len(labels_a) - expected_correct),
        "mnn_accepted": None, "mnn_precision": None, "mnn_recall": None,
        "mnn_false_matches": None, "mnn_unmatched_a": None,
        "note": "analytic random choice within the disclosed public cell",
    }


def add_record(records: list[dict], meta: dict, method: str, metrics: dict) -> None:
    record = dict(meta)
    record["linkage_method"] = method
    record.update(metrics)
    records.append(record)
    LOG.info(
        "%s N=%d ov=%.2f seed=%d %-8s %-27s R1=%.4f MRR=%s MNN-P/R=%s/%s false=%s",
        meta["encoder"], meta["n"], meta["overlap"], meta["seed"],
        meta["condition"], method,
        float(metrics.get("top1_recall", float("nan"))),
        f"{metrics['mrr']:.4f}" if metrics.get("mrr") is not None else "NA",
        f"{metrics['mnn_precision']:.4f}" if metrics.get("mnn_precision") is not None else "NA",
        f"{metrics['mnn_recall']:.4f}" if metrics.get("mnn_recall") is not None else "NA",
        metrics.get("mnn_false_matches", "NA"),
    )


def aggregate(records: list[dict]) -> list[dict]:
    group_keys = ("encoder", "scheme", "n", "overlap", "condition", "linkage_method")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in group_keys)].append(record)
    metric_names = (
        "top1_precision", "top1_recall", "top5_recall", "top10_recall", "mrr", "auc",
        "forced_false_matches", "mnn_accepted", "mnn_precision", "mnn_recall",
        "mnn_false_matches", "mnn_unmatched_a",
    )
    summary: list[dict] = []
    for group, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        item = dict(zip(group_keys, group))
        item["runs"] = len(rows)
        for name in metric_names:
            values = [row.get(name) for row in rows]
            values = [float(value) for value in values if value is not None and np.isfinite(value)]
            item[f"{name}_mean"] = float(np.mean(values)) if values else None
            item[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
            item[f"{name}_min"] = float(np.min(values)) if values else None
            item[f"{name}_max"] = float(np.max(values)) if values else None
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(out: Path, records: list[dict]) -> None:
    (out / "raw_metrics.json").write_text(
        json.dumps(records, indent=2, default=json_default), encoding="utf-8",
    )
    write_csv(out / "metrics.csv", records)


def prepare_encoder(
    encoder: str,
    pool: int,
    pca_sample: int,
    d_pub: int,
    cells: int,
    universe_size: int,
    selection_seed: int,
    cell_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, Path]:
    source = Path(S.CACHE) / S.ENC[encoder][0]
    LOG.info("Loading %s pool=%d from %s", encoder, pool, source)
    x, _, d = S.load(encoder, n=pool)
    if not 0 < d_pub < d:
        raise ValueError(f"invalid d_pub={d_pub} for d={d}")
    rng = np.random.default_rng(selection_seed)
    mu = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    fit_n = min(pca_sample, len(x))
    fit_ids = rng.choice(len(x), size=fit_n, replace=False)
    LOG.info("Fitting %s PCA d=%d on %d documents", encoder, d, fit_n)
    basis, _ = S.pca_basis((x[fit_ids] - mu).astype(np.float32))
    # Evaluation IDs are sampled independently of their later release role.
    eval_ids = rng.choice(len(x), size=universe_size, replace=False)
    rotated = ((x[eval_ids] - mu) @ basis).astype(np.float32)
    u = np.ascontiguousarray(rotated[:, :d_pub])
    r = np.ascontiguousarray(rotated[:, d_pub:])
    # A fixed public clustering rule is trained without pair labels.
    train_ids = rng.choice(len(x), size=min(len(x), max(10_000, pca_sample)), replace=False)
    train_u = ((x[train_ids] - mu) @ basis[:, :d_pub]).astype(np.float32)
    _, centroids = S.kmeans_cells(train_u, cells, seed=cell_seed, n_train=len(train_u))
    labels = assign_cells(u, centroids)
    del x, rotated, train_u
    occupancy = np.bincount(labels, minlength=cells)
    LOG.info(
        "%s universe=%d d_pub=%d d_priv=%d cell occupancy min/median/max=%d/%.1f/%d",
        encoder, universe_size, d_pub, d - d_pub, int(occupancy.min()),
        float(np.median(occupancy)), int(occupancy.max()),
    )
    return u, r, labels, d, source


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out = Path(args.output).resolve()
    setup_logging(out)
    encoders = csv_strings(args.encoders)
    sizes = sorted(set(csv_ints(args.sizes)))
    overlaps = sorted(set(csv_floats(args.overlaps)))
    seeds = csv_ints(args.seeds)
    conditions = csv_strings(args.conditions)
    d_pub_values = csv_ints(args.d_pub)
    if len(d_pub_values) == 1:
        d_pub_values *= len(encoders)
    if len(d_pub_values) != len(encoders):
        raise ValueError("--d-pub must contain one value or one per encoder")
    if args.quick:
        encoders = encoders[:1]
        d_pub_values = d_pub_values[:1]
        sizes = [min(sizes)]
        overlaps = [0.5, 1.0]
        seeds = seeds[:1]
        conditions = ["clean", "int8"]
    if any(overlap <= 0 or overlap > 1 for overlap in overlaps):
        raise ValueError("overlaps must be in (0,1]")
    max_n = max(sizes)
    universe_size = int(max(round(max_n * (2.0 - overlap)) for overlap in overlaps))
    if args.pool < universe_size:
        raise ValueError("--pool is smaller than the required churn universe")

    reproduction = (
        f"{sys.executable} {Path(__file__).resolve()} "
        f"--encoders {args.encoders} --pool {args.pool} --pca-sample {args.pca_sample} "
        f"--sizes {args.sizes} --overlaps {args.overlaps} --d-pub {args.d_pub} "
        f"--cells {args.cells} --seeds {args.seeds} --conditions {args.conditions} "
        f"--quantile-count {args.quantile_count} --neighbour-width {args.neighbour_width} "
        f"--output {out}"
    )
    config = {
        "experiment": "exp28_cross_release_churn",
        "encoders": encoders,
        "pool": args.pool,
        "pca_sample": args.pca_sample,
        "sizes": sizes,
        "overlaps": overlaps,
        "d_pub": dict(zip(encoders, d_pub_values)),
        "cells": args.cells,
        "seeds": seeds,
        "conditions": conditions,
        "quantile_count": args.quantile_count,
        "neighbour_width": args.neighbour_width,
        "universe_size": universe_size,
        "selection_seed": args.selection_seed,
        "cell_seed": args.cell_seed,
        "git_head_before_run": git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "protocol": {
            "release_rows_each": "N",
            "shared_rows": "round(overlap*N)",
            "a_only_deletions": "N-shared",
            "b_only_insertions": "N-shared",
            "independent_rekeying": True,
            "independent_row_permutation": True,
            "persistent_ids_visible_to_attacker": False,
            "public_cell_rule_fixed_across_releases": True,
            "mnn_acceptance": "reciprocal nearest neighbour; no distance threshold",
            "feature_scaling": "unsupervised pooled marginal standardisation; no correspondences",
        },
        "limitations": [
            "The corpus embedding pool is fixed; churn changes membership, not document text.",
            "AUC uses ten sampled within-cell non-mates per shared row.",
            "Microkey Haar images are sampled directly on the equal-norm sphere, which is distributionally exact per row.",
            "FP16 and INT8 are storage controls, not privacy mechanisms.",
        ],
        "reproduction_command": reproduction,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "reproduce.txt").write_text(reproduction + "\n", encoding="utf-8")
    LOG.info("Configuration: %s", json.dumps(config))

    records: list[dict] = []
    input_files: dict[str, dict] = {}
    for encoder_index, (encoder, d_pub) in enumerate(zip(encoders, d_pub_values)):
        u, r, labels, d, source = prepare_encoder(
            encoder, args.pool, args.pca_sample, d_pub, args.cells,
            universe_size, args.selection_seed + 1000 * encoder_index,
            args.cell_seed + 1000 * encoder_index,
        )
        input_files[encoder] = {
            "path": str(source),
            "sha256": sha256(source) if args.hash_input else None,
            "size_bytes": source.stat().st_size,
        }

        for seed in seeds:
            LOG.info("Generating independent %s release keys for seed=%d", encoder, seed)
            cell_a_all = S.apply_keys(r, labels, master_seed=2_800_000 + 2 * seed)
            cell_b_all = S.apply_keys(r, labels, master_seed=2_800_001 + 2 * seed)
            micro_a_all = microkey_release(r, seed=2_900_000 + 2 * seed)
            micro_b_all = microkey_release(r, seed=2_900_001 + 2 * seed)

            for n in sizes:
                for overlap in overlaps:
                    overlap_code = int(round(overlap * 10_000))
                    ids_a, ids_b, shared_n = make_membership(
                        universe_size, n, overlap,
                        3_000_000 + encoder_index * 100_000 + seed * 1000 + n + overlap_code,
                    )
                    schemes = {
                        f"cell_C{args.cells}": (cell_a_all, cell_b_all),
                        "microkey_perdoc": (micro_a_all, micro_b_all),
                    }
                    for scheme_index, (scheme, (za_all, zb_all)) in enumerate(schemes.items()):
                        rel_a = shuffle_release(
                            ids_a, u, za_all, labels,
                            3_100_000 + scheme_index * 100_000 + seed * 1000 + n + overlap_code,
                        )
                        rel_b = shuffle_release(
                            ids_b, u, zb_all, labels,
                            3_200_000 + scheme_index * 100_000 + seed * 1000 + n + overlap_code,
                        )
                        truth = truth_from_hidden(rel_a["hidden_ids"], rel_b["hidden_ids"])
                        if int(np.sum(truth >= 0)) != shared_n:
                            raise RuntimeError("hidden truth cardinality mismatch")

                        for condition_index, condition in enumerate(conditions):
                            rng_a = np.random.default_rng(
                                3_300_000 + condition_index * 100_000 + seed * 1000 + n + overlap_code
                            )
                            rng_b = np.random.default_rng(
                                3_400_000 + condition_index * 100_000 + seed * 1000 + n + overlap_code
                            )
                            ua = condition_view(rel_a["u"], condition, rng_a)
                            ub = condition_view(rel_b["u"], condition, rng_b)
                            za = condition_view(rel_a["z"], condition, rng_a)
                            zb = condition_view(rel_b["z"], condition, rng_b)
                            la, lb = rel_a["labels"], rel_b["labels"]

                            norm_a = np.linalg.norm(za, axis=1).astype(np.float32)[:, None]
                            norm_b = np.linalg.norm(zb, axis=1).astype(np.float32)[:, None]
                            gram_a, neigh_a = signature_features(
                                za, la, args.quantile_count, args.neighbour_width,
                            )
                            gram_b, neigh_b = signature_features(
                                zb, lb, args.quantile_count, args.neighbour_width,
                            )
                            prefix_a, prefix_b = standardize_observed_pair(ua, ub)
                            scaled_norm_a, scaled_norm_b = standardize_observed_pair(norm_a, norm_b)
                            scaled_gram_a, scaled_gram_b = standardize_observed_pair(gram_a, gram_b)
                            scaled_neigh_a, scaled_neigh_b = standardize_observed_pair(neigh_a, neigh_b)
                            combined_res_a = np.concatenate(
                                [scaled_norm_a, scaled_gram_a, scaled_neigh_a], axis=1,
                            )
                            combined_res_b = np.concatenate(
                                [scaled_norm_b, scaled_gram_b, scaled_neigh_b], axis=1,
                            )
                            combined_all_a = np.concatenate([prefix_a, combined_res_a], axis=1)
                            combined_all_b = np.concatenate([prefix_b, combined_res_b], axis=1)

                            meta = {
                                "encoder": encoder, "d": d, "d_pub": d_pub,
                                "d_priv": d - d_pub, "scheme": scheme,
                                "n": n, "overlap": overlap, "shared_n": shared_n,
                                "a_only": n - shared_n, "b_only": n - shared_n,
                                "seed": seed, "condition": condition,
                                "candidate_scope": "same public cell",
                            }
                            add_record(records, meta, "cell_label_chance_control",
                                       chance_control(la, lb, truth))
                            methods = {
                                "public_prefix_nn": (prefix_a, prefix_b),
                                "residual_norm_nn": (scaled_norm_a, scaled_norm_b),
                                "gram_quantile_signature_nn": (scaled_gram_a, scaled_gram_b),
                                "robust_neighbour_signature_nn": (scaled_neigh_a, scaled_neigh_b),
                                "combined_residual_invariants_nn": (combined_res_a, combined_res_b),
                                "combined_prefix_residual_nn": (combined_all_a, combined_all_b),
                            }
                            for method_index, (method, (fa, fb)) in enumerate(methods.items()):
                                metrics = evaluate_linkage(
                                    fa, fb, la, lb, truth,
                                    rng_seed=3_500_000 + method_index * 100_000 + seed * 1000 + n + overlap_code,
                                )
                                add_record(records, meta, method, metrics)
                            if condition == "clean":
                                add_record(
                                    records, meta, "raw_cross_key_cosine_control",
                                    evaluate_linkage(
                                        za, zb, la, lb, truth, cosine=True,
                                        rng_seed=3_600_000 + seed * 1000 + n + overlap_code,
                                    ),
                                )
                        checkpoint(out, records)

            del cell_a_all, cell_b_all, micro_a_all, micro_b_all
        del u, r, labels

    config["input_files"] = input_files
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = aggregate(records)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default), encoding="utf-8",
    )
    write_csv(out / "summary.csv", summary)
    elapsed = time.time() - started
    run_info = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "records": len(records),
        "summary_rows": len(summary),
        "git_head_after_run": git_head(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "result_files": [
            "README.md", "config.json", "metrics.csv", "raw_metrics.json",
            "summary.csv", "summary.json", "run.log", "run_info.json", "reproduce.txt",
        ],
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    LOG.info("Complete: %d raw records, %d summary rows in %.1f s", len(records), len(summary), elapsed)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoders", default="e5-small,e5-base")
    p.add_argument("--pool", type=int, default=50_000)
    p.add_argument("--pca-sample", type=int, default=50_000)
    p.add_argument("--sizes", default="2000,10000")
    p.add_argument("--overlaps", default="0.25,0.5,0.75,0.9,1.0")
    p.add_argument("--d-pub", default="96,192")
    p.add_argument("--cells", type=int, default=64)
    p.add_argument("--seeds", default="11,23,47")
    p.add_argument("--conditions", default="clean,fp16,int8")
    p.add_argument("--quantile-count", type=int, default=17)
    p.add_argument("--neighbour-width", type=int, default=12)
    p.add_argument("--selection-seed", type=int, default=2801)
    p.add_argument("--cell-seed", type=int, default=2802)
    p.add_argument("--output", default=str(ROOT / "results" / "exp28_cross_release_churn"))
    p.add_argument("--hash-input", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
