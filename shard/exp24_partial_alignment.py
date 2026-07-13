"""Experiment 24: partial alignment attacks without an artificial rank gate.

This experiment revisits SHARD's known-pair alignment threat.  Unlike exp13,
an attacker is fitted in every target cell that contains at least one anchor;
there is no ``m_cell >= d_priv`` gate.  All SHARD variants use the same
residual dimensionality, gallery, targets, and nested global anchor sets.

The fitted reverse maps (stored residual z -> native residual r) are:

* rank-deficient orthogonal Procrustes;
* minimum-norm OLS via the Moore--Penrose pseudoinverse;
* Ridge(alpha=0), reported explicitly (mathematically equal to min-norm OLS);
* Ridge with a relative alpha selected only from held-out anchors in the same
  cell and then refitted on all available anchors;
* the orthogonal polar projection of the OLS map.

For cells with no known pair, an attacker cannot fit a map and the stored
keyed residual is retained as the prediction.  ``coverage_any`` reports this
case separately.  ``coverage_full_rank`` is the fraction of targets in cells
whose anchor design matrix is numerically full rank.  The primary metrics are
per-target reconstruction cosine and residual-only gallery R@1/R@10.

Outputs (all under results/exp24_partial_alignment by default):
  config.json, metrics_per_target.csv, cell_fits.csv, summary.csv,
  summary.json, and run.log.

Example full run:
  python shard/exp24_partial_alignment.py

Small smoke test:
  python shard/exp24_partial_alignment.py --n-pool 6000 --gallery-size 500 \
      --n-target 32 --cells 1,16 --m-grid 8,32,128,512,2048 \
      --seeds 11,23 --bootstrap-reps 50 --kmeans-iters 5 \
      --out results/exp24_partial_alignment/smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS  # noqa: E402
import shard_lib as S  # noqa: E402


METHODS = (
    "no_attack",
    "procrustes_partial",
    "ols_pinv",
    "ridge_alpha0",
    "ridge_cv",
    "polar_ols",
    "oracle_true_residual",
)


def robust_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD with a slower QR-based LAPACK fallback for rare gesdd failures."""
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        # NumPy normally calls divide-and-conquer GESDD.  On heavily loaded
        # BLAS runtimes it can very occasionally fail even for a finite
        # 288-by-288 matrix; QR-based GESVD is slower but more robust.
        return scipy.linalg.svd(
            matrix,
            full_matrices=False,
            check_finite=True,
            lapack_driver="gesvd",
        )


def comma_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def comma_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=os.environ.get("E24_ENC", "e5-small"))
    parser.add_argument("--n-pool", type=int, default=100_000)
    parser.add_argument("--d-pub", type=int, default=96)
    parser.add_argument("--cells", default="1,16,64,256")
    parser.add_argument(
        "--m-grid",
        default="32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,90000",
    )
    parser.add_argument("--seeds", default="11,23,31")
    parser.add_argument("--gallery-size", type=int, default=2_000)
    parser.add_argument("--n-target", type=int, default=128)
    parser.add_argument("--selection-seed", type=int, default=20260712)
    parser.add_argument("--key-seed", type=int, default=240024)
    parser.add_argument("--cell-seed", type=int, default=0)
    parser.add_argument("--kmeans-iters", type=int, default=15)
    parser.add_argument("--kmeans-train", type=int, default=100_000)
    parser.add_argument(
        "--ridge-grid",
        default="0,1e-8,1e-6,1e-4,1e-2,1,100",
        help="Relative alpha grid; absolute alpha is relative alpha times the "
        "mean non-zero squared singular value of the cell's training design.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=240025)
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "exp24_partial_alignment",
    )
    return parser.parse_args()


def setup_logging(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("exp24")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def linear_maps(
    z: np.ndarray,
    r: np.ndarray,
    relative_alphas: Iterable[float],
    rtol: float,
) -> tuple[dict[float, np.ndarray], int, dict[float, float]]:
    """Compute several min-norm/ridge maps from one decomposition.

    A thin SVD is most stable in the underdetermined regime.  Once a cell has
    substantially more rows than columns, a d-by-d Gram eigendecomposition
    avoids materialising the enormous U matrix and makes the large-m end of
    the experiment practical.  Both branches implement
    ``(Z'Z + alpha I)^+ Z'R``.
    """
    alphas = list(relative_alphas)
    z64 = z.astype(np.float64, copy=False)
    r64 = r.astype(np.float64, copy=False)
    n, d = z.shape
    maps: dict[float, np.ndarray] = {}
    absolute: dict[float, float] = {}

    if n <= 2 * d:
        u, singular, vt = robust_svd(z64)
        if singular.size == 0:
            zero = np.zeros((d, r.shape[1]), dtype=np.float32)
            return {float(a): zero.copy() for a in alphas}, 0, {float(a): 0.0 for a in alphas}
        keep = singular > rtol * singular[0]
        rank = int(keep.sum())
        scale = float(np.mean(singular[keep] ** 2)) if rank else 1.0
        projected = u.T @ r64
        for relative in alphas:
            relative = float(relative)
            alpha = relative * scale
            factor = np.zeros_like(singular)
            if alpha == 0.0:
                factor[keep] = 1.0 / singular[keep]
            else:
                factor = singular / (singular**2 + alpha)
            maps[relative] = (vt.T @ (factor[:, None] * projected)).astype(np.float32)
            absolute[relative] = float(alpha)
        return maps, rank, absolute

    gram = z64.T @ z64
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vectors = vectors[:, order]
    if not eigenvalues.size or eigenvalues[0] == 0.0:
        zero = np.zeros((d, r.shape[1]), dtype=np.float32)
        return {float(a): zero.copy() for a in alphas}, 0, {float(a): 0.0 for a in alphas}
    keep = eigenvalues > (rtol**2) * eigenvalues[0]
    rank = int(keep.sum())
    scale = float(np.mean(eigenvalues[keep])) if rank else 1.0
    projected = vectors.T @ (z64.T @ r64)
    for relative in alphas:
        relative = float(relative)
        alpha = relative * scale
        factor = np.zeros_like(eigenvalues)
        if alpha == 0.0:
            factor[keep] = 1.0 / eigenvalues[keep]
        else:
            factor = 1.0 / (eigenvalues + alpha)
        maps[relative] = (vectors @ (factor[:, None] * projected)).astype(np.float32)
        absolute[relative] = float(alpha)
    return maps, rank, absolute


def svd_pinv_map(z: np.ndarray, r: np.ndarray, rtol: float) -> tuple[np.ndarray, int]:
    """Return the minimum-Frobenius-norm W in z @ W ~= r and rank(z)."""
    maps, rank, _ = linear_maps(z, r, [0.0], rtol)
    return maps[0.0], rank


def ridge_map(
    z: np.ndarray, r: np.ndarray, relative_alpha: float, rtol: float
) -> tuple[np.ndarray, int, float]:
    maps, rank, absolute = linear_maps(z, r, [relative_alpha], rtol)
    relative_alpha = float(relative_alpha)
    return maps[relative_alpha], rank, absolute[relative_alpha]


def procrustes_map(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Orthogonal z->r map, including SVD completion when rank deficient."""
    left, _, right_t = robust_svd(
        z.astype(np.float64).T @ r.astype(np.float64)
    )
    return (left @ right_t).astype(np.float32)


def polar_projection(w: np.ndarray) -> np.ndarray:
    left, _, right_t = robust_svd(w.astype(np.float64))
    return (left @ right_t).astype(np.float32)


def choose_ridge_alpha(
    z: np.ndarray,
    r: np.ndarray,
    relative_grid: list[float],
    rtol: float,
) -> tuple[float, dict[str, float]]:
    """Tune only on a deterministic within-cell split of the known anchors."""
    n = len(z)
    if n < 3:
        return 0.0, {str(a): float("nan") for a in relative_grid}
    # Anchor rows arrive in a seeded random order.  Every fifth row is held out
    # for selection; all anchors are used again after alpha has been selected.
    val_mask = np.arange(n) % 5 == 0
    if val_mask.all() or not val_mask.any():
        val_mask[0] = True
        val_mask[-1] = False
    train = ~val_mask
    losses: dict[str, float] = {}
    best_alpha = relative_grid[0]
    best_loss = float("inf")
    candidate_maps, _, _ = linear_maps(z[train], r[train], relative_grid, rtol)
    for alpha in relative_grid:
        w = candidate_maps[float(alpha)]
        error = z[val_mask] @ w - r[val_mask]
        loss = float(np.mean(error.astype(np.float64) ** 2))
        losses[str(alpha)] = loss
        # Prefer less regularisation on numerical ties in this noiseless game.
        if loss < best_loss - 1e-15 or (abs(loss - best_loss) <= 1e-15 and alpha < best_alpha):
            best_loss = loss
            best_alpha = alpha
    return float(best_alpha), losses


def row_cosines(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction.astype(np.float64) * truth.astype(np.float64), axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def retrieval_hits(
    prediction: np.ndarray, gallery_unit: np.ndarray, target_gallery_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(prediction, axis=1, keepdims=True)
    prediction_unit = np.divide(
        prediction, norm, out=np.zeros_like(prediction), where=norm > 0
    )
    scores = prediction_unit @ gallery_unit.T
    top1 = np.argmax(scores, axis=1)
    k = min(10, scores.shape[1])
    topk = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    r1 = top1 == target_gallery_pos
    r10 = np.any(topk == target_gallery_pos[:, None], axis=1)
    return r1, r10


def apply_cell_keys(
    residual: np.ndarray, labels: np.ndarray, key_seed: int, logger: logging.Logger
) -> np.ndarray:
    keyed = np.empty_like(residual)
    cells = np.unique(labels)
    started = time.perf_counter()
    for index, cell in enumerate(cells, start=1):
        mask = labels == cell
        key = S.cell_key(int(cell), residual.shape[1], master_seed=key_seed)
        keyed[mask] = residual[mask] @ key.T
        if index % 64 == 0:
            logger.info("keyed %d/%d cells", index, len(cells))
    logger.info("key transform finished in %.1fs", time.perf_counter() - started)
    return keyed


def make_labels(
    prefix: np.ndarray,
    cells: int,
    seed: int,
    iterations: int,
    n_train: int,
) -> np.ndarray:
    if cells == 1:
        return np.zeros(len(prefix), dtype=np.int32)
    labels, _ = S.kmeans_cells(
        prefix, cells, seed=seed, iters=iterations, n_train=n_train
    )
    return labels


def fit_cell_maps(
    z_anchor: np.ndarray,
    r_anchor: np.ndarray,
    ridge_grid: list[float],
    rtol: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    selected, losses = choose_ridge_alpha(z_anchor, r_anchor, ridge_grid, rtol)
    linear, rank, absolute = linear_maps(z_anchor, r_anchor, [0.0, selected], rtol)
    ols = linear[0.0]
    ridge = linear[selected]
    absolute_alpha = absolute[selected]
    maps = {
        "procrustes_partial": procrustes_map(z_anchor, r_anchor),
        "ols_pinv": ols,
        # In an underdetermined linear least-squares problem, Ridge(alpha=0)
        # with a minimum-norm solver is exactly the Moore--Penrose solution.
        "ridge_alpha0": ols,
        "ridge_cv": ridge,
        "polar_ols": polar_projection(ols),
    }
    details: dict[str, object] = {
        "rank": rank,
        "ridge_rank": rank,
        "ridge_selected_relative_alpha": selected,
        "ridge_selected_absolute_alpha": absolute_alpha,
        "ridge_cv_losses": losses,
    }
    return maps, details


def bootstrap_interval(
    matrix: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Hierarchical bootstrap: resample seeds, then targets within each seed."""
    n_seed, n_target = matrix.shape
    if repetitions <= 0:
        return float("nan"), float("nan")
    values = np.empty(repetitions, dtype=np.float64)
    for b in range(repetitions):
        sampled_seeds = rng.integers(0, n_seed, size=n_seed)
        total = 0.0
        count = 0
        for seed_index in sampled_seeds:
            sampled_targets = rng.integers(0, n_target, size=n_target)
            total += float(matrix[seed_index, sampled_targets].sum())
            count += n_target
        values[b] = total / count
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def summarise(
    per_target: list[dict[str, object]],
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_target:
        grouped[(int(row["cells"]), int(row["m"]), str(row["method"]))].append(row)

    output: list[dict[str, object]] = []
    rng = np.random.default_rng(bootstrap_seed)
    metric_names = ("cosine", "r1", "r10", "coverage_any", "coverage_full_rank")
    for (cells, m, method), rows in sorted(grouped.items()):
        seeds = sorted({int(row["seed"]) for row in rows})
        targets = sorted({int(row["target_index"]) for row in rows})
        seed_pos = {seed: i for i, seed in enumerate(seeds)}
        target_pos = {target: i for i, target in enumerate(targets)}
        summary: dict[str, object] = {
            "cells": cells,
            "m": m,
            "method": method,
            "n_seeds": len(seeds),
            "n_targets_per_seed": len(targets),
            "mean_anchors_in_target_cell": float(
                np.mean([float(row["anchors_in_cell"]) for row in rows])
            ),
        }
        for metric in metric_names:
            matrix = np.empty((len(seeds), len(targets)), dtype=np.float64)
            matrix.fill(np.nan)
            for row in rows:
                matrix[seed_pos[int(row["seed"])], target_pos[int(row["target_index"])]] = float(
                    row[metric]
                )
            if np.isnan(matrix).any():
                raise RuntimeError(f"incomplete bootstrap matrix for {(cells, m, method, metric)}")
            low, high = bootstrap_interval(matrix, bootstrap_reps, rng)
            summary[f"{metric}_mean"] = float(matrix.mean())
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        output.append(summary)
    return output


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    logger = setup_logging(out)
    started = time.perf_counter()

    cells_grid = sorted(set(comma_ints(args.cells)))
    m_requested = sorted(set(comma_ints(args.m_grid)))
    seeds = comma_ints(args.seeds)
    ridge_grid = sorted(set(comma_floats(args.ridge_grid)))
    if not cells_grid or not m_requested or not seeds or not ridge_grid:
        raise ValueError("cells, m-grid, seeds, and ridge-grid must be non-empty")
    if min(m_requested) <= 0:
        raise ValueError("all anchor counts must be positive")
    if min(ridge_grid) < 0:
        raise ValueError("ridge alphas must be non-negative")

    repo = Path(__file__).resolve().parents[1]
    logger.info("loading encoder=%s n_pool=%d", args.encoder, args.n_pool)
    x, _, dimension = S.load(args.encoder, n=args.n_pool)
    if args.d_pub <= 0 or args.d_pub >= dimension:
        raise ValueError("d_pub must be in (0, embedding dimension)")
    if args.gallery_size >= len(x):
        raise ValueError("gallery must be smaller than n_pool")
    if args.n_target > args.gallery_size:
        raise ValueError("n_target must not exceed gallery_size")

    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    logger.info("fitting PCA on %d centered embeddings", len(x))
    pca_started = time.perf_counter()
    basis, eigenvalues = S.pca_basis((x - mean).astype(np.float32))
    rotated = ((x - mean) @ basis).astype(np.float32)
    del x
    prefix = np.ascontiguousarray(rotated[:, : args.d_pub])
    residual = np.ascontiguousarray(rotated[:, args.d_pub :])
    del rotated
    d_priv = residual.shape[1]
    logger.info(
        "PCA finished in %.1fs; d=%d d_pub=%d d_priv=%d",
        time.perf_counter() - pca_started,
        dimension,
        args.d_pub,
        d_priv,
    )

    selection_rng = np.random.default_rng(args.selection_seed)
    gallery_ids = np.sort(
        selection_rng.choice(len(residual), size=args.gallery_size, replace=False)
    )
    target_ids = np.sort(
        selection_rng.choice(gallery_ids, size=args.n_target, replace=False)
    )
    gallery_position = {int(doc): pos for pos, doc in enumerate(gallery_ids)}
    target_gallery_pos = np.asarray(
        [gallery_position[int(doc)] for doc in target_ids], dtype=np.int64
    )
    anchor_pool = np.setdiff1d(
        np.arange(len(residual), dtype=np.int64), gallery_ids, assume_unique=True
    )
    m_grid = [m for m in m_requested if m <= len(anchor_pool)]
    if not m_grid:
        raise ValueError("no m-grid value fits the available anchor pool")
    omitted = sorted(set(m_requested) - set(m_grid))
    if omitted:
        logger.warning("omitting anchor counts larger than pool: %s", omitted)

    anchor_orders: dict[int, np.ndarray] = {}
    for seed in seeds:
        anchor_orders[seed] = np.random.default_rng(seed).permutation(anchor_pool)

    gallery = residual[gallery_ids]
    gallery_norm = np.linalg.norm(gallery, axis=1, keepdims=True)
    gallery_unit = np.divide(
        gallery, gallery_norm, out=np.zeros_like(gallery), where=gallery_norm > 0
    )
    truth = residual[target_ids]

    config: dict[str, object] = {
        "experiment": "exp24_partial_alignment",
        "encoder": args.encoder,
        "n_pool": len(residual),
        "embedding_dimension": dimension,
        "d_pub": args.d_pub,
        "d_priv": d_priv,
        "cells": cells_grid,
        "m_grid": m_grid,
        "seeds": seeds,
        "gallery_size": args.gallery_size,
        "n_target": args.n_target,
        "selection_seed": args.selection_seed,
        "key_seed": args.key_seed,
        "cell_seed": args.cell_seed,
        "kmeans_iters": args.kmeans_iters,
        "kmeans_train": min(args.kmeans_train, len(prefix)),
        "ridge_relative_alpha_grid": ridge_grid,
        "ridge_cv": "within-cell deterministic 80/20 split; refit on all anchors",
        "rank_rtol": args.rank_rtol,
        "bootstrap": "hierarchical seed-then-target percentile bootstrap",
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "methods": list(METHODS),
        "retrieval_metric": "cosine against native residual gallery",
        "uncovered_cell_fallback": "stored keyed residual (no fitted map)",
        "pca_eigenvalues": eigenvalues.tolist(),
        "target_ids": target_ids.tolist(),
        "gallery_ids_sha256_note": "gallery IDs are regenerated from selection_seed",
        "git_commit": git_commit(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "command": [sys.executable, *sys.argv],
    }
    with (out / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    per_target: list[dict[str, object]] = []
    cell_fits: list[dict[str, object]] = []

    for cells in cells_grid:
        c_started = time.perf_counter()
        logger.info("C=%d: fitting cells", cells)
        labels = make_labels(
            prefix,
            cells,
            args.cell_seed,
            args.kmeans_iters,
            min(args.kmeans_train, len(prefix)),
        )
        sizes = np.bincount(labels, minlength=cells)
        logger.info(
            "C=%d cell sizes min/median/max=%d/%d/%d",
            cells,
            int(sizes.min()),
            int(np.median(sizes)),
            int(sizes.max()),
        )
        keyed = apply_cell_keys(residual, labels, args.key_seed, logger)
        target_cells = labels[target_ids]
        target_cell_values = np.unique(target_cells)

        for seed in seeds:
            order = anchor_orders[seed]
            for m in m_grid:
                fit_started = time.perf_counter()
                anchors = order[:m]
                anchor_cells = labels[anchors]
                predictions = {
                    method: keyed[target_ids].copy()
                    for method in METHODS
                    if method not in ("no_attack", "oracle_true_residual")
                }
                predictions["no_attack"] = keyed[target_ids].copy()
                predictions["oracle_true_residual"] = truth.copy()
                target_anchor_count = np.zeros(args.n_target, dtype=np.int32)
                target_rank = np.zeros(args.n_target, dtype=np.int32)

                for cell in target_cell_values:
                    target_mask = target_cells == cell
                    cell_anchors = anchors[anchor_cells == cell]
                    target_anchor_count[target_mask] = len(cell_anchors)
                    if not len(cell_anchors):
                        cell_fits.append(
                            {
                                "cells": cells,
                                "seed": seed,
                                "m": m,
                                "cell": int(cell),
                                "n_targets": int(target_mask.sum()),
                                "n_anchors": 0,
                                "rank": 0,
                                "ridge_selected_relative_alpha": "",
                                "ridge_selected_absolute_alpha": "",
                                "ridge_cv_losses_json": "{}",
                            }
                        )
                        continue

                    maps, details = fit_cell_maps(
                        keyed[cell_anchors],
                        residual[cell_anchors],
                        ridge_grid,
                        args.rank_rtol,
                    )
                    target_rank[target_mask] = int(details["rank"])
                    z_target = keyed[target_ids[target_mask]]
                    for method, mapping in maps.items():
                        predictions[method][target_mask] = z_target @ mapping
                    cell_fits.append(
                        {
                            "cells": cells,
                            "seed": seed,
                            "m": m,
                            "cell": int(cell),
                            "n_targets": int(target_mask.sum()),
                            "n_anchors": len(cell_anchors),
                            "rank": int(details["rank"]),
                            "ridge_selected_relative_alpha": details[
                                "ridge_selected_relative_alpha"
                            ],
                            "ridge_selected_absolute_alpha": details[
                                "ridge_selected_absolute_alpha"
                            ],
                            "ridge_cv_losses_json": json.dumps(details["ridge_cv_losses"]),
                        }
                    )

                coverage_any = target_anchor_count > 0
                coverage_full = target_rank >= d_priv
                for method in METHODS:
                    cosine = row_cosines(predictions[method], truth)
                    r1, r10 = retrieval_hits(
                        predictions[method], gallery_unit, target_gallery_pos
                    )
                    for target_index, target_id in enumerate(target_ids):
                        per_target.append(
                            {
                                "cells": cells,
                                "seed": seed,
                                "m": m,
                                "method": method,
                                "target_index": target_index,
                                "target_id": int(target_id),
                                "target_cell": int(target_cells[target_index]),
                                "anchors_in_cell": int(target_anchor_count[target_index]),
                                "anchor_rank": int(target_rank[target_index]),
                                "coverage_any": int(coverage_any[target_index]),
                                "coverage_full_rank": int(coverage_full[target_index]),
                                "cosine": float(cosine[target_index]),
                                "r1": int(r1[target_index]),
                                "r10": int(r10[target_index]),
                            }
                        )
                pro_cos = row_cosines(predictions["procrustes_partial"], truth).mean()
                ols_cos = row_cosines(predictions["ols_pinv"], truth).mean()
                logger.info(
                    "C=%d seed=%d m=%d coverage=%.3f full-rank=%.3f "
                    "Procrustes cos=%.3f OLS cos=%.3f (%.1fs)",
                    cells,
                    seed,
                    m,
                    float(coverage_any.mean()),
                    float(coverage_full.mean()),
                    float(pro_cos),
                    float(ols_cos),
                    time.perf_counter() - fit_started,
                )

        del keyed, labels
        logger.info("C=%d finished in %.1fs", cells, time.perf_counter() - c_started)

    metric_fields = (
        "cells",
        "seed",
        "m",
        "method",
        "target_index",
        "target_id",
        "target_cell",
        "anchors_in_cell",
        "anchor_rank",
        "coverage_any",
        "coverage_full_rank",
        "cosine",
        "r1",
        "r10",
    )
    cell_fields = (
        "cells",
        "seed",
        "m",
        "cell",
        "n_targets",
        "n_anchors",
        "rank",
        "ridge_selected_relative_alpha",
        "ridge_selected_absolute_alpha",
        "ridge_cv_losses_json",
    )
    write_csv(out / "metrics_per_target.csv", per_target, metric_fields)
    write_csv(out / "cell_fits.csv", cell_fits, cell_fields)

    logger.info("computing %d-repetition hierarchical bootstrap", args.bootstrap_reps)
    summary = summarise(per_target, args.bootstrap_reps, args.bootstrap_seed)
    summary_fields = (
        "cells",
        "m",
        "method",
        "n_seeds",
        "n_targets_per_seed",
        "mean_anchors_in_target_cell",
        "cosine_mean",
        "cosine_ci_low",
        "cosine_ci_high",
        "r1_mean",
        "r1_ci_low",
        "r1_ci_high",
        "r10_mean",
        "r10_ci_low",
        "r10_ci_high",
        "coverage_any_mean",
        "coverage_any_ci_low",
        "coverage_any_ci_high",
        "coverage_full_rank_mean",
        "coverage_full_rank_ci_low",
        "coverage_full_rank_ci_high",
    )
    write_csv(out / "summary.csv", summary, summary_fields)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    alpha_counts: Counter[str] = Counter(
        str(row["ridge_selected_relative_alpha"])
        for row in cell_fits
        if row["ridge_selected_relative_alpha"] != ""
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "elapsed_seconds": elapsed,
        "n_per_target_rows": len(per_target),
        "n_cell_fit_rows": len(cell_fits),
        "n_summary_rows": len(summary),
        "ridge_selected_relative_alpha_counts": dict(alpha_counts),
        "files": [
            "config.json",
            "metrics_per_target.csv",
            "cell_fits.csv",
            "summary.csv",
            "summary.json",
            "run.log",
            "manifest.json",
        ],
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("saved outputs to %s", out)
    logger.info("total runtime %.1fs", elapsed)


if __name__ == "__main__":
    main()
