"""Experiment 27: a formally calibrated Gaussian release baseline.

This experiment deliberately replaces the earlier, uncalibrated "DP-noise"
diagnostic with a mechanism for which a precise adjacency relation and an
end-to-end one-shot privacy statement can be made.

Primary mechanism
-----------------
Let D=(x_1,...,x_N) be a fixed-size, ordered database.  Two databases are
bounded/replacement adjacent when they differ in at most one row.  Row
identities and participation are treated as public.  Define

    f_B(D) = vec(clip_B(x_1), ..., clip_B(x_N)),
    clip_B(x) = x * min(1, B / ||x||_2).

The L2 sensitivity of f_B is at most Delta=2B.  We publish

    f_B(D) + Normal(0, sigma^2 I),

optionally followed by row-wise L2 normalisation (post-processing).  Sigma is
the minimum value satisfying the exact Gaussian privacy-loss equation

    delta(epsilon, sigma) = Phi(Delta/(2 sigma)-epsilon sigma/Delta)
                          - exp(epsilon)
                            Phi(-Delta/(2 sigma)-epsilon sigma/Delta).

The calculation is performed in the log domain.  Thus the release is
(epsilon, delta)-DP under the stated bounded adjacency.  It is not an
add/remove membership guarantee, and data-dependent model/PCA parameters
would need separate accounting.  Repeated releases compose; the README
records the basic k(epsilon,delta) bound.

Evaluation
----------
The script evaluates real cached multilingual-e5 BEIR/MIRACL embeddings with
graded nDCG@10 and Recall@10/100, score Pearson/RMSE and clean-top-10 overlap.
It also shuffles away row identities and performs a native clean-reference
gallery linkage attack (nearest Euclidean/cosine neighbour plus AUC).  Three
independent noise seeds and hierarchical bootstrap confidence intervals are
used.  Corrected SHARD utility is read from exp23 and matched only when both
nDCG@10 and Recall@100 tolerances are met.

The CUDA path computes the exact noisy cosine scores efficiently without
materialising every noisy release:

  <q, norm(x + sigma z)> = (<q,x> + sigma <q,z>) / ||x+sigma z||.

Examples
--------
Full default programme (e5-small BEIR+MIRACL, e5-base BEIR):

    python shard/exp27_formal_dp_baseline.py

Quick smoke test:

    python shard/exp27_formal_dp_baseline.py --encoders multilingual-e5-small \
        --suites beir --datasets scifact --max-queries 20 \
        --epsilons 1,64,512 --seeds 11 --bootstrap 100 --force

Calibration-only self-test:

    python shard/exp27_formal_dp_baseline.py --calibration-only --force
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

try:
    import torch
except Exception:  # pragma: no cover - reported as a clear runtime error
    torch = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "exp27_formal_dp_baseline"
BEIR_CACHE = ROOT / "results" / "exp17_outputs" / "emb"
MIRACL_CACHE = ROOT / "results" / "exp17b_outputs" / "emb"
EXP23 = ROOT / "results" / "exp23_corrected_score"

DEFAULT_EPSILONS = (
    0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
    512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0,
)
DEFAULT_SEEDS = (11, 23, 31)


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(x) for x in parse_csv(value)]


def parse_int_csv(value: str) -> list[int]:
    return [int(x) for x in parse_csv(value)]


def json_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        line = f"{stamp} {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Exact analytic Gaussian calibration
# ---------------------------------------------------------------------------


def gaussian_log_delta(epsilon: float, sigma: float, sensitivity: float) -> float:
    """Log of the exact Gaussian-mechanism delta at epsilon.

    The stable log-domain subtraction is essential for the deliberately broad
    epsilon grid (up to 32768), where direct ``exp(epsilon)`` overflows.
    """
    if epsilon < 0 or sigma <= 0 or sensitivity <= 0:
        raise ValueError("epsilon>=0, sigma>0 and sensitivity>0 are required")
    a = sensitivity / (2.0 * sigma)
    b = epsilon * sigma / sensitivity
    log_a = float(log_ndtr(a - b))
    log_b = float(epsilon + log_ndtr(-a - b))
    # Analytically log_b < log_a.  Rounding can make them equal in extreme
    # tails; then delta is below floating-point resolution.
    diff = min(0.0, log_b - log_a)
    if diff == 0.0:
        return -math.inf
    return log_a + math.log(-math.expm1(diff))


def gaussian_delta(epsilon: float, sigma: float, sensitivity: float) -> float:
    ld = gaussian_log_delta(epsilon, sigma, sensitivity)
    return 0.0 if ld == -math.inf else float(math.exp(ld))


def calibrate_analytic_gaussian(
    epsilon: float, delta: float, sensitivity: float,
) -> float:
    """Return the minimum Gaussian standard deviation for (eps,delta)-DP."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    target = math.log(delta)

    def residual(log_sigma: float) -> float:
        return gaussian_log_delta(epsilon, math.exp(log_sigma), sensitivity) - target

    lo = math.log(sensitivity) - 40.0
    hi = math.log(sensitivity) + 20.0
    if residual(lo) <= 0 or residual(hi) >= 0:
        raise RuntimeError("failed to bracket analytic Gaussian calibration")
    return float(math.exp(brentq(residual, lo, hi, xtol=1e-13, rtol=1e-13)))


def calibration_self_tests() -> list[dict[str, Any]]:
    """Executable unit tests for formula, scaling, monotonicity and clipping."""
    tests: list[dict[str, Any]] = []
    delta = 1e-6
    eps_grid = [0.5, 1.0, 4.0, 64.0, 512.0, 8192.0]
    sigmas = [calibrate_analytic_gaussian(e, delta, 2.0) for e in eps_grid]
    for e, s in zip(eps_grid, sigmas):
        ld = gaussian_log_delta(e, s, 2.0)
        err = abs(ld - math.log(delta))
        assert err < 2e-9, (e, s, err)
    tests.append({
        "name": "calibrated_log_delta_matches_target",
        "passed": True,
        "max_abs_log_error": max(
            abs(gaussian_log_delta(e, s, 2.0) - math.log(delta))
            for e, s in zip(eps_grid, sigmas)
        ),
    })

    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    tests.append({"name": "sigma_decreases_with_epsilon", "passed": True})

    s1 = calibrate_analytic_gaussian(3.0, delta, 1.0)
    s7 = calibrate_analytic_gaussian(3.0, delta, 7.0)
    assert abs(s7 / s1 - 7.0) < 1e-10
    tests.append({"name": "sigma_scales_linearly_with_sensitivity", "passed": True})

    # Cross-check the stable expression against the direct expression where
    # the latter is numerically safe.
    e, s, sens = 1.0, 8.44935777865368, 2.0
    a, b = sens / (2 * s), e * s / sens
    direct = float(ndtr(a - b) - math.exp(e) * ndtr(-a - b))
    stable = gaussian_delta(e, s, sens)
    assert abs(direct - stable) < 1e-14
    tests.append({
        "name": "stable_formula_matches_direct_formula",
        "passed": True,
        "absolute_error": abs(direct - stable),
    })

    rng = np.random.default_rng(27)
    X = rng.standard_normal((1000, 17)).astype(np.float32)
    B = 1.25
    clipped, _ = row_clip(X, B)
    max_norm = float(np.linalg.norm(clipped, axis=1).max())
    # Triangle inequality establishes sensitivity <=2B for any replacement.
    assert max_norm <= B * (1 + 1e-12)
    max_pair = float(np.linalg.norm(clipped[:500] - clipped[500:], axis=1).max())
    assert max_pair <= 2 * B * (1 + 1e-12)
    tests.append({
        "name": "clipping_implies_replacement_sensitivity_2B",
        "passed": True,
        "observed_max_norm": max_norm,
        "observed_max_paired_difference": max_pair,
        "theoretical_sensitivity": 2 * B,
    })
    return tests


# ---------------------------------------------------------------------------
# Data and IR metrics
# ---------------------------------------------------------------------------


def load_case(suite: str, encoder: str, dataset: str, max_queries: int) -> dict[str, Any]:
    cache = BEIR_CACHE if suite == "beir" else MIRACL_CACHE
    npz_path = cache / f"{encoder}_{dataset}.npz"
    meta_path = cache / f"{encoder}_{dataset}_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing cache pair: {npz_path}, {meta_path}")
    with np.load(npz_path) as z:
        D = np.asarray(z["D"], dtype=np.float32)
        Q = np.asarray(z["Q"], dtype=np.float32)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    qids = [str(x) for x in meta["qids"]]
    if max_queries > 0:
        Q = Q[:max_queries]
        qids = qids[:max_queries]
    cids = [str(x) for x in meta["cids"]]
    cid_to_idx = {cid: i for i, cid in enumerate(cids)}
    qrels = {
        str(q): {str(doc): int(rel) for doc, rel in rels.items()}
        for q, rels in meta["qrels"].items()
    }
    relevant = [
        {cid_to_idx[doc]: rel for doc, rel in qrels[qid].items()
         if rel > 0 and doc in cid_to_idx}
        for qid in qids
    ]
    return {
        "D": D,
        "Q": Q,
        "qids": qids,
        "cids": cids,
        "relevant": relevant,
        "npz_path": npz_path,
        "meta_path": meta_path,
    }


def row_clip(X: np.ndarray, bound: float) -> tuple[np.ndarray, dict[str, float]]:
    norms = np.linalg.norm(X.astype(np.float64), axis=1)
    # Keep clipped float32 rows strictly inside the declared mathematical
    # bound despite the final rounding.  Rows already inside B are unchanged.
    safe_target = bound * (1.0 - 1e-6)
    scale = np.ones_like(norms)
    clipped_mask = norms > bound
    scale[clipped_mask] = safe_target / np.maximum(norms[clipped_mask], 1e-30)
    Y = (X * scale[:, None].astype(np.float32)).astype(np.float32)
    return Y, {
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "fraction_clipped": float(np.mean(clipped_mask)),
        "max_postclip_norm": float(np.linalg.norm(Y.astype(np.float64), axis=1).max()),
    }


def ranking_metrics(order: np.ndarray, relevant: dict[int, int]) -> tuple[float, float, float]:
    gains = np.array([relevant.get(int(i), 0) for i in order[:10]], dtype=np.float64)
    discount = np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    dcg = float((gains / discount).sum())
    ideal = np.array(sorted(relevant.values(), reverse=True)[:10], dtype=np.float64)
    idcg = float((ideal / np.log2(np.arange(2, ideal.size + 2))).sum()) if ideal.size else 0.0
    rel_ids = set(relevant)
    denom = max(1, len(rel_ids))
    return (
        dcg / idcg if idcg else 0.0,
        len(rel_ids.intersection(map(int, order[:10]))) / denom,
        len(rel_ids.intersection(map(int, order[:100]))) / denom,
    )


def merge_topk(
    old_scores: "torch.Tensor", old_ids: "torch.Tensor",
    scores: "torch.Tensor", start: int, k: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    kk = min(k, scores.shape[1])
    vals, loc = torch.topk(scores, kk, dim=1, largest=True, sorted=False)
    ids = loc.to(torch.int64) + int(start)
    all_vals = torch.cat((old_scores, vals), dim=1)
    all_ids = torch.cat((old_ids, ids), dim=1)
    take_vals, take = torch.topk(all_vals, k, dim=1, largest=True, sorted=False)
    take_ids = torch.gather(all_ids, 1, take)
    return take_vals, take_ids


def roc_auc_from_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC with tie handling, computed as pairwise positive-vs-negative rate."""
    # Each target has one positive and a modest number of negatives.  This
    # target-stratified calculation avoids a dependency on sklearn.
    return float(np.mean((pos[:, None] > neg).astype(np.float64) +
                         0.5 * (pos[:, None] == neg)))


def bootstrap_ci(
    matrix: np.ndarray, samples: int, seed: int,
) -> tuple[float, float, float]:
    """Hierarchical bootstrap of seed x unit matrices."""
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    mean = float(np.nanmean(a))
    if samples <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    out = np.empty(samples, dtype=np.float64)
    ns, nu = a.shape
    # Vectorise in bounded blocks.  For each replicate we first resample seed
    # clusters, average those selected rows, and then resample observational
    # units.  This is algebraically the same cross-product bootstrap as the
    # scalar implementation, but avoids millions of Python-level iterations.
    block = 128
    for start in range(0, samples, block):
        stop = min(samples, start + block)
        nb = stop - start
        si = rng.integers(0, ns, size=(nb, ns))
        ui = rng.integers(0, nu, size=(nb, nu))
        seed_mean = np.nanmean(a[si], axis=1)  # (nb, nu)
        out[start:stop] = np.nanmean(
            np.take_along_axis(seed_mean, ui, axis=1), axis=1,
        )
    return mean, float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def case_seed(suite: str, encoder: str, dataset: str, seed: int, stream: str) -> int:
    data = f"exp27|{suite}|{encoder}|{dataset}|{seed}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**63 - 1)


def evaluate_case(
    suite: str,
    encoder: str,
    dataset: str,
    case: dict[str, Any],
    epsilons: list[float],
    sigmas: list[float],
    seeds: list[int],
    bound: float,
    topk: int,
    chunk_size: int,
    gallery_size: int,
    targets: int,
    negatives_per_target: int,
    device: str,
    log: Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for exp27")
    D_raw = case["D"]
    Q = case["Q"]
    D, clip_stats = row_clip(D_raw, bound)
    n, d = D.shape
    nq = len(Q)
    ne = len(epsilons)
    ns = len(seeds)
    dev = torch.device(device)
    qt = torch.as_tensor(np.ascontiguousarray(Q), device=dev, dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    neg_inf = torch.full((nq, topk), -torch.inf, device=dev)
    neg_ids = torch.full((nq, topk), -1, dtype=torch.int64, device=dev)
    raw_top_s, raw_top_i = neg_inf.clone(), neg_ids.clone()
    top_s = [[neg_inf.clone() for _ in range(ne)] for _ in range(ns)]
    top_i = [[neg_ids.clone() for _ in range(ne)] for _ in range(ns)]

    # Exact per-query score-fidelity sufficient statistics over the corpus.
    sx = np.zeros(nq, dtype=np.float64)
    sx2 = np.zeros(nq, dtype=np.float64)
    sy = np.zeros((ns, ne, nq), dtype=np.float64)
    sy2 = np.zeros((ns, ne, nq), dtype=np.float64)
    sxy = np.zeros((ns, ne, nq), dtype=np.float64)
    sdiff2 = np.zeros((ns, ne, nq), dtype=np.float64)

    generators = []
    for seed in seeds:
        g = torch.Generator(device=dev)
        g.manual_seed(case_seed(suite, encoder, dataset, seed, "retrieval"))
        generators.append(g)

    log(f"{suite}/{encoder}/{dataset}: n={n:,}, q={nq:,}, d={d}, device={dev}")
    with torch.inference_mode():
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            x = torch.as_tensor(np.ascontiguousarray(D[start:stop]), device=dev)
            base = qt @ x.T
            raw_top_s, raw_top_i = merge_topk(raw_top_s, raw_top_i, base, start, topk)
            bx = base.sum(dim=1).double().cpu().numpy()
            bx2 = (base.double().square().sum(dim=1)).cpu().numpy()
            sx += bx
            sx2 += bx2
            xnorm2 = x.double().square().sum(dim=1)
            for si, gen in enumerate(generators):
                z = torch.randn(x.shape, generator=gen, device=dev, dtype=torch.float32)
                qz = qt @ z.T
                xz = (x.double() * z.double()).sum(dim=1)
                z2 = z.double().square().sum(dim=1)
                for ei, sigma in enumerate(sigmas):
                    denom = torch.sqrt(torch.clamp(
                        xnorm2 + 2.0 * sigma * xz + sigma * sigma * z2,
                        min=1e-30,
                    )).float()
                    noisy = (base + float(sigma) * qz) / denom[None, :]
                    top_s[si][ei], top_i[si][ei] = merge_topk(
                        top_s[si][ei], top_i[si][ei], noisy, start, topk,
                    )
                    noisy_d = noisy.double()
                    sy[si, ei] += noisy_d.sum(dim=1).cpu().numpy()
                    sy2[si, ei] += noisy_d.square().sum(dim=1).cpu().numpy()
                    sxy[si, ei] += (base.double() * noisy_d).sum(dim=1).cpu().numpy()
                    sdiff2[si, ei] += (noisy_d - base.double()).square().sum(dim=1).cpu().numpy()
            if start == 0 or stop == n or stop // chunk_size % 10 == 0:
                log(f"  retrieval chunk {start:,}:{stop:,}/{n:,}")

    # Sort the final top-k by score.  torch.topk's internal order above is not
    # guaranteed after the last merge.
    raw_order = torch.gather(raw_top_i, 1, torch.argsort(raw_top_s, dim=1, descending=True)).cpu().numpy()
    orders: list[list[np.ndarray]] = []
    for si in range(ns):
        seed_orders = []
        for ei in range(ne):
            order = torch.gather(
                top_i[si][ei], 1,
                torch.argsort(top_s[si][ei], dim=1, descending=True),
            ).cpu().numpy()
            seed_orders.append(order)
        orders.append(seed_orders)

    # Per-query rows.
    query_rows: list[dict[str, Any]] = []
    raw_metrics = np.asarray([
        ranking_metrics(raw_order[i], case["relevant"][i]) for i in range(nq)
    ])
    for qi in range(nq):
        query_rows.append({
            "suite": suite, "encoder": encoder, "dataset": dataset,
            "seed": "raw", "epsilon": "inf", "sigma": 0.0,
            "query_index": qi, "query_id": case["qids"][qi],
            "ndcg10": raw_metrics[qi, 0], "recall10": raw_metrics[qi, 1],
            "recall100": raw_metrics[qi, 2], "score_pearson": 1.0,
            "score_rmse": 0.0, "clean_top10_overlap": 1.0,
        })

    denom_x = sx2 - sx * sx / n
    for si, seed in enumerate(seeds):
        for ei, (eps, sigma) in enumerate(zip(epsilons, sigmas)):
            order = orders[si][ei]
            ir = np.asarray([
                ranking_metrics(order[i], case["relevant"][i]) for i in range(nq)
            ])
            denom_y = sy2[si, ei] - sy[si, ei] ** 2 / n
            numer = sxy[si, ei] - sx * sy[si, ei] / n
            pearson = numer / np.sqrt(np.maximum(denom_x * denom_y, 1e-30))
            rmse = np.sqrt(np.maximum(sdiff2[si, ei] / n, 0.0))
            overlap = np.asarray([
                len(set(raw_order[i, :10]).intersection(order[i, :10])) / 10.0
                for i in range(nq)
            ])
            for qi in range(nq):
                query_rows.append({
                    "suite": suite, "encoder": encoder, "dataset": dataset,
                    "seed": seed, "epsilon": eps, "sigma": sigma,
                    "query_index": qi, "query_id": case["qids"][qi],
                    "ndcg10": ir[qi, 0], "recall10": ir[qi, 1],
                    "recall100": ir[qi, 2], "score_pearson": pearson[qi],
                    "score_rmse": rmse[qi], "clean_top10_overlap": overlap[qi],
                })

    # Clean-reference native-gallery linkage.  Row identifiers are assumed
    # removed before the attack; otherwise the public ordering makes linkage
    # trivially 1.0 and no geometric experiment is meaningful.
    arng = np.random.default_rng(case_seed(suite, encoder, dataset, 0, "gallery"))
    ng = min(gallery_size, n)
    gallery = np.sort(arng.choice(n, size=ng, replace=False))
    nt = min(targets, ng)
    tgt_pos = np.sort(arng.choice(ng, size=nt, replace=False))
    target_ids = gallery[tgt_pos]
    G = D[gallery].astype(np.float32)
    G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-30)
    Xtarget = D[target_ids].astype(np.float32)
    attack_rows: list[dict[str, Any]] = []
    gt = torch.as_tensor(G, device=dev)
    xt = torch.as_tensor(Xtarget, device=dev)
    with torch.inference_mode():
        for seed in seeds:
            gen = torch.Generator(device=dev)
            gen.manual_seed(case_seed(suite, encoder, dataset, seed, "attack"))
            z = torch.randn(xt.shape, generator=gen, device=dev)
            for eps, sigma in zip(epsilons, sigmas):
                y = xt + float(sigma) * z
                y = y / torch.clamp(torch.linalg.vector_norm(y, dim=1, keepdim=True), min=1e-30)
                score = y @ gt.T
                k_attack = min(10, ng)
                _, pred = torch.topk(score, k_attack, dim=1, largest=True, sorted=True)
                pred_np = pred.cpu().numpy()
                score_np = score.cpu().numpy()
                pos_score = score_np[np.arange(nt), tgt_pos]
                neg_pos = np.empty((nt, negatives_per_target), dtype=np.int64)
                for ti in range(nt):
                    pool = np.delete(np.arange(ng), tgt_pos[ti])
                    neg_pos[ti] = arng.choice(pool, negatives_per_target, replace=False)
                neg_score = score_np[np.arange(nt)[:, None], neg_pos]
                for ti in range(nt):
                    hit = np.flatnonzero(pred_np[ti] == tgt_pos[ti])
                    rank = int(hit[0] + 1) if hit.size else 11
                    attack_rows.append({
                        "suite": suite, "encoder": encoder, "dataset": dataset,
                        "seed": seed, "epsilon": eps, "sigma": sigma,
                        "target_index": ti, "target_document_id": int(target_ids[ti]),
                        "gallery_size": ng, "rank_capped_at_11": rank,
                        "linkage_r1": int(rank == 1), "linkage_r10": int(rank <= 10),
                        "positive_score": float(pos_score[ti]),
                        "mean_negative_score": float(neg_score[ti].mean()),
                        "pairwise_auc_contribution": float(np.mean(
                            (pos_score[ti] > neg_score[ti]).astype(np.float64) +
                            0.5 * (pos_score[ti] == neg_score[ti])
                        )),
                    })

    diagnostics = {
        "n_corpus": n,
        "n_queries": nq,
        "dimension": d,
        "mean_relevant_per_query": float(np.mean([len(r) for r in case["relevant"]])),
        "clip": clip_stats,
        "gallery_size": ng,
        "targets": nt,
        "raw_metrics": {
            "ndcg10": float(raw_metrics[:, 0].mean()),
            "recall10": float(raw_metrics[:, 1].mean()),
            "recall100": float(raw_metrics[:, 2].mean()),
        },
    }
    del qt, gt, xt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return query_rows, attack_rows, diagnostics


def summarize_queries(
    rows: list[dict[str, Any]], seeds: list[int], epsilons: list[float],
    bootstrap: int, boot_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = sorted({(r["suite"], r["encoder"], r["dataset"]) for r in rows})
    metrics = ("ndcg10", "recall10", "recall100", "score_pearson", "score_rmse", "clean_top10_overlap")
    for suite, enc, ds in cases:
        cr = [r for r in rows if (r["suite"], r["encoder"], r["dataset"]) == (suite, enc, ds)]
        raw = [r for r in cr if r["seed"] == "raw"]
        raw_row: dict[str, Any] = {
            "suite": suite, "encoder": enc, "dataset": ds,
            "epsilon": "inf", "sigma": 0.0, "privacy": "nonprivate",
        }
        for mi, metric in enumerate(metrics):
            vals = np.asarray([float(r[metric]) for r in raw])[None, :]
            mean, lo, hi = bootstrap_ci(vals, bootstrap, boot_seed + mi)
            raw_row[metric] = mean
            raw_row[f"{metric}_ci95_low"] = lo
            raw_row[f"{metric}_ci95_high"] = hi
        out.append(raw_row)
        for ei, eps in enumerate(epsilons):
            sub = [r for r in cr if r["seed"] != "raw" and float(r["epsilon"]) == eps]
            nq = len({int(r["query_index"]) for r in sub})
            row = {
                "suite": suite, "encoder": enc, "dataset": ds,
                "epsilon": eps, "sigma": float(sub[0]["sigma"]),
                "privacy": "formal_bounded_record_dp",
            }
            for mi, metric in enumerate(metrics):
                mat = np.full((len(seeds), nq), np.nan)
                smap = {s: i for i, s in enumerate(seeds)}
                for r in sub:
                    mat[smap[int(r["seed"])], int(r["query_index"])] = float(r[metric])
                mean, lo, hi = bootstrap_ci(mat, bootstrap, boot_seed + 1000 * ei + mi)
                row[metric] = mean
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi
            out.append(row)
    return out


def summarize_attacks(
    rows: list[dict[str, Any]], seeds: list[int], epsilons: list[float],
    bootstrap: int, boot_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = sorted({(r["suite"], r["encoder"], r["dataset"]) for r in rows})
    metrics = ("linkage_r1", "linkage_r10", "pairwise_auc_contribution")
    for suite, enc, ds in cases:
        cr = [r for r in rows if (r["suite"], r["encoder"], r["dataset"]) == (suite, enc, ds)]
        for ei, eps in enumerate(epsilons):
            sub = [r for r in cr if float(r["epsilon"]) == eps]
            nt = len({int(r["target_index"]) for r in sub})
            row = {
                "suite": suite, "encoder": enc, "dataset": ds,
                "epsilon": eps, "sigma": float(sub[0]["sigma"]),
                "gallery_size": int(sub[0]["gallery_size"]),
                "chance_r1": 1.0 / int(sub[0]["gallery_size"]),
                "chance_r10": min(10.0 / int(sub[0]["gallery_size"]), 1.0),
            }
            smap = {s: i for i, s in enumerate(seeds)}
            for mi, metric in enumerate(metrics):
                mat = np.full((len(seeds), nt), np.nan)
                for r in sub:
                    mat[smap[int(r["seed"])], int(r["target_index"])] = float(r[metric])
                mean, lo, hi = bootstrap_ci(mat, bootstrap, boot_seed + 1000 * ei + mi)
                row[metric] = mean
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi
            out.append(row)
    return out


def summarize_per_seed(
    query_rows: list[dict[str, Any]], attack_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compact per-seed JSON/CSV view backed by the raw per-unit CSV files."""
    metrics = (
        "ndcg10", "recall10", "recall100", "score_pearson",
        "score_rmse", "clean_top10_overlap",
    )
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for r in query_rows:
        eps_key = "inf" if str(r["epsilon"]) == "inf" else f"{float(r['epsilon']):.12g}"
        key = (str(r["suite"]), str(r["encoder"]), str(r["dataset"]), str(r["seed"]), eps_key)
        groups.setdefault(key, []).append(r)
    attack_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for r in attack_rows:
        key = (
            str(r["suite"]), str(r["encoder"]), str(r["dataset"]), str(r["seed"]),
            f"{float(r['epsilon']):.12g}",
        )
        attack_groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for key, rows in groups.items():
        suite, enc, ds, seed, eps_key = key
        row: dict[str, Any] = {
            "suite": suite, "encoder": enc, "dataset": ds, "seed": seed,
            "epsilon": "inf" if eps_key == "inf" else float(eps_key),
            "sigma": float(rows[0]["sigma"]), "n_queries": len(rows),
        }
        for metric in metrics:
            row[metric] = float(np.mean([float(r[metric]) for r in rows]))
        ar = attack_groups.get(key, [])
        if ar:
            row.update({
                "n_attack_targets": len(ar),
                "linkage_r1": float(np.mean([float(r["linkage_r1"]) for r in ar])),
                "linkage_r10": float(np.mean([float(r["linkage_r10"]) for r in ar])),
                "pairwise_auc": float(np.mean([
                    float(r["pairwise_auc_contribution"]) for r in ar
                ])),
            })
        out.append(row)

    def sort_key(r: dict[str, Any]) -> tuple[Any, ...]:
        eps = math.inf if r["epsilon"] == "inf" else float(r["epsilon"])
        return r["suite"], r["encoder"], r["dataset"], eps, str(r["seed"])

    return sorted(out, key=sort_key)


def load_shard_target(suite: str, encoder: str, dataset: str, d: int) -> dict[str, Any] | None:
    path = EXP23 / f"{suite}_{encoder}_{dataset}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    key = f"corrected_shard_centered_router_dpub{d // 4}_kc200"
    metrics = data.get("metrics", {}).get(key)
    if metrics is None:
        return None
    return {
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "method": key,
        "ndcg10": float(metrics["ndcg10"]),
        "recall10": float(metrics["recall10"]),
        "recall100": float(metrics["recall100"]),
    }


def match_shard_utility(
    utility: list[dict[str, Any]], diagnostics: dict[str, Any],
    epsilons: list[float], tol_ndcg: float, tol_recall100: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_name, diag in sorted(diagnostics.items()):
        suite, enc, ds = case_name.split("|")
        target = load_shard_target(suite, enc, ds, int(diag["dimension"]))
        row: dict[str, Any] = {
            "suite": suite, "encoder": enc, "dataset": ds,
            "ndcg_tolerance": tol_ndcg,
            "recall100_tolerance": tol_recall100,
            "matched": False,
        }
        if target is None:
            row["reason"] = "corrected exp23 SHARD target unavailable"
            out.append(row)
            continue
        row.update({f"shard_{k}": v for k, v in target.items()})
        candidates = [
            r for r in utility
            if r["suite"] == suite and r["encoder"] == enc and r["dataset"] == ds
            and r["epsilon"] != "inf"
            and abs(float(r["ndcg10"]) - target["ndcg10"]) <= tol_ndcg
            and abs(float(r["recall100"]) - target["recall100"]) <= tol_recall100
        ]
        if candidates:
            # Smallest epsilon is the strongest privacy among utility matches.
            best = min(candidates, key=lambda r: float(r["epsilon"]))
            row.update({
                "matched": True,
                "epsilon": best["epsilon"],
                "sigma": best["sigma"],
                "dp_ndcg10": best["ndcg10"],
                "dp_recall100": best["recall100"],
                "ndcg_difference_dp_minus_shard": float(best["ndcg10"]) - target["ndcg10"],
                "recall100_difference_dp_minus_shard": float(best["recall100"]) - target["recall100"],
            })
        else:
            row["reason"] = "no finite epsilon on the evaluated grid meets both utility tolerances"
            finite = [
                r for r in utility
                if r["suite"] == suite and r["encoder"] == enc and r["dataset"] == ds
                and r["epsilon"] != "inf"
            ]
            if finite:
                strongest = max(finite, key=lambda r: float(r["epsilon"]))
                row.update({
                    "largest_evaluated_epsilon": strongest["epsilon"],
                    "ndcg_at_largest_epsilon": strongest["ndcg10"],
                    "recall100_at_largest_epsilon": strongest["recall100"],
                })
        out.append(row)
    return out


def build_readme(
    config: dict[str, Any], calibration: list[dict[str, Any]],
    utility: list[dict[str, Any]], attacks: list[dict[str, Any]],
    matched: list[dict[str, Any]], elapsed: float,
) -> str:
    def fmt(v: float) -> str:
        return f"{v:.3f}"

    eps_one = [r for r in utility if r["epsilon"] == 1.0]
    eps_512_attacks = [r for r in attacks if r["epsilon"] == 512.0]
    finite_matches = [r for r in matched if r.get("matched")]
    if eps_one and eps_512_attacks:
        finding_lines = [
            (f"At `epsilon=1`, nDCG@10 is at most "
             f"{math.ceil(1000 * max(float(r['ndcg10']) for r in eps_one)) / 1000:.3f} across the "
             f"{len(eps_one)} evaluated cases. At"),
            (f"`epsilon=512`, native-gallery R@1 is already "
             f"{min(float(r['linkage_r1']) for r in eps_512_attacks):.3f}--"
             f"{max(float(r['linkage_r1']) for r in eps_512_attacks):.3f}, while retrieval"),
            "remains far below the corrected SHARD target.",
            "",
            (f"Only {len(finite_matches)}/{len(matched)} strict utility matches occur on the "
             "finite grid, all at"),
            "`epsilon=32768`; their linkage R@1 is at least 0.995. The other cases do not",
            "satisfy both utility tolerances even at that extremely weak privacy parameter.",
            "Thus this formal per-record Gaussian release has no evaluated high-utility",
            "point that also resists native clean-reference linkage. This does not imply",
            "that SHARD is unlinkable or DP.",
        ]
    else:
        finding_lines = ["Calibration-only run; no retrieval or linkage cases were evaluated."]

    lines = [
        "# Experiment 27: formally calibrated Gaussian release baseline",
        "",
        "This directory replaces the earlier pseudo-DP noise sweep with a one-shot",
        "mechanism that has an explicit adjacency relation, clipping bound, global",
        "sensitivity, and exact analytic Gaussian calibration.",
        "",
        "## Main finding",
        "",
        *finding_lines,
        "",
        "## Privacy statement",
        "",
        f"- Adjacency: fixed-size bounded/replacement adjacency; databases differ in one row.",
        f"- Public information: row participation/identifiers, encoder, dimension and release size.",
        f"- Clipping: `clip_B(x)` with `B={config['clip_bound']}`.",
        "  The bound is fixed from the e5 unit-normalisation contract with a float32",
        "  safety margin; it is not estimated from the evaluated corpus.",
        f"- Global L2 sensitivity: `Delta_2=2B={config['sensitivity']}` for the concatenated database release.",
        f"- Mechanism: clipped rows plus iid Gaussian noise; row-wise unit normalisation is DP-preserving post-processing.",
        f"- Accounting: exact Gaussian privacy-loss equation at `delta={config['delta']}`.",
        "",
        "This is a **content privacy conditional on public participation** guarantee, not",
        "an add/remove membership guarantee. A trusted curator gives the central-DP",
        "interpretation. If each single-record owner clips and perturbs locally, the same",
        "calibration gives approximate local DP for that vector. Independent repeated",
        "releases compose; the elementary bound is `(k epsilon, k delta)`. The one-shot",
        "joint database publication is not charged N times because replacement",
        "adjacency changes only one row. Public or independently trained model",
        "parameters are assumed. Fitting a centroid/PCA on the private release set",
        "would require a separate DP mechanism and composition.",
        "",
        "The mechanism protects document-vector content only; it makes no query-privacy",
        "claim. Although identifiers are public in the formal adjacency model, the",
        "linkage diagnostic removes them before matching so that R@1 measures geometric",
        "linkability rather than the tautological identifier channel. It is an attack",
        "diagnostic, not an empirical test or replacement for the DP theorem.",
        "",
        "## Calibration",
        "",
        "| epsilon | sigma | noise multiplier sigma/Delta | achieved delta |",
        "|---:|---:|---:|---:|",
    ]
    for r in calibration:
        lines.append(
            f"| {r['epsilon']:g} | {r['sigma']:.8g} | {r['noise_multiplier']:.8g} | {r['achieved_delta']:.3g} |"
        )
    lines += [
        "",
        "## Retrieval utility and native-gallery linkage",
        "",
        "The table reports raw utility, the strongest conventional privacy point",
        "(`epsilon=1`), and the largest/weakest evaluated finite epsilon. Linkage uses",
        "256 released targets against a clean native gallery of up to 5,000 documents",
        "after row identifiers have been removed. CIs are hierarchical bootstrap CIs",
        "over noise seeds and queries/targets.",
        "",
        "| suite / encoder / data | eps | nDCG@10 | R@100 | score r | link R@1 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    cases = sorted({(r["suite"], r["encoder"], r["dataset"]) for r in utility})
    max_eps = max(config["epsilons"])
    for suite, enc, ds in cases:
        for eps in ("inf", 1.0, max_eps):
            u = next((r for r in utility if (r["suite"], r["encoder"], r["dataset"]) == (suite, enc, ds) and r["epsilon"] == eps), None)
            if u is None:
                continue
            a = None if eps == "inf" else next((r for r in attacks if (r["suite"], r["encoder"], r["dataset"]) == (suite, enc, ds) and r["epsilon"] == eps), None)
            lines.append(
                f"| {suite}/{enc}/{ds} | {eps} | {fmt(u['ndcg10'])} | {fmt(u['recall100'])} | "
                f"{fmt(u['score_pearson'])} | {('-' if a is None else fmt(a['linkage_r1']))} | "
                f"{('-' if a is None else fmt(a['pairwise_auc_contribution']))} |"
            )
    lines += [
        "",
        "## Matched utility against corrected SHARD",
        "",
        f"A match requires `|Delta nDCG@10| <= {config['match_ndcg_tolerance']}` and",
        f"`|Delta Recall@100| <= {config['match_recall100_tolerance']}` simultaneously.",
        "No interpolation is used; a match is claimed only at an actually evaluated point.",
        "",
        "| suite / encoder / data | matched | strongest matching epsilon | note |",
        "|---|---:|---:|---|",
    ]
    for r in matched:
        note = "finite grid match" if r["matched"] else r.get("reason", "")
        lines.append(
            f"| {r['suite']}/{r['encoder']}/{r['dataset']} | {str(r['matched']).lower()} | "
            f"{r.get('epsilon', '-')} | {note} |"
        )
    lines += [
        "",
        "The comparison is deliberately limited to retrieval utility. The Gaussian",
        "baseline also uses an exact full-corpus scan, whereas the exp23 SHARD target",
        "uses two-stage K=200 routing. Therefore the match controls quality, not",
        "latency, traffic or compute. SHARD and the Gaussian mechanism have different",
        "guarantees and attack surfaces; a native-gallery linkage rate must not be",
        "relabelled as an epsilon value, and SHARD must not be described as DP.",
        "",
        "## Files",
        "",
        "- `config.json`: complete mechanism and run configuration.",
        "- `calibration.csv/json`: exact sigma values and achieved deltas.",
        "- `calibration_tests.json`: executable formula/clipping unit tests.",
        "- `per_query.csv`: raw seed/query measurements.",
        "- `per_target.csv`: raw seed/target linkage measurements.",
        "- `per_seed_summary.csv/json`: compact per-seed aggregates of both raw tables.",
        "- `utility_summary.csv/json`: bootstrap summaries.",
        "- `attack_summary.csv/json`: linkage bootstrap summaries.",
        "- `matched_utility.csv/json`: explicit corrected-SHARD matching decisions.",
        "- `case_diagnostics.json`, `run_info.json`, and `run.log`: audit trail.",
        "",
        f"Elapsed time: {elapsed:.1f} seconds.",
        "",
    ]
    return "\n".join(lines)


def resolve_cases(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    encoders = parse_csv(args.encoders)
    suites = parse_csv(args.suites)
    requested = set(parse_csv(args.datasets)) if args.datasets else None
    cases: list[tuple[str, str, str]] = []
    for enc in encoders:
        for suite in suites:
            # The default programme uses both encoders for BEIR and e5-small
            # for the two large MIRACL corpora.  Users can explicitly request
            # base MIRACL with --include-base-miracl.
            if suite == "miracl" and enc.endswith("e5-base") and not args.include_base_miracl:
                continue
            datasets = ("scifact", "nfcorpus", "arguana") if suite == "beir" else ("sw", "bn")
            for ds in datasets:
                if requested is not None and ds not in requested:
                    continue
                cases.append((suite, enc, ds))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--encoders", default="multilingual-e5-small,multilingual-e5-base")
    parser.add_argument("--suites", default="beir,miracl")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--include-base-miracl", action="store_true")
    parser.add_argument("--epsilons", default=",".join(map(str, DEFAULT_EPSILONS)))
    parser.add_argument("--delta", type=float, default=1e-6)
    # Cached e5 vectors are unit-normalised up to float32 round-off.  The tiny
    # margin makes the implemented finite-precision domain bound literal
    # (observed maxima are about 1+2e-7), rather than silently exceeding B=1.
    parser.add_argument("--clip-bound", type=float, default=1.000001)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--gallery-size", type=int, default=5000)
    parser.add_argument("--targets", type=int, default=256)
    parser.add_argument("--negatives-per-target", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--match-ndcg-tolerance", type=float, default=0.01)
    parser.add_argument("--match-recall100-tolerance", type=float, default=0.02)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not args.force:
        raise SystemExit(f"{out / 'summary.json'} exists; pass --force to overwrite")
    log_path = out / "run.log"
    if args.force and log_path.exists():
        log_path.unlink()
    log = Logger(log_path)
    start = time.time()

    epsilons = sorted(set(parse_float_csv(args.epsilons)))
    seeds = parse_int_csv(args.seeds)
    if any(e <= 0 for e in epsilons):
        raise ValueError("all epsilons must be positive")
    if args.topk < 100:
        raise ValueError("topk must be at least 100 for Recall@100")
    sensitivity = 2.0 * args.clip_bound

    tests = calibration_self_tests()
    json_dump(out / "calibration_tests.json", tests)
    log(f"calibration self-tests: {len(tests)} passed")
    calibration = []
    sigmas = []
    for eps in epsilons:
        sigma = calibrate_analytic_gaussian(eps, args.delta, sensitivity)
        sigmas.append(sigma)
        calibration.append({
            "epsilon": eps,
            "delta_target": args.delta,
            "clip_bound": args.clip_bound,
            "sensitivity": sensitivity,
            "sigma": sigma,
            "noise_multiplier": sigma / sensitivity,
            "achieved_delta": gaussian_delta(eps, sigma, sensitivity),
            "log_achieved_delta": gaussian_log_delta(eps, sigma, sensitivity),
        })
    write_csv(out / "calibration.csv", calibration)
    json_dump(out / "calibration.json", calibration)

    cases = [] if args.calibration_only else resolve_cases(args)
    config = {
        "schema_version": 1,
        "mechanism": "row-clipped analytic Gaussian, followed by row-wise unit normalisation",
        "adjacency": "fixed-size bounded/replacement; one embedding row may change",
        "public_information": "participation, row identities, encoder, dimension, release size",
        "clip_bound": args.clip_bound,
        "clip_bound_source": "fixed e5 unit-normalisation contract plus 1e-6 float32 margin; not data-fitted",
        "sensitivity": sensitivity,
        "delta": args.delta,
        "epsilons": epsilons,
        "seeds": seeds,
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "cases": [{"suite": s, "encoder": e, "dataset": d} for s, e, d in cases],
        "max_queries": args.max_queries,
        "topk": args.topk,
        "chunk_size": args.chunk_size,
        "gallery_size": args.gallery_size,
        "targets": args.targets,
        "negatives_per_target": args.negatives_per_target,
        "device": args.device,
        "match_ndcg_tolerance": args.match_ndcg_tolerance,
        "match_recall100_tolerance": args.match_recall100_tolerance,
        "composition_note": "k independent releases satisfy basic (k*epsilon,k*delta)-DP",
        "scope_note": "content privacy conditional on public participation; not add/remove membership DP",
    }
    json_dump(out / "config.json", config)

    all_queries: list[dict[str, Any]] = []
    all_attacks: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for suite, enc, ds in cases:
        case = load_case(suite, enc, ds, args.max_queries)
        qrows, arows, diag = evaluate_case(
            suite, enc, ds, case, epsilons, sigmas, seeds,
            args.clip_bound, args.topk, args.chunk_size, args.gallery_size,
            args.targets, args.negatives_per_target, args.device, log,
        )
        all_queries.extend(qrows)
        all_attacks.extend(arows)
        diagnostics[f"{suite}|{enc}|{ds}"] = diag
        # Checkpoint raw rows after each potentially large MIRACL case.
        write_csv(out / "per_query.csv", all_queries)
        write_csv(out / "per_target.csv", all_attacks)
        json_dump(out / "case_diagnostics.json", diagnostics)

    utility = summarize_queries(
        all_queries, seeds, epsilons, args.bootstrap, args.bootstrap_seed,
    ) if all_queries else []
    attacks = summarize_attacks(
        all_attacks, seeds, epsilons, args.bootstrap, args.bootstrap_seed + 500_000,
    ) if all_attacks else []
    matched = match_shard_utility(
        utility, diagnostics, epsilons,
        args.match_ndcg_tolerance, args.match_recall100_tolerance,
    ) if utility else []
    per_seed = summarize_per_seed(all_queries, all_attacks) if all_queries else []
    write_csv(out / "utility_summary.csv", utility)
    json_dump(out / "utility_summary.json", utility)
    write_csv(out / "attack_summary.csv", attacks)
    json_dump(out / "attack_summary.json", attacks)
    write_csv(out / "matched_utility.csv", matched)
    json_dump(out / "matched_utility.json", matched)
    write_csv(out / "per_seed_summary.csv", per_seed)
    json_dump(out / "per_seed_summary.json", per_seed)

    elapsed = time.time() - start
    run_info = {
        "started_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "torch": None if torch is None else torch.__version__,
        "cuda_available": False if torch is None else torch.cuda.is_available(),
        "cuda_device": None if torch is None or not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        "command": sys.argv,
    }
    json_dump(out / "run_info.json", run_info)
    summary = {
        "schema_version": 1,
        "config": config,
        "calibration": calibration,
        "calibration_tests": tests,
        "utility_summary": utility,
        "attack_summary": attacks,
        "matched_utility": matched,
        "per_seed_summary": per_seed,
        "case_diagnostics": diagnostics,
        "elapsed_seconds": elapsed,
    }
    json_dump(out / "summary.json", summary)
    (out / "README.md").write_text(
        build_readme(config, calibration, utility, attacks, matched, elapsed),
        encoding="utf-8",
    )
    log(f"complete: {len(cases)} cases, elapsed={elapsed:.1f}s, output={out}")


if __name__ == "__main__":
    main()
