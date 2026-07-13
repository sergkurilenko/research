"""Experiment 23: correct the centering convention in the SHARD score.

The PCA basis V is fitted on centred documents.  Three full-space scores are
then compared on cached BEIR and MIRACL embeddings:

    raw:        <q, x>
    old:        <V^T(q-mu), V^T(x-mu)>
    corrected:  <V^T q, V^T(x-mu)>

For an orthogonal V, corrected = raw - <q,mu>; the difference from the raw
score is constant for every document of a query and therefore preserves the
ranking.  The old convention additionally contains the document-dependent
term -<mu,x>, and does not in general preserve the raw ranking.

The experiment verifies the identity numerically, reports nDCG@10 and
Recall@10/100, and also evaluates (i) old and corrected half-PCA truncation
baselines and (ii) two corrected public-prefix routing choices plus full-space
reranking for the paper's d_pub and K_c settings.  A centred routing vector
can be derived from the uncentred scoring prefix by subtracting the public
constant V_pub^T mu; it therefore requires no extra query disclosure.  The
experiment reads only existing caches and never downloads or re-encodes data.

Example (all cached BEIR and MIRACL cases):

    python shard/exp23_corrected_score.py --suite all

Fast smoke test:

    python shard/exp23_corrected_score.py --suite beir \
        --encoders multilingual-e5-small --datasets scifact \
        --max-queries 12 --max-pca-docs 2000 --force
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "exp23_corrected_score"
BEIR_CACHE = ROOT / "results" / "exp17_outputs" / "emb"
MIRACL_CACHE = ROOT / "results" / "exp17b_outputs" / "emb"

DEFAULT_ENCODERS = ("multilingual-e5-small", "multilingual-e5-base")
DEFAULT_BEIR = ("scifact", "nfcorpus", "arguana")
DEFAULT_MIRACL = ("sw", "bn")
DEFAULT_DPUB_FRACS = (1 / 8, 1 / 4)
DEFAULT_KC = (100, 200)


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(x) for x in parse_csv(value)]


def parse_float_csv(value: str) -> list[float]:
    return [float(x) for x in parse_csv(value)]


def get_git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def json_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        line = f"{stamp} {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


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
    qrels = {str(q): {str(d): int(r) for d, r in rels.items()}
             for q, rels in meta["qrels"].items()}
    cid_to_idx = {cid: i for i, cid in enumerate(cids)}
    relevant = []
    for qid in qids:
        relevant.append({cid_to_idx[cid]: rel for cid, rel in qrels[qid].items()
                         if rel > 0 and cid in cid_to_idx})

    return {
        "D": D, "Q": Q, "cids": cids, "qids": qids,
        "relevant": relevant,
        "npz_path": npz_path, "meta_path": meta_path,
    }


def fit_pca(D: np.ndarray, mu: np.ndarray, max_docs: int, seed: int,
            chunk_size: int, log: Logger) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a full PCA basis on a deterministic document sample.

    The covariance multiplication intentionally uses float64 inputs.  This is
    slower than the earlier experiment's float32 covariance, but substantially
    reduces numerical ambiguity in the rank-equivalence check.
    """
    n, d = D.shape
    n_fit = min(n, max_docs) if max_docs > 0 else n
    rng = np.random.default_rng(seed)
    indices = np.arange(n, dtype=np.int64) if n_fit == n else np.sort(
        rng.choice(n, size=n_fit, replace=False)
    )
    cov = np.zeros((d, d), dtype=np.float64)
    log(f"PCA covariance: n_fit={n_fit:,}, d={d}, chunk={chunk_size:,}")
    for start in range(0, n_fit, chunk_size):
        idx = indices[start:start + chunk_size]
        X = np.asarray(D[idx], dtype=np.float64)
        X -= mu.astype(np.float64, copy=False)
        cov += X.T @ X
    cov /= float(max(1, n_fit))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    V64 = evecs[:, order]
    evals = evals[order]
    orth_err64 = float(np.max(np.abs(V64.T @ V64 - np.eye(d))))
    V = V64.astype(np.float32)
    orth_err32 = float(np.max(np.abs(V.T @ V - np.eye(d, dtype=np.float32))))
    return V, V64, evals, {
        "n_fit": int(n_fit),
        "seed": int(seed),
        "orthogonality_max_abs_float64": orth_err64,
        "orthogonality_max_abs_float32": orth_err32,
        "explained_variance_total": float(evals.sum()),
        "top_eigenvalues": [float(x) for x in evals[:10]],
    }


def transform_documents(D: np.ndarray, mu: np.ndarray, V: np.ndarray,
                        chunk_size: int, log: Logger) -> np.ndarray:
    out = np.empty_like(D, dtype=np.float32)
    log(f"Transforming {len(D):,} documents to centred PCA coordinates")
    for start in range(0, len(D), chunk_size):
        stop = min(len(D), start + chunk_size)
        out[start:stop] = (D[start:stop] - mu) @ V
    return out


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Deterministic top-k: descending score, ascending document id on ties."""
    k = min(k, scores.size)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.lexsort((part, -scores[part]))]


def ranking_metrics(order: np.ndarray, relevant: dict[int, int]) -> tuple[float, float, float]:
    gains = np.array([relevant.get(int(i), 0) for i in order[:10]], dtype=np.float64)
    discount = np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    dcg = float((gains / discount).sum())
    ideal = np.array(sorted(relevant.values(), reverse=True)[:10], dtype=np.float64)
    idcg = float((ideal / np.log2(np.arange(2, ideal.size + 2))).sum()) if ideal.size else 0.0
    ndcg10 = dcg / idcg if idcg else 0.0
    rel_ids = set(relevant)
    denom = max(1, len(rel_ids))
    recall10 = len(rel_ids.intersection(map(int, order[:10]))) / denom
    recall100 = len(rel_ids.intersection(map(int, order[:100]))) / denom
    return ndcg10, recall10, recall100


def bootstrap_delta(a: np.ndarray, b: np.ndarray, samples: int,
                    seed: int) -> dict[str, float]:
    diff = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if samples <= 0 or len(diff) == 0:
        return {"delta": float(diff.mean()) if len(diff) else 0.0,
                "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    # Chunk bootstrap draws to avoid a large samples x queries allocation.
    means = np.empty(samples, dtype=np.float64)
    block = 2048
    for start in range(0, samples, block):
        stop = min(samples, start + block)
        draw = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        means[start:stop] = diff[draw].mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def aggregate_metrics(per_query: dict[str, list[tuple[float, float, float]]],
                      boot_samples: int, boot_seed: int) -> dict[str, Any]:
    names = ("ndcg10", "recall10", "recall100")
    arrays = {method: np.asarray(rows, dtype=np.float64)
              for method, rows in per_query.items()}
    result: dict[str, Any] = {}
    raw = arrays["raw"]
    for method, values in arrays.items():
        result[method] = {name: float(values[:, j].mean())
                          for j, name in enumerate(names)}
        if method != "raw":
            result[method]["vs_raw"] = {
                name: bootstrap_delta(raw[:, j], values[:, j], boot_samples,
                                      boot_seed + 101 * j)
                for j, name in enumerate(names)
            }
    return result


def full_order_diagnostics(raw: np.ndarray, corrected: np.ndarray) -> dict[str, Any]:
    ids = np.arange(raw.size)
    raw_order = np.lexsort((ids, -raw))
    corr_order = np.lexsort((ids, -corrected))
    exact = bool(np.array_equal(raw_order, corr_order))
    same_position = float(np.mean(raw_order == corr_order))
    inv_raw = np.empty_like(raw_order)
    inv_corr = np.empty_like(corr_order)
    inv_raw[raw_order] = np.arange(raw_order.size)
    inv_corr[corr_order] = np.arange(corr_order.size)
    displacement = np.abs(inv_raw - inv_corr)
    return {
        "exact": exact,
        "same_position_fraction": same_position,
        "max_rank_displacement": int(displacement.max(initial=0)),
        "mean_rank_displacement": float(displacement.mean()),
    }


def float64_reference_check(D: np.ndarray, Q: np.ndarray, mu: np.ndarray,
                            V64: np.ndarray, n_queries: int, chunk_size: int,
                            log: Logger) -> dict[str, Any]:
    """Independent high-precision check of the transformed-coordinate score.

    This deliberately recomputes both raw and corrected scores in float64 for
    a small deterministic query subset.  It prevents a claim of equivalence
    from resting on an algebraically constructed ``raw - constant`` array.
    """
    nq = min(n_queries, len(Q))
    if nq <= 0:
        return {"queries": 0}
    n = len(D)
    Q64 = Q[:nq].astype(np.float64)
    mu64 = mu.astype(np.float64)
    Qrot64 = Q64 @ V64
    raw = np.empty((nq, n), dtype=np.float64)
    corrected = np.empty((nq, n), dtype=np.float64)
    log(f"Float64 identity check: queries={nq}, documents={n:,}")
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        D64 = D[start:stop].astype(np.float64)
        raw[:, start:stop] = Q64 @ D64.T
        Drot64 = (D64 - mu64) @ V64
        corrected[:, start:stop] = Qrot64 @ Drot64.T

    expected = raw - (Q64 @ mu64)[:, None]
    error = corrected - expected
    top10_equal = 0
    top100_equal = 0
    top100_overlap = 0.0
    full_checks = []
    for qi in range(nq):
        ro = topk(raw[qi], 100)
        co = topk(corrected[qi], 100)
        top10_equal += int(np.array_equal(ro[:10], co[:10]))
        top100_equal += int(np.array_equal(ro[:100], co[:100]))
        top100_overlap += len(set(map(int, ro)).intersection(map(int, co))) / min(100, n)
        diag = full_order_diagnostics(raw[qi], corrected[qi])
        diag["query_index"] = qi
        full_checks.append(diag)
    return {
        "queries": nq,
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "top10_exact_fraction": top10_equal / nq,
        "top100_exact_fraction": top100_equal / nq,
        "top100_mean_set_overlap": top100_overlap / nq,
        "full_order_exact_fraction": sum(int(x["exact"]) for x in full_checks) / nq,
        "full_order_checks": full_checks,
    }


def evaluate_case(case: dict[str, Any], suite: str, encoder: str, dataset: str,
                  args: argparse.Namespace, log: Logger) -> dict[str, Any]:
    started = time.time()
    D, Q = case["D"], case["Q"]
    n, d = D.shape
    mu = D.mean(axis=0, dtype=np.float64).astype(np.float32)
    V, V64, evals, pca_info = fit_pca(D, mu, args.max_pca_docs, args.seed,
                                      args.pca_chunk, log)
    Drot = transform_documents(D, mu, V, args.transform_chunk, log)
    Q_corrected = Q @ V
    Q_old = (Q - mu) @ V
    q_mu = (Q.astype(np.float64) @ mu.astype(np.float64)).reshape(-1)
    float64_check = float64_reference_check(
        D, Q, mu, V64, args.full_order_queries, args.transform_chunk, log
    )

    max_k = min(max([100, *args.kc]), n)
    per_query: dict[str, list[tuple[float, float, float]]] = {
        "raw": [], "old_centered_full": [], "corrected_full": [],
        "old_centered_half_pca": [], "corrected_half_pca": [],
    }
    d_half = d // 2
    stage_keys: list[tuple[int, int]] = []
    for frac in args.dpub_fracs:
        d_pub = max(1, min(d, int(round(d * frac))))
        for kc in args.kc:
            kc_eff = min(kc, n)
            stage_keys.append((d_pub, kc_eff))
            per_query[f"legacy_hybrid_dpub{d_pub}_kc{kc_eff}"] = []
            per_query[f"corrected_shard_centered_router_dpub{d_pub}_kc{kc_eff}"] = []
            per_query[f"corrected_shard_uncentered_router_dpub{d_pub}_kc{kc_eff}"] = []

    score_error_abs_max = 0.0
    score_error_abs_sum = 0.0
    score_error_sq_sum = 0.0
    score_count = 0
    shift_std_max = 0.0
    shift_range_max = 0.0
    top10_equal = 0
    top100_equal = 0
    top100_overlap_sum = 0.0
    full_order_checks: list[dict[str, Any]] = []

    log(f"Scoring n_q={len(Q):,}, n_docs={n:,}, d={d}, query_batch={args.query_batch}")
    for q_start in range(0, len(Q), args.query_batch):
        q_stop = min(len(Q), q_start + args.query_batch)
        raw_batch = Q[q_start:q_stop] @ D.T
        corr_batch = Q_corrected[q_start:q_stop] @ Drot.T
        old_batch = Q_old[q_start:q_stop] @ Drot.T
        corr_half_batch = Q_corrected[q_start:q_stop, :d_half] @ Drot[:, :d_half].T
        old_half_batch = Q_old[q_start:q_stop, :d_half] @ Drot[:, :d_half].T
        prefix_corr: dict[int, np.ndarray] = {}
        prefix_old: dict[int, np.ndarray] = {}
        for d_pub in sorted({x[0] for x in stage_keys}):
            prefix_corr[d_pub] = Q_corrected[q_start:q_stop, :d_pub] @ Drot[:, :d_pub].T
            prefix_old[d_pub] = Q_old[q_start:q_stop, :d_pub] @ Drot[:, :d_pub].T

        for local, qi in enumerate(range(q_start, q_stop)):
            raw_scores = raw_batch[local]
            corr_scores = corr_batch[local]
            old_scores = old_batch[local]
            corr_half_scores = corr_half_batch[local]
            old_half_scores = old_half_batch[local]
            expected = raw_scores.astype(np.float64) - q_mu[qi]
            err = corr_scores.astype(np.float64) - expected
            score_error_abs_max = max(score_error_abs_max, float(np.max(np.abs(err))))
            score_error_abs_sum += float(np.abs(err).sum())
            score_error_sq_sum += float(np.square(err).sum())
            score_count += err.size
            shift = raw_scores.astype(np.float64) - corr_scores.astype(np.float64)
            shift_std_max = max(shift_std_max, float(shift.std()))
            shift_range_max = max(shift_range_max, float(np.ptp(shift)))

            raw_order = topk(raw_scores, max_k)
            corr_order = topk(corr_scores, max_k)
            old_order = topk(old_scores, max_k)
            corr_half_order = topk(corr_half_scores, max_k)
            old_half_order = topk(old_half_scores, max_k)
            top10_equal += int(np.array_equal(raw_order[:10], corr_order[:10]))
            top100_equal += int(np.array_equal(raw_order[:100], corr_order[:100]))
            top100_overlap_sum += len(set(map(int, raw_order[:100])).intersection(
                map(int, corr_order[:100]))) / min(100, n)

            if qi < args.full_order_queries:
                diag = full_order_diagnostics(raw_scores, corr_scores)
                diag["query_index"] = int(qi)
                full_order_checks.append(diag)

            rel = case["relevant"][qi]
            per_query["raw"].append(ranking_metrics(raw_order, rel))
            per_query["old_centered_full"].append(ranking_metrics(old_order, rel))
            per_query["corrected_full"].append(ranking_metrics(corr_order, rel))
            per_query["old_centered_half_pca"].append(ranking_metrics(old_half_order, rel))
            per_query["corrected_half_pca"].append(ranking_metrics(corr_half_order, rel))

            for d_pub, kc in stage_keys:
                old_candidates = topk(prefix_old[d_pub][local], kc)
                corr_candidates = topk(prefix_corr[d_pub][local], kc)
                legacy_order = old_candidates[topk(raw_scores[old_candidates], min(max_k, kc))]
                corrected_centered_router = old_candidates[
                    topk(corr_scores[old_candidates], min(max_k, kc))
                ]
                corrected_uncentered_router = corr_candidates[
                    topk(corr_scores[corr_candidates], min(max_k, kc))
                ]
                per_query[f"legacy_hybrid_dpub{d_pub}_kc{kc}"].append(
                    ranking_metrics(legacy_order, rel))
                per_query[f"corrected_shard_centered_router_dpub{d_pub}_kc{kc}"].append(
                    ranking_metrics(corrected_centered_router, rel))
                per_query[f"corrected_shard_uncentered_router_dpub{d_pub}_kc{kc}"].append(
                    ranking_metrics(corrected_uncentered_router, rel))

        log(f"  queries {q_start + 1}-{q_stop}/{len(Q)}")

    metrics = aggregate_metrics(per_query, args.bootstrap, args.seed + 23)
    full_exact = sum(int(x["exact"]) for x in full_order_checks)
    result = {
        "schema_version": 1,
        "suite": suite,
        "encoder": encoder,
        "dataset": dataset,
        "n_corpus": int(n),
        "n_queries": int(len(Q)),
        "dimension": int(d),
        "mean_relevant_per_query": float(np.mean([len(x) for x in case["relevant"]])),
        "cache": {
            "npz": str(case["npz_path"].relative_to(ROOT)).replace("\\", "/"),
            "npz_bytes": int(case["npz_path"].stat().st_size),
            "meta": str(case["meta_path"].relative_to(ROOT)).replace("\\", "/"),
            "meta_bytes": int(case["meta_path"].stat().st_size),
        },
        "pca": pca_info,
        "score_identity": {
            "formula": "corrected(q,x) = <V^T q,V^T(x-mu)> = raw(q,x)-<q,mu>",
            "max_abs_error": score_error_abs_max,
            "mean_abs_error": score_error_abs_sum / score_count,
            "rmse": float(np.sqrt(score_error_sq_sum / score_count)),
            "max_within_query_std_of_raw_minus_corrected": shift_std_max,
            "max_within_query_range_of_raw_minus_corrected": shift_range_max,
        },
        "rank_equivalence": {
            "queries": int(len(Q)),
            "top10_exact_fraction": top10_equal / len(Q),
            "top100_exact_fraction": top100_equal / len(Q),
            "top100_mean_set_overlap": top100_overlap_sum / len(Q),
            "full_order_queries_checked": len(full_order_checks),
            "full_order_exact_fraction": full_exact / len(full_order_checks)
            if full_order_checks else None,
            "full_order_checks": full_order_checks,
            "note": "Full-order differences can arise only from finite-precision near-tie swaps; top-k and IR metrics are the operational checks.",
        },
        "float64_reference_check": float64_check,
        "metrics": metrics,
        "settings": {
            "half_pca_dimension": int(d_half),
            "d_pub_fractions": args.dpub_fracs,
            "candidate_counts": args.kc,
            "bootstrap_samples": args.bootstrap,
            "seed": args.seed,
        },
        "elapsed_seconds": time.time() - started,
    }
    return result


def flatten_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        base = {
            "suite": result["suite"], "encoder": result["encoder"],
            "dataset": result["dataset"], "n_corpus": result["n_corpus"],
            "n_queries": result["n_queries"], "dimension": result["dimension"],
            "score_max_abs_error": result["score_identity"]["max_abs_error"],
            "top10_exact_fraction": result["rank_equivalence"]["top10_exact_fraction"],
            "top100_exact_fraction": result["rank_equivalence"]["top100_exact_fraction"],
        }
        for method, metrics in result["metrics"].items():
            row = dict(base)
            row.update({"method": method, "ndcg10": metrics["ndcg10"],
                        "recall10": metrics["recall10"],
                        "recall100": metrics["recall100"]})
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=("all", "beir", "miracl"), default="all")
    p.add_argument("--encoders", type=parse_csv, default=list(DEFAULT_ENCODERS))
    p.add_argument("--datasets", type=parse_csv, default=list(DEFAULT_BEIR))
    p.add_argument("--langs", type=parse_csv, default=list(DEFAULT_MIRACL))
    p.add_argument("--dpub-fracs", type=parse_float_csv, default=list(DEFAULT_DPUB_FRACS))
    p.add_argument("--kc", type=parse_int_csv, default=list(DEFAULT_KC))
    p.add_argument("--max-pca-docs", type=int, default=200_000)
    p.add_argument("--max-queries", type=int, default=0,
                   help="0 uses every cached judged query")
    p.add_argument("--pca-chunk", type=int, default=4096)
    p.add_argument("--transform-chunk", type=int, default=32768)
    p.add_argument("--query-batch", type=int, default=8)
    p.add_argument("--full-order-queries", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true", help="rerun completed cases")
    return p


def main() -> int:
    args = build_parser().parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    log = Logger(args.output / "run.log")
    config = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_revision": get_git_revision(),
        "parameters": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
        "score_conventions": {
            "raw": "q^T x",
            "old_centered": "(V^T(q-mu))^T(V^T(x-mu))",
            "corrected": "(V^T q)^T(V^T(x-mu))",
        },
    }
    json_dump(args.output / "config.json", config)

    cases: list[tuple[str, str, str]] = []
    if args.suite in ("all", "beir"):
        cases.extend(("beir", enc, ds) for enc in args.encoders for ds in args.datasets)
    if args.suite in ("all", "miracl"):
        cases.extend(("miracl", enc, lang) for enc in args.encoders for lang in args.langs)
    log(f"START cases={len(cases)} output={args.output}")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for suite, encoder, dataset in cases:
        tag = f"{suite}_{encoder}_{dataset}"
        case_path = args.output / f"{tag}.json"
        if case_path.exists() and not args.force:
            log(f"SKIP {tag}: completed result exists")
            with open(case_path, encoding="utf-8") as fh:
                results.append(json.load(fh))
            continue
        log(f"CASE {tag}")
        try:
            case = load_case(suite, encoder, dataset, args.max_queries)
            result = evaluate_case(case, suite, encoder, dataset, args, log)
            json_dump(case_path, result)
            results.append(result)
            log(f"DONE {tag} elapsed={result['elapsed_seconds']:.1f}s "
                f"raw={result['metrics']['raw']['ndcg10']:.6f} "
                f"old={result['metrics']['old_centered_full']['ndcg10']:.6f} "
                f"corrected={result['metrics']['corrected_full']['ndcg10']:.6f}")
            del case
        except Exception as exc:
            failures.append({"case": tag, "type": type(exc).__name__, "message": str(exc)})
            log(f"FAIL {tag}: {type(exc).__name__}: {exc}")

        summary = {"schema_version": 1, "results": results, "failures": failures}
        json_dump(args.output / "summary.json", summary)
        write_csv(args.output / "summary.csv", flatten_results(results))

    log(f"FINISH completed={len(results)} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
