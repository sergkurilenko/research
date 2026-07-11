"""Measure semantic leakage from the candidate identifiers already exposed.

The protocol intentionally sends an *ordered* public-PQ shortlist to the
provider.  This audit asks how much of the projected query direction can be
estimated from those identifiers when the provider already owns the exact
candidate vectors.  It never decrypts a ciphertext and never uses PQ scores.

Three estimators are deliberately simple and reproducible:

``centroid``
    Mean of unit-normalised candidate vectors.
``log_rank``
    Candidate mean weighted by ``1/log2(rank + 1)``.
``ridge_rank_ls``
    A regularised least-squares fit from exact candidate vectors to fixed
    normal-score rank targets, with the log-rank direction as a prior.

The audit reports direction cosine, full-corpus retrieval overlap, a relevance
proxy, and disjoint-view linkability.  The last metric reconstructs two views
from alternating (therefore non-overlapping) candidate identifiers and asks
whether cosine similarity links views of the same query better than views of
different queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score, roc_curve

from system.graded_ir_bench import exact_topk_search
from system.retrieval_bench import (
    deterministic_query_split,
    deterministic_self_qrels,
    load_projection_basis,
    project_queries,
)


SCHEMA = "candidate_id_semantic_leakage.v1"
METHODS = ("centroid", "log_rank", "ridge_rank_ls")
METHOD_METADATA = {
    "centroid": {
        "observation": "set_only",
        "uses_candidate_order": False,
        "description": "unweighted centroid of the exposed candidate-ID set",
    },
    "log_rank": {
        "observation": "order_aware",
        "uses_candidate_order": True,
        "description": "log-discounted centroid of the ordered candidate list",
    },
    "ridge_rank_ls": {
        "observation": "order_aware",
        "uses_candidate_order": True,
        "description": "regularised least-squares fit to fixed rank targets",
    },
}


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        return np.zeros_like(value, dtype=np.float64)
    return value / norm


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def reconstruct_from_candidates(
    candidate_vectors: np.ndarray,
    method: str,
    ridge_fraction: float = 0.10,
) -> np.ndarray:
    """Estimate a query direction from ordered exact candidate vectors only."""

    rows = _unit_rows(candidate_vectors)
    if rows.ndim != 2 or len(rows) < 1:
        raise ValueError("candidate_vectors must be a non-empty matrix")
    if method == "centroid":
        return _unit(rows.mean(axis=0))

    ranks = np.arange(1, len(rows) + 1, dtype=np.float64)
    weights = 1.0 / np.log2(ranks + 1.0)
    prior = _unit(np.average(rows, axis=0, weights=weights))
    if method == "log_rank":
        return prior
    if method != "ridge_rank_ls":
        raise ValueError(f"unknown reconstruction method: {method}")
    if ridge_fraction <= 0:
        raise ValueError("ridge_fraction must be positive")

    # Fixed normal-score targets preserve rank but assume no unavailable score
    # values.  Solve the regularised problem in the K-dimensional dual:
    # min_q ||Xq-y||^2 + alpha ||q-prior||^2.
    centered = rows - rows.mean(axis=0, keepdims=True)
    probabilities = (len(rows) - ranks + 0.5) / len(rows)
    targets = ndtri(probabilities)
    targets -= targets.mean()
    gram = centered @ centered.T
    scale = max(float(np.trace(gram) / len(rows)), np.finfo(np.float64).eps)
    alpha = ridge_fraction * scale
    residual = targets - centered @ prior
    try:
        dual = np.linalg.solve(
            gram + alpha * np.eye(len(rows), dtype=np.float64), residual
        )
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(
            gram + alpha * np.eye(len(rows), dtype=np.float64),
            residual,
            rcond=None,
        )[0]
    return _unit(prior + centered.T @ dual)


def reconstruct_batch(
    projected_docs: np.ndarray,
    candidates: np.ndarray,
    method: str,
    ridge_fraction: float,
) -> np.ndarray:
    if candidates.ndim != 2 or np.any(candidates < 0) or np.any(
        candidates >= len(projected_docs)
    ):
        raise ValueError("candidate ids must be a valid two-dimensional matrix")
    output = np.empty((len(candidates), projected_docs.shape[1]), dtype=np.float32)
    for row, ids in enumerate(candidates):
        vectors = np.asarray(projected_docs[ids], dtype=np.float32)
        output[row] = reconstruct_from_candidates(
            vectors, method, ridge_fraction
        ).astype(np.float32)
    return output


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("cosine inputs must be aligned matrices")
    numerator = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("cannot summarize an empty vector")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def ranking_overlap(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    if reference.shape[0] != candidate.shape[0] or k < 1:
        raise ValueError("rankings must align and k must be positive")
    output = np.empty(len(reference), dtype=np.float64)
    for row in range(len(reference)):
        expected = set(int(x) for x in reference[row, :k])
        observed = set(int(x) for x in candidate[row, :k])
        output[row] = len(expected & observed) / k
    return output


def relevance_proxy(
    rankings: np.ndarray,
    relevant_local_ids: Sequence[set[int]],
    cutoffs: Sequence[int] = (10, 100),
) -> dict[str, Any]:
    if len(rankings) != len(relevant_local_ids):
        raise ValueError("rankings and relevance sets must align")
    result: dict[str, Any] = {}
    reciprocal_ranks: list[float] = []
    for row, relevant in zip(rankings, relevant_local_ids, strict=True):
        rank = next(
            (position for position, doc in enumerate(row, start=1) if int(doc) in relevant),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    result["reciprocal_rank_at_100"] = summarize(reciprocal_ranks)
    for cutoff in cutoffs:
        hits = [
            float(any(int(doc) in relevant for doc in row[:cutoff]))
            for row, relevant in zip(rankings, relevant_local_ids, strict=True)
        ]
        result[f"hit_at_{cutoff}"] = summarize(hits)
    return result


def linkability_metrics(view_a: np.ndarray, view_b: np.ndarray, seed: int) -> dict[str, Any]:
    """Same-query versus different-query linking from disjoint ID views."""

    if view_a.shape != view_b.shape or len(view_a) < 3:
        raise ValueError("linkability requires at least three aligned query views")
    positive = cosine_rows(view_a, view_b)
    rng = np.random.default_rng(seed)
    negative_parts: list[np.ndarray] = []
    for _ in range(min(10, len(view_a) - 1)):
        shift = int(rng.integers(1, len(view_a)))
        negative_parts.append(cosine_rows(view_a, np.roll(view_b, shift, axis=0)))
    negative = np.concatenate(negative_parts)
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    scores = np.concatenate((positive, negative))
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    allowed = np.flatnonzero(fpr <= 0.01)
    index = int(allowed[np.argmax(tpr[allowed])]) if len(allowed) else 0
    return {
        "protocol": (
            "positive=alternating, non-overlapping candidate-ID views of the same "
            "query; negative=view B cyclically reassigned to another query"
        ),
        "roc_auc": auc,
        "tpr_at_fpr_le_0_01": float(tpr[index]),
        "threshold": float(thresholds[index]),
        "positive_cosine": summarize(positive),
        "negative_cosine": summarize(negative),
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
    }


def _sha256_ids(ids: np.ndarray) -> str:
    values = np.ascontiguousarray(ids, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _filter_identical(
    rankings: np.ndarray,
    query_ids: Sequence[str] | None,
    corpus_ids: Sequence[str] | None,
    keep: int,
) -> np.ndarray:
    if query_ids is None or corpus_ids is None:
        return rankings[:, :keep].copy()
    corpus = np.asarray([str(x) for x in corpus_ids], dtype=object)
    output = np.empty((len(rankings), keep), dtype=np.int64)
    for row, (query_id, ranked) in enumerate(zip(query_ids, rankings, strict=True)):
        retained = [int(x) for x in ranked if str(corpus[int(x)]) != str(query_id)]
        if len(retained) < keep:
            raise RuntimeError("insufficient results after identical-id filtering")
        output[row] = retained[:keep]
    return output


def evaluate_collection(
    *,
    name: str,
    projected_docs: np.ndarray,
    projected_queries: np.ndarray,
    pq_index: Any,
    ks: Sequence[int],
    relevant_local_ids: Sequence[set[int]],
    exact_backend: str,
    exact_query_batch_size: int,
    exact_doc_chunk_size: int,
    ridge_fraction: float,
    link_seed: int,
    query_ids: Sequence[str] | None = None,
    corpus_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if projected_queries.ndim != 2 or projected_docs.ndim != 2:
        raise ValueError("projected arrays must be matrices")
    if projected_queries.shape[1] != projected_docs.shape[1]:
        raise ValueError("projected query/document dimensions differ")
    if len(relevant_local_ids) != len(projected_queries):
        raise ValueError("relevance sets must align with queries")
    max_k = max(int(x) for x in ks)
    spare = 1 if query_ids is not None and corpus_ids is not None else 0
    started = time.perf_counter()
    _, raw_candidates = pq_index.search(
        np.ascontiguousarray(projected_queries, dtype=np.float32), max_k + spare
    )
    candidates = _filter_identical(
        raw_candidates, query_ids, corpus_ids, max_k
    )
    candidate_ms = (time.perf_counter() - started) * 1000.0
    if np.any(candidates < 0):
        raise RuntimeError("PQ returned an incomplete candidate list")

    reconstructions: dict[tuple[int, str], np.ndarray] = {}
    reconstruction_ms: dict[tuple[int, str], float] = {}
    linkability: dict[str, dict[str, Any]] = {}
    per_query: list[dict[str, Any]] = []
    true_unit = _unit_rows(projected_queries).astype(np.float32)
    for k in ks:
        subset = candidates[:, :k]
        linkability[str(k)] = {}
        for method in METHODS:
            reconstruction_started = time.perf_counter()
            estimated = reconstruct_batch(
                projected_docs, subset, method, ridge_fraction
            )
            reconstruction_ms[(k, method)] = (
                time.perf_counter() - reconstruction_started
            ) * 1000.0
            reconstructions[(k, method)] = estimated
            # Alternating ranks create two non-overlapping identifier views.
            view_a = reconstruct_batch(
                projected_docs, subset[:, 0::2], method, ridge_fraction
            )
            view_b = reconstruct_batch(
                projected_docs, subset[:, 1::2], method, ridge_fraction
            )
            linkability[str(k)][method] = linkability_metrics(
                view_a, view_b, link_seed + k
            )

    labels: list[tuple[str, int | None, str | None]] = [("true", None, None)]
    stacked = [np.asarray(projected_queries, dtype=np.float32)]
    for k in ks:
        for method in METHODS:
            labels.append((f"K{k}:{method}", k, method))
            stacked.append(reconstructions[(k, method)])
    all_queries = np.concatenate(stacked, axis=0)
    retrieve_k = min(100, len(projected_docs) - spare)
    _, all_rankings_raw, exact_ms, resolved_backend = exact_topk_search(
        projected_docs,
        all_queries,
        retrieve_k + spare,
        backend=exact_backend,
        query_batch_size=exact_query_batch_size,
        doc_chunk_size=exact_doc_chunk_size,
    )
    n_queries = len(projected_queries)
    all_rankings = [
        _filter_identical(
            all_rankings_raw[i * n_queries : (i + 1) * n_queries],
            query_ids,
            corpus_ids,
            retrieve_k,
        )
        for i in range(len(labels))
    ]
    true_ranking = all_rankings[0]
    methods: dict[str, Any] = {}
    for label_index, (label, k, method) in enumerate(labels[1:], start=1):
        assert k is not None and method is not None
        estimate = reconstructions[(k, method)]
        cosines = cosine_rows(true_unit, estimate)
        overlap_10 = ranking_overlap(true_ranking, all_rankings[label_index], 10)
        overlap_100 = ranking_overlap(
            true_ranking, all_rankings[label_index], retrieve_k
        )
        key = f"K{k}:{method}"
        methods[key] = {
            "k": int(k),
            "method": method,
            **METHOD_METADATA[method],
            "direction_cosine": summarize(cosines),
            "exact_top10_overlap_fraction": summarize(overlap_10),
            f"exact_top{retrieve_k}_overlap_fraction": summarize(overlap_100),
            "relevance_proxy": relevance_proxy(
                all_rankings[label_index], relevant_local_ids, (10, retrieve_k)
            ),
            "linkability": linkability[str(k)][method],
            "reconstruction_total_ms": reconstruction_ms[(k, method)],
            "reconstruction_mean_ms": reconstruction_ms[(k, method)] / n_queries,
        }
        for row in range(n_queries):
            per_query.append(
                {
                    "collection": name,
                    "query_row": row,
                    "k": int(k),
                    "method": method,
                    "direction_cosine": float(cosines[row]),
                    "exact_top10_overlap_fraction": float(overlap_10[row]),
                    f"exact_top{retrieve_k}_overlap_fraction": float(overlap_100[row]),
                    "candidate_ids_sha256": _sha256_ids(candidates[row, :k]),
                }
            )

    result = {
        "collection": name,
        "query_count": int(n_queries),
        "document_count": int(len(projected_docs)),
        "dimension": int(projected_docs.shape[1]),
        "candidate_ks": [int(x) for x in ks],
        "threat_observation": (
            "ordered candidate identifiers and provider-owned exact candidate "
            "vectors only; no PQ distances, CKKS ciphertexts, plaintext scores, "
            "secret key, or decryptor are accessed"
        ),
        "candidate_search_total_ms": candidate_ms,
        "exact_search_total_ms": exact_ms,
        "exact_search_backend": resolved_backend,
        "true_query_relevance_proxy": relevance_proxy(
            true_ranking, relevant_local_ids, (10, retrieve_k)
        ),
        "methods": methods,
    }
    return result, per_query


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_id_sidecars(directory: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    return (
        _load_json(directory / "query_ids.json"),
        _load_json(directory / "corpus_ids.json"),
        _load_json(directory / "qrels.json"),
    )


def _relevance_from_external_ids(
    selected_query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
) -> list[set[int]]:
    local = {str(doc_id): index for index, doc_id in enumerate(corpus_ids)}
    output: list[set[int]] = []
    for query_id in selected_query_ids:
        relevant = {
            local[str(doc_id)]
            for doc_id, grade in qrels[str(query_id)].items()
            if float(grade) > 0 and str(doc_id) in local
        }
        output.append(relevant)
    return output


def run_million(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import faiss  # type: ignore

    docs = np.load(args.projected_docs, mmap_mode="r", allow_pickle=False)
    raw_queries = np.load(args.queries, mmap_mode="r", allow_pickle=False)
    basis = load_projection_basis(args.basis)
    selected = deterministic_query_split(
        len(raw_queries), args.validation_size, args.split_seed
    )[args.split][: args.max_queries]
    projected_queries = np.ascontiguousarray(
        project_queries(
            np.asarray(raw_queries[selected], dtype=np.float32),
            basis,
            center_online_queries=False,
        ),
        dtype=np.float32,
    )
    all_qrels = deterministic_self_qrels(len(docs), len(raw_queries), args.qrel_seed)
    relevant = [{int(all_qrels[index])} for index in selected]
    index = faiss.read_index(str(args.pq_index))
    if args.faiss_threads > 0:
        faiss.omp_set_num_threads(args.faiss_threads)
    collection, rows = evaluate_collection(
        name="million_vector_heldout",
        projected_docs=docs,
        projected_queries=projected_queries,
        pq_index=index,
        ks=args.k,
        relevant_local_ids=relevant,
        exact_backend=args.exact_backend,
        exact_query_batch_size=args.exact_query_batch_size,
        exact_doc_chunk_size=args.exact_doc_chunk_size,
        ridge_fraction=args.ridge_fraction,
        link_seed=args.link_seed,
    )
    collection["selected_query_rows_sha256"] = _sha256_ids(selected)
    collection["split"] = {
        "name": args.split,
        "validation_size": args.validation_size,
        "seed": args.split_seed,
        "qrel_seed": args.qrel_seed,
    }
    return {"collections": {collection["collection"]: collection}}, rows


def run_beir(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import faiss  # type: ignore

    summary = _load_json(args.graded_summary)
    available = summary["datasets"]
    names = args.dataset or list(available)
    output: dict[str, Any] = {"collections": {}}
    all_rows: list[dict[str, Any]] = []
    for dataset_offset, name in enumerate(names):
        entry = available[name]
        cache = entry["cache"]
        projection_dir = Path(cache["projection_basis"]).parent
        docs = np.load(projection_dir / "corpus.npy", mmap_mode="r", allow_pickle=False)
        all_queries = np.load(
            projection_dir / Path(cache["query_embeddings"]).name,
            mmap_mode="r",
            allow_pickle=False,
        )
        sidecar_dir = Path(cache["data_sidecars"])
        query_ids, corpus_ids, qrels = _load_id_sidecars(sidecar_dir)
        if len(query_ids) != len(all_queries):
            raise ValueError(f"BEIR query sidecar/cache mismatch for {name}")
        confirmatory_path = args.confirmatory_split_dir / f"{name}_confirmatory_qids.txt"
        confirmatory_ids = [
            value.strip()
            for value in confirmatory_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        query_row = {str(query_id): row for row, query_id in enumerate(query_ids)}
        unavailable = [query_id for query_id in confirmatory_ids if query_id not in query_row]
        evaluable = [query_id for query_id in confirmatory_ids if query_id in query_row]
        if args.max_queries is not None:
            rng = np.random.default_rng(args.beir_query_seed + dataset_offset)
            order = rng.permutation(len(evaluable))[: min(args.max_queries, len(evaluable))]
            evaluable = [evaluable[int(index)] for index in order]
        selected = np.asarray([query_row[query_id] for query_id in evaluable], dtype=np.int64)
        selected_queries = np.ascontiguousarray(all_queries[selected], dtype=np.float32)
        selected_query_ids = [str(query_ids[int(index)]) for index in selected]
        relevant = _relevance_from_external_ids(
            selected_query_ids, corpus_ids, qrels
        )
        index = faiss.read_index(str(cache["pq_index"]))
        if args.faiss_threads > 0:
            faiss.omp_set_num_threads(args.faiss_threads)
        collection, rows = evaluate_collection(
            name=f"beir_{name}",
            projected_docs=docs,
            projected_queries=selected_queries,
            pq_index=index,
            ks=args.k,
            relevant_local_ids=relevant,
            exact_backend=args.exact_backend,
            exact_query_batch_size=args.exact_query_batch_size,
            exact_doc_chunk_size=args.exact_doc_chunk_size,
            ridge_fraction=args.ridge_fraction,
            link_seed=args.link_seed + dataset_offset,
            query_ids=selected_query_ids,
            corpus_ids=corpus_ids,
        )
        collection["selected_query_rows_sha256"] = _sha256_ids(selected)
        collection["confirmatory_split"] = {
            "path": str(confirmatory_path.resolve()),
            "sha256": hashlib.sha256(confirmatory_path.read_bytes()).hexdigest(),
            "requested_query_count": len(confirmatory_ids),
            "evaluable_query_count": len(selected_query_ids),
            "unavailable_query_ids": unavailable,
            "selection": (
                "all strict confirmatory IDs"
                if args.max_queries is None
                else f"deterministic subset of at most {args.max_queries} strict confirmatory IDs"
            ),
        }
        output["collections"][collection["collection"]] = collection
        all_rows.extend(rows)
    return output, all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--k", type=int, nargs="+", default=[20, 50, 100, 200])
        target.add_argument("--ridge-fraction", type=float, default=0.10)
        target.add_argument("--link-seed", type=int, default=2026)
        target.add_argument("--faiss-threads", type=int, default=1)
        target.add_argument(
            "--exact-backend", choices=["auto", "numpy", "cuda"], default="cuda"
        )
        target.add_argument("--exact-query-batch-size", type=int, default=128)
        target.add_argument("--exact-doc-chunk-size", type=int, default=16384)
        target.add_argument("--output", type=Path, required=True)
        target.add_argument("--per-query-output", type=Path)

    million = subparsers.add_parser("million", help="million-vector held-out audit")
    common(million)
    million.add_argument("--projected-docs", type=Path, required=True)
    million.add_argument("--queries", type=Path, required=True)
    million.add_argument("--basis", type=Path, required=True)
    million.add_argument("--pq-index", type=Path, required=True)
    million.add_argument("--split", choices=["validation", "test"], default="test")
    million.add_argument("--validation-size", type=int, default=100)
    million.add_argument("--split-seed", type=int, default=2026)
    million.add_argument("--qrel-seed", type=int, default=42)
    million.add_argument("--max-queries", type=int, default=400)

    beir = subparsers.add_parser("beir", help="audit cached BEIR collections")
    common(beir)
    beir.add_argument("--graded-summary", type=Path, required=True)
    beir.add_argument("--dataset", nargs="+")
    beir.add_argument("--beir-query-seed", type=int, default=2026)
    beir.add_argument("--confirmatory-split-dir", type=Path, required=True)
    beir.add_argument("--max-queries", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.max_queries is not None and args.max_queries < 3) or any(
        k < 4 for k in args.k
    ):
        raise ValueError("max_queries must be >=3 and every K must be >=4")
    started = time.perf_counter()
    result, per_query = run_million(args) if args.mode == "million" else run_beir(args)
    result.update(
        {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "methods": METHOD_METADATA,
            "elapsed_seconds": time.perf_counter() - started,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.per_query_output:
        args.per_query_output.parent.mkdir(parents=True, exist_ok=True)
        args.per_query_output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_query),
            encoding="utf-8",
        )
    compact = {
        name: {
            key: {
                "cosine_mean": value["direction_cosine"]["mean"],
                "top10_overlap_mean": value["exact_top10_overlap_fraction"]["mean"],
                "link_auc": value["linkability"]["roc_auc"],
            }
            for key, value in collection["methods"].items()
        }
        for name, collection in result["collections"].items()
    }
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
