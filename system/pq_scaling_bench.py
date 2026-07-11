"""Measure client-side FAISS PQ scan scaling using prefix replicas of one index.

Only latency and serialized index size are compared across ``N``.  Prefix
indices are cloned from the same trained codebook and truncated to the first
``N`` codes, so no retraining or corpus-dependent utility comparison is mixed
into the scaling result.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from system.retrieval_bench import load_projection_basis, project_queries


SCHEMA = "pq_scan_scaling.v1"


def prefix_index(index: Any, n_docs: int, faiss_module: Any) -> Any:
    """Clone an IndexPQ and retain its first ``n_docs`` encoded rows."""

    if not 1 <= n_docs <= int(index.ntotal):
        raise ValueError("n_docs must be within the source index")
    if not hasattr(index, "codes") or not hasattr(index, "code_size"):
        raise TypeError("source must expose IndexPQ codes")
    clone = faiss_module.clone_index(index)
    codes = faiss_module.vector_to_array(clone.codes)
    needed = int(n_docs) * int(clone.code_size)
    faiss_module.copy_array_to_vector(codes[:needed], clone.codes)
    clone.ntotal = int(n_docs)
    if faiss_module.vector_to_array(clone.codes).size != needed:
        raise RuntimeError("truncated index code length is inconsistent")
    return clone


def timing_summary(samples: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size < 1:
        raise ValueError("at least one timing sample is required")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "samples": [float(x) for x in values],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FAISS is required") from exc

    if args.repeats < 1 or args.warmups < 0 or args.max_queries < 1:
        raise ValueError("repeats/max_queries must be positive; warmups non-negative")
    source = faiss.read_index(str(args.pq_index))
    basis = load_projection_basis(args.basis)
    queries = np.load(args.queries, mmap_mode="r", allow_pickle=False)
    selected = np.asarray(queries[: min(len(queries), args.max_queries)], dtype=np.float32)
    projected = np.ascontiguousarray(
        project_queries(selected, basis, center_online_queries=False), dtype=np.float32
    )
    if projected.shape[1] != int(source.d):
        raise ValueError("query projection and PQ dimensions differ")
    if args.faiss_threads > 0:
        faiss.omp_set_num_threads(args.faiss_threads)

    requested = sorted(set(int(x) for x in args.n_docs))
    if requested[-1] > int(source.ntotal):
        raise ValueError("a requested scale exceeds source ntotal")
    points: list[dict[str, Any]] = []
    for n_docs in requested:
        index = prefix_index(source, n_docs, faiss)
        for i in range(min(args.warmups, len(projected))):
            index.search(projected[i : i + 1], args.shortlist_k)
        samples: list[float] = []
        checksums: list[int] = []
        for repeat in range(args.repeats):
            order = np.random.default_rng(args.seed + repeat).permutation(len(projected))
            for query_id in order:
                start = time.perf_counter_ns()
                _, ids = index.search(
                    projected[query_id : query_id + 1], args.shortlist_k
                )
                samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
                checksums.append(int(np.sum(ids, dtype=np.int64)))
        serialized_bytes = int(np.asarray(faiss.serialize_index(index)).nbytes)
        points.append(
            {
                "n_docs": n_docs,
                "serialized_index_bytes": serialized_bytes,
                "code_bytes_per_document": int(index.code_size),
                "search_ms": timing_summary(samples),
                "result_checksum": int(sum(checksums) % (2**63 - 1)),
            }
        )
        del index

    n = np.asarray([point["n_docs"] for point in points], dtype=np.float64)
    medians = np.asarray([point["search_ms"]["p50"] for point in points])
    slope, intercept = np.polyfit(n, medians, deg=1)
    predicted = slope * n + intercept
    ss_res = float(np.sum((medians - predicted) ** 2))
    ss_tot = float(np.sum((medians - np.mean(medians)) ** 2))
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "single-query client-side exhaustive IndexPQ scan; index cloning, "
            "deserialization, projection, and network time excluded"
        ),
        "config": {
            "source_ntotal": int(source.ntotal),
            "dimension": int(source.d),
            "shortlist_k": int(args.shortlist_k),
            "query_count": int(len(projected)),
            "repeats": int(args.repeats),
            "warmups": int(args.warmups),
            "seed": int(args.seed),
            "faiss_threads_requested": int(args.faiss_threads),
            "n_docs": requested,
        },
        "points": points,
        "linear_fit_median_ms_vs_n_docs": {
            "slope_ms_per_document": float(slope),
            "intercept_ms": float(intercept),
            "r_squared": None if ss_tot == 0 else float(1.0 - ss_res / ss_tot),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "platform": platform.platform(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-index", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--n-docs", type=int, nargs="+", default=[100_000, 250_000, 500_000, 1_000_000]
    )
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--faiss-threads", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = [
        {"n_docs": p["n_docs"], "p50_ms": p["search_ms"]["p50"], "p95_ms": p["search_ms"]["p95"]}
        for p in result["points"]
    ]
    print(json.dumps(compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
