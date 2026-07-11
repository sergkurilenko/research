"""Audit exact and near-exact document-vector disclosure baselines.

This script consumes the frozen test-query log from ``retrieval_bench.py``;
it never regenerates or changes PQ candidate IDs.  For every query it really
serializes a server response to ``bytes``, parses that response client-side,
and reranks the fixed candidates.  Five disclosure baselines are evaluated:

* ``projected_float32_return``: exact projected float32 vectors;
* ``projected_float16_return``: near-exact projected float16 vectors;
* ``projected_int8_symmetric_return``: per-vector symmetric int8 projected
  vectors plus one little-endian float32 scale per vector;
* ``raw_float32_return``: exact raw float32 document vectors;
* ``raw_float16_return``: near-exact raw float16 document vectors.

These methods disclose exact or near-exact document representations.  They
are dominance/bandwidth baselines, **not privacy protocols**.  The client
already has its raw query and the frozen PQ candidate IDs; application payload
therefore consists of candidate IDs in the request and document vectors in the
response.  HTTP/TLS framing is deliberately excluded.

Projected reranking uses the deployable uncentred online query ``q @ V_k`` and
the cached exact document projection ``(d - passage_mean) @ V_k``.  Numerical
error and top-10 stability are paired against the corresponding projected or
raw float32 reference on the same query/candidates.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_CACHE = Path(
    os.environ.get(
        "PACKRERANK_CACHE",
        str(WORKSPACE / "cache" / "million_vector"),
    )
)
DEFAULT_RESULTS = WORKSPACE / "results" / "system_revision"
DEFAULT_FROZEN_LOG = DEFAULT_RESULTS / "retrieval_e5base_k672_K100_test.jsonl"
DEFAULT_BASIS = DEFAULT_RESULTS / "cache" / "e5base_k672_basis.npz"
DEFAULT_PROJECTED_DOCS = (
    DEFAULT_RESULTS / "cache" / "E_docs_e5base_proj672_1000000.npy"
)
DEFAULT_RAW_DOCS = SOURCE_CACHE / "E_docs_e5base_1000000.npy"
DEFAULT_RAW_QUERIES = SOURCE_CACHE / "E_queries_e5base_self_q500.npy"
DEFAULT_SUMMARY = DEFAULT_RESULTS / "vector_return_baselines_e5base_K100_test.json"
DEFAULT_QUERY_LOG = (
    DEFAULT_RESULTS / "vector_return_baselines_e5base_K100_test.jsonl"
)


METHODS = (
    "projected_float32_return",
    "projected_float16_return",
    "projected_int8_symmetric_return",
    "raw_float32_return",
    "raw_float16_return",
)

METHOD_SPEC: dict[str, dict[str, str]] = {
    "projected_float32_return": {
        "space": "projected",
        "encoding": "float32",
        "reference": "projected_float32_return",
        "disclosure": "exact vector disclosure baseline; not a privacy protocol",
    },
    "projected_float16_return": {
        "space": "projected",
        "encoding": "float16",
        "reference": "projected_float32_return",
        "disclosure": "near-exact vector disclosure baseline; not a privacy protocol",
    },
    "projected_int8_symmetric_return": {
        "space": "projected",
        "encoding": "symmetric_int8_per_vector",
        "reference": "projected_float32_return",
        "disclosure": "near-exact vector disclosure baseline; not a privacy protocol",
    },
    "raw_float32_return": {
        "space": "raw",
        "encoding": "float32",
        "reference": "raw_float32_return",
        "disclosure": "exact vector disclosure baseline; not a privacy protocol",
    },
    "raw_float16_return": {
        "space": "raw",
        "encoding": "float16",
        "reference": "raw_float32_return",
        "disclosure": "near-exact vector disclosure baseline; not a privacy protocol",
    },
}


@dataclass(frozen=True)
class FrozenQuery:
    query_index: int
    ground_truth_id: int
    candidate_ids: np.ndarray
    split: str = "test"


@dataclass
class AuditConfig:
    shortlist_k: int = 100
    top_k: int = 10
    expected_queries: int = 400
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    validate_projection_vectors: int = 16
    validation_atol: float = 5e-5

    def validate(self) -> None:
        if not 1 <= self.top_k <= self.shortlist_k:
            raise ValueError("Expected 1 <= top_k <= shortlist_k")
        if self.expected_queries < 1:
            raise ValueError("expected_queries must be positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if self.validate_projection_vectors < 1:
            raise ValueError("validate_projection_vectors must be positive")


@dataclass
class MethodExecution:
    method: str
    scores: np.ndarray
    predictions: np.ndarray
    latency_ms: dict[str, float]
    payload_bytes: dict[str, int]


def load_float32_memmap(path: Path) -> np.memmap:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap) or array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional NPY memmap: {path}")
    if array.dtype != np.float32:
        raise ValueError(f"Expected float32 at {path}, got {array.dtype}")
    return array


def load_basis(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    data = np.load(path, allow_pickle=False)
    required = {"passage_mean", "vectors"}
    if not required.issubset(data.files):
        raise ValueError(f"Basis must contain {sorted(required)}: {path}")
    mean = np.asarray(data["passage_mean"], dtype=np.float32)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    if mean.ndim == 1:
        mean = mean.reshape(1, -1)
    if mean.shape != (1, vectors.shape[0]):
        raise ValueError("Basis mean/vectors shape mismatch")

    def scalar(name: str, default: int) -> int:
        return int(np.asarray(data[name]).item()) if name in data.files else default

    metadata = {
        "fit_sample_size": scalar("fit_sample_size", -1),
        "fit_sample_seed": scalar("fit_sample_seed", -1),
        "svd_seed": scalar("svd_seed", -1),
    }
    return mean, vectors, metadata


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = sha256_file(path)
    return result


def load_frozen_queries(
    path: Path,
    shortlist_k: int,
    expected_queries: int,
    *,
    expected_split: str = "test",
    max_queries: int | None = None,
) -> list[FrozenQuery]:
    """Load and cross-check candidates repeated across methods in a JSONL log."""

    order: list[int] = []
    entries: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_index = int(record["query_index"])
            ground_truth = int(record["ground_truth_id"])
            split = str(record["split"])
            if split != expected_split:
                raise ValueError(
                    f"Unexpected split {split!r} at line {line_number}; "
                    f"expected {expected_split!r}"
                )
            if query_index not in entries:
                order.append(query_index)
                entries[query_index] = {
                    "ground_truth_id": ground_truth,
                    "split": split,
                    "candidate_ids": None,
                }
            entry = entries[query_index]
            if entry["ground_truth_id"] != ground_truth:
                raise ValueError(f"Inconsistent ground truth for query {query_index}")
            if "candidate_ids" in record:
                candidates = np.asarray(record["candidate_ids"], dtype=np.int64)
                if candidates.shape != (shortlist_k,):
                    raise ValueError(
                        f"Query {query_index} has {len(candidates)} candidates; "
                        f"expected {shortlist_k}"
                    )
                previous = entry["candidate_ids"]
                if previous is None:
                    entry["candidate_ids"] = candidates
                elif not np.array_equal(previous, candidates):
                    raise ValueError(
                        f"Candidate IDs differ across frozen methods for query {query_index}"
                    )

    if len(order) != expected_queries:
        raise ValueError(f"Frozen log has {len(order)} queries; expected {expected_queries}")
    frozen: list[FrozenQuery] = []
    for query_index in order:
        entry = entries[query_index]
        candidates = entry["candidate_ids"]
        if candidates is None:
            raise ValueError(f"No candidate IDs for query {query_index}")
        if len(np.unique(candidates)) != shortlist_k:
            raise ValueError(f"Duplicate candidate IDs for query {query_index}")
        frozen.append(
            FrozenQuery(
                query_index=query_index,
                ground_truth_id=entry["ground_truth_id"],
                candidate_ids=candidates,
                split=entry["split"],
            )
        )
    if max_queries is not None:
        if not 1 <= max_queries <= len(frozen):
            raise ValueError("max_queries must fit the frozen query set")
        frozen = frozen[:max_queries]
    return frozen


def frozen_candidates_sha256(queries: Sequence[FrozenQuery]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(np.asarray(query.query_index, dtype="<i8").tobytes())
        digest.update(np.asarray(query.ground_truth_id, dtype="<i8").tobytes())
        digest.update(
            np.asarray(query.candidate_ids, dtype="<i8").tobytes(order="C")
        )
    return digest.hexdigest()


def encode_vector_payload(vectors: np.ndarray, encoding: str) -> bytes:
    """Encode a response payload with no hidden Python-object accounting.

    The candidate count and vector dimension are fixed by the request/config,
    so float payloads need no per-response header.  The int8 layout is
    ``K*d int8 codes || K little-endian float32 scales``.
    """

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a matrix")
    if encoding == "float32":
        return vectors.astype("<f4", copy=False).tobytes(order="C")
    if encoding == "float16":
        return vectors.astype("<f2").tobytes(order="C")
    if encoding == "symmetric_int8_per_vector":
        maxima = np.max(np.abs(vectors), axis=1)
        scales = np.where(maxima > 0.0, maxima / 127.0, 1.0).astype("<f4")
        codes = np.rint(vectors / scales[:, None])
        codes = np.clip(codes, -127, 127).astype(np.int8)
        return codes.tobytes(order="C") + scales.tobytes(order="C")
    raise ValueError(f"Unsupported encoding: {encoding}")


def decode_vector_payload(
    payload: bytes, encoding: str, n_vectors: int, dimension: int
) -> np.ndarray:
    if encoding == "float32":
        expected = n_vectors * dimension * 4
        if len(payload) != expected:
            raise ValueError(f"float32 payload has {len(payload)} bytes, expected {expected}")
        return np.frombuffer(payload, dtype="<f4").reshape(n_vectors, dimension)
    if encoding == "float16":
        expected = n_vectors * dimension * 2
        if len(payload) != expected:
            raise ValueError(f"float16 payload has {len(payload)} bytes, expected {expected}")
        encoded = np.frombuffer(payload, dtype="<f2").reshape(n_vectors, dimension)
        return encoded.astype(np.float32)
    if encoding == "symmetric_int8_per_vector":
        code_bytes = n_vectors * dimension
        expected = code_bytes + n_vectors * 4
        if len(payload) != expected:
            raise ValueError(f"int8 payload has {len(payload)} bytes, expected {expected}")
        codes = np.frombuffer(payload, dtype=np.int8, count=code_bytes).reshape(
            n_vectors, dimension
        )
        scales = np.frombuffer(
            payload, dtype="<f4", count=n_vectors, offset=code_bytes
        )
        return codes.astype(np.float32) * scales[:, None]
    raise ValueError(f"Unsupported encoding: {encoding}")


def execute_method(
    method: str,
    candidate_ids: np.ndarray,
    query_vector: np.ndarray,
    source_vectors: np.ndarray,
    top_k: int,
) -> MethodExecution:
    spec = METHOD_SPEC[method]
    encoding = spec["encoding"]
    candidates = np.asarray(candidate_ids, dtype=np.int64)
    request_payload = candidates.astype("<u8", copy=False).tobytes(order="C")

    started = time.perf_counter_ns()
    gathered = np.ascontiguousarray(source_vectors[candidates], dtype=np.float32)
    gather_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    started = time.perf_counter_ns()
    response_payload = encode_vector_payload(gathered, encoding)
    encode_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    started = time.perf_counter_ns()
    received = decode_vector_payload(
        response_payload, encoding, len(candidates), source_vectors.shape[1]
    )
    parse_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    started = time.perf_counter_ns()
    scores = np.asarray(received @ np.asarray(query_vector, dtype=np.float32))
    order = np.argsort(-scores, kind="stable")[:top_k]
    predictions = candidates[order]
    rerank_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    return MethodExecution(
        method=method,
        scores=scores.astype(np.float32, copy=False),
        predictions=predictions,
        latency_ms={
            "server_gather_ms": gather_ms,
            "server_encode_serialize_ms": encode_ms,
            "client_parse_dequantize_ms": parse_ms,
            "client_rerank_ms": rerank_ms,
            "end_to_end_compute_ms": gather_ms + encode_ms + parse_ms + rerank_ms,
        },
        payload_bytes={
            "request_candidate_ids_bytes": len(request_payload),
            "response_vector_bytes": len(response_payload),
            "total_application_bytes": len(request_payload) + len(response_payload),
        },
    )


def ranking_metrics(predictions: np.ndarray, ground_truth_id: int) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.int64)
    positions = np.flatnonzero(predictions == ground_truth_id)
    rank = int(positions[0]) + 1 if len(positions) else None
    return {
        "hit_at_1": float(rank == 1),
        f"hit_at_{len(predictions)}": float(rank is not None),
        f"mrr_at_{len(predictions)}": 0.0 if rank is None else 1.0 / rank,
        f"ndcg_at_{len(predictions)}": (
            0.0 if rank is None else 1.0 / np.log2(rank + 1.0)
        ),
    }


def numerical_comparison(
    execution: MethodExecution, reference: MethodExecution
) -> dict[str, float | bool | str]:
    error = np.asarray(execution.scores, dtype=np.float64) - np.asarray(
        reference.scores, dtype=np.float64
    )
    reference_norm = float(np.linalg.norm(reference.scores.astype(np.float64)))
    predicted = np.asarray(execution.predictions)
    reference_predicted = np.asarray(reference.predictions)
    return {
        "reference_method": reference.method,
        "score_mae": float(np.mean(np.abs(error))),
        "score_rmse": float(np.sqrt(np.mean(error * error))),
        "score_max_abs": float(np.max(np.abs(error))),
        "score_relative_l2": float(
            np.linalg.norm(error) / max(reference_norm, np.finfo(np.float64).eps)
        ),
        "top1_match": bool(predicted[0] == reference_predicted[0]),
        "top10_set_match": bool(set(predicted) == set(reference_predicted)),
        "top10_exact_order_match": bool(np.array_equal(predicted, reference_predicted)),
    }


def validate_projected_cache(
    raw_docs: np.ndarray,
    projected_docs: np.ndarray,
    passage_mean: np.ndarray,
    basis_vectors: np.ndarray,
    frozen: Sequence[FrozenQuery],
    n_vectors: int,
    atol: float,
) -> dict[str, Any]:
    ids = np.unique(np.concatenate([q.candidate_ids for q in frozen]))[:n_vectors]
    expected = (
        (np.asarray(raw_docs[ids], dtype=np.float32) - passage_mean) @ basis_vectors
    ).astype(np.float32)
    observed = np.asarray(projected_docs[ids], dtype=np.float32)
    difference = np.abs(expected - observed)
    max_abs = float(np.max(difference))
    if not np.allclose(expected, observed, rtol=1e-4, atol=atol):
        raise ValueError(
            f"Projected cache does not match (raw - mean) @ V; max abs={max_abs:.3g}"
        )
    return {
        "checked_vectors": int(len(ids)),
        "max_abs_error": max_abs,
        "atol": atol,
        "rtol": 1e-4,
    }


def bootstrap_mean_ci(
    values: np.ndarray, sample_indices: np.ndarray, confidence: float = 0.95
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty vector")
    bootstrap = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(bootstrap, alpha)),
        "ci_high": float(np.quantile(bootstrap, 1.0 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": int(len(sample_indices)),
    }


def latency_summary(values: np.ndarray, sample_indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        **bootstrap_mean_ci(values, sample_indices),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _records_to_method_arrays(
    records: Sequence[Mapping[str, Any]], method: str
) -> dict[str, dict[str, np.ndarray]]:
    selected = [record for record in records if record["method"] == method]
    if not selected:
        raise ValueError(f"No records for {method}")
    metric_keys = selected[0]["metrics"].keys()
    latency_keys = selected[0]["latency_ms"].keys()
    numerical_keys = (
        "score_mae",
        "score_rmse",
        "score_max_abs",
        "score_relative_l2",
        "top1_match",
        "top10_set_match",
        "top10_exact_order_match",
    )
    return {
        "metrics": {
            key: np.asarray([row["metrics"][key] for row in selected], dtype=np.float64)
            for key in metric_keys
        },
        "latency": {
            key: np.asarray(
                [row["latency_ms"][key] for row in selected], dtype=np.float64
            )
            for key in latency_keys
        },
        "numerical": {
            key: np.asarray(
                [row["numerical_vs_float32"][key] for row in selected],
                dtype=np.float64,
            )
            for key in numerical_keys
        },
        "payload": {
            key: np.asarray(
                [row["payload_bytes"][key] for row in selected], dtype=np.int64
            )
            for key in selected[0]["payload_bytes"]
        },
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    config: AuditConfig,
    frozen: Sequence[FrozenQuery],
    input_metadata: Mapping[str, Any],
    projection_validation: Mapping[str, Any],
    basis_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    n_queries = len(frozen)
    sample_indices = np.random.default_rng(config.bootstrap_seed).integers(
        0,
        n_queries,
        size=(config.bootstrap_samples, n_queries),
        dtype=np.int64,
    )
    arrays = {method: _records_to_method_arrays(records, method) for method in METHODS}
    methods: dict[str, Any] = {}
    for method in METHODS:
        values = arrays[method]
        methods[method] = {
            "space": METHOD_SPEC[method]["space"],
            "wire_encoding": METHOD_SPEC[method]["encoding"],
            "float32_reference": METHOD_SPEC[method]["reference"],
            "disclosure_class": METHOD_SPEC[method]["disclosure"],
            "metrics": {
                key: bootstrap_mean_ci(vector, sample_indices)
                for key, vector in values["metrics"].items()
            },
            "numerical_vs_corresponding_float32": {
                key: bootstrap_mean_ci(vector, sample_indices)
                for key, vector in values["numerical"].items()
            },
            "latency_ms": {
                key: latency_summary(vector, sample_indices)
                for key, vector in values["latency"].items()
            },
            "payload_bytes": {
                key: {
                    "mean": float(vector.mean()),
                    "min": int(vector.min()),
                    "max": int(vector.max()),
                }
                for key, vector in values["payload"].items()
            },
        }

    paired: dict[str, Any] = {}
    comparisons = {
        "projected_float16_minus_projected_float32": (
            "projected_float16_return",
            "projected_float32_return",
        ),
        "projected_int8_minus_projected_float32": (
            "projected_int8_symmetric_return",
            "projected_float32_return",
        ),
        "raw_float16_minus_raw_float32": (
            "raw_float16_return",
            "raw_float32_return",
        ),
        "raw_float32_minus_projected_float32": (
            "raw_float32_return",
            "projected_float32_return",
        ),
    }
    for label, (method, reference) in comparisons.items():
        paired[label] = {
            key: bootstrap_mean_ci(
                arrays[method]["metrics"][key] - arrays[reference]["metrics"][key],
                sample_indices,
            )
            for key in arrays[method]["metrics"]
        }

    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit_type": "exact_and_near_exact_vector_disclosure_baselines",
        "privacy_statement": (
            "All five methods disclose exact or near-exact document vectors. "
            "They are bandwidth/utility dominance baselines, not privacy protocols."
        ),
        "protocol": {
            "candidate_source": "frozen test JSONL; candidates never regenerated",
            "request": "K little-endian uint64 candidate IDs",
            "response": "serialized document vectors in the method wire encoding",
            "projected_query": "q @ V_k (uncentred deployable protocol)",
            "projected_documents": "cached (document - passage_mean) @ V_k",
            "payload_scope": "application bytes only; HTTP/TLS framing excluded",
            "latency_scope": (
                "warm-process local gather, real in-memory encode/serialize, real "
                "parse/dequantize, and client rerank; network transfer excluded"
            ),
            "int8_layout": "K*d int8 codes followed by K little-endian float32 scales",
        },
        "config": {
            **dataclasses.asdict(config),
            "actual_queries": n_queries,
            "split": "test",
        },
        "frozen_candidates_sha256": frozen_candidates_sha256(frozen),
        "basis": dict(basis_metadata),
        "projection_cache_validation": dict(projection_validation),
        "inputs": dict(input_metadata),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "methods": methods,
        "paired_query_bootstrap_metric_deltas": paired,
    }


def run_audit(
    frozen: Sequence[FrozenQuery],
    raw_queries: np.ndarray,
    raw_docs: np.ndarray,
    projected_docs: np.ndarray,
    basis_vectors: np.ndarray,
    config: AuditConfig,
) -> list[dict[str, Any]]:
    """Execute all wire encodings for the frozen queries."""

    records: list[dict[str, Any]] = []
    for ordinal, frozen_query in enumerate(frozen):
        query_index = frozen_query.query_index
        raw_query = np.asarray(raw_queries[query_index], dtype=np.float32)
        projected_query = np.asarray(raw_query @ basis_vectors, dtype=np.float32)
        executions: dict[str, MethodExecution] = {}
        # References first: their scores define paired numerical error.
        for method in METHODS:
            spec = METHOD_SPEC[method]
            source = projected_docs if spec["space"] == "projected" else raw_docs
            query = projected_query if spec["space"] == "projected" else raw_query
            executions[method] = execute_method(
                method,
                frozen_query.candidate_ids,
                query,
                source,
                config.top_k,
            )

        for method in METHODS:
            execution = executions[method]
            reference = executions[METHOD_SPEC[method]["reference"]]
            records.append(
                {
                    "query_ordinal": ordinal,
                    "query_index": query_index,
                    "split": frozen_query.split,
                    "ground_truth_id": frozen_query.ground_truth_id,
                    "method": method,
                    "space": METHOD_SPEC[method]["space"],
                    "wire_encoding": METHOD_SPEC[method]["encoding"],
                    "disclosure_class": METHOD_SPEC[method]["disclosure"],
                    "is_privacy_protocol": False,
                    "candidate_ids": [int(x) for x in frozen_query.candidate_ids],
                    "predictions": [int(x) for x in execution.predictions],
                    "metrics": ranking_metrics(
                        execution.predictions, frozen_query.ground_truth_id
                    ),
                    "numerical_vs_float32": numerical_comparison(
                        execution, reference
                    ),
                    "latency_ms": execution.latency_ms,
                    "payload_bytes": execution.payload_bytes,
                }
            )
    return records


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def print_summary(summary: Mapping[str, Any]) -> None:
    print(summary["privacy_statement"])
    top_k = int(summary["config"]["top_k"])
    for method in METHODS:
        item = summary["methods"][method]
        metrics = item["metrics"]
        numerical = item["numerical_vs_corresponding_float32"]
        payload = item["payload_bytes"]["response_vector_bytes"]["mean"]
        latency = item["latency_ms"]["end_to_end_compute_ms"]["p50"]
        print(
            f"{method:36s} Hit@1={metrics['hit_at_1']['mean']:.4f} "
            f"Hit@{top_k}={metrics[f'hit_at_{top_k}']['mean']:.4f} "
            f"MRR={metrics[f'mrr_at_{top_k}']['mean']:.4f} "
            f"order-match={numerical['top10_exact_order_match']['mean']:.4f} "
            f"payload={payload / 1024:.2f} KiB p50={latency:.3f} ms"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact/near-exact vector-return disclosure baseline audit"
    )
    parser.add_argument("--frozen-query-log", type=Path, default=DEFAULT_FROZEN_LOG)
    parser.add_argument("--raw-queries", type=Path, default=DEFAULT_RAW_QUERIES)
    parser.add_argument("--raw-docs", type=Path, default=DEFAULT_RAW_DOCS)
    parser.add_argument("--projected-docs", type=Path, default=DEFAULT_PROJECTED_DOCS)
    parser.add_argument("--basis", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--query-log-jsonl", type=Path, default=DEFAULT_QUERY_LOG)
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--expected-queries", type=int, default=400)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--validate-projection-vectors", type=int, default=16)
    parser.add_argument("--validation-atol", type=float, default=5e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AuditConfig(
        shortlist_k=args.shortlist_k,
        top_k=args.top_k,
        expected_queries=args.expected_queries,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        validate_projection_vectors=args.validate_projection_vectors,
        validation_atol=args.validation_atol,
    )
    config.validate()
    frozen = load_frozen_queries(
        args.frozen_query_log,
        config.shortlist_k,
        config.expected_queries,
        max_queries=args.max_queries,
    )
    raw_queries = load_float32_memmap(args.raw_queries)
    raw_docs = load_float32_memmap(args.raw_docs)
    projected_docs = load_float32_memmap(args.projected_docs)
    passage_mean, basis_vectors, basis_recipe = load_basis(args.basis)
    if raw_docs.shape[0] != projected_docs.shape[0]:
        raise ValueError("Raw/projected document counts differ")
    if raw_docs.shape[1] != raw_queries.shape[1]:
        raise ValueError("Raw query/document dimensions differ")
    if basis_vectors.shape != (raw_docs.shape[1], projected_docs.shape[1]):
        raise ValueError("Basis does not map raw to projected dimensions")
    all_candidates = np.concatenate([query.candidate_ids for query in frozen])
    if np.min(all_candidates) < 0 or np.max(all_candidates) >= len(raw_docs):
        raise ValueError("Frozen candidate ID is outside the document arrays")
    if max(query.query_index for query in frozen) >= len(raw_queries):
        raise ValueError("Frozen query index is outside the query array")

    projection_validation = validate_projected_cache(
        raw_docs,
        projected_docs,
        passage_mean,
        basis_vectors,
        frozen,
        config.validate_projection_vectors,
        config.validation_atol,
    )
    started = time.perf_counter()
    records = run_audit(
        frozen, raw_queries, raw_docs, projected_docs, basis_vectors, config
    )
    elapsed = time.perf_counter() - started
    input_metadata = {
        "frozen_query_log": file_metadata(args.frozen_query_log, include_sha256=True),
        "raw_queries": file_metadata(args.raw_queries),
        "raw_documents": file_metadata(args.raw_docs),
        "projected_documents": file_metadata(args.projected_docs),
        "basis": file_metadata(args.basis, include_sha256=True),
    }
    basis_metadata = {
        **basis_recipe,
        "raw_dimension": int(basis_vectors.shape[0]),
        "projected_dimension": int(basis_vectors.shape[1]),
    }
    summary = build_summary(
        records,
        config,
        frozen,
        input_metadata,
        projection_validation,
        basis_metadata,
    )
    summary["wall_elapsed_seconds"] = elapsed
    write_json(args.summary_json, summary)
    write_jsonl(args.query_log_jsonl, records)
    print_summary(summary)
    print(f"summary: {args.summary_json.resolve()}")
    print(f"query log: {args.query_log_jsonl.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
