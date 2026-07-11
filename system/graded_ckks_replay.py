"""Replay graded BEIR PQ shortlists through the real spawned CKKS server.

This harness consumes artifacts emitted by :mod:`system.graded_ir_bench`.  It
does not repeat encoding, projection, or PQ search.  Instead it takes each
``pq_only`` top-100 list as the frozen client-side shortlist and replays that
request through :class:`system.unified_ckks_bench.SpawnedCKKSServer` using the
``segmented_score_packed`` kernel.  The server owns the exact projected corpus
memmap and a public-only CKKS context; the client owns the secret context and
the already projected query.

The recorded ``pq_projected_rerank`` output is the paired plaintext reference.
An untimed post-protocol audit recomputes candidate scores to report CKKS error
and rank-order agreement.  It is benchmark instrumentation, not a production
protocol step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from system.graded_ir_bench import (
    METRIC_NAMES,
    canonical_dataset_name,
    evaluate_rankings,
    load_projection_basis,
    projection_fingerprint,
    summarise_method_metrics,
)
from system.packed_ckks import (
    CKKSParameters,
    EncryptedScoreResponse,
    client_decrypt_scores,
    create_context_pair,
    encrypt_query,
)
from system.unified_ckks_bench import SpawnedCKKSServer


SCHEMA = "graded_ckks_replay.v1"
CKKS_METHOD = "segmented_score_packed"


@dataclass(frozen=True)
class ReplayConfig:
    dataset: str
    max_queries: int | None = None
    selection_seed: int = 2026
    warmup_queries: int = 1
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    response_timeout_s: float = 600.0

    def validate(self) -> None:
        canonical_dataset_name(self.dataset)
        if self.max_queries is not None and self.max_queries < 1:
            raise ValueError("max_queries must be positive")
        if self.warmup_queries < 0:
            raise ValueError("warmup_queries cannot be negative")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")


@dataclass(frozen=True)
class ReplayArtifacts:
    dataset_summary: Mapping[str, Any]
    data_sidecars: Path
    corpus_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    qrels: Mapping[str, Mapping[str, float]]
    projection_basis: Path
    projected_corpus: Path
    projected_queries: Path
    pq_candidates: Mapping[str, tuple[str, ...]]
    plaintext_rankings: Mapping[str, tuple[str, ...]]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at line {line_number} is not an object")
            records.append(value)
    return records


def _resolve_stored_path(stored: str | Path, summary_path: Path) -> Path:
    path = Path(stored).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    return path.resolve()


def _dataset_summary(aggregate: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    canonical = canonical_dataset_name(dataset)
    datasets = aggregate.get("datasets")
    if isinstance(datasets, Mapping):
        if canonical not in datasets:
            raise KeyError(f"dataset {canonical!r} is absent from aggregate summary")
        summary = datasets[canonical]
    elif aggregate.get("dataset"):
        metadata = aggregate["dataset"]
        name = metadata.get("name") if isinstance(metadata, Mapping) else None
        if canonical_dataset_name(str(name)) != canonical:
            raise KeyError(f"per-dataset summary is for {name!r}, not {canonical!r}")
        summary = aggregate
    else:
        raise ValueError("input is neither an aggregate nor per-dataset graded summary")
    if not isinstance(summary, Mapping):
        raise ValueError("dataset summary must be a JSON object")
    return summary


def _load_string_list(path: Path, label: str) -> tuple[str, ...]:
    values = _read_json(path)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} sidecar must be a non-empty JSON list")
    result = tuple(str(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} sidecar contains duplicate identifiers")
    return result


def _load_qrels(path: Path) -> dict[str, dict[str, float]]:
    raw = _read_json(path)
    if not isinstance(raw, Mapping):
        raise ValueError("qrels sidecar must be a JSON object")
    result: dict[str, dict[str, float]] = {}
    for qid, judgments in raw.items():
        if not isinstance(judgments, Mapping):
            raise ValueError(f"qrels for {qid!r} must be an object")
        result[str(qid)] = {
            str(docid): float(grade) for docid, grade in judgments.items()
        }
    return result


def _rankings_from_records(
    records: Iterable[Mapping[str, Any]], dataset: str
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    candidates: dict[str, tuple[str, ...]] = {}
    reference: dict[str, tuple[str, ...]] = {}
    for record in records:
        try:
            record_dataset = canonical_dataset_name(str(record["dataset"]))
        except (KeyError, ValueError):
            continue
        if record_dataset != dataset:
            continue
        method = str(record.get("method"))
        if method not in {"pq_only", "pq_projected_rerank"}:
            continue
        qid = str(record.get("query_id"))
        ranked = record.get("ranked_doc_ids")
        if not isinstance(ranked, list):
            raise ValueError(f"{method} record for {qid!r} has no ranked_doc_ids")
        values = tuple(str(docid) for docid in ranked)
        target = candidates if method == "pq_only" else reference
        if qid in target:
            raise ValueError(f"duplicate {method} record for query {qid!r}")
        target[qid] = values
    return candidates, reference


def load_replay_artifacts(
    aggregate_json: Path,
    per_query_jsonl: Path,
    dataset: str,
) -> ReplayArtifacts:
    aggregate_json = Path(aggregate_json).expanduser().resolve()
    per_query_jsonl = Path(per_query_jsonl).expanduser().resolve()
    canonical = canonical_dataset_name(dataset)
    aggregate = _read_json(aggregate_json)
    if not isinstance(aggregate, Mapping):
        raise ValueError("graded aggregate must be a JSON object")
    summary = _dataset_summary(aggregate, canonical)
    cache = summary.get("cache")
    config = summary.get("config")
    if not isinstance(cache, Mapping) or not isinstance(config, Mapping):
        raise ValueError("graded summary lacks cache/config metadata")
    candidate_k = int(config.get("candidate_k", -1))
    retrieve_k = int(config.get("retrieve_k", -1))
    if candidate_k != 100 or retrieve_k != 100:
        raise ValueError(
            "CKKS replay requires the frozen candidate_k=retrieve_k=100 protocol"
        )

    data_sidecars = _resolve_stored_path(cache["data_sidecars"], aggregate_json)
    basis_path = _resolve_stored_path(cache["projection_basis"], aggregate_json)
    corpus_ids = _load_string_list(data_sidecars / "corpus_ids.json", "corpus_ids")
    query_ids = _load_string_list(data_sidecars / "query_ids.json", "query_ids")
    qrels = _load_qrels(data_sidecars / "qrels.json")
    missing_qrels = [qid for qid in query_ids if qid not in qrels]
    if missing_qrels:
        raise ValueError(f"qrels sidecar lacks {len(missing_qrels)} selected queries")

    projection_dir = basis_path.parent
    projected_corpus = (projection_dir / "corpus.npy").resolve()
    query_embedding_path = _resolve_stored_path(
        cache["query_embeddings"], aggregate_json
    )
    projected_queries = (projection_dir / query_embedding_path.name).resolve()
    if not projected_queries.is_file():
        alternatives = sorted(projection_dir.glob("queries-*.npy"))
        if len(alternatives) == 1:
            projected_queries = alternatives[0].resolve()
        else:
            raise FileNotFoundError(
                f"cannot identify projected query artifact {projected_queries}"
            )
    for path in (basis_path, projected_corpus, projected_queries):
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates, reference = _rankings_from_records(
        _read_jsonl(per_query_jsonl), canonical
    )
    for qid in query_ids:
        if qid not in candidates or qid not in reference:
            raise ValueError(f"per-query log lacks paired PQ records for {qid!r}")
        if len(candidates[qid]) != 100 or len(reference[qid]) != 100:
            raise ValueError(f"query {qid!r} does not have exactly 100 ranked docs")
        if len(set(candidates[qid])) != 100:
            raise ValueError(f"pq_only shortlist for {qid!r} contains duplicates")
        if set(candidates[qid]) != set(reference[qid]):
            raise ValueError(
                f"pq_projected_rerank for {qid!r} does not rerank the same shortlist"
            )
    return ReplayArtifacts(
        dataset_summary=summary,
        data_sidecars=data_sidecars,
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        qrels=qrels,
        projection_basis=basis_path,
        projected_corpus=projected_corpus,
        projected_queries=projected_queries,
        pq_candidates=candidates,
        plaintext_rankings=reference,
    )


def _select_query_ids(
    query_ids: Sequence[str], max_queries: int | None, seed: int
) -> list[str]:
    if max_queries is None or max_queries >= len(query_ids):
        return list(query_ids)
    chosen = np.random.default_rng(seed).choice(
        len(query_ids), size=max_queries, replace=False
    )
    # Preserve sidecar order so projected-query row access stays sequential and
    # logs are stable apart from the explicitly seeded subset choice.
    return [str(query_ids[index]) for index in sorted(int(x) for x in chosen)]


def _npy_memmap(path: Path, expected_rows: int | None = None) -> np.memmap:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(values, np.memmap) or values.ndim != 2:
        raise ValueError(f"expected a two-dimensional NPY memmap: {path}")
    if values.dtype != np.float32:
        raise ValueError(f"expected float32 matrix at {path}, got {values.dtype}")
    if expected_rows is not None and len(values) != expected_rows:
        raise ValueError(
            f"row count mismatch at {path}: {len(values)} != {expected_rows}"
        )
    return values


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _payload_distribution(values: Sequence[int]) -> dict[str, float | int]:
    result = _distribution(values)
    return {
        key: int(value) if key in {"count", "min", "max"} else value
        for key, value in result.items()
    }


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _ids_fingerprint(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def run_graded_ckks_replay(
    artifacts: ReplayArtifacts,
    config: ReplayConfig,
    *,
    ckks_parameters: CKKSParameters | None = None,
    input_summary_path: Path | None = None,
    input_jsonl_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config.validate()
    dataset = canonical_dataset_name(config.dataset)
    ckks_parameters = ckks_parameters or CKKSParameters()
    selected_qids = _select_query_ids(
        artifacts.query_ids, config.max_queries, config.selection_seed
    )
    if not selected_qids:
        raise ValueError("query selection is empty")
    query_row = {qid: row for row, qid in enumerate(artifacts.query_ids)}
    corpus_row = {docid: row for row, docid in enumerate(artifacts.corpus_ids)}

    projected_corpus = _npy_memmap(
        artifacts.projected_corpus, len(artifacts.corpus_ids)
    )
    projected_queries = _npy_memmap(
        artifacts.projected_queries, len(artifacts.query_ids)
    )
    if projected_corpus.shape[1] != projected_queries.shape[1]:
        raise ValueError("projected corpus/query dimensions differ")
    dimension = int(projected_corpus.shape[1])
    basis = load_projection_basis(artifacts.projection_basis)
    if basis.vectors.shape[1] != dimension:
        raise ValueError("projection basis dimension differs from projected arrays")

    candidate_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    query_vectors: list[np.ndarray] = []
    for qid in selected_qids:
        try:
            candidates = np.asarray(
                [corpus_row[docid] for docid in artifacts.pq_candidates[qid]],
                dtype=np.int64,
            )
            reference = np.asarray(
                [corpus_row[docid] for docid in artifacts.plaintext_rankings[qid]],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(f"ranked document id is absent from corpus: {exc}") from exc
        candidate_rows.append(candidates)
        reference_rows.append(reference)
        query_vectors.append(
            np.asarray(projected_queries[query_row[qid]], dtype=np.float32)
        )
    # The client no longer holds either memmap during the timed online replay.
    del projected_queries, projected_corpus

    context_started = time.perf_counter_ns()
    pair = create_context_pair(ckks_parameters, max_query_dimension=dimension)
    context_setup_ms = _elapsed_ms(context_started)
    server = SpawnedCKKSServer(
        public_context=pair.serialized_server_context,
        projected_docs_path=artifacts.projected_corpus,
        method=CKKS_METHOD,
        response_timeout_s=config.response_timeout_s,
    )
    server.start()
    timings: dict[str, list[float]] = {}
    request_bytes: list[int] = []
    response_bytes: list[int] = []
    ciphertext_counts: list[int] = []
    decrypted_rows: list[np.ndarray] = []
    ckks_rankings: list[np.ndarray] = []

    def record(name: str, value: float) -> None:
        timings.setdefault(name, []).append(float(value))

    try:
        warmups = min(config.warmup_queries, len(selected_qids))
        for row in range(warmups):
            encrypted = encrypt_query(pair.client, query_vectors[row])
            response_wire, _, _, _ = server.score(encrypted, candidate_rows[row])
            client_decrypt_scores(pair.client, response_wire)

        for query, candidates in zip(query_vectors, candidate_rows):
            online_started = time.perf_counter_ns()
            started = time.perf_counter_ns()
            encrypted = encrypt_query(pair.client, query)
            record("client_encrypt_serialize", _elapsed_ms(started))

            response_wire, phases, metadata, ipc_ms = server.score(
                encrypted, candidates
            )
            record("ipc_roundtrip", ipc_ms)
            for name, value in phases.items():
                record(name, value)

            started = time.perf_counter_ns()
            parsed = EncryptedScoreResponse.from_bytes(response_wire)
            record("client_response_frame_parse", _elapsed_ms(started))
            started = time.perf_counter_ns()
            scores = client_decrypt_scores(pair.client, parsed)
            record("client_decrypt_decode", _elapsed_ms(started))
            if len(scores) != 100:
                raise RuntimeError("CKKS response did not contain 100 scores")
            started = time.perf_counter_ns()
            order = np.lexsort((candidates, -scores))
            ranking = candidates[order]
            record("client_rank", _elapsed_ms(started))
            record("online_end_to_end", _elapsed_ms(online_started))

            request_bytes.append(len(encrypted) + metadata["candidate_ids_bytes"])
            response_bytes.append(len(response_wire))
            ciphertext_counts.append(metadata["seal_ciphertext_count"])
            decrypted_rows.append(np.asarray(scores, dtype=np.float64))
            ckks_rankings.append(ranking)
        if server.attestation is None:
            raise RuntimeError("server did not produce a public-only attestation")
        attestation = dict(server.attestation)
    finally:
        server.close()

    # Untimed post-protocol audit.  Float64 scores measure CKKS approximation
    # error; float32 scores reproduce graded_ir_bench's reference ranking.
    audit_corpus = _npy_memmap(
        artifacts.projected_corpus, len(artifacts.corpus_ids)
    )
    max_abs_error: list[float] = []
    mean_abs_error: list[float] = []
    rmse: list[float] = []
    top100_match: list[float] = []
    top10_match: list[float] = []
    recomputed_reference_match: list[float] = []
    for query, candidates, recorded_reference, scores, ckks_ranking in zip(
        query_vectors,
        candidate_rows,
        reference_rows,
        decrypted_rows,
        ckks_rankings,
    ):
        docs32 = np.asarray(audit_corpus[candidates], dtype=np.float32)
        reference_scores32 = np.asarray(docs32 @ query, dtype=np.float32)
        reference_order = np.lexsort((candidates, -reference_scores32))
        recomputed = candidates[reference_order]
        exact64 = np.asarray(docs32, dtype=np.float64) @ np.asarray(
            query, dtype=np.float64
        )
        error = scores - exact64
        max_abs_error.append(float(np.max(np.abs(error))))
        mean_abs_error.append(float(np.mean(np.abs(error))))
        rmse.append(float(np.sqrt(np.mean(np.square(error)))))
        top100_match.append(float(np.array_equal(ckks_ranking, recorded_reference)))
        top10_match.append(
            float(np.array_equal(ckks_ranking[:10], recorded_reference[:10]))
        )
        recomputed_reference_match.append(
            float(np.array_equal(recomputed, recorded_reference))
        )
    del audit_corpus

    ckks_matrix = np.asarray(ckks_rankings, dtype=np.int64)
    reference_matrix = np.asarray(reference_rows, dtype=np.int64)
    selected_qrels = {qid: artifacts.qrels[qid] for qid in selected_qids}
    metrics_by_method = {
        "pq_projected_rerank": evaluate_rankings(
            reference_matrix,
            selected_qids,
            artifacts.corpus_ids,
            selected_qrels,
        ),
        "ckks_segmented_score_packed": evaluate_rankings(
            ckks_matrix,
            selected_qids,
            artifacts.corpus_ids,
            selected_qrels,
        ),
    }
    metric_summary, paired = summarise_method_metrics(
        metrics_by_method,
        config.bootstrap_samples,
        config.bootstrap_seed,
        reference="pq_projected_rerank",
    )
    numerical_vectors = {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "top100_exact_order_match": top100_match,
        "top10_exact_order_match": top10_match,
        "recomputed_plaintext_matches_recorded_reference": recomputed_reference_match,
    }

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "config": {
            "max_queries": config.max_queries,
            "selection_seed": config.selection_seed,
            "selected_queries": len(selected_qids),
            "warmup_queries": min(config.warmup_queries, len(selected_qids)),
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "candidate_k": 100,
            "retrieve_k": 100,
            "projected_dimension": dimension,
            "ckks": ckks_parameters.as_json(),
        },
        "protocol": {
            "method": CKKS_METHOD,
            "client": (
                "projected query, frozen public PQ shortlist, private CKKS context"
            ),
            "server": (
                "read-only exact projected corpus and serialized public-only CKKS context"
            ),
            "process_boundary": "Windows-compatible multiprocessing spawn + Pipe",
            "server_process_arguments": ["Pipe endpoint only"],
            "request": "encrypted projected query + 100 uint64 candidate row IDs",
            "response": "one segmented score-packed encrypted response",
            "plaintext_audit_in_online_latency": False,
            "query_encoding_projection_and_pq_search_in_latency": False,
            "payload_scope": "application bytes; Pipe/pickle/OS framing excluded",
        },
        "inputs": {
            "graded_summary_json": (
                _file_metadata(Path(input_summary_path)) if input_summary_path else None
            ),
            "graded_per_query_jsonl": (
                _file_metadata(Path(input_jsonl_path)) if input_jsonl_path else None
            ),
            "data_sidecars": str(artifacts.data_sidecars),
            "projection_basis": _file_metadata(artifacts.projection_basis),
            "projection_basis_sha256": projection_fingerprint(basis),
            "projected_corpus": _file_metadata(artifacts.projected_corpus),
            "projected_queries": _file_metadata(artifacts.projected_queries),
            "selected_query_ids_sha256": _ids_fingerprint(selected_qids),
        },
        "server_attestation": attestation,
        "context_setup_ms_excluded": context_setup_ms,
        "metrics": metric_summary,
        "paired_query_bootstrap_deltas": paired,
        "numerical_audit": {
            name: _distribution(values) for name, values in numerical_vectors.items()
        },
        "latency_ms": {
            name: _distribution(values) for name, values in timings.items()
        },
        "payload_bytes": {
            "request": _payload_distribution(request_bytes),
            "response": _payload_distribution(response_bytes),
            "total": _payload_distribution(
                np.asarray(request_bytes) + np.asarray(response_bytes)
            ),
        },
        "seal_ciphertext_count": _payload_distribution(ciphertext_counts),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "client_pid": os.getpid(),
        },
    }

    records: list[dict[str, Any]] = []
    for row, qid in enumerate(selected_qids):
        records.append(
            {
                "schema": SCHEMA,
                "dataset": dataset,
                "query_id": qid,
                "candidate_doc_ids": list(artifacts.pq_candidates[qid]),
                "ranked_doc_ids": [
                    artifacts.corpus_ids[int(index)] for index in ckks_matrix[row]
                ],
                "plaintext_reference_doc_ids": list(
                    artifacts.plaintext_rankings[qid]
                ),
                "metrics": {
                    method: {
                        metric: float(values[metric][row])
                        for metric in METRIC_NAMES
                    }
                    for method, values in metrics_by_method.items()
                },
                "numerical_audit": {
                    name: float(values[row])
                    for name, values in numerical_vectors.items()
                },
                "latency_ms": {
                    name: float(values[row]) for name, values in timings.items()
                },
                "payload_bytes": {
                    "request": int(request_bytes[row]),
                    "response": int(response_bytes[row]),
                    "total": int(request_bytes[row] + response_bytes[row]),
                },
                "seal_ciphertext_count": int(ciphertext_counts[row]),
            }
        )
    return summary, records


def replay_from_files(
    aggregate_json: Path,
    per_query_jsonl: Path,
    config: ReplayConfig,
    *,
    ckks_parameters: CKKSParameters | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts = load_replay_artifacts(
        aggregate_json, per_query_jsonl, config.dataset
    )
    return run_graded_ckks_replay(
        artifacts,
        config,
        ckks_parameters=ckks_parameters,
        input_summary_path=Path(aggregate_json).resolve(),
        input_jsonl_path=Path(per_query_jsonl).resolve(),
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay graded-IR PQ top-100 candidates through public-server CKKS"
    )
    parser.add_argument("--graded-summary-json", type=Path, required=True)
    parser.add_argument("--per-query-jsonl", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--response-timeout-s", type=float, default=600.0)
    parser.add_argument("--poly-modulus-degree", type=int, default=8192)
    parser.add_argument(
        "--coeff-mod-bit-sizes", type=int, nargs="+", default=[60, 40, 60]
    )
    parser.add_argument("--scale-bits", type=int, default=40)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ReplayConfig(
        dataset=args.dataset,
        max_queries=args.max_queries,
        selection_seed=args.selection_seed,
        warmup_queries=args.warmup_queries,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        response_timeout_s=args.response_timeout_s,
    )
    ckks = CKKSParameters(
        poly_modulus_degree=args.poly_modulus_degree,
        coeff_mod_bit_sizes=tuple(args.coeff_mod_bit_sizes),
        scale_bits=args.scale_bits,
    )
    summary, records = replay_from_files(
        args.graded_summary_json,
        args.per_query_jsonl,
        config,
        ckks_parameters=ckks,
    )
    _atomic_write_json(args.output_json, summary)
    _atomic_write_jsonl(args.output_jsonl, records)
    print(
        json.dumps(
            {
                "dataset": summary["dataset"],
                "queries": summary["config"]["selected_queries"],
                "ndcg_at_10": summary["metrics"]
                ["ckks_segmented_score_packed"]["ndcg_at_10"]["mean"],
                "top10_order_match": summary["numerical_audit"]
                ["top10_exact_order_match"]["mean"],
                "online_p50_ms": summary["latency_ms"]["online_end_to_end"]["p50"],
                "online_p95_ms": summary["latency_ms"]["online_end_to_end"]["p95"],
                "response_mean_bytes": summary["payload_bytes"]["response"]["mean"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    sys.exit(main())
