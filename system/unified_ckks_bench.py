"""True two-process benchmark for client-side PQ plus CKKS reranking.

The online protocol has a strict process boundary.  The parent process is the
client and owns the secret CKKS context, projection basis, and public FAISS PQ
index.  A fresh ``multiprocessing`` worker is started with the Windows-compatible
``spawn`` method and receives *only through a Pipe*:

* a serialized public CKKS context and the projected-document NPY path at
  initialization; and
* an encrypted query plus serialized candidate identifiers for every request.

The worker never receives a Python object containing a secret key and its
attestation explicitly checks that the reconstructed context exposes neither a
secret key nor a decryptor.  It gathers candidate rows from a read-only memmap,
runs one of the kernels in :mod:`system.packed_ckks`, and returns only the
serialized encrypted response and phase timings.

Plaintext scores used to measure CKKS numerical error are computed by an
explicit, untimed post-protocol audit after the server has stopped.  That audit
is experimental instrumentation, not part of the deployable client protocol.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from system.packed_ckks import (
    CKKSParameters,
    EncryptedScoreResponse,
    client_decrypt_scores,
    create_context_pair,
    deserialize_server_context,
    encrypt_query,
    server_block_packed_scores,
    server_naive_scores,
    server_segmented_score_packed,
)
from system.retrieval_bench import (
    DEFAULT_CACHE,
    MODEL_SPECS,
    ProjectionBasis,
    basis_fingerprint,
    bootstrap_mean_ci,
    configure_faiss_threads,
    deterministic_self_qrels,
    load_memmap,
    load_pq_index,
    load_projection_basis,
    metric_vectors,
    project_queries,
    select_query_indices,
    summarize_latency,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "unified_ckks_benchmark.v1"
_INIT = "initialize"
_READY = "ready"
_SCORE = "score"
_RESULT = "result"
_STOP = "stop"
_STOPPED = "stopped"

METHODS = (
    "naive_per_candidate",
    "block_packed",
    "segmented_score_packed",
)


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "type": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _load_projected_docs(path: str | Path) -> np.memmap:
    docs = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    if not isinstance(docs, np.memmap):
        raise TypeError("projected document artifact must be an NPY memmap")
    if docs.ndim != 2 or docs.shape[0] < 1 or docs.shape[1] < 1:
        raise ValueError("projected document artifact must be a non-empty matrix")
    if docs.dtype != np.float32:
        raise ValueError(f"projected documents must be float32, got {docs.dtype}")
    return docs


def _server_worker(connection: Connection) -> None:
    """Spawn target.  Its only process argument is the Pipe endpoint."""

    try:
        message = connection.recv()
        if not isinstance(message, dict) or message.get("type") != _INIT:
            raise ValueError("first server message must initialize the worker")
        public_wire = message.get("public_context")
        if not isinstance(public_wire, bytes):
            raise TypeError("public_context must be serialized bytes")
        docs_path = message.get("projected_docs_path")
        if not isinstance(docs_path, str):
            raise TypeError("projected_docs_path must be a string")
        method = str(message.get("method"))
        if method not in METHODS:
            raise ValueError(f"unknown CKKS method {method!r}")
        block_size_raw = message.get("block_size")
        block_size = None if block_size_raw is None else int(block_size_raw)

        started = time.perf_counter_ns()
        context = deserialize_server_context(public_wire)
        docs = _load_projected_docs(docs_path)
        startup_ms = _elapsed_ms(started)
        attestation = {
            "pid": os.getpid(),
            "process_name": mp.current_process().name,
            "start_method": mp.get_start_method(allow_none=True),
            "context_class": type(context).__name__,
            "context_is_public": bool(context.is_public()),
            "has_secret_key": bool(context.has_secret_key()),
            "has_secret_key_attribute": hasattr(context, "secret_key"),
            "has_decryptor_attribute": hasattr(context, "decryptor"),
            "received_public_context_bytes": len(public_wire),
            "projected_docs_shape": [int(x) for x in docs.shape],
            "projected_docs_dtype": str(docs.dtype),
            "projected_docs_read_only": not bool(docs.flags.writeable),
            "startup_ms": startup_ms,
        }
        if (
            not attestation["context_is_public"]
            or attestation["has_secret_key"]
            or attestation["has_secret_key_attribute"]
            or attestation["has_decryptor_attribute"]
        ):
            raise RuntimeError("server context failed public-only attestation")
        connection.send({"type": _READY, "attestation": attestation})

        while True:
            message = connection.recv()
            if not isinstance(message, dict):
                raise TypeError("server request must be a dictionary")
            if message.get("type") == _STOP:
                connection.send({"type": _STOPPED})
                break
            if message.get("type") != _SCORE:
                raise ValueError("unknown server request type")
            try:
                total_started = time.perf_counter_ns()
                encrypted_query = message.get("encrypted_query")
                candidate_wire = message.get("candidate_ids")
                if not isinstance(encrypted_query, bytes):
                    raise TypeError("encrypted_query must be bytes")
                if not isinstance(candidate_wire, bytes) or not candidate_wire:
                    raise TypeError("candidate_ids must be non-empty bytes")
                if len(candidate_wire) % np.dtype("<u8").itemsize:
                    raise ValueError("candidate_ids byte length is invalid")
                ids = np.frombuffer(candidate_wire, dtype="<u8").astype(
                    np.int64, copy=False
                )
                if np.any(ids < 0) or np.any(ids >= len(docs)):
                    raise IndexError("candidate identifier is outside the corpus")

                gather_started = time.perf_counter_ns()
                candidates = np.ascontiguousarray(docs[ids], dtype=np.float64)
                gather_ms = _elapsed_ms(gather_started)

                kernel_started = time.perf_counter_ns()
                if method == "naive_per_candidate":
                    response = server_naive_scores(
                        context, encrypted_query, candidates
                    )
                elif method == "block_packed":
                    response = server_block_packed_scores(
                        context,
                        encrypted_query,
                        candidates,
                        block_size=block_size,
                    )
                else:
                    response = server_segmented_score_packed(
                        context,
                        encrypted_query,
                        candidates,
                        block_size=block_size,
                    )
                kernel_ms = _elapsed_ms(kernel_started)

                frame_started = time.perf_counter_ns()
                response_wire = response.to_bytes()
                frame_ms = _elapsed_ms(frame_started)
                phase_ms = {
                    "server_gather": gather_ms,
                    **{
                        f"server_{name}": float(value)
                        for name, value in response.server_phase_ms.items()
                    },
                    "server_kernel_total": kernel_ms,
                    "server_response_frame_serialize": frame_ms,
                    "server_total": _elapsed_ms(total_started),
                }
                connection.send(
                    {
                        "type": _RESULT,
                        "response": response_wire,
                        "phase_ms": phase_ms,
                        "response_metadata": {
                            "score_count": response.score_count,
                            "seal_ciphertext_count": response.seal_ciphertext_count,
                            "raw_ciphertext_payload_bytes": response.payload_bytes,
                        },
                    }
                )
            except BaseException as exc:  # keep worker diagnostics observable
                connection.send(_error_payload(exc))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send(_error_payload(exc))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class SpawnedCKKSServer:
    """Lifecycle wrapper for the public-only spawned scoring worker."""

    def __init__(
        self,
        *,
        public_context: bytes,
        projected_docs_path: Path,
        method: str,
        block_size: int | None = None,
        response_timeout_s: float = 600.0,
    ) -> None:
        if not isinstance(public_context, bytes) or not public_context:
            raise TypeError("public_context must be non-empty serialized bytes")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if block_size is not None and block_size < 1:
            raise ValueError("block_size must be positive")
        self.public_context = public_context
        self.projected_docs_path = Path(projected_docs_path).resolve()
        self.method = method
        self.block_size = block_size
        self.response_timeout_s = float(response_timeout_s)
        self._connection: Connection | None = None
        self._process: mp.Process | None = None
        self.attestation: dict[str, Any] | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _receive(self) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("server is not running")
        if not self._connection.poll(self.response_timeout_s):
            raise TimeoutError("timed out waiting for the CKKS server worker")
        result = self._connection.recv()
        if not isinstance(result, dict):
            raise RuntimeError("server returned a malformed response")
        if result.get("type") == "error":
            raise RuntimeError(
                f"server {result.get('error_type')}: {result.get('message')}\n"
                f"{result.get('traceback', '')}"
            )
        return result

    def start(self) -> dict[str, Any]:
        if self._process is not None:
            raise RuntimeError("server was already started")
        if not self.projected_docs_path.is_file():
            raise FileNotFoundError(self.projected_docs_path)
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_server_worker,
            args=(child,),
            name="public-ckks-rerank-server",
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        parent.send(
            {
                "type": _INIT,
                "public_context": self.public_context,
                "projected_docs_path": str(self.projected_docs_path),
                "method": self.method,
                "block_size": self.block_size,
            }
        )
        ready = self._receive()
        if ready.get("type") != _READY:
            raise RuntimeError("server did not acknowledge initialization")
        self.attestation = dict(ready["attestation"])
        return self.attestation

    def score(
        self, encrypted_query: bytes, candidate_ids: np.ndarray
    ) -> tuple[bytes, dict[str, float], dict[str, int], float]:
        if self._connection is None:
            raise RuntimeError("server is not running")
        ids = np.asarray(candidate_ids, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or np.any(ids < 0):
            raise ValueError("candidate_ids must be a non-empty nonnegative vector")
        candidate_wire = ids.astype("<u8", copy=False).tobytes(order="C")
        started = time.perf_counter_ns()
        self._connection.send(
            {
                "type": _SCORE,
                "encrypted_query": encrypted_query,
                "candidate_ids": candidate_wire,
            }
        )
        result = self._receive()
        ipc_roundtrip_ms = _elapsed_ms(started)
        if result.get("type") != _RESULT:
            raise RuntimeError("server returned an unexpected response type")
        response = result.get("response")
        if not isinstance(response, bytes):
            raise RuntimeError("server response payload is not bytes")
        phases = {str(k): float(v) for k, v in result["phase_ms"].items()}
        metadata = {
            str(k): int(v) for k, v in result["response_metadata"].items()
        }
        metadata["candidate_ids_bytes"] = len(candidate_wire)
        return response, phases, metadata, ipc_roundtrip_ms

    def close(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            try:
                if process is not None and process.is_alive():
                    connection.send({"type": _STOP})
                    if connection.poll(min(self.response_timeout_s, 10.0)):
                        connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                connection.close()
        if process is not None:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)

    def __enter__(self) -> "SpawnedCKKSServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@dataclass(frozen=True)
class UnifiedBenchmarkConfig:
    method: str = "segmented_score_packed"
    shortlist_k: int = 100
    top_k: int = 10
    split: str = "validation"
    warmup_queries: int = 1
    block_size: int | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    faiss_threads: int = 0
    center_online_queries: bool = False
    response_timeout_s: float = 600.0

    def validate(self, *, n_docs: int, n_queries: int, dimension: int) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not 1 <= self.top_k <= self.shortlist_k <= n_docs:
            raise ValueError("expected 1 <= top_k <= shortlist_k <= n_docs")
        if n_queries < 1 or dimension < 1:
            raise ValueError("queries and projected dimension must be non-empty")
        if self.warmup_queries < 0:
            raise ValueError("warmup_queries cannot be negative")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")


def _timing_summary(
    values: Sequence[float], sample_indices: np.ndarray
) -> dict[str, float]:
    return summarize_latency(np.asarray(values, dtype=np.float64), sample_indices)


def _distribution_summary(
    values: Sequence[float], sample_indices: np.ndarray
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        **summarize_latency(array, sample_indices),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _payload_summary(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "mean": float(array.mean()),
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def _metric_summary(
    vectors: Mapping[str, np.ndarray], sample_indices: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        key: bootstrap_mean_ci(value, sample_indices=sample_indices)
        for key, value in vectors.items()
    }


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _environment_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "client_pid": os.getpid(),
        "logical_cpus": os.cpu_count(),
    }
    for distribution, key in (("tenseal", "tenseal"), ("faiss-cpu", "faiss")):
        try:
            result[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[key] = None
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["cuda_available"] = False
    return result


def run_unified_ckks_benchmark(
    *,
    projected_docs_path: Path,
    queries: np.ndarray,
    query_indices: np.ndarray,
    ground_truth_ids: np.ndarray,
    basis: ProjectionBasis,
    pq_index: Any,
    config: UnifiedBenchmarkConfig | None = None,
    ckks_parameters: CKKSParameters | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute the paired online protocol and return summary plus JSONL rows."""

    config = config or UnifiedBenchmarkConfig()
    ckks_parameters = ckks_parameters or CKKSParameters()
    projected_docs_path = Path(projected_docs_path).resolve()
    queries = np.asarray(queries, dtype=np.float32)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    ground_truth_ids = np.asarray(ground_truth_ids, dtype=np.int64)
    if queries.ndim != 2 or queries.shape[0] < 1:
        raise ValueError("queries must be a non-empty matrix")
    if len(query_indices) != len(queries) or len(ground_truth_ids) != len(queries):
        raise ValueError("queries, query_indices, and ground_truth_ids must align")
    if basis.vectors.shape[0] != queries.shape[1]:
        raise ValueError("basis input dimension does not match queries")
    projected_shape = _load_projected_docs(projected_docs_path).shape
    n_docs, projected_dimension = (int(projected_shape[0]), int(projected_shape[1]))
    if basis.vectors.shape[1] != projected_dimension:
        raise ValueError("basis output dimension does not match projected documents")
    if int(pq_index.d) != projected_dimension or int(pq_index.ntotal) != n_docs:
        raise ValueError("PQ index shape/count does not match projected corpus")
    if np.any(ground_truth_ids < 0) or np.any(ground_truth_ids >= n_docs):
        raise ValueError("ground-truth identifiers are outside the corpus")
    config.validate(
        n_docs=n_docs, n_queries=len(queries), dimension=projected_dimension
    )
    configure_faiss_threads(config.faiss_threads)

    pair = create_context_pair(
        ckks_parameters, max_query_dimension=projected_dimension
    )
    timings: dict[str, list[float]] = {}
    request_bytes: list[int] = []
    response_bytes: list[int] = []
    ciphertext_counts: list[int] = []
    projected_queries: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    decrypted_rows: list[np.ndarray] = []
    ckks_predictions: list[np.ndarray] = []
    pq_predictions: list[np.ndarray] = []
    server_attestation: dict[str, Any]

    def record_timing(name: str, value: float) -> None:
        timings.setdefault(name, []).append(float(value))

    server = SpawnedCKKSServer(
        public_context=pair.serialized_server_context,
        projected_docs_path=projected_docs_path,
        method=config.method,
        block_size=config.block_size,
        response_timeout_s=config.response_timeout_s,
    )
    server.start()
    try:
        assert server.attestation is not None
        server_attestation = dict(server.attestation)

        warmup_count = min(config.warmup_queries, len(queries))
        for i in range(warmup_count):
            projected = project_queries(
                queries[i : i + 1], basis, config.center_online_queries
            )[0]
            _, ids = pq_index.search(
                np.ascontiguousarray(projected.reshape(1, -1)), config.shortlist_k
            )
            if np.any(ids[0] < 0):
                raise RuntimeError("PQ warmup returned missing candidates")
            encrypted = encrypt_query(pair.client, projected)
            response_wire, _, _, _ = server.score(encrypted, ids[0])
            client_decrypt_scores(pair.client, response_wire)

        for i, query in enumerate(queries):
            online_started = time.perf_counter_ns()

            started = time.perf_counter_ns()
            projected = project_queries(
                query.reshape(1, -1), basis, config.center_online_queries
            )[0]
            record_timing("client_query_projection", _elapsed_ms(started))

            started = time.perf_counter_ns()
            _, ids_matrix = pq_index.search(
                np.ascontiguousarray(projected.reshape(1, -1)), config.shortlist_k
            )
            record_timing("client_pq_search", _elapsed_ms(started))
            ids = np.asarray(ids_matrix[0], dtype=np.int64)
            if np.any(ids < 0):
                raise RuntimeError("FAISS returned missing candidates")

            started = time.perf_counter_ns()
            encrypted = encrypt_query(pair.client, projected)
            record_timing("client_encrypt_serialize", _elapsed_ms(started))

            response_wire, server_phases, response_meta, ipc_ms = server.score(
                encrypted, ids
            )
            record_timing("ipc_roundtrip", ipc_ms)
            for name, value in server_phases.items():
                record_timing(name, value)

            started = time.perf_counter_ns()
            parsed = EncryptedScoreResponse.from_bytes(response_wire)
            record_timing("client_response_frame_parse", _elapsed_ms(started))

            started = time.perf_counter_ns()
            scores = client_decrypt_scores(pair.client, parsed)
            record_timing("client_decrypt_decode", _elapsed_ms(started))
            if len(scores) != config.shortlist_k:
                raise RuntimeError("decrypted CKKS score count differs from shortlist")

            started = time.perf_counter_ns()
            order = np.argsort(-scores, kind="stable")[: config.top_k]
            prediction = ids[order]
            record_timing("client_rank", _elapsed_ms(started))
            record_timing("online_end_to_end", _elapsed_ms(online_started))

            request_bytes.append(len(encrypted) + response_meta["candidate_ids_bytes"])
            response_bytes.append(len(response_wire))
            ciphertext_counts.append(response_meta["seal_ciphertext_count"])
            projected_queries.append(np.asarray(projected, dtype=np.float32))
            candidate_rows.append(ids.copy())
            decrypted_rows.append(np.asarray(scores, dtype=np.float64))
            ckks_predictions.append(prediction.copy())
            pq_predictions.append(ids[: config.top_k].copy())
    finally:
        server.close()

    # Explicit untimed audit; no projected document row was opened in the
    # client while the deployable online protocol was running.
    audit_docs = _load_projected_docs(projected_docs_path)
    plaintext_rows: list[np.ndarray] = []
    plaintext_predictions: list[np.ndarray] = []
    max_abs_error: list[float] = []
    mean_abs_error: list[float] = []
    rmse: list[float] = []
    topk_exact_match: list[float] = []
    for projected, ids, decrypted, ckks_prediction in zip(
        projected_queries, candidate_rows, decrypted_rows, ckks_predictions
    ):
        reference = np.asarray(
            np.asarray(audit_docs[ids], dtype=np.float64)
            @ np.asarray(projected, dtype=np.float64),
            dtype=np.float64,
        )
        error = decrypted - reference
        reference_order = np.argsort(-reference, kind="stable")[: config.top_k]
        reference_prediction = ids[reference_order]
        plaintext_rows.append(reference)
        plaintext_predictions.append(reference_prediction)
        max_abs_error.append(float(np.max(np.abs(error))))
        mean_abs_error.append(float(np.mean(np.abs(error))))
        rmse.append(float(np.sqrt(np.mean(np.square(error)))))
        topk_exact_match.append(float(np.array_equal(ckks_prediction, reference_prediction)))
    del audit_docs

    ckks_prediction_matrix = np.asarray(ckks_predictions, dtype=np.int64)
    plaintext_prediction_matrix = np.asarray(plaintext_predictions, dtype=np.int64)
    pq_prediction_matrix = np.asarray(pq_predictions, dtype=np.int64)
    candidates_matrix = np.asarray(candidate_rows, dtype=np.int64)
    shortlist_recall = np.any(
        candidates_matrix == ground_truth_ids[:, None], axis=1
    ).astype(np.float64)
    recall_key = f"shortlist_recall_at_{config.shortlist_k}"
    ckks_metric_vectors = {
        **metric_vectors(ckks_prediction_matrix, ground_truth_ids),
        recall_key: shortlist_recall,
    }
    plaintext_metric_vectors = {
        **metric_vectors(plaintext_prediction_matrix, ground_truth_ids),
        recall_key: shortlist_recall,
    }
    pq_metric_vectors = {
        **metric_vectors(pq_prediction_matrix, ground_truth_ids),
        recall_key: shortlist_recall,
    }

    rng = np.random.default_rng(config.bootstrap_seed)
    sample_indices = rng.integers(
        0,
        len(queries),
        size=(config.bootstrap_samples, len(queries)),
        dtype=np.int64,
    )
    paired_delta = {
        key: bootstrap_mean_ci(
            ckks_metric_vectors[key] - plaintext_metric_vectors[key],
            sample_indices=sample_indices,
        )
        for key in plaintext_metric_vectors
    }
    error_vectors = {
        "max_abs_error": np.asarray(max_abs_error, dtype=np.float64),
        "mean_abs_error": np.asarray(mean_abs_error, dtype=np.float64),
        "rmse": np.asarray(rmse, dtype=np.float64),
        "top_k_exact_order_match": np.asarray(topk_exact_match, dtype=np.float64),
    }

    summary: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "roles": {
                "client": (
                    "private CKKS context, projection basis, public PQ index; "
                    "does not open projected document rows during online requests"
                ),
                "server": (
                    "serialized public CKKS context and read-only exact projected "
                    "document memmap; no secret key or decryptor"
                ),
                "post_protocol_auditor": (
                    "untimed benchmark-only plaintext numerical reference"
                ),
            },
            "process_boundary": "multiprocessing Pipe with Windows spawn",
            "server_process_arguments": ["Pipe endpoint only"],
            "server_initialization_over_pipe": [
                "serialized public CKKS context",
                "projected document NPY path",
                "kernel method and public packing parameter",
            ],
            "per_query_request_over_pipe": [
                "serialized encrypted query",
                "uint64 candidate identifiers",
            ],
            "per_query_response_over_pipe": [
                "serialized encrypted score response",
                "server phase timings",
            ],
            "document_projection": "(document - passage_mean) @ V_k",
            "online_query_projection": (
                "(query - passage_mean) @ V_k [centering ablation]"
                if config.center_online_queries
                else "query @ V_k [deployable default]"
            ),
            "payload_scope": (
                "application payload only: encrypted query + candidate-id bytes; "
                "encrypted response framing; multiprocessing pickle/OS headers excluded"
            ),
            "latency_scope": (
                "measured local spawned-process online path; context/key setup, "
                "query embedding inference, network transport, and plaintext audit excluded"
            ),
            "plaintext_audit_is_deployable_protocol": False,
        },
        "config": {
            **dataclasses.asdict(config),
            "n_docs": n_docs,
            "n_queries": len(queries),
            "raw_query_dimension": int(queries.shape[1]),
            "projected_dimension": projected_dimension,
            "ckks": ckks_parameters.as_json(),
        },
        "artifacts": {
            "projected_documents": _file_metadata(projected_docs_path),
            "basis_sha256": basis_fingerprint(basis),
            **dict(source_metadata or {}),
        },
        "environment": _environment_metadata(),
        "server_attestation": server_attestation,
        "methods": {
            "pq_only": {"metrics": _metric_summary(pq_metric_vectors, sample_indices)},
            "plaintext_projected_shortlist": {
                "role": "untimed paired computational/numerical reference",
                "metrics": _metric_summary(
                    plaintext_metric_vectors, sample_indices
                ),
            },
            "ckks_projected_shortlist": {
                "kernel": config.method,
                "metrics": _metric_summary(ckks_metric_vectors, sample_indices),
                "latency_ms": {
                    name: _timing_summary(values, sample_indices)
                    for name, values in timings.items()
                },
                "payload_bytes": {
                    "request": _payload_summary(request_bytes),
                    "response": _payload_summary(response_bytes),
                    "total": _payload_summary(
                        np.asarray(request_bytes) + np.asarray(response_bytes)
                    ),
                },
                "seal_ciphertext_count": _payload_summary(ciphertext_counts),
            },
        },
        "numerical_audit": {
            name: _distribution_summary(values, sample_indices)
            for name, values in error_vectors.items()
        },
        "paired_query_bootstrap": {
            "ckks_minus_plaintext_projected_shortlist": paired_delta
        },
    }

    records: list[dict[str, Any]] = []
    for i in range(len(queries)):
        record = {
            "schema": SCHEMA_VERSION,
            "query_index": int(query_indices[i]),
            "split": config.split,
            "ground_truth_id": int(ground_truth_ids[i]),
            "candidate_ids": [int(x) for x in candidate_rows[i]],
            "predictions": {
                "pq_only": [int(x) for x in pq_predictions[i]],
                "plaintext_projected_shortlist": [
                    int(x) for x in plaintext_predictions[i]
                ],
                "ckks_projected_shortlist": [int(x) for x in ckks_predictions[i]],
            },
            "metrics": {
                "ckks_projected_shortlist": {
                    key: float(value[i]) for key, value in ckks_metric_vectors.items()
                },
                "plaintext_projected_shortlist": {
                    key: float(value[i])
                    for key, value in plaintext_metric_vectors.items()
                },
                "pq_only": {
                    key: float(value[i]) for key, value in pq_metric_vectors.items()
                },
            },
            "numerical_audit": {
                name: float(value[i]) for name, value in error_vectors.items()
            },
            "latency_ms": {name: float(value[i]) for name, value in timings.items()},
            "payload_bytes": {
                "request": int(request_bytes[i]),
                "response": int(response_bytes[i]),
                "total": int(request_bytes[i] + response_bytes[i]),
            },
            "seal_ciphertext_count": int(ciphertext_counts[i]),
        }
        records.append(record)
    return summary, records


def run_cached_cli(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache_dir = Path(args.cache_dir)
    spec = MODEL_SPECS[args.model]
    queries_path = cache_dir / spec.queries_file
    all_queries = load_memmap(queries_path)
    query_indices = select_query_indices(
        len(all_queries),
        args.split,
        args.validation_size,
        args.split_seed,
        args.max_queries,
    )
    queries = np.asarray(all_queries[query_indices], dtype=np.float32)
    projected_docs_path = Path(args.projected_docs_path)
    projected_docs = _load_projected_docs(projected_docs_path)
    qrels = deterministic_self_qrels(
        len(projected_docs), len(all_queries), args.qrel_seed
    )[query_indices]
    del projected_docs

    basis = load_projection_basis(Path(args.basis_path))
    pq_index = load_pq_index(
        Path(args.pq_index_path), basis.vectors.shape[1],
        int(np.load(projected_docs_path, mmap_mode="r", allow_pickle=False).shape[0])
    )
    config = UnifiedBenchmarkConfig(
        method=args.method,
        shortlist_k=args.shortlist_k,
        top_k=args.top_k,
        split=args.split,
        warmup_queries=args.warmup_queries,
        block_size=args.block_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        faiss_threads=args.faiss_threads,
        center_online_queries=args.center_online_queries,
        response_timeout_s=args.response_timeout_s,
    )
    ckks = CKKSParameters(
        poly_modulus_degree=args.poly_modulus_degree,
        coeff_mod_bit_sizes=tuple(args.coeff_mod_bit_sizes),
        scale_bits=args.scale_bits,
    )
    return run_unified_ckks_benchmark(
        projected_docs_path=projected_docs_path,
        queries=queries,
        query_indices=query_indices,
        ground_truth_ids=qrels,
        basis=basis,
        pq_index=pq_index,
        config=config,
        ckks_parameters=ckks,
        source_metadata={
            "query_embeddings": _file_metadata(queries_path),
            "basis": _file_metadata(Path(args.basis_path)),
            "pq_index": _file_metadata(Path(args.pq_index_path)),
            "split_recipe": {
                "validation_size": args.validation_size,
                "split_seed": args.split_seed,
                "qrel_seed": args.qrel_seed,
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spawned client-PQ/public-server CKKS retrieval benchmark"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="e5-base")
    parser.add_argument("--projected-docs-path", type=Path, required=True)
    parser.add_argument("--basis-path", type=Path, required=True)
    parser.add_argument("--pq-index-path", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="segmented_score_packed")
    parser.add_argument("--shortlist-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--split", choices=("validation", "test", "all"), default="validation")
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--qrel-seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--faiss-threads", type=int, default=0)
    parser.add_argument("--center-online-queries", action="store_true")
    parser.add_argument("--response-timeout-s", type=float, default=600.0)
    parser.add_argument("--poly-modulus-degree", type=int, default=8192)
    parser.add_argument(
        "--coeff-mod-bit-sizes", type=int, nargs="+", default=[60, 40, 60]
    )
    parser.add_argument("--scale-bits", type=int, default=40)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--query-log-jsonl", type=Path)
    return parser


def _metric_mean(summary: Mapping[str, Any], method: str, key: str) -> float:
    return float(summary["methods"][method]["metrics"][key]["mean"])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, records = run_cached_cli(args)
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    if args.query_log_jsonl:
        write_jsonl(Path(args.query_log_jsonl), records)
    hit_key = f"hit_at_{args.top_k}"
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "method": args.method,
                "split": args.split,
                "n_queries": summary["config"]["n_queries"],
                "hit_at_1": _metric_mean(
                    summary, "ckks_projected_shortlist", "hit_at_1"
                ),
                hit_key: _metric_mean(
                    summary, "ckks_projected_shortlist", hit_key
                ),
                "online_p50_ms": summary["methods"]["ckks_projected_shortlist"]
                ["latency_ms"]["online_end_to_end"]["p50"],
                "online_p95_ms": summary["methods"]["ckks_projected_shortlist"]
                ["latency_ms"]["online_end_to_end"]["p95"],
                "response_mean_bytes": summary["methods"]
                ["ckks_projected_shortlist"]["payload_bytes"]["response"]["mean"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
