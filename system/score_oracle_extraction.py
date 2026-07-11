"""Chosen-query score-oracle extraction against the public CKKS server.

This is a limitation experiment, not a prescribed-client feature.  For a
known candidate row ``x`` in ``R^d``, an adaptive client submits the encrypted
standard-basis queries ``e_0, ..., e_{d-1}`` with a one-element candidate list.
The decrypted responses are ``<x, e_j> = x_j`` and therefore reconstruct the
entire projected document vector in exactly ``d`` score-oracle calls.

The experiment intentionally uses the same spawned, public-only server as the
main protocol.  It demonstrates that query confidentiality and provider-side
exact-index non-disclosure do not imply document/database privacy when clients
may choose arbitrary encrypted queries and candidate identifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from system.packed_ckks import (
    CKKSParameters,
    EncryptedScoreResponse,
    client_decrypt_scores,
    create_context_pair,
    encrypt_query,
)
from system.unified_ckks_bench import SpawnedCKKSServer


SCHEMA = "score_oracle_extraction.v1"
METHOD = "segmented_score_packed"
DISCLAIMER = (
    "This adaptive chosen-query attack is outside the prescribed-client model. "
    "It shows that the protocol provides neither cryptographic database privacy "
    "nor document-vector privacy: a client that knows a candidate identifier can "
    "recover its projected vector through d basis-vector score queries. Rate "
    "limiting, admission control, and anomaly detection can raise extraction cost "
    "but are operational mitigations only; they do not establish document privacy."
)


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _load_projected_docs(path: Path) -> np.memmap:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(values, np.memmap) or values.ndim != 2:
        raise ValueError("projected-docs must be a two-dimensional NPY memmap")
    if values.dtype != np.float32:
        raise ValueError(f"projected-docs must be float32, got {values.dtype}")
    if not values.shape[0] or not values.shape[1]:
        raise ValueError("projected-docs matrix is empty")
    return values


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": int(len(array)),
        "sum": float(array.sum()),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _byte_summary(values: Sequence[int]) -> dict[str, float | int]:
    summary = _distribution(values)
    return {
        key: int(value) if key in {"count", "sum", "min", "max"} else value
        for key, value in summary.items()
    }


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _reconstruction_metrics(
    reconstructed: np.ndarray, exact: np.ndarray
) -> dict[str, float]:
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if reconstructed.shape != exact.shape or reconstructed.ndim != 1:
        raise ValueError("reconstructed and exact vectors must be aligned")
    error = reconstructed - exact
    exact_norm = float(np.linalg.norm(exact))
    reconstructed_norm = float(np.linalg.norm(reconstructed))
    denominator = exact_norm * reconstructed_norm
    cosine = (
        float(np.dot(reconstructed, exact) / denominator)
        if denominator > 0.0
        else (1.0 if exact_norm == reconstructed_norm else 0.0)
    )
    return {
        "cosine_similarity": cosine,
        "relative_l2_error": (
            float(np.linalg.norm(error) / exact_norm)
            if exact_norm > 0.0
            else float(np.linalg.norm(error))
        ),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
        "exact_l2_norm": exact_norm,
        "reconstructed_l2_norm": reconstructed_norm,
    }


def run_score_oracle_extraction(
    *,
    projected_docs_path: Path,
    candidate_ids: Sequence[int],
    max_dimension: int | None = None,
    warmup_queries: int = 1,
    ckks_parameters: CKKSParameters | None = None,
    response_timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Recover selected projected rows through adaptive basis-vector queries."""

    projected_docs_path = Path(projected_docs_path).expanduser().resolve()
    docs = _load_projected_docs(projected_docs_path)
    n_docs, stored_dimension = (int(docs.shape[0]), int(docs.shape[1]))
    ids = [int(value) for value in candidate_ids]
    if not ids:
        raise ValueError("candidate_ids cannot be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")
    if any(value < 0 or value >= n_docs for value in ids):
        raise ValueError("candidate id is outside the projected corpus")
    dimension = stored_dimension if max_dimension is None else int(max_dimension)
    if not 1 <= dimension <= stored_dimension:
        raise ValueError("max_dimension must lie within the stored dimension")
    if warmup_queries < 0:
        raise ValueError("warmup_queries cannot be negative")
    if response_timeout_s <= 0:
        raise ValueError("response_timeout_s must be positive")
    # Exact rows are deliberately not read until the online extraction is over.
    del docs

    ckks_parameters = ckks_parameters or CKKSParameters()
    setup_started = time.perf_counter_ns()
    pair = create_context_pair(ckks_parameters, max_query_dimension=dimension)
    context_setup_ms = _elapsed_ms(setup_started)
    server = SpawnedCKKSServer(
        public_context=pair.serialized_server_context,
        projected_docs_path=projected_docs_path,
        method=METHOD,
        response_timeout_s=response_timeout_s,
    )
    server_started = time.perf_counter_ns()
    server.start()
    server_startup_wall_ms = _elapsed_ms(server_started)

    phase_values: dict[str, list[float]] = {}
    request_sizes: list[int] = []
    response_sizes: list[int] = []
    ciphertext_counts: list[int] = []
    reconstructed: dict[int, np.ndarray] = {
        candidate_id: np.empty(dimension, dtype=np.float64)
        for candidate_id in ids
    }
    document_wall_ms: dict[int, float] = {}

    def record(name: str, value: float) -> None:
        phase_values.setdefault(name, []).append(float(value))

    try:
        basis_query = np.zeros(dimension, dtype=np.float64)
        for index in range(min(warmup_queries, dimension)):
            basis_query.fill(0.0)
            basis_query[index] = 1.0
            encrypted = encrypt_query(pair.client, basis_query)
            response_wire, _, _, _ = server.score(
                encrypted, np.asarray([ids[0]], dtype=np.int64)
            )
            client_decrypt_scores(pair.client, response_wire)

        attack_started = time.perf_counter_ns()
        for candidate_id in ids:
            document_started = time.perf_counter_ns()
            for coordinate in range(dimension):
                online_started = time.perf_counter_ns()
                started = time.perf_counter_ns()
                basis_query.fill(0.0)
                basis_query[coordinate] = 1.0
                record("client_construct_basis_query", _elapsed_ms(started))

                started = time.perf_counter_ns()
                encrypted = encrypt_query(pair.client, basis_query)
                record("client_encrypt_serialize", _elapsed_ms(started))

                response_wire, phases, metadata, ipc_ms = server.score(
                    encrypted, np.asarray([candidate_id], dtype=np.int64)
                )
                record("ipc_roundtrip", ipc_ms)
                for name, value in phases.items():
                    record(name, value)

                started = time.perf_counter_ns()
                parsed = EncryptedScoreResponse.from_bytes(response_wire)
                record("client_response_frame_parse", _elapsed_ms(started))
                started = time.perf_counter_ns()
                score = client_decrypt_scores(pair.client, parsed)
                record("client_decrypt_decode", _elapsed_ms(started))
                if score.shape != (1,):
                    raise RuntimeError("K=1 extraction response did not contain one score")
                reconstructed[candidate_id][coordinate] = float(score[0])
                record("online_end_to_end", _elapsed_ms(online_started))

                request_sizes.append(len(encrypted) + metadata["candidate_ids_bytes"])
                response_sizes.append(len(response_wire))
                ciphertext_counts.append(metadata["seal_ciphertext_count"])
            document_wall_ms[candidate_id] = _elapsed_ms(document_started)
        attack_wall_ms = _elapsed_ms(attack_started)
        if server.attestation is None:
            raise RuntimeError("server did not return public-only attestation")
        attestation = dict(server.attestation)
    finally:
        server.close()

    audit_docs = _load_projected_docs(projected_docs_path)
    documents: list[dict[str, Any]] = []
    for candidate_id in ids:
        exact = np.asarray(
            audit_docs[candidate_id, :dimension], dtype=np.float64
        ).copy()
        recovered = reconstructed[candidate_id]
        documents.append(
            {
                "candidate_id": candidate_id,
                "queries": dimension,
                "wall_ms": document_wall_ms[candidate_id],
                "metrics": _reconstruction_metrics(recovered, exact),
            }
        )
    del audit_docs

    query_count = dimension * len(ids)
    request_total = int(sum(request_sizes))
    response_total = int(sum(response_sizes))
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "name": "adaptive standard-basis score-oracle extraction",
            "construction": (
                "for each target candidate x and coordinate j, submit encrypted "
                "e_j with K=1 and set reconstructed[j] = decrypt(<x,e_j>)"
            ),
            "method": METHOD,
            "candidate_ids": ids,
            "stored_dimension": stored_dimension,
            "reconstructed_dimension": dimension,
            "queries_per_document": dimension,
            "actual_online_query_count": query_count,
            "warmup_queries_excluded": min(warmup_queries, dimension),
            "disclaimer": DISCLAIMER,
            "security_conclusion": {
                "query_confidentiality_refuted": False,
                "document_privacy_provided": False,
                "database_privacy_provided": False,
                "provider_exact_index_non_disclosure_alone_prevents_extraction": False,
                "rate_limiting_is_cryptographic_mitigation": False,
            },
        },
        "protocol": {
            "online_client_has_exact_projected_rows": False,
            "exact_rows_loaded_only_for_post_protocol_audit": True,
            "server_process_boundary": "Windows-compatible spawn + Pipe",
            "server_process_arguments": ["Pipe endpoint only"],
            "request": "one encrypted basis query + one uint64 candidate ID",
            "response": "one encrypted dot score",
            "payload_scope": "application bytes; Pipe/pickle/OS framing excluded",
        },
        "input": _file_metadata(projected_docs_path),
        "ckks": ckks_parameters.as_json(),
        "server_attestation": attestation,
        "setup_ms_excluded": {
            "client_context_and_keys": context_setup_ms,
            "spawn_and_public_server_initialization": server_startup_wall_ms,
            "serialized_public_context_bytes": len(pair.serialized_server_context),
        },
        "wall_latency_ms": {
            "attack_online_total": attack_wall_ms,
            "mean_per_document": float(
                np.mean(list(document_wall_ms.values()), dtype=np.float64)
            ),
            "mean_per_query": attack_wall_ms / query_count,
        },
        "phase_latency_ms": {
            name: _distribution(values) for name, values in phase_values.items()
        },
        "application_payload_bytes": {
            "request": _byte_summary(request_sizes),
            "response": _byte_summary(response_sizes),
            "total": _byte_summary(
                np.asarray(request_sizes, dtype=np.int64)
                + np.asarray(response_sizes, dtype=np.int64)
            ),
            "online_request_total": request_total,
            "online_response_total": response_total,
            "online_bidirectional_total": request_total + response_total,
            "public_context_setup_excluded": len(pair.serialized_server_context),
        },
        "seal_ciphertext_count": _byte_summary(ciphertext_counts),
        "documents": documents,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "client_pid": os.getpid(),
            "logical_cpus": os.cpu_count(),
        },
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demonstrate adaptive CKKS score-oracle document extraction"
    )
    parser.add_argument("--projected-docs", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=int, nargs="+", required=True)
    parser.add_argument("--max-dimension", type=int)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--response-timeout-s", type=float, default=600.0)
    parser.add_argument("--poly-modulus-degree", type=int, default=8192)
    parser.add_argument(
        "--coeff-mod-bit-sizes", type=int, nargs="+", default=[60, 40, 60]
    )
    parser.add_argument("--scale-bits", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_score_oracle_extraction(
        projected_docs_path=args.projected_docs,
        candidate_ids=args.candidate_ids,
        max_dimension=args.max_dimension,
        warmup_queries=args.warmup_queries,
        ckks_parameters=CKKSParameters(
            poly_modulus_degree=args.poly_modulus_degree,
            coeff_mod_bit_sizes=tuple(args.coeff_mod_bit_sizes),
            scale_bits=args.scale_bits,
        ),
        response_timeout_s=args.response_timeout_s,
    )
    _atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "queries": result["experiment"]["actual_online_query_count"],
                "attack_wall_ms": result["wall_latency_ms"]["attack_online_total"],
                "application_bytes": result["application_payload_bytes"]
                ["online_bidirectional_total"],
                "documents": [
                    {
                        "candidate_id": row["candidate_id"],
                        "cosine": row["metrics"]["cosine_similarity"],
                        "relative_l2": row["metrics"]["relative_l2_error"],
                    }
                    for row in result["documents"]
                ],
                "output": str(Path(args.output).resolve()),
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
