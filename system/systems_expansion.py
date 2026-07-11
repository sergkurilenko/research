"""Reproducible systems expansion for the packed-CKKS reranker.

This benchmark complements :mod:`system.unified_ckks_bench` with four
measurements that are deliberately kept separate from retrieval quality:

* closed-loop provider-side worker scaling at several concurrency levels;
* fresh-session ("cold") versus persistent-session ("warm") latency;
* exact serialized public-context/key sizes and setup amortization; and
* real persistent TCP and TLS 1.3 transports on the loopback interface.

The concurrency experiment excludes client encryption/decryption so it can
measure saturation of the provider-side CKKS service on one host.  Each
closed-loop client thread owns one independently spawned public-only worker.
The session experiment separately reports the complete client path.

Run this module with thread-pool environment variables fixed to one, e.g.::

    $env:OMP_NUM_THREADS='1'
    $env:MKL_NUM_THREADS='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:NUMEXPR_NUM_THREADS='1'
    python -m system.systems_expansion ...

Validated with Python 3.11.15, TenSEAL 0.3.16, and Windows ``spawn``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import socket
import ssl
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil

from system.packed_ckks import (
    CKKSParameters,
    _CONTEXT_MAGIC,
    _decode_frame,
    _seal_save,
    client_decrypt_scores,
    create_context_pair,
    deserialize_server_context,
    encrypt_query,
    server_block_packed_scores,
    server_naive_scores,
    server_segmented_score_packed,
)
from system.unified_ckks_bench import METHODS, SpawnedCKKSServer
from system.tls_test_material import generate_loopback_tls_material


SCHEMA_VERSION = "systems_expansion.v1"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_FRAME_MAGIC = b"SXF1"
_FRAME_PREFIX = struct.Struct("!4sBH")
_FRAME_LENGTH = struct.Struct("!Q")
_MAX_FRAME_PART = 512 * 1024 * 1024
_MSG_INIT = 1
_MSG_READY = 2
_MSG_SCORE = 3
_MSG_RESULT = 4
_MSG_STOP = 5
_MSG_STOPPED = 6
_MSG_ERROR = 255


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def distribution(values: Sequence[float]) -> dict[str, Any]:
    """Return a JSON-safe latency/size distribution with retained samples."""

    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "samples": [],
        }
    return {
        "count": len(samples),
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "p50": _percentile(samples, 50),
        "p95": _percentile(samples, 95),
        "p99": _percentile(samples, 99),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _file_metadata(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _gpu_metadata() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    line = result.stdout.strip().splitlines()[0]
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 3:
        return {"raw": line}
    return {
        "name": fields[0],
        "driver_version": fields[1],
        "memory_mib": int(fields[2]),
        "used_by_ckks": False,
    }


def environment_metadata() -> dict[str, Any]:
    try:
        from threadpoolctl import threadpool_info

        pools: list[dict[str, Any]] = threadpool_info()
    except ImportError:
        pools = []
    process = psutil.Process()
    try:
        affinity: list[int] | None = process.cpu_affinity()
    except (AttributeError, psutil.Error):
        affinity = None
    try:
        priority: Any = int(process.nice())
    except psutil.Error:
        priority = None
    return {
        "captured_utc": _utc_now(),
        "platform": platform.platform(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu": {
            "name": _cpu_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "affinity_logical_cpus": affinity,
        },
        "memory_bytes": int(psutil.virtual_memory().total),
        "python": sys.version,
        "packages": {
            "tenseal": _package_version("tenseal"),
            "numpy": np.__version__,
            "psutil": psutil.__version__,
            "threadpoolctl": _package_version("threadpoolctl"),
        },
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "loaded_native_threadpools": pools,
        "process": {
            "pid": os.getpid(),
            "priority": priority,
            "affinity_was_modified": False,
        },
        "gpu": _gpu_metadata(),
        "ckks_backend": "Microsoft SEAL through TenSEAL sealapi; CPU only",
    }


def validate_thread_environment() -> None:
    invalid = {
        name: os.environ.get(name)
        for name in THREAD_ENV_VARS[:4]
        if os.environ.get(name) != "1"
    }
    if invalid:
        detail = ", ".join(f"{key}={value!r}" for key, value in invalid.items())
        raise RuntimeError(
            "set all primary thread-pool variables to '1' before benchmarking: "
            + detail
        )


def _load_docs(path: Path) -> np.memmap:
    docs = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    if not isinstance(docs, np.memmap) or docs.ndim != 2 or docs.dtype != np.float32:
        raise ValueError("projected docs must be a two-dimensional float32 NPY memmap")
    return docs


def _build_request_pool(
    pair: Any,
    *,
    dimension: int,
    n_docs: int,
    candidate_count: int,
    pool_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    for _ in range(pool_size):
        query = rng.normal(size=dimension)
        query /= np.linalg.norm(query)
        ids = rng.choice(n_docs, size=candidate_count, replace=False).astype(np.int64)
        wire = encrypt_query(pair.client, query)
        result.append({"query": query, "query_wire": wire, "candidate_ids": ids})
    return result


def _process_cpu_seconds(processes: Sequence[psutil.Process]) -> float:
    total = 0.0
    for process in processes:
        try:
            times = process.cpu_times()
            total += float(times.user + times.system)
        except psutil.Error:
            continue
    return total


class _RSSMonitor:
    def __init__(self, processes: Sequence[psutil.Process], interval_s: float = 0.02):
        self.processes = tuple(processes)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.total_samples: list[int] = []
        self.max_worker_samples: list[int] = []

    def _sample(self) -> None:
        values: list[int] = []
        for process in self.processes:
            try:
                values.append(int(process.memory_info().rss))
            except psutil.Error:
                continue
        if values:
            self.total_samples.append(sum(values))
            self.max_worker_samples.append(max(values))

    def start(self) -> None:
        self._sample()

        def run() -> None:
            while not self.stop_event.wait(self.interval_s):
                self._sample()

        self.thread = threading.Thread(target=run, name="ckks-rss-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._sample()
        return {
            "sampling_interval_ms": self.interval_s * 1000,
            "sample_count": len(self.total_samples),
            "peak_sum_worker_rss_bytes": max(self.total_samples, default=0),
            "peak_single_worker_rss_bytes": max(self.max_worker_samples, default=0),
            "scope": (
                "sum of per-worker OS RSS/working-set samples; shared pages can be "
                "counted once per process"
            ),
        }


def _closed_loop_workload(
    servers: Sequence[SpawnedCKKSServer],
    request_pool: Sequence[Mapping[str, Any]],
    *,
    requests_per_worker: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
    list[dict[str, Any]],
]:
    """Run one closed-loop client thread per independently spawned worker."""

    start_event = threading.Event()

    def run_one(worker_index: int) -> dict[str, Any]:
        start_event.wait()
        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        last_response: bytes | None = None
        last_request_index: int | None = None
        for request_number in range(requests_per_worker):
            request_index = (
                worker_index * requests_per_worker + request_number
            ) % len(request_pool)
            request = request_pool[request_index]
            try:
                response, phases, metadata, roundtrip_ms = servers[worker_index].score(
                    request["query_wire"], request["candidate_ids"]
                )
                records.append(
                    {
                        "worker_index": worker_index,
                        "request_number": request_number,
                        "request_pool_index": request_index,
                        "roundtrip_ms": float(roundtrip_ms),
                        "server_total_ms": float(phases["server_total"]),
                        "server_kernel_total_ms": float(
                            phases["server_kernel_total"]
                        ),
                        "response_bytes": len(response),
                        "request_bytes": len(request["query_wire"])
                        + int(metadata["candidate_ids_bytes"]),
                        "seal_ciphertext_count": int(
                            metadata["seal_ciphertext_count"]
                        ),
                    }
                )
                last_response = response
                last_request_index = request_index
            except BaseException as exc:
                failures.append(
                    {
                        "worker_index": worker_index,
                        "request_number": request_number,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        return {
            "records": records,
            "failures": failures,
            "last_response": last_response,
            "last_request_index": last_request_index,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
        futures = [executor.submit(run_one, index) for index in range(len(servers))]
        wall_started = time.perf_counter_ns()
        start_event.set()
        worker_results = [future.result() for future in futures]
        wall_ms = _elapsed_ms(wall_started)

    records = [record for result in worker_results for record in result["records"]]
    failures = [failure for result in worker_results for failure in result["failures"]]
    audit = [
        {
            "last_response": result["last_response"],
            "last_request_index": result["last_request_index"],
        }
        for result in worker_results
    ]
    return records, failures, wall_ms, audit


def run_scaling_condition(
    *,
    public_context: bytes,
    client_context: Any,
    projected_docs_path: Path,
    method: str,
    concurrency: int,
    requests_per_worker: int,
    warmup_requests_per_worker: int,
    request_pool: Sequence[Mapping[str, Any]],
    restart: int,
    logical_cpus: int,
    response_timeout_s: float,
) -> dict[str, Any]:
    servers = [
        SpawnedCKKSServer(
            public_context=public_context,
            projected_docs_path=projected_docs_path,
            method=method,
            response_timeout_s=response_timeout_s,
        )
        for _ in range(concurrency)
    ]
    startup_started = time.perf_counter_ns()
    try:
        for server in servers:
            server.start()
        startup_ms = _elapsed_ms(startup_started)
        processes = [psutil.Process(int(server.pid)) for server in servers if server.pid]

        if warmup_requests_per_worker:
            warmup_records, warmup_failures, _, _ = _closed_loop_workload(
                servers,
                request_pool,
                requests_per_worker=warmup_requests_per_worker,
            )
            if warmup_failures or len(warmup_records) != (
                concurrency * warmup_requests_per_worker
            ):
                raise RuntimeError(f"scaling warmup failed: {warmup_failures}")

        cpu_before = _process_cpu_seconds(processes)
        monitor = _RSSMonitor(processes)
        monitor.start()
        records, failures, wall_ms, audit = _closed_loop_workload(
            servers,
            request_pool,
            requests_per_worker=requests_per_worker,
        )
        memory = monitor.stop()
        cpu_after = _process_cpu_seconds(processes)

        numerical_errors: list[float] = []
        docs = _load_docs(projected_docs_path)
        for item in audit:
            if item["last_response"] is None:
                continue
            request = request_pool[int(item["last_request_index"])]
            decrypted = client_decrypt_scores(
                client_context, item["last_response"]
            )
            reference = (
                np.asarray(docs[request["candidate_ids"]], dtype=np.float64)
                @ np.asarray(request["query"], dtype=np.float64)
            )
            numerical_errors.append(float(np.max(np.abs(decrypted - reference))))
        del docs

        successful = len(records)
        attempted = concurrency * requests_per_worker
        wall_s = wall_ms / 1000.0
        cpu_seconds = max(0.0, cpu_after - cpu_before)
        effective_cores = cpu_seconds / wall_s if wall_s else 0.0
        return {
            "method": method,
            "concurrency": concurrency,
            "restart": restart,
            "workers": concurrency,
            "closed_loop_clients": concurrency,
            "requests_per_worker": requests_per_worker,
            "attempted_requests": attempted,
            "successful_requests": successful,
            "failed_requests": len(failures),
            "failure_rate": len(failures) / attempted,
            "failures": failures,
            "startup_ms_excluded_from_workload": startup_ms,
            "workload_wall_ms": wall_ms,
            "throughput_qps": successful / wall_s if wall_s else 0.0,
            "latency_ms": {
                "pipe_roundtrip": distribution(
                    [record["roundtrip_ms"] for record in records]
                ),
                "server_total": distribution(
                    [record["server_total_ms"] for record in records]
                ),
                "server_kernel_total": distribution(
                    [record["server_kernel_total_ms"] for record in records]
                ),
            },
            "payload_bytes": {
                "request": distribution(
                    [float(record["request_bytes"]) for record in records]
                ),
                "response": distribution(
                    [float(record["response_bytes"]) for record in records]
                ),
            },
            "cpu": {
                "aggregate_worker_cpu_seconds": cpu_seconds,
                "effective_busy_cores": effective_cores,
                "host_normalized_percent": (
                    100.0 * effective_cores / logical_cpus if logical_cpus else None
                ),
                "normalization_denominator": "host logical CPUs",
            },
            "memory": memory,
            "correctness": {
                "audited_responses": len(numerical_errors),
                "max_abs_error": max(numerical_errors, default=None),
            },
            "server_attestations": [server.attestation for server in servers],
        }
    finally:
        for server in servers:
            server.close()


def aggregate_scaling(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["concurrency"]))].append(row)
    result: list[dict[str, Any]] = []
    for (method, concurrency), group in sorted(grouped.items()):
        latencies = [
            float(value)
            for row in group
            for value in row["latency_ms"]["pipe_roundtrip"]["samples"]
        ]
        server_latencies = [
            float(value)
            for row in group
            for value in row["latency_ms"]["server_total"]["samples"]
        ]
        attempted = sum(int(row["attempted_requests"]) for row in group)
        failed = sum(int(row["failed_requests"]) for row in group)
        result.append(
            {
                "method": method,
                "concurrency": concurrency,
                "restart_count": len(group),
                "attempted_requests": attempted,
                "failed_requests": failed,
                "failure_rate": failed / attempted,
                "throughput_qps_across_restarts": distribution(
                    [float(row["throughput_qps"]) for row in group]
                ),
                "pooled_pipe_roundtrip_ms": distribution(latencies),
                "pooled_server_total_ms": distribution(server_latencies),
                "effective_busy_cores_across_restarts": distribution(
                    [float(row["cpu"]["effective_busy_cores"]) for row in group]
                ),
                "peak_sum_worker_rss_bytes": max(
                    int(row["memory"]["peak_sum_worker_rss_bytes"])
                    for row in group
                ),
                "peak_single_worker_rss_bytes": max(
                    int(row["memory"]["peak_single_worker_rss_bytes"])
                    for row in group
                ),
                "max_abs_error": max(
                    float(row["correctness"]["max_abs_error"])
                    for row in group
                    if row["correctness"]["max_abs_error"] is not None
                ),
            }
        )
    return result


def inspect_key_material(pair: Any) -> dict[str, Any]:
    header, payloads = _decode_frame(
        _CONTEXT_MAGIC, pair.serialized_server_context
    )
    roles = list(header["payload_roles"])
    components = {role: len(payload) for role, payload in zip(roles, payloads)}
    envelope_bytes = len(pair.serialized_server_context)
    return {
        "serialized_public_server_context_bytes": envelope_bytes,
        "context_envelope_overhead_bytes": envelope_bytes - sum(components.values()),
        "public_components_bytes": components,
        "client_secret_key_serialized_bytes": len(_seal_save(pair.client.secret_key)),
        "galois_key_objects": int(pair.client.galois_keys.size()),
        "rotation_steps": list(pair.client.rotation_steps),
        "contains_relinearization_keys": False,
        "contains_secret_key_in_server_context": False,
        "exact_serialization_scope": (
            "SEAL save() payloads plus packed_ckks public-context envelope"
        ),
    }


def run_session_benchmark(
    *,
    projected_docs_path: Path,
    dimension: int,
    candidate_count: int,
    repeats: int,
    warm_requests: int,
    seed: int,
    parameters: CKKSParameters,
    response_timeout_s: float,
) -> dict[str, Any]:
    docs_shape = _load_docs(projected_docs_path).shape
    if int(docs_shape[1]) != dimension:
        raise ValueError("session dimension differs from projected document dimension")
    sessions: list[dict[str, Any]] = []
    key_material: dict[str, Any] | None = None
    payload_reference: dict[str, int] | None = None
    rng = np.random.default_rng(seed)

    for restart in range(repeats):
        setup_started = time.perf_counter_ns()
        pair = create_context_pair(parameters, max_query_dimension=dimension)
        context_setup_ms = _elapsed_ms(setup_started)
        if key_material is None:
            key_material = inspect_key_material(pair)

        queries = rng.normal(size=(warm_requests + 1, dimension))
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        candidate_rows = [
            rng.choice(docs_shape[0], size=candidate_count, replace=False).astype(
                np.int64
            )
            for _ in range(warm_requests + 1)
        ]
        server = SpawnedCKKSServer(
            public_context=pair.serialized_server_context,
            projected_docs_path=projected_docs_path,
            method="segmented_score_packed",
            response_timeout_s=response_timeout_s,
        )
        startup_started = time.perf_counter_ns()
        server.start()
        server_startup_ms = _elapsed_ms(startup_started)
        try:
            request_rows: list[dict[str, Any]] = []
            for request_index, (query, ids) in enumerate(zip(queries, candidate_rows)):
                e2e_started = time.perf_counter_ns()
                encrypt_started = time.perf_counter_ns()
                query_wire = encrypt_query(pair.client, query)
                encrypt_ms = _elapsed_ms(encrypt_started)
                response_wire, phases, metadata, pipe_ms = server.score(query_wire, ids)
                decrypt_started = time.perf_counter_ns()
                scores = client_decrypt_scores(pair.client, response_wire)
                decrypt_ms = _elapsed_ms(decrypt_started)
                e2e_ms = _elapsed_ms(e2e_started)
                request_rows.append(
                    {
                        "request_index": request_index,
                        "session_state": "first" if request_index == 0 else "warm",
                        "client_encrypt_serialize_ms": encrypt_ms,
                        "pipe_roundtrip_ms": pipe_ms,
                        "server_total_ms": float(phases["server_total"]),
                        "client_decrypt_decode_ms": decrypt_ms,
                        "client_end_to_end_ms": e2e_ms,
                        "query_wire_bytes": len(query_wire),
                        "candidate_id_bytes": int(metadata["candidate_ids_bytes"]),
                        "response_wire_bytes": len(response_wire),
                        "score_count": len(scores),
                    }
                )
            if payload_reference is None:
                first = request_rows[0]
                payload_reference = {
                    "query_wire_bytes": int(first["query_wire_bytes"]),
                    "candidate_id_bytes": int(first["candidate_id_bytes"]),
                    "request_application_bytes": int(first["query_wire_bytes"])
                    + int(first["candidate_id_bytes"]),
                    "response_wire_bytes": int(first["response_wire_bytes"]),
                }
            server_rss = None
            if server.pid:
                try:
                    server_rss = int(psutil.Process(server.pid).memory_info().rss)
                except psutil.Error:
                    pass
            sessions.append(
                {
                    "restart": restart,
                    "context_key_setup_and_local_verification_ms": context_setup_ms,
                    "spawned_server_startup_ms": server_startup_ms,
                    "ready_to_first_result_ms": context_setup_ms
                    + server_startup_ms
                    + float(request_rows[0]["client_end_to_end_ms"]),
                    "server_rss_after_requests_bytes": server_rss,
                    "server_attestation": server.attestation,
                    "requests": request_rows,
                }
            )
        finally:
            server.close()

    assert key_material is not None and payload_reference is not None
    first_rows = [session["requests"][0] for session in sessions]
    warm_rows = [row for session in sessions for row in session["requests"][1:]]
    context_bytes = int(key_material["serialized_public_server_context_bytes"])
    per_request_bytes = (
        payload_reference["request_application_bytes"]
        + payload_reference["response_wire_bytes"]
    )
    warm_e2e_median = distribution(
        [float(row["client_end_to_end_ms"]) for row in warm_rows]
    )["median"]
    setup_time_median = statistics.median(
        float(session["context_key_setup_and_local_verification_ms"])
        + float(session["spawned_server_startup_ms"])
        for session in sessions
    )
    amortization: dict[str, Any] = {}
    for request_count in (1, 10, 100):
        amortization[str(request_count)] = {
            "context_bytes_per_request": context_bytes / request_count,
            "effective_total_bytes_per_request": context_bytes / request_count
            + per_request_bytes,
            "setup_ms_per_request": setup_time_median / request_count,
            "effective_warm_path_plus_setup_ms_per_request": (
                float(warm_e2e_median) + setup_time_median / request_count
            ),
        }
    return {
        "definition": {
            "fresh_session": (
                "new CKKS key material, serialized public context, newly spawned "
                "server, then first full client request; OS page cache is not flushed"
            ),
            "warm_session": (
                "subsequent full encrypt + Pipe + server + decrypt requests on the "
                "same keys and persistent server process"
            ),
            "setup_timer_scope": (
                "create_context_pair includes key generation, SEAL serialization, and "
                "a local public-context deserialize/role check"
            ),
        },
        "restart_count": repeats,
        "warm_requests_per_restart": warm_requests,
        "key_material": key_material,
        "payload_reference": payload_reference,
        "summary_ms": {
            "context_key_setup_and_local_verification": distribution(
                [
                    float(session["context_key_setup_and_local_verification_ms"])
                    for session in sessions
                ]
            ),
            "spawned_server_startup": distribution(
                [float(session["spawned_server_startup_ms"]) for session in sessions]
            ),
            "ready_to_first_result": distribution(
                [float(session["ready_to_first_result_ms"]) for session in sessions]
            ),
            "first_full_client_request": distribution(
                [float(row["client_end_to_end_ms"]) for row in first_rows]
            ),
            "warm_full_client_request": distribution(
                [float(row["client_end_to_end_ms"]) for row in warm_rows]
            ),
            "warm_client_encrypt_serialize": distribution(
                [float(row["client_encrypt_serialize_ms"]) for row in warm_rows]
            ),
            "warm_pipe_roundtrip": distribution(
                [float(row["pipe_roundtrip_ms"]) for row in warm_rows]
            ),
            "warm_server_total": distribution(
                [float(row["server_total_ms"]) for row in warm_rows]
            ),
            "warm_client_decrypt_decode": distribution(
                [float(row["client_decrypt_decode_ms"]) for row in warm_rows]
            ),
        },
        "amortization": amortization,
        "sessions": sessions,
    }


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while receiving a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(connection: socket.socket, message_type: int, parts: Sequence[bytes]) -> int:
    if not 0 <= message_type <= 255:
        raise ValueError("message type must fit in one byte")
    if len(parts) > 65535:
        raise ValueError("too many frame parts")
    if any(not isinstance(part, bytes) for part in parts):
        raise TypeError("all frame parts must be bytes")
    header = _FRAME_PREFIX.pack(_FRAME_MAGIC, message_type, len(parts)) + b"".join(
        _FRAME_LENGTH.pack(len(part)) for part in parts
    )
    connection.sendall(header)
    for part in parts:
        connection.sendall(part)
    return len(header) + sum(len(part) for part in parts)


def recv_frame(connection: socket.socket) -> tuple[int, tuple[bytes, ...], int]:
    prefix = _recv_exact(connection, _FRAME_PREFIX.size)
    magic, message_type, part_count = _FRAME_PREFIX.unpack(prefix)
    if magic != _FRAME_MAGIC:
        raise ValueError("invalid systems-expansion socket frame magic")
    raw_lengths = _recv_exact(connection, _FRAME_LENGTH.size * part_count)
    lengths = [
        _FRAME_LENGTH.unpack_from(raw_lengths, index * _FRAME_LENGTH.size)[0]
        for index in range(part_count)
    ]
    if any(length > _MAX_FRAME_PART for length in lengths):
        raise ValueError("socket frame part exceeds the safety bound")
    parts = tuple(_recv_exact(connection, int(length)) for length in lengths)
    return message_type, parts, len(prefix) + len(raw_lengths) + sum(lengths)


def _socket_error_payload(exc: BaseException) -> bytes:
    return json.dumps(
        {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _socket_server_worker(
    control: Any,
    *,
    tls_enabled: bool,
    certificate_path: str | None,
    private_key_path: str | None,
) -> None:
    listener: socket.socket | None = None
    connection: socket.socket | None = None
    try:
        server_tls_context: ssl.SSLContext | None = None
        tls_context_setup_ms = 0.0
        if tls_enabled:
            if not certificate_path or not private_key_path:
                raise ValueError("TLS server needs a certificate and private key")
            tls_context_started = time.perf_counter_ns()
            server_tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server_tls_context.minimum_version = ssl.TLSVersion.TLSv1_3
            server_tls_context.load_cert_chain(certificate_path, private_key_path)
            tls_context_setup_ms = _elapsed_ms(tls_context_started)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])
        control.send(
            {
                "type": "listening",
                "port": port,
                "pid": os.getpid(),
                "server_tls_context_and_cert_load_ms": tls_context_setup_ms,
            }
        )
        raw_connection, peer = listener.accept()
        raw_connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if tls_enabled:
            assert server_tls_context is not None
            connection = server_tls_context.wrap_socket(raw_connection, server_side=True)
        else:
            connection = raw_connection

        message_type, parts, init_frame_bytes = recv_frame(connection)
        if message_type != _MSG_INIT or len(parts) != 2:
            raise ValueError("first socket frame must be INIT(metadata, public_context)")
        metadata = json.loads(parts[0].decode("utf-8"))
        public_context = parts[1]
        startup_started = time.perf_counter_ns()
        ckks_context = deserialize_server_context(public_context)
        docs = _load_docs(Path(metadata["projected_docs_path"]))
        method = str(metadata["method"])
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        block_size_raw = metadata.get("block_size")
        block_size = None if block_size_raw is None else int(block_size_raw)
        startup_ms = _elapsed_ms(startup_started)
        ready = {
            "pid": os.getpid(),
            "peer": [str(peer[0]), int(peer[1])],
            "bound_address": "127.0.0.1",
            "loopback_only": True,
            "tls": tls_enabled,
            "tls_version": connection.version() if tls_enabled else None,
            "cipher": list(connection.cipher()) if tls_enabled else None,
            "context_is_public": bool(ckks_context.is_public()),
            "has_secret_key": bool(ckks_context.has_secret_key()),
            "has_decryptor_attribute": hasattr(ckks_context, "decryptor"),
            "received_context_bytes": len(public_context),
            "received_init_frame_bytes": init_frame_bytes,
            "docs_shape": [int(value) for value in docs.shape],
            "docs_read_only": not bool(docs.flags.writeable),
            "context_deserialize_and_memmap_ms": startup_ms,
        }
        send_frame(
            connection,
            _MSG_READY,
            (json.dumps(ready, sort_keys=True).encode("utf-8"),),
        )

        while True:
            message_type, parts, request_frame_bytes = recv_frame(connection)
            if message_type == _MSG_STOP:
                send_frame(connection, _MSG_STOPPED, ())
                break
            if message_type != _MSG_SCORE or len(parts) != 2:
                raise ValueError("expected SCORE(query, candidate_ids)")
            request_started = time.perf_counter_ns()
            query_wire, candidate_wire = parts
            if len(candidate_wire) % 8:
                raise ValueError("candidate-id payload is not uint64 aligned")
            ids = np.frombuffer(candidate_wire, dtype="<u8").astype(np.int64, copy=False)
            if not len(ids) or np.any(ids >= len(docs)):
                raise IndexError("candidate ID outside projected corpus")
            candidates = np.ascontiguousarray(docs[ids], dtype=np.float64)
            if method == "naive_per_candidate":
                response = server_naive_scores(ckks_context, query_wire, candidates)
            elif method == "block_packed":
                response = server_block_packed_scores(
                    ckks_context, query_wire, candidates, block_size=block_size
                )
            else:
                response = server_segmented_score_packed(
                    ckks_context, query_wire, candidates, block_size=block_size
                )
            response_wire = response.to_bytes()
            server_process_ms = _elapsed_ms(request_started)
            response_metadata = json.dumps(
                {
                    "server_process_ms": server_process_ms,
                    "request_frame_bytes": request_frame_bytes,
                    "response_wire_bytes": len(response_wire),
                },
                sort_keys=True,
            ).encode("utf-8")
            send_frame(
                connection,
                _MSG_RESULT,
                (response_wire, response_metadata),
            )
    except EOFError:
        pass
    except BaseException as exc:
        if connection is not None:
            try:
                send_frame(connection, _MSG_ERROR, (_socket_error_payload(exc),))
            except (OSError, EOFError):
                pass
        try:
            control.send({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if listener is not None:
            listener.close()
        control.close()


class LoopbackSocketServer:
    def __init__(
        self,
        *,
        public_context: bytes,
        projected_docs_path: Path,
        method: str,
        tls_enabled: bool,
        certificate_path: Path | None = None,
        private_key_path: Path | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.public_context = public_context
        self.projected_docs_path = Path(projected_docs_path).resolve()
        self.method = method
        self.tls_enabled = tls_enabled
        self.certificate_path = certificate_path
        self.private_key_path = private_key_path
        self.timeout_s = timeout_s
        self.process: mp.Process | None = None
        self.control: Any = None
        self.connection: socket.socket | None = None
        self.attestation: dict[str, Any] | None = None
        self.startup: dict[str, Any] | None = None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def start(self) -> dict[str, Any]:
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_socket_server_worker,
            kwargs={
                "control": child,
                "tls_enabled": self.tls_enabled,
                "certificate_path": (
                    str(self.certificate_path) if self.certificate_path else None
                ),
                "private_key_path": (
                    str(self.private_key_path) if self.private_key_path else None
                ),
            },
            name="public-ckks-loopback-socket-server",
        )
        process.start()
        child.close()
        self.process = process
        self.control = parent
        if not parent.poll(self.timeout_s):
            raise TimeoutError("socket server did not bind in time")
        listening = parent.recv()
        if listening.get("type") != "listening":
            raise RuntimeError(f"socket server failed before bind: {listening}")
        port = int(listening["port"])

        connect_started = time.perf_counter_ns()
        raw = socket.create_connection(("127.0.0.1", port), timeout=self.timeout_s)
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        tcp_connect_ms = _elapsed_ms(connect_started)
        tls_handshake_ms = 0.0
        if self.tls_enabled:
            client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            client_context.minimum_version = ssl.TLSVersion.TLSv1_3
            client_context.check_hostname = False
            client_context.verify_mode = ssl.CERT_NONE
            tls_started = time.perf_counter_ns()
            connection = client_context.wrap_socket(raw, server_hostname="localhost")
            tls_handshake_ms = _elapsed_ms(tls_started)
        else:
            connection = raw
        connection.settimeout(self.timeout_s)
        self.connection = connection

        init_metadata = json.dumps(
            {
                "projected_docs_path": str(self.projected_docs_path),
                "method": self.method,
                "block_size": None,
            },
            sort_keys=True,
        ).encode("utf-8")
        init_started = time.perf_counter_ns()
        init_sent_bytes = send_frame(
            connection, _MSG_INIT, (init_metadata, self.public_context)
        )
        message_type, parts, ready_frame_bytes = recv_frame(connection)
        init_roundtrip_ms = _elapsed_ms(init_started)
        if message_type == _MSG_ERROR:
            raise RuntimeError(parts[0].decode("utf-8", errors="replace"))
        if message_type != _MSG_READY or len(parts) != 1:
            raise RuntimeError("socket server did not return READY")
        self.attestation = json.loads(parts[0].decode("utf-8"))
        self.startup = {
            "server_tls_context_and_cert_load_ms": float(
                listening["server_tls_context_and_cert_load_ms"]
            ),
            "tcp_connect_ms": tcp_connect_ms,
            "tls_handshake_ms": tls_handshake_ms,
            "context_init_roundtrip_ms": init_roundtrip_ms,
            "init_application_frame_bytes": init_sent_bytes,
            "ready_application_frame_bytes": ready_frame_bytes,
            "tls_version": connection.version() if self.tls_enabled else None,
            "cipher": list(connection.cipher()) if self.tls_enabled else None,
            "certificate_verification": (
                "disabled for self-signed loopback benchmark certificate"
                if self.tls_enabled
                else None
            ),
        }
        return self.attestation

    def score(self, query_wire: bytes, candidate_ids: np.ndarray) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("socket server is not connected")
        candidate_wire = np.asarray(candidate_ids, dtype="<u8").tobytes(order="C")
        started = time.perf_counter_ns()
        request_frame_bytes = send_frame(
            self.connection, _MSG_SCORE, (query_wire, candidate_wire)
        )
        message_type, parts, response_frame_bytes = recv_frame(self.connection)
        roundtrip_ms = _elapsed_ms(started)
        if message_type == _MSG_ERROR:
            raise RuntimeError(parts[0].decode("utf-8", errors="replace"))
        if message_type != _MSG_RESULT or len(parts) != 2:
            raise RuntimeError("socket server returned an invalid result frame")
        metadata = json.loads(parts[1].decode("utf-8"))
        return {
            "response": parts[0],
            "roundtrip_ms": roundtrip_ms,
            "server_process_ms": float(metadata["server_process_ms"]),
            "transport_and_framing_ms": roundtrip_ms
            - float(metadata["server_process_ms"]),
            "request_application_frame_bytes": request_frame_bytes,
            "response_application_frame_bytes": response_frame_bytes,
        }

    def close(self) -> None:
        connection, process, control = self.connection, self.process, self.control
        self.connection = None
        self.process = None
        self.control = None
        if connection is not None:
            try:
                send_frame(connection, _MSG_STOP, ())
                recv_frame(connection)
            except (OSError, EOFError, TimeoutError):
                pass
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
        if control is not None:
            control.close()


def run_loopback_benchmark(
    *,
    pair: Any,
    projected_docs_path: Path,
    request_pool: Sequence[Mapping[str, Any]],
    warmups: int,
    repeats: int,
    certificate_path: Path,
    private_key_path: Path,
    response_timeout_s: float,
) -> dict[str, Any]:
    transports: dict[str, Any] = {}
    for name, tls_enabled in (("tcp", False), ("tls13", True)):
        server = LoopbackSocketServer(
            public_context=pair.serialized_server_context,
            projected_docs_path=projected_docs_path,
            method="segmented_score_packed",
            tls_enabled=tls_enabled,
            certificate_path=certificate_path if tls_enabled else None,
            private_key_path=private_key_path if tls_enabled else None,
            timeout_s=response_timeout_s,
        )
        server.start()
        try:
            for index in range(warmups):
                request = request_pool[index % len(request_pool)]
                server.score(request["query_wire"], request["candidate_ids"])
            records: list[dict[str, Any]] = []
            for index in range(repeats):
                request = request_pool[(warmups + index) % len(request_pool)]
                row = server.score(request["query_wire"], request["candidate_ids"])
                decrypt_started = time.perf_counter_ns()
                scores = client_decrypt_scores(pair.client, row["response"])
                row["client_decrypt_decode_ms"] = _elapsed_ms(decrypt_started)
                row["score_count"] = len(scores)
                row.pop("response")
                records.append(row)
            transports[name] = {
                "tls": tls_enabled,
                "startup": server.startup,
                "server_attestation": server.attestation,
                "roundtrip_ms": distribution(
                    [float(row["roundtrip_ms"]) for row in records]
                ),
                "server_process_ms": distribution(
                    [float(row["server_process_ms"]) for row in records]
                ),
                "transport_and_framing_ms": distribution(
                    [float(row["transport_and_framing_ms"]) for row in records]
                ),
                "client_decrypt_decode_ms": distribution(
                    [float(row["client_decrypt_decode_ms"]) for row in records]
                ),
                "request_application_frame_bytes": distribution(
                    [float(row["request_application_frame_bytes"]) for row in records]
                ),
                "response_application_frame_bytes": distribution(
                    [float(row["response_application_frame_bytes"]) for row in records]
                ),
                "records": records,
            }
        finally:
            server.close()
    tcp_overhead = transports["tcp"]["transport_and_framing_ms"]["median"]
    tls_overhead = transports["tls13"]["transport_and_framing_ms"]["median"]
    return {
        "scope": {
            "network": (
                "real persistent IPv4 stream sockets bound to 127.0.0.1; this is a "
                "loopback protocol/framing benchmark, not a LAN or WAN claim"
            ),
            "framing": (
                "custom deterministic binary frame with message type, part count, "
                "uint64 part lengths, and existing serialized CKKS envelopes"
            ),
            "byte_counts": (
                "application-frame bytes; TCP/IP headers and TLS record overhead are "
                "not packet-captured"
            ),
            "persistent_connection": True,
            "tcp_nodelay": True,
        },
        "warmups": warmups,
        "repeats": repeats,
        "transports": transports,
        "comparison": {
            "tls_minus_tcp_median_transport_and_framing_ms": float(tls_overhead)
            - float(tcp_overhead),
            "tls_over_tcp_median_roundtrip_ratio": float(
                transports["tls13"]["roundtrip_ms"]["median"]
            )
            / float(transports["tcp"]["roundtrip_ms"]["median"]),
        },
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_unfixed_threads:
        validate_thread_environment()
    projected_docs_path = Path(args.projected_docs_path).resolve()
    docs = _load_docs(projected_docs_path)
    n_docs, dimension = int(docs.shape[0]), int(docs.shape[1])
    del docs
    if dimension != args.dimension:
        raise ValueError(
            f"requested dimension {args.dimension} does not match docs {dimension}"
        )
    if args.candidates > n_docs:
        raise ValueError("candidate count exceeds corpus size")
    parameters = CKKSParameters(
        poly_modulus_degree=args.poly_modulus_degree,
        coeff_mod_bit_sizes=tuple(args.coeff_mod_bit_sizes),
        scale_bits=args.scale_bits,
    )
    common_pair = create_context_pair(parameters, max_query_dimension=dimension)
    request_pool = _build_request_pool(
        common_pair,
        dimension=dimension,
        n_docs=n_docs,
        candidate_count=args.candidates,
        pool_size=args.request_pool_size,
        seed=args.seed,
    )
    environment = environment_metadata()
    logical_cpus = int(environment["cpu"]["logical_cpus"])
    scaling_rows: list[dict[str, Any]] = []
    schedule: list[tuple[str, int, int, int]] = []
    for restart in range(args.scaling_restarts):
        levels = list(args.concurrency)
        if restart % 2:
            levels.reverse()
        schedule.extend(
            (
                "segmented_score_packed",
                level,
                restart,
                args.requests_per_worker,
            )
            for level in levels
        )
    for restart in range(args.naive_restarts):
        levels = list(args.naive_concurrency)
        if restart % 2:
            levels.reverse()
        schedule.extend(
            ("naive_per_candidate", level, restart, args.naive_requests_per_worker)
            for level in levels
        )

    for method, concurrency, restart, requests_per_worker in schedule:
        print(
            json.dumps(
                {
                    "event": "scaling_condition_start",
                    "method": method,
                    "concurrency": concurrency,
                    "restart": restart,
                    "requests_per_worker": requests_per_worker,
                }
            ),
            flush=True,
        )
        row = run_scaling_condition(
            public_context=common_pair.serialized_server_context,
            client_context=common_pair.client,
            projected_docs_path=projected_docs_path,
            method=method,
            concurrency=concurrency,
            requests_per_worker=requests_per_worker,
            warmup_requests_per_worker=(
                args.warmup_requests_per_worker
                if method == "segmented_score_packed"
                else args.naive_warmup_requests_per_worker
            ),
            request_pool=request_pool,
            restart=restart,
            logical_cpus=logical_cpus,
            response_timeout_s=args.response_timeout_s,
        )
        scaling_rows.append(row)
        print(
            json.dumps(
                {
                    "event": "scaling_condition_done",
                    "method": method,
                    "concurrency": concurrency,
                    "restart": restart,
                    "qps": row["throughput_qps"],
                    "p50_ms": row["latency_ms"]["pipe_roundtrip"]["p50"],
                    "p95_ms": row["latency_ms"]["pipe_roundtrip"]["p95"],
                    "failures": row["failed_requests"],
                }
            ),
            flush=True,
        )

    session = run_session_benchmark(
        projected_docs_path=projected_docs_path,
        dimension=dimension,
        candidate_count=args.candidates,
        repeats=args.session_restarts,
        warm_requests=args.session_warm_requests,
        seed=args.seed + 1000,
        parameters=parameters,
        response_timeout_s=args.response_timeout_s,
    )
    loopback = run_loopback_benchmark(
        pair=common_pair,
        projected_docs_path=projected_docs_path,
        request_pool=request_pool,
        warmups=args.socket_warmups,
        repeats=args.socket_repeats,
        certificate_path=Path(args.tls_certificate).resolve(),
        private_key_path=Path(args.tls_private_key).resolve(),
        response_timeout_s=args.response_timeout_s,
    )
    return {
        "schema": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "config": {
            "dimension": dimension,
            "candidates": args.candidates,
            "ckks": parameters.as_json(),
            "concurrency": list(args.concurrency),
            "scaling_restarts": args.scaling_restarts,
            "requests_per_worker": args.requests_per_worker,
            "warmup_requests_per_worker": args.warmup_requests_per_worker,
            "naive_concurrency": list(args.naive_concurrency),
            "naive_restarts": args.naive_restarts,
            "naive_requests_per_worker": args.naive_requests_per_worker,
            "naive_warmup_requests_per_worker": args.naive_warmup_requests_per_worker,
            "session_restarts": args.session_restarts,
            "session_warm_requests": args.session_warm_requests,
            "socket_warmups": args.socket_warmups,
            "socket_repeats": args.socket_repeats,
            "request_pool_size": args.request_pool_size,
            "seed": args.seed,
            "response_timeout_s": args.response_timeout_s,
        },
        "methodology": {
            "scaling_scope": (
                "provider-side closed-loop saturation: one parent client thread and "
                "one independent public-only spawned CKKS worker per concurrency slot; "
                "query encryption and response decryption excluded from workload timer"
            ),
            "oversubscription_control": (
                "each worker executes the single-threaded SEAL kernel; BLAS/OpenMP "
                "thread pools fixed to one; concurrency 16 equals logical CPU count "
                "but exceeds the 10 physical cores on this hybrid CPU"
            ),
            "level_order": (
                "ascending on even restarts and descending on odd restarts to reduce "
                "monotone thermal/order bias"
            ),
            "rss_scope": (
                "sampled spawned-worker RSS/working set; sums may double-count shared pages"
            ),
            "no_affinity_or_priority_changes": True,
            "gpu_used": False,
        },
        "environment": environment,
        "artifact": {"projected_docs": _file_metadata(projected_docs_path)},
        "request_pool": {
            "count": len(request_pool),
            "query_wire_bytes": distribution(
                [float(len(request["query_wire"])) for request in request_pool]
            ),
            "candidate_id_bytes": args.candidates * 8,
            "queries_and_candidate_ids": "deterministic NumPy RNG; normalized queries",
        },
        "scaling": {
            "conditions": scaling_rows,
            "aggregate": aggregate_scaling(scaling_rows),
        },
        "sessions": session,
        "loopback_transport": loopback,
    }


def run_loopback_only(args: argparse.Namespace) -> dict[str, Any]:
    """Run only the real TCP/TLS transport stage for focused replication."""

    if not args.allow_unfixed_threads:
        validate_thread_environment()
    projected_docs_path = Path(args.projected_docs_path).resolve()
    docs = _load_docs(projected_docs_path)
    n_docs, dimension = int(docs.shape[0]), int(docs.shape[1])
    del docs
    if dimension != args.dimension:
        raise ValueError(
            f"requested dimension {args.dimension} does not match docs {dimension}"
        )
    parameters = CKKSParameters(
        poly_modulus_degree=args.poly_modulus_degree,
        coeff_mod_bit_sizes=tuple(args.coeff_mod_bit_sizes),
        scale_bits=args.scale_bits,
    )
    pair = create_context_pair(parameters, max_query_dimension=dimension)
    pool = _build_request_pool(
        pair,
        dimension=dimension,
        n_docs=n_docs,
        candidate_count=args.candidates,
        pool_size=args.request_pool_size,
        seed=args.seed,
    )
    return {
        "schema": "systems_expansion.loopback.v1",
        "created_utc": _utc_now(),
        "config": {
            "dimension": dimension,
            "candidates": args.candidates,
            "ckks": parameters.as_json(),
            "socket_warmups": args.socket_warmups,
            "socket_repeats": args.socket_repeats,
            "request_pool_size": args.request_pool_size,
            "seed": args.seed,
        },
        "environment": environment_metadata(),
        "artifact": {"projected_docs": _file_metadata(projected_docs_path)},
        "key_material": inspect_key_material(pair),
        "loopback_transport": run_loopback_benchmark(
            pair=pair,
            projected_docs_path=projected_docs_path,
            request_pool=pool,
            warmups=args.socket_warmups,
            repeats=args.socket_repeats,
            certificate_path=Path(args.tls_certificate).resolve(),
            private_key_path=Path(args.tls_private_key).resolve(),
            response_timeout_s=args.response_timeout_s,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Packed-CKKS concurrency, session, and loopback transport study"
    )
    parser.add_argument("--projected-docs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=672)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--scaling-restarts", type=int, default=3)
    parser.add_argument("--requests-per-worker", type=int, default=12)
    parser.add_argument("--warmup-requests-per-worker", type=int, default=2)
    parser.add_argument("--naive-concurrency", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--naive-restarts", type=int, default=2)
    parser.add_argument("--naive-requests-per-worker", type=int, default=3)
    parser.add_argument("--naive-warmup-requests-per-worker", type=int, default=1)
    parser.add_argument("--session-restarts", type=int, default=5)
    parser.add_argument("--session-warm-requests", type=int, default=10)
    parser.add_argument("--socket-warmups", type=int, default=3)
    parser.add_argument("--socket-repeats", type=int, default=20)
    parser.add_argument("--request-pool-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--response-timeout-s", type=float, default=600.0)
    parser.add_argument("--poly-modulus-degree", type=int, default=8192)
    parser.add_argument(
        "--coeff-mod-bit-sizes", type=int, nargs="+", default=[60, 40, 60]
    )
    parser.add_argument("--scale-bits", type=int, default=40)
    parser.add_argument(
        "--tls-certificate",
        type=Path,
        default=None,
        help=(
            "optional loopback certificate; omit together with --tls-private-key "
            "to generate a disposable pair"
        ),
    )
    parser.add_argument(
        "--tls-private-key",
        type=Path,
        default=None,
        help=(
            "optional loopback private key; omit together with --tls-certificate "
            "to generate a disposable pair"
        ),
    )
    parser.add_argument(
        "--allow-unfixed-threads",
        action="store_true",
        help="permit execution without all primary thread-pool variables set to 1",
    )
    parser.add_argument(
        "--loopback-only",
        action="store_true",
        help="run only the focused real TCP/TLS loopback stage",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "dimension": args.dimension,
        "candidates": args.candidates,
        "scaling_restarts": args.scaling_restarts,
        "requests_per_worker": args.requests_per_worker,
        "session_restarts": args.session_restarts,
        "session_warm_requests": args.session_warm_requests,
        "socket_repeats": args.socket_repeats,
        "request_pool_size": args.request_pool_size,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError("positive arguments required: " + ", ".join(invalid))
    if any(value < 1 for value in (*args.concurrency, *args.naive_concurrency)):
        raise ValueError("concurrency values must be positive")
    logical = psutil.cpu_count(logical=True) or 1
    if max(args.concurrency) > logical:
        raise ValueError(
            "provider-worker concurrency exceeds logical CPU count; this benchmark "
            "refuses avoidable oversubscription"
        )
    supplied_tls_paths = (args.tls_certificate is not None, args.tls_private_key is not None)
    if supplied_tls_paths[0] != supplied_tls_paths[1]:
        raise ValueError("provide both TLS paths or omit both for disposable material")
    if all(supplied_tls_paths) and (
        not Path(args.tls_certificate).is_file()
        or not Path(args.tls_private_key).is_file()
    ):
        raise FileNotFoundError("loopback TLS certificate/key files are missing")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tls_certificate is None and args.tls_private_key is None:
        with tempfile.TemporaryDirectory(prefix="packrerank_tls_") as directory:
            args.tls_certificate, args.tls_private_key = generate_loopback_tls_material(
                Path(directory)
            )
            _validate_args(args)
            payload = run_loopback_only(args) if args.loopback_only else run_experiment(args)
    else:
        _validate_args(args)
        payload = run_loopback_only(args) if args.loopback_only else run_experiment(args)
    _write_json(args.output, payload)
    if args.loopback_only:
        print(
            json.dumps(
                {
                    "schema": payload["schema"],
                    "output": str(Path(args.output).resolve()),
                    "tcp_p50_ms": payload["loopback_transport"]["transports"]
                    ["tcp"]["roundtrip_ms"]["p50"],
                    "tls_p50_ms": payload["loopback_transport"]["transports"]
                    ["tls13"]["roundtrip_ms"]["p50"],
                }
            ),
            flush=True,
        )
        return 0
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "output": str(Path(args.output).resolve()),
                "packed_scaling": [
                    {
                        "concurrency": row["concurrency"],
                        "qps_median": row["throughput_qps_across_restarts"]["median"],
                        "p50_ms": row["pooled_pipe_roundtrip_ms"]["p50"],
                        "p95_ms": row["pooled_pipe_roundtrip_ms"]["p95"],
                        "p99_ms": row["pooled_pipe_roundtrip_ms"]["p99"],
                    }
                    for row in payload["scaling"]["aggregate"]
                    if row["method"] == "segmented_score_packed"
                ],
                "warm_client_p50_ms": payload["sessions"]["summary_ms"]
                ["warm_full_client_request"]["p50"],
                "tcp_p50_ms": payload["loopback_transport"]["transports"]["tcp"]
                ["roundtrip_ms"]["p50"],
                "tls_p50_ms": payload["loopback_transport"]["transports"]["tls13"]
                ["roundtrip_ms"]["p50"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
