"""Experiment 26: measured end-to-end CKKS reranking for SHARD.

This experiment uses Microsoft SEAL through TenSEAL; it is not a timing
model.  It compares two query-ciphertext layouts on cached, real SciFact
embeddings from multilingual-e5-small and multilingual-e5-base:

* width 1: one active cell-specific transformed residual per ciphertext;
* width B: B transformed residual blocks in one CKKS ciphertext.  A document
  is placed in the matching plaintext block and all other blocks are masked
  to zero before the ciphertext-plaintext dot product.

The online path is split into measured client transform, pack, encrypt and
serialize phases; server deserialize, plaintext-mask preparation, ct-pt dot,
public-prefix addition and response serialization phases; and client result
deserialize/decrypt phases.  Offline PCA, clustering, key generation, keyed
document preparation and CKKS context/key generation are reported separately
and are never folded into online latency.

Default full run (three independent key/query seeds):

    D:/PHD/research/RES/experiments/.venv/Scripts/python.exe \
        shard/exp26_ckks_blocksimd.py --force

Quick validation:

    D:/PHD/research/RES/experiments/.venv/Scripts/python.exe \
        shard/exp26_ckks_blocksimd.py --smoke --force
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import psutil
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("psutil is required: python -m pip install psutil") from exc

try:
    import tenseal as ts
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "TenSEAL is required: python -m pip install --no-deps tenseal==0.3.16"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
SHARD = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARD))
from shard_lib import cell_key, kmeans_cells  # noqa: E402


DEFAULT_OUT = ROOT / "results" / "exp26_ckks_blocksimd"
CACHE = ROOT / "results" / "exp17_outputs" / "emb"
ENCODERS = {
    "multilingual-e5-small": CACHE / "multilingual-e5-small_scifact.npz",
    "multilingual-e5-base": CACHE / "multilingual-e5-base_scifact.npz",
}


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(x) for x in parse_csv(value)]


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


class Logger:
    def __init__(self, path: Path, reset: bool = False):
        self.path = path
        if reset:
            path.write_text("", encoding="utf-8")

    def __call__(self, message: str) -> None:
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        line = f"{stamp} {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def get_git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def get_gpu_inventory() -> list[dict[str, str]]:
    """Record available GPUs while making explicit that SEAL itself is CPU-only."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ], text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except Exception:
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({
                "name": fields[0], "driver_version": fields[1],
                "memory_mib": fields[2],
            })
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, q))


def topk_desc(scores: np.ndarray, k: int) -> np.ndarray:
    """Descending score with ascending local id as deterministic tie-break."""
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.lexsort((part, -scores[part]))]


def ranking_statistics(reference: np.ndarray, measured: np.ndarray) -> dict[str, Any]:
    n = len(reference)
    ref_order = np.lexsort((np.arange(n), -reference))
    got_order = np.lexsort((np.arange(n), -measured))
    k = min(10, n)
    ref10 = ref_order[:k]
    got10 = got_order[:k]
    ref_pos = np.empty(n, dtype=np.int64)
    got_pos = np.empty(n, dtype=np.int64)
    ref_pos[ref_order] = np.arange(n)
    got_pos[got_order] = np.arange(n)

    concordant = 0
    discordant = 0
    tied = 0
    for i in range(n - 1):
        ref_sign = np.sign(reference[i] - reference[i + 1:])
        got_sign = np.sign(measured[i] - measured[i + 1:])
        usable = (ref_sign != 0) & (got_sign != 0)
        concordant += int(np.count_nonzero(ref_sign[usable] == got_sign[usable]))
        discordant += int(np.count_nonzero(ref_sign[usable] != got_sign[usable]))
        tied += int(np.count_nonzero(~usable))
    denom = concordant + discordant
    tau = (concordant - discordant) / denom if denom else 1.0
    pairwise_flip = discordant / denom if denom else 0.0
    return {
        "top1_flip": int(ref_order[0] != got_order[0]),
        "top10_overlap": float(len(set(map(int, ref10)) & set(map(int, got10))) / k),
        "top10_exact_order": int(np.array_equal(ref10, got10)),
        "full_rank_position_flip_fraction": float(np.mean(ref_pos != got_pos)),
        "kendall_tau": float(tau),
        "pairwise_order_flip_fraction": float(pairwise_flip),
        "ranking_tied_pair_count": int(tied),
    }


def score_statistics(reference: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    error = measured - reference
    abs_error = np.abs(error)
    relative = abs_error / np.maximum(np.abs(reference), 1e-8)
    return {
        "score_max_abs_error": float(abs_error.max(initial=0.0)),
        "score_rmse": float(np.sqrt(np.mean(error * error))),
        "score_mean_abs_error": float(abs_error.mean()),
        "score_p99_abs_error": float(np.percentile(abs_error, 99)),
        "score_mean_relative_error": float(relative.mean()),
        "score_p99_relative_error": float(np.percentile(relative, 99)),
    }


def create_ckks_context(poly_degree: int, coeff_bits: list[int], scale_bits: int) -> tuple[Any, Any, dict[str, Any]]:
    start = time.perf_counter()
    client = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_degree,
        coeff_mod_bit_sizes=coeff_bits,
    )
    client.global_scale = 2**scale_bits
    client.generate_galois_keys()
    public_blob = client.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=False,
    )
    server = ts.context_from(public_blob)
    elapsed = time.perf_counter() - start
    metadata = {
        "scheme": "CKKS",
        "poly_modulus_degree": poly_degree,
        "slot_capacity": poly_degree // 2,
        "coeff_mod_bit_sizes": coeff_bits,
        "global_scale_bits": scale_bits,
        "context_and_key_setup_seconds": elapsed,
        "public_server_context_bytes": len(public_blob),
        "server_has_secret_key": bool(server.has_secret_key()),
        "server_has_public_key": bool(server.has_public_key()),
        "server_has_galois_keys": bool(server.has_galois_keys()),
        "server_has_relin_keys": bool(server.has_relin_keys()),
    }
    if metadata["server_has_secret_key"]:
        raise AssertionError("server context unexpectedly contains the secret key")
    if not metadata["server_has_galois_keys"]:
        raise AssertionError("server context lacks the rotations required by dot()")
    return client, server, metadata


def run_unit_checks(client: Any, server: Any) -> dict[str, Any]:
    """Algebraic mask checks plus a real encrypted mask-isolation check."""
    rng = np.random.default_rng(26001)
    d = 32
    width = 4
    H = cell_key(7, d, master_seed=26001).astype(np.float64)
    q = rng.standard_normal(d)
    r = rng.standard_normal(d)
    qh = q @ H.T
    zh = r @ H.T
    key_error = abs(float(qh @ zh) - float(q @ r))
    orth_error = float(np.max(np.abs(H @ H.T - np.eye(d))))

    blocks = [rng.standard_normal(d) for _ in range(width)]
    slot = 2
    packed = np.concatenate(blocks)
    mask = np.zeros(width * d, dtype=np.float64)
    mask[slot * d:(slot + 1) * d] = zh
    mask_error = abs(float(packed @ mask) - float(blocks[slot] @ zh))

    encrypted = ts.ckks_vector(client, packed)
    query_blob = encrypted.serialize()
    encrypted_server = ts.ckks_vector_from(server, query_blob)
    result_server = encrypted_server.dot(mask)
    response_blob = result_server.serialize()
    decrypted = float(ts.ckks_vector_from(client, response_blob).decrypt()[0])
    ckks_error = abs(decrypted - float(blocks[slot] @ zh))

    checks = {
        "dense_key_orthogonality_max_abs": orth_error,
        "keyed_dot_identity_abs_error": key_error,
        "packed_plaintext_mask_identity_abs_error": mask_error,
        "ckks_packed_mask_abs_error": ckks_error,
        "ckks_query_ciphertext_bytes": len(query_blob),
        "ckks_scalar_response_bytes": len(response_blob),
        "thresholds": {
            "orthogonality": 2e-6,
            "keyed_dot": 2e-6,
            "plaintext_mask": 1e-10,
            "ckks_mask": 2e-3,
        },
    }
    checks["passed"] = bool(
        orth_error < checks["thresholds"]["orthogonality"]
        and key_error < checks["thresholds"]["keyed_dot"]
        and mask_error < checks["thresholds"]["plaintext_mask"]
        and ckks_error < checks["thresholds"]["ckks_mask"]
    )
    if not checks["passed"]:
        raise AssertionError(f"unit checks failed: {checks}")
    return checks


def prepare_encoder(path: Path, cells: int, d_pub_fraction: float, cell_seed: int,
                    kmeans_iters: int, log: Logger) -> dict[str, Any]:
    start = time.perf_counter()
    with np.load(path) as data:
        documents = np.asarray(data["D"], dtype=np.float32)
        queries = np.asarray(data["Q"], dtype=np.float32)
    n, d = documents.shape
    d_pub = int(round(d * d_pub_fraction))
    d_priv = d - d_pub
    mu = documents.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = documents - mu
    covariance = (centered.T @ centered).astype(np.float64) / float(n)
    eigenvalues, basis = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = basis[:, order].astype(np.float32)
    rotated = centered @ basis
    public = rotated[:, :d_pub]
    residual = rotated[:, d_pub:]
    labels, centroids = kmeans_cells(
        public, cells, seed=cell_seed, iters=kmeans_iters, n_train=n,
    )
    query_score = queries @ basis
    query_route = (queries - mu) @ basis

    # The corrected full PCA score differs from raw q.x only by q.mu.
    check_q = min(8, len(queries))
    check_d = min(256, len(documents))
    raw = queries[:check_q] @ documents[:check_d].T
    split = query_score[:check_q] @ rotated[:check_d].T
    corrected = raw - (queries[:check_q] @ mu)[:, None]
    score_identity_error = float(np.max(np.abs(split - corrected)))
    orth_error = float(np.max(np.abs(basis.T @ basis - np.eye(d, dtype=np.float32))))
    elapsed = time.perf_counter() - start
    counts = np.bincount(labels, minlength=cells)
    log(
        f"prepared {path.stem}: n={n:,} q={len(queries)} d={d} "
        f"d_pub={d_pub} d_priv={d_priv} in {elapsed:.3f}s"
    )
    return {
        "D": documents,
        "Q": queries,
        "mu": mu,
        "V": basis,
        "U": public,
        "R": residual,
        "labels": labels,
        "centroids": centroids,
        "Qscore": query_score,
        "Qroute": query_route,
        "d": d,
        "d_pub": d_pub,
        "d_priv": d_priv,
        "n_documents": n,
        "n_queries": len(queries),
        "offline_prepare_seconds": elapsed,
        "basis_orthogonality_max_abs_float32": orth_error,
        "corrected_score_identity_max_abs_float32": score_identity_error,
        "cell_size_min": int(counts.min()),
        "cell_size_median": float(np.median(counts)),
        "cell_size_max": int(counts.max()),
    }


def stable_candidates(route_scores: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(route_scores))
    part = np.argpartition(-route_scores, count - 1)[:count]
    return part[np.lexsort((part, -route_scores[part]))]


def make_query_case(data: dict[str, Any], seed: int, candidate_count: int,
                    cells: int, key_seed_offset: int) -> tuple[dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    query_id = int(rng.integers(0, data["n_queries"]))
    route_q = data["Qroute"][query_id]
    score_q = data["Qscore"][query_id]
    route_scores = data["U"] @ route_q[:data["d_pub"]]
    candidate_ids = stable_candidates(route_scores, candidate_count)
    candidate_cells = data["labels"][candidate_ids]
    active_cells = list(dict.fromkeys(map(int, candidate_cells)))
    master_key_seed = key_seed_offset + seed

    setup_start = time.perf_counter()
    keys = {
        c: cell_key(c, data["d_priv"], master_seed=master_key_seed)
        for c in active_cells
    }
    keyed_documents = np.empty((len(candidate_ids), data["d_priv"]), dtype=np.float32)
    for c in active_cells:
        mask = candidate_cells == c
        keyed_documents[mask] = data["R"][candidate_ids[mask]] @ keys[c].T
    offline_keyed_setup_seconds = time.perf_counter() - setup_start

    public_scores = data["U"][candidate_ids] @ score_q[:data["d_pub"]]
    residual_scores = data["R"][candidate_ids] @ score_q[data["d_pub"]:]
    reference_scores = public_scores + residual_scores
    keyed_reference = np.empty(len(candidate_ids), dtype=np.float64)
    for i, c in enumerate(candidate_cells):
        keyed_q = score_q[data["d_pub"]:] @ keys[int(c)].T
        keyed_reference[i] = float(public_scores[i]) + float(
            keyed_q.astype(np.float64) @ keyed_documents[i].astype(np.float64)
        )
    keying_error = float(np.max(np.abs(keyed_reference - reference_scores)))
    return {
        "query_id": query_id,
        "candidate_ids": candidate_ids,
        "candidate_cells": candidate_cells,
        "active_cells": active_cells,
        "keys": keys,
        "keyed_documents": keyed_documents,
        "public_scores": public_scores.astype(np.float64),
        "reference_scores": reference_scores.astype(np.float64),
        "query_residual": score_q[data["d_pub"]:].astype(np.float32),
        "plaintext_keying_max_abs_error": keying_error,
        "master_key_seed": master_key_seed,
    }, offline_keyed_setup_seconds


def warmup_ckks(client: Any, server: Any, packed_length: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=0.05, size=packed_length)
    p = rng.normal(scale=0.05, size=packed_length)
    query = ts.ckks_vector(client, q)
    server_query = ts.ckks_vector_from(server, query.serialize())
    result = server_query.dot(p)
    result += 0.01
    value = ts.ckks_vector_from(client, result.serialize()).decrypt()[0]
    if abs(value - (float(q @ p) + 0.01)) > 2e-3:
        raise AssertionError("CKKS warmup failed")


def online_trial(client: Any, server: Any, case: dict[str, Any], packing_width: int,
                 d_priv: int) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    rss_samples = [rss_mb()]
    peak_before = float(getattr(process.memory_info(), "peak_wset", 0)) / (1024.0**2)
    total_start = time.perf_counter_ns()

    # Client: transform the residual once for every active cell.
    phase = time.perf_counter_ns()
    transformed_queries = {
        c: case["query_residual"] @ case["keys"][c].T
        for c in case["active_cells"]
    }
    client_transform_ms = (time.perf_counter_ns() - phase) / 1e6
    rss_samples.append(rss_mb())

    # Client: fixed-capacity blocks.  The final group is zero padded so that
    # logical CKKS vector length (and reduction cost) is truly width*d_priv.
    phase = time.perf_counter_ns()
    groups = [
        case["active_cells"][i:i + packing_width]
        for i in range(0, len(case["active_cells"]), packing_width)
    ]
    cell_location: dict[int, tuple[int, int]] = {}
    packed_queries: list[np.ndarray] = []
    for group_id, group_cells in enumerate(groups):
        packed = np.zeros(packing_width * d_priv, dtype=np.float64)
        for slot, cell in enumerate(group_cells):
            packed[slot * d_priv:(slot + 1) * d_priv] = transformed_queries[cell]
            cell_location[cell] = (group_id, slot)
        packed_queries.append(packed)
    client_pack_ms = (time.perf_counter_ns() - phase) / 1e6

    phase = time.perf_counter_ns()
    client_ciphertexts = [ts.ckks_vector(client, packed) for packed in packed_queries]
    client_encrypt_ms = (time.perf_counter_ns() - phase) / 1e6
    rss_samples.append(rss_mb())

    phase = time.perf_counter_ns()
    query_blobs = [ciphertext.serialize() for ciphertext in client_ciphertexts]
    client_serialize_ms = (time.perf_counter_ns() - phase) / 1e6
    upload_bytes = int(sum(map(len, query_blobs)))
    del client_ciphertexts

    # Server: it owns only the public context and Galois keys.
    phase = time.perf_counter_ns()
    server_queries = [ts.ckks_vector_from(server, blob) for blob in query_blobs]
    server_query_deserialize_ms = (time.perf_counter_ns() - phase) / 1e6
    rss_samples.append(rss_mb())

    phase = time.perf_counter_ns()
    masks = np.zeros(
        (len(case["candidate_ids"]), packing_width * d_priv), dtype=np.float64,
    )
    group_ids = np.empty(len(case["candidate_ids"]), dtype=np.int32)
    for i, c_raw in enumerate(case["candidate_cells"]):
        group_id, slot = cell_location[int(c_raw)]
        group_ids[i] = group_id
        masks[i, slot * d_priv:(slot + 1) * d_priv] = case["keyed_documents"][i]
    server_plaintext_prepare_ms = (time.perf_counter_ns() - phase) / 1e6
    rss_samples.append(rss_mb())

    # Dot and public-score addition are timed without serialization.
    response_blobs: list[bytes] = []
    server_eval_ns = 0
    server_response_serialize_ns = 0
    for i in range(len(case["candidate_ids"])):
        phase = time.perf_counter_ns()
        result = server_queries[int(group_ids[i])].dot(masks[i])
        result += float(case["public_scores"][i])
        server_eval_ns += time.perf_counter_ns() - phase

        phase = time.perf_counter_ns()
        response_blobs.append(result.serialize())
        server_response_serialize_ns += time.perf_counter_ns() - phase
    server_eval_ms = server_eval_ns / 1e6
    server_response_serialize_ms = server_response_serialize_ns / 1e6
    response_bytes = int(sum(map(len, response_blobs)))
    rss_samples.append(rss_mb())

    phase = time.perf_counter_ns()
    client_results = [ts.ckks_vector_from(client, blob) for blob in response_blobs]
    client_result_deserialize_ms = (time.perf_counter_ns() - phase) / 1e6
    rss_samples.append(rss_mb())

    phase = time.perf_counter_ns()
    measured_scores = np.array(
        [float(result.decrypt()[0]) for result in client_results], dtype=np.float64,
    )
    client_decrypt_ms = (time.perf_counter_ns() - phase) / 1e6
    end_to_end_ms = (time.perf_counter_ns() - total_start) / 1e6
    rss_samples.append(rss_mb())
    peak_after = float(getattr(process.memory_info(), "peak_wset", 0)) / (1024.0**2)

    timings = {
        "client_transform_ms": client_transform_ms,
        "client_pack_ms": client_pack_ms,
        "client_encrypt_ms": client_encrypt_ms,
        "client_serialize_ms": client_serialize_ms,
        "server_query_deserialize_ms": server_query_deserialize_ms,
        "server_plaintext_prepare_ms": server_plaintext_prepare_ms,
        "server_ct_pt_eval_ms": server_eval_ms,
        "server_response_serialize_ms": server_response_serialize_ms,
        "client_result_deserialize_ms": client_result_deserialize_ms,
        "client_decrypt_ms": client_decrypt_ms,
        "end_to_end_ms": end_to_end_ms,
        "server_eval_candidates_per_second": len(case["candidate_ids"]) / (server_eval_ms / 1000.0),
        "end_to_end_candidates_per_second": len(case["candidate_ids"]) / (end_to_end_ms / 1000.0),
    }
    memory = {
        "rss_start_mb": rss_samples[0],
        "rss_end_mb": rss_samples[-1],
        "rss_sampled_max_mb": max(rss_samples),
        "rss_sampled_increment_mb": max(rss_samples) - rss_samples[0],
        "process_peak_wset_before_mb": peak_before,
        "process_peak_wset_after_mb": peak_after,
    }
    result = {
        **timings,
        **memory,
        **score_statistics(case["reference_scores"], measured_scores),
        **ranking_statistics(case["reference_scores"], measured_scores),
        "query_ciphertext_count": len(query_blobs),
        "query_ciphertext_bytes": upload_bytes,
        "query_ciphertext_bytes_each": json.dumps(list(map(len, query_blobs))),
        "response_ciphertext_count": len(response_blobs),
        "response_ciphertext_bytes": response_bytes,
        "response_ciphertext_bytes_each_min": min(map(len, response_blobs)),
        "response_ciphertext_bytes_each_max": max(map(len, response_blobs)),
        "plaintext_keying_max_abs_error": case["plaintext_keying_max_abs_error"],
    }
    return result


def summarize(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        key = (str(row["encoder"]), int(row["candidate_count"]), int(row["packing_width"]))
        groups[key].append(row)

    fields_for_percentiles = [
        "client_transform_ms", "client_pack_ms", "client_encrypt_ms",
        "client_serialize_ms", "server_query_deserialize_ms",
        "server_plaintext_prepare_ms", "server_ct_pt_eval_ms",
        "server_response_serialize_ms", "client_result_deserialize_ms",
        "client_decrypt_ms", "end_to_end_ms",
        "server_eval_candidates_per_second", "end_to_end_candidates_per_second",
        "query_ciphertext_bytes", "response_ciphertext_bytes",
        "rss_sampled_increment_mb",
    ]
    summary: list[dict[str, Any]] = []
    for (encoder, candidate_count, width), rows in sorted(groups.items()):
        out: dict[str, Any] = {
            "encoder": encoder,
            "candidate_count": candidate_count,
            "packing_width": width,
            "measurement_count": len(rows),
            "seed_count": len(set(int(r["seed"]) for r in rows)),
            "active_cells_mean": float(np.mean([r["active_cell_count"] for r in rows])),
            "query_ciphertext_count_median": percentile(
                (r["query_ciphertext_count"] for r in rows), 50,
            ),
        }
        for field in fields_for_percentiles:
            values = [float(r[field]) for r in rows]
            out[field + "_mean"] = float(np.mean(values))
            out[field + "_p50"] = percentile(values, 50)
            out[field + "_p95"] = percentile(values, 95)
            out[field + "_p99"] = percentile(values, 99)
        out.update({
            "score_max_abs_error_max": max(float(r["score_max_abs_error"]) for r in rows),
            "score_rmse_mean": float(np.mean([r["score_rmse"] for r in rows])),
            "score_p99_abs_error_max": max(float(r["score_p99_abs_error"]) for r in rows),
            "score_mean_relative_error_mean": float(np.mean([
                r["score_mean_relative_error"] for r in rows
            ])),
            "top1_flip_count": sum(int(r["top1_flip"]) for r in rows),
            "top10_overlap_min": min(float(r["top10_overlap"]) for r in rows),
            "top10_overlap_mean": float(np.mean([r["top10_overlap"] for r in rows])),
            "top10_exact_order_rate": float(np.mean([r["top10_exact_order"] for r in rows])),
            "kendall_tau_min": min(float(r["kendall_tau"]) for r in rows),
            "kendall_tau_mean": float(np.mean([r["kendall_tau"] for r in rows])),
            "pairwise_order_flip_fraction_max": max(
                float(r["pairwise_order_flip_fraction"]) for r in rows
            ),
            "plaintext_keying_max_abs_error_max": max(
                float(r["plaintext_keying_max_abs_error"]) for r in rows
            ),
        })
        summary.append(out)

    # Width-1 is the empirical reference for ratios; no modelled quantities.
    baseline: dict[tuple[str, int], dict[str, Any]] = {
        (r["encoder"], int(r["candidate_count"])): r
        for r in summary if int(r["packing_width"]) == 1
    }
    for row in summary:
        ref = baseline[(row["encoder"], int(row["candidate_count"]))]
        row["upload_bytes_ratio_vs_width1_measured"] = (
            row["query_ciphertext_bytes_p50"] / ref["query_ciphertext_bytes_p50"]
        )
        row["upload_reduction_vs_width1_measured"] = 1.0 - row[
            "upload_bytes_ratio_vs_width1_measured"
        ]
        row["end_to_end_speedup_vs_width1_measured_p50"] = (
            ref["end_to_end_ms_p50"] / row["end_to_end_ms_p50"]
        )
        row["server_eval_speedup_vs_width1_measured_p50"] = (
            ref["server_ct_pt_eval_ms_p50"] / row["server_ct_pt_eval_ms_p50"]
        )
    return summary


def format_readme(config: dict[str, Any], ckks: dict[str, Any], unit: dict[str, Any],
                  geometry: list[dict[str, Any]], summary: list[dict[str, Any]],
                  elapsed: float) -> str:
    def widest_row(encoder: str, candidates: int) -> dict[str, Any]:
        rows = [r for r in summary if r["encoder"] == encoder and r["candidate_count"] == candidates]
        return max(rows, key=lambda r: r["packing_width"])

    lines = [
        "# Experiment 26: measured CKKS block-SIMD reranking",
        "",
        "This directory contains **measured Microsoft SEAL/TenSEAL CKKS runs**, not a",
        "latency or ciphertext-size model.  The server context contains a public key and",
        "Galois keys but no secret key.  Cached real SciFact document/query embeddings",
        "from multilingual-e5-small and multilingual-e5-base are used throughout.",
        "",
        "## Protocol",
        "",
        "Documents use `(x-mu) @ V`; final query scoring uses `q @ V`, while shortlist",
        "routing uses `(q-mu) @ V`.  The private residual occupies 3/4 of the embedding.",
        "For width 1, each active cell-specific `H_c r_q` is encrypted separately.  For",
        "width B, B such vectors are concatenated in one ciphertext.  For a candidate",
        "in block b, the plaintext contains `H_c r_i` only in b and zeros in every other",
        "block; a real ciphertext-plaintext `dot` therefore returns the residual score.",
        "The public-prefix score is then added to that ciphertext before it is returned.",
        "",
        "The online p50/p95/p99 measurements include client transformation, packing,",
        "encryption and serialization; server query deserialization, mask construction,",
        "ct-pt evaluation and response serialization; and client response deserialization",
        "and decryption.  PCA, k-means, dense key generation, keyed-document preparation",
        "and CKKS context/key generation are measured separately and excluded online.",
        "",
        "## Main measured results",
        "",
        f"CKKS parameters: N={ckks['poly_modulus_degree']}, slots={ckks['slot_capacity']}, ",
        f"coefficient bits={ckks['coeff_mod_bit_sizes']}, scale=2^{ckks['global_scale_bits']}.",
        f"The provisioned public server context is {ckks['public_server_context_bytes'] / 1e6:.2f} MB; ",
        "it is a one-time setup object and is not counted as per-query traffic.",
        "",
    ]
    for encoder in config["encoders"]:
        largest = max(config["candidate_counts"])
        b1 = next(r for r in summary if r["encoder"] == encoder and r["candidate_count"] == largest and r["packing_width"] == 1)
        packed = widest_row(encoder, largest)
        latency_change = 100.0 * (packed["end_to_end_ms_p50"] / b1["end_to_end_ms_p50"] - 1.0)
        encrypt_reduction = 100.0 * (
            1.0 - packed["client_encrypt_ms_p50"] / b1["client_encrypt_ms_p50"]
        )
        lines.extend([
            f"* `{encoder}`, {largest} candidates: width 1 used a median of ",
            f"  {b1['query_ciphertext_count_median']:.0f} query ciphertexts and ",
            f"  {b1['query_ciphertext_bytes_p50'] / 1e6:.2f} MB upload; its measured ",
            f"  end-to-end p50/p95/p99 was {b1['end_to_end_ms_p50']:.1f}/",
            f"  {b1['end_to_end_ms_p95']:.1f}/{b1['end_to_end_ms_p99']:.1f} ms.",
            f"  Width {packed['packing_width']} reduced query upload to ",
            f"  {packed['query_ciphertext_bytes_p50'] / 1e6:.2f} MB ",
            f"  ({100 * packed['upload_reduction_vs_width1_measured']:.1f}% reduction) ",
            f"  and client encryption time by {encrypt_reduction:.1f}%, but its p50 ",
            f"  end-to-end latency was {packed['end_to_end_ms_p50']:.1f} ms ",
            f"  ({latency_change:+.1f}%).  The encrypted scalar responses occupied ",
            f"  {packed['response_ciphertext_bytes_p50'] / 1e6:.2f} MB in either layout.",
        ])
    max_error = max(r["score_max_abs_error_max"] for r in summary)
    min_overlap = min(r["top10_overlap_min"] for r in summary)
    top1_flips = sum(int(r["top1_flip_count"]) for r in summary)
    min_tau = min(r["kendall_tau_min"] for r in summary)
    lines.extend([
        "",
        f"Across all measured trials, maximum absolute score error was {max_error:.3e}, ",
        f"minimum top-10 overlap was {min_overlap:.3f}, total top-1 flips were ",
        f"{top1_flips}, and minimum Kendall tau was {min_tau:.6f}.",
        "",
        "See `summary.csv` for every encoder/candidate-count/width cell and",
        "`raw_measurements.csv` for individual seed/repetition measurements.",
        "",
        "## Interpretation and limitations",
        "",
        "Block-SIMD here reduces the number of uploaded query ciphertexts.  Each candidate",
        "still produces one encrypted scalar response, and TenSEAL's `dot` reduces the",
        "entire logical packed vector.  Consequently, larger packing widths can make each",
        "candidate evaluation slower even while query upload and encryption become smaller.",
        "This experiment does not claim batched multi-score output, network RTT, TLS, ANN",
        "index latency, multi-client concurrency, GPU acceleration, or production service",
        "throughput.  Point-sampled RSS is not an exact high-frequency peak-memory trace.",
        "TenSEAL/SEAL executes this CKKS path on the CPU; the installed RTX GPU is not used.",
        f"Trials were pinned to logical CPUs {config['cpu_affinity_logical_processors']} ",
        "(one hardware thread per i5-14400F P-core) to prevent P/E-core migration;",
        "the deterministic trial order was shuffled within each comparison cell.",
        "",
        "## Reproduction",
        "",
        "From the repository root:",
        "",
        "```powershell",
        "D:\\PHD\\research\\RES\\experiments\\.venv\\Scripts\\python.exe shard\\exp26_ckks_blocksimd.py --force",
        "```",
        "",
        "Dependencies added to that environment without upgrading other packages:",
        "",
        "```powershell",
        "python -m pip install --no-deps tenseal==0.3.16 psutil==7.2.2",
        "```",
        "",
        f"Unit/algebra checks passed: `{unit['passed']}`. Total wall time: {elapsed:.1f} s.",
        "",
        "## Files",
        "",
        "* `config.json`: frozen protocol and CLI configuration.",
        "* `run_info.json`: software/hardware, hashes and CKKS context metadata.",
        "* `unit_checks.json`: orthogonality, keyed-dot and real encrypted-mask checks.",
        "* `geometry.json`: dataset/PCA/cell preparation diagnostics.",
        "* `raw_measurements.csv` / `.json`: every actual timed trial.",
        "* `summary.csv` / `.json`: aggregate p50/p95/p99 and measured width-1 ratios.",
        "* `run.log`: timestamped execution log.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--encoders", default=",".join(ENCODERS))
    parser.add_argument("--seeds", default="11,23,31")
    parser.add_argument("--candidate-counts", default="16,64,128")
    parser.add_argument("--packing-widths", default="1,2,4,8")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--d-pub-fraction", type=float, default=0.25)
    parser.add_argument("--cell-seed", type=int, default=42)
    parser.add_argument("--key-seed-offset", type=int, default=260000)
    parser.add_argument("--kmeans-iters", type=int, default=15)
    parser.add_argument("--poly-degree", type=int, default=8192)
    parser.add_argument("--coeff-bits", default="60,40,40,60")
    parser.add_argument("--scale-bits", type=int, default=40)
    parser.add_argument(
        "--cpu-affinity", default="0,2,4,6,8,10",
        help=(
            "logical CPUs used by the measured process; the default selects one "
            "hardware thread from each P-core of the experiment host's i5-14400F"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    encoders = parse_csv(args.encoders)
    for encoder in encoders:
        if encoder not in ENCODERS:
            raise SystemExit(f"unknown encoder: {encoder}")
        if not ENCODERS[encoder].exists():
            raise SystemExit(f"missing cached embeddings: {ENCODERS[encoder]}")
    seeds = parse_int_csv(args.seeds)
    candidate_counts = parse_int_csv(args.candidate_counts)
    packing_widths = parse_int_csv(args.packing_widths)
    coeff_bits = parse_int_csv(args.coeff_bits)
    cpu_affinity = parse_int_csv(args.cpu_affinity) if args.cpu_affinity.strip() else []
    process = psutil.Process(os.getpid())
    available_affinity = process.cpu_affinity()
    if cpu_affinity:
        invalid_affinity = sorted(set(cpu_affinity) - set(available_affinity))
        if invalid_affinity:
            raise SystemExit(
                f"requested CPU affinity {invalid_affinity} is unavailable; "
                f"available logical CPUs are {available_affinity}"
            )
        process.cpu_affinity(cpu_affinity)
    applied_affinity = process.cpu_affinity()
    repetitions = args.repetitions
    if args.smoke:
        encoders = encoders[:1]
        seeds = seeds[:1]
        candidate_counts = [min(candidate_counts)]
        packing_widths = [w for w in packing_widths if w <= 2]
        repetitions = 1

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and not args.force:
        raise SystemExit(f"output exists: {out}; pass --force to overwrite generated files")
    log = Logger(out / "run.log", reset=args.force)
    wall_start = time.perf_counter()
    start_utc = dt.datetime.now(dt.timezone.utc)

    config = {
        "experiment": "exp26_ckks_blocksimd",
        "measurement_kind": "actual TenSEAL/Microsoft SEAL execution; no timing model",
        "encoders": encoders,
        "dataset": "SciFact",
        "embedding_cache": {e: str(ENCODERS[e]) for e in encoders},
        "seeds": seeds,
        "candidate_counts": candidate_counts,
        "packing_widths_requested": packing_widths,
        "repetitions": repetitions,
        "warmup_per_encoder_width": 1,
        "trial_order": "deterministically shuffled within each encoder/seed/candidate-count cell",
        "cpu_affinity_logical_processors": applied_affinity,
        "cpu_affinity_rationale": (
            "one hardware thread per i5-14400F P-core to avoid P/E-core migration"
            if cpu_affinity else "operating-system default"
        ),
        "cells": args.cells,
        "d_pub_fraction": args.d_pub_fraction,
        "cell_seed": args.cell_seed,
        "key_seed_offset": args.key_seed_offset,
        "kmeans_iters": args.kmeans_iters,
        "ckks": {
            "poly_modulus_degree": args.poly_degree,
            "coeff_mod_bit_sizes": coeff_bits,
            "global_scale_bits": args.scale_bits,
        },
        "online_latency_includes": [
            "client cell-key transforms", "fixed-width zero-padded block construction",
            "actual CKKS encryption", "query ciphertext serialization",
            "server ciphertext deserialization", "plaintext-mask construction",
            "actual ct-pt dot and public-prefix scalar addition",
            "response ciphertext serialization", "client result deserialization",
            "actual CKKS decryption",
        ],
        "online_latency_excludes": [
            "embedding inference", "PCA", "cell k-means", "ANN index search",
            "dense cell-key generation", "stored document key transform",
            "CKKS context and key generation", "network/TLS/RTT",
        ],
        "smoke": bool(args.smoke),
    }
    atomic_json(out / "config.json", config)
    log(f"starting exp26; output={out}")
    log(f"TenSEAL={getattr(ts, '__version__', 'unknown')} numpy={np.__version__}")
    log(f"CPU affinity logical processors={applied_affinity}")

    client, server, ckks_metadata = create_ckks_context(
        args.poly_degree, coeff_bits, args.scale_bits,
    )
    log(
        "CKKS context ready: "
        f"N={args.poly_degree}, slots={ckks_metadata['slot_capacity']}, "
        f"public context={ckks_metadata['public_server_context_bytes'] / 1e6:.2f} MB, "
        f"setup={ckks_metadata['context_and_key_setup_seconds']:.3f}s"
    )
    unit = run_unit_checks(client, server)
    atomic_json(out / "unit_checks.json", unit)
    log(f"unit/algebra checks passed; CKKS mask error={unit['ckks_packed_mask_abs_error']:.3e}")
    if args.unit_only:
        atomic_json(out / "run_info.json", {"ckks": ckks_metadata})
        log("unit-only run complete")
        return

    raw: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    warmed: set[tuple[str, int]] = set()

    for encoder in encoders:
        data = prepare_encoder(
            ENCODERS[encoder], args.cells, args.d_pub_fraction, args.cell_seed,
            args.kmeans_iters, log,
        )
        geometry.append({
            "encoder": encoder,
            "cache_path": str(ENCODERS[encoder]),
            "cache_sha256": sha256_file(ENCODERS[encoder]),
            **{k: v for k, v in data.items() if not isinstance(v, np.ndarray)},
        })
        max_width = ckks_metadata["slot_capacity"] // data["d_priv"]
        valid_widths = []
        for width in packing_widths:
            if width * data["d_priv"] <= ckks_metadata["slot_capacity"]:
                valid_widths.append(width)
            else:
                skipped.append({
                    "encoder": encoder, "packing_width": width,
                    "reason": (
                        f"width*d_priv={width * data['d_priv']} exceeds single-"
                        f"ciphertext slot capacity={ckks_metadata['slot_capacity']}"
                    ),
                    "maximum_single_ciphertext_width": max_width,
                })
                log(f"skip {encoder} width={width}: exceeds {max_width} blocks/ciphertext")
        if 1 not in valid_widths:
            raise AssertionError("width 1 must fit")

        for seed in seeds:
            for candidate_count in candidate_counts:
                case, offline_key_seconds = make_query_case(
                    data, seed, candidate_count, args.cells, args.key_seed_offset,
                )
                log(
                    f"case encoder={encoder} seed={seed} q={case['query_id']} "
                    f"K={candidate_count} active_cells={len(case['active_cells'])} "
                    f"offline_keys/docs={offline_key_seconds:.3f}s"
                )
                for width in valid_widths:
                    packed_length = width * data["d_priv"]
                    warm_key = (encoder, width)
                    if warm_key not in warmed:
                        warmup_ckks(client, server, packed_length, 26026 + width)
                        warmed.add(warm_key)
                # Interleave layouts so thermal/frequency drift cannot
                # systematically favour width 1 or the largest width.
                schedule = [
                    (repeat, width)
                    for repeat in range(repetitions)
                    for width in valid_widths
                ]
                schedule_rng = np.random.default_rng(
                    26_000_000 + seed * 10_000 + candidate_count,
                )
                schedule_rng.shuffle(schedule)
                for repeat, width in schedule:
                    packed_length = width * data["d_priv"]
                    gc.collect()
                    metrics = online_trial(client, server, case, width, data["d_priv"])
                    row = {
                        "encoder": encoder,
                        "dataset": "SciFact",
                        "seed": seed,
                        "repeat": repeat,
                        "query_id": case["query_id"],
                        "candidate_count": len(case["candidate_ids"]),
                        "active_cell_count": len(case["active_cells"]),
                        "packing_width": width,
                        "packed_vector_length": packed_length,
                        "d": data["d"],
                        "d_pub": data["d_pub"],
                        "d_priv": data["d_priv"],
                        "offline_key_and_document_prepare_seconds": offline_key_seconds,
                        **metrics,
                    }
                    raw.append(row)
                    log(
                        f"measured {encoder} seed={seed} K={candidate_count} "
                        f"width={width} rep={repeat}: e2e={metrics['end_to_end_ms']:.1f}ms "
                        f"server={metrics['server_ct_pt_eval_ms']:.1f}ms "
                        f"upload={metrics['query_ciphertext_bytes']/1e6:.2f}MB "
                        f"maxerr={metrics['score_max_abs_error']:.2e} "
                        f"top10={metrics['top10_overlap']:.3f}"
                    )

    summary = summarize(raw)
    atomic_json(out / "geometry.json", geometry)
    atomic_json(out / "skipped_layouts.json", skipped)
    atomic_json(out / "raw_measurements.json", raw)
    write_csv(out / "raw_measurements.csv", raw)
    atomic_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)

    elapsed = time.perf_counter() - wall_start
    end_utc = dt.datetime.now(dt.timezone.utc)
    memory_info = psutil.virtual_memory()
    run_info = {
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "wall_seconds": elapsed,
        "git_revision": get_git_revision(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "cpu_affinity_logical_processors": psutil.Process(os.getpid()).cpu_affinity(),
        "system_memory_bytes": memory_info.total,
        "numpy": np.__version__,
        "tenseal": getattr(ts, "__version__", "unknown"),
        "psutil": psutil.__version__,
        "gpu_inventory": get_gpu_inventory(),
        "ckks": ckks_metadata,
        "raw_measurement_count": len(raw),
        "summary_cell_count": len(summary),
        "measured_not_modelled": True,
        "gpu_used_by_tenseal": False,
    }
    atomic_json(out / "run_info.json", run_info)
    (out / "README.md").write_text(
        format_readme(config, ckks_metadata, unit, geometry, summary, elapsed),
        encoding="utf-8",
    )
    log(f"complete: measurements={len(raw)} summaries={len(summary)} wall={elapsed:.1f}s")


if __name__ == "__main__":
    main()
