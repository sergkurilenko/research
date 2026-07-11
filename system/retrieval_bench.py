"""Unified go/no-go retrieval baselines for the revised paper.

The primary cached pilot is multilingual-e5-base over one million passages.
The online protocol is deliberately fixed to an *uncentred* query projection::

    z_q = q @ V_k
    z_d = (d - passage_mean) @ V_k

The passage mean may be used while fitting the SVD basis, but it is not
subtracted from online queries unless ``--center-online-queries`` is supplied
as an explicit ablation.  Subtracting the document mean from every projected
document changes every score for a fixed query by the same constant and hence
does not change the ranking.

The harness evaluates three paired methods on exactly the same queries:

``exact_projected``
    Exhaustive inner-product search in the projected space.  This is a local
    quality reference, not a proposed network protocol.
``pq_only``
    Search the public FAISS PQ artifact and return its top-k ordering.
``return_full_vectors``
    Use PQ for a client-side shortlist, request those full float32 passage
    vectors, and rerank locally with the original query embedding.  This is
    the mandatory dominance baseline for an encrypted reranker.
``projected_shortlist_rerank``
    Use the same PQ shortlist, gather exact ``(d - passage_mean) @ V_k``
    passage vectors, and score them with deployable ``q @ V_k``.  This is the
    plaintext computational reference for the CKKS dot-product kernel, not a
    claimed online protocol.  Its reported vector payload is explicitly a
    counterfactual return-vectors baseline.

Large ``.npy`` arrays are opened with ``mmap_mode='r'``.  Exact search and PQ
construction project documents in chunks, so the projected million-document
matrix need not be resident in RAM.  Every metric has a paired query-bootstrap
confidence interval; per-query latency and application-payload records can be
written as JSONL.

Safe first command (small deterministic cache-backed smoke run)::

    python system/retrieval_bench.py --smoke

The full pilot is intentionally opt-in::

    python system/retrieval_bench.py --split validation \
      --summary-json results/retrieval_e5base_validation.json \
      --query-log-jsonl results/retrieval_e5base_validation.jsonl

After all choices are frozen, rerun with ``--split test``.  The bundled legacy
e5-base PQ artifact is used only with its exact deterministic SVD recipe
(k=672, 200k fit passages, seeds 0/42); otherwise ``--allow-build-pq`` is
required.
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


DEFAULT_CACHE = Path(
    os.environ.get(
        "PACKRERANK_CACHE",
        str(Path(__file__).resolve().parents[1] / "cache" / "million_vector"),
    )
)


@dataclass(frozen=True)
class ModelSpec:
    """Paths and the paper's primary projection/PQ operating point."""

    tag: str
    docs_file: str
    queries_file: str
    dimension: int
    projection_dim: int
    pq_m: int
    pq_nbits: int = 8
    legacy_pq_file: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "e5-base": ModelSpec(
        tag="e5-base",
        docs_file="E_docs_e5base_1000000.npy",
        queries_file="E_queries_e5base_self_q500.npy",
        dimension=768,
        projection_dim=672,
        pq_m=96,
        legacy_pq_file="v2_indexpq_e5base_proj672_M96_b8.faiss",
    ),
    "e5-small": ModelSpec(
        tag="e5-small",
        docs_file="E_docs_e5small_1000000.npy",
        queries_file="E_queries_e5small_self_q500.npy",
        dimension=384,
        projection_dim=336,
        pq_m=84,
        legacy_pq_file="v2_indexpq_e5small_proj336_M84_b8.faiss",
    ),
    "mpnet": ModelSpec(
        tag="mpnet",
        docs_file="E_docs_mpnet_1000000.npy",
        queries_file="E_queries_mpnet_self_q500.npy",
        dimension=768,
        projection_dim=672,
        pq_m=96,
        legacy_pq_file="v2_indexpq_mpnet_proj672_M96_b8.faiss",
    ),
}


@dataclass
class BenchmarkConfig:
    """Configuration shared by cache-backed and in-memory benchmark runs."""

    model: str = "e5-base"
    projection_dim: int = 672
    pq_m: int = 96
    pq_nbits: int = 8
    shortlist_k: int = 40
    top_k: int = 10
    center_online_queries: bool = False
    exact_backend: str = "auto"  # auto, numpy, cuda
    doc_chunk_size: int = 16_384
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    fit_sample_size: int = 200_000
    fit_sample_seed: int = 0
    svd_seed: int = 42
    pq_train_size: int = 200_000
    pq_train_seed: int = 42
    faiss_threads: int = 0
    network_mbps: float | None = None
    network_rtt_ms: float = 0.0

    def validate(self, dimension: int, n_docs: int) -> None:
        if not 1 <= self.projection_dim <= dimension:
            raise ValueError(
                f"projection_dim must be in [1, {dimension}], got "
                f"{self.projection_dim}"
            )
        if self.projection_dim % self.pq_m:
            raise ValueError(
                f"PQ M={self.pq_m} must divide projection_dim="
                f"{self.projection_dim}"
            )
        if not 1 <= self.top_k <= self.shortlist_k <= n_docs:
            raise ValueError(
                "Expected 1 <= top_k <= shortlist_k <= number of documents"
            )
        if self.exact_backend not in {"auto", "numpy", "cuda"}:
            raise ValueError("exact_backend must be one of auto, numpy, cuda")
        if self.doc_chunk_size < 1:
            raise ValueError("doc_chunk_size must be positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if self.network_mbps is not None and self.network_mbps <= 0:
            raise ValueError("network_mbps must be positive when supplied")
        if self.network_rtt_ms < 0:
            raise ValueError("network_rtt_ms cannot be negative")


@dataclass(frozen=True)
class ProjectionBasis:
    """A passage-centred SVD basis and its deterministic construction data."""

    passage_mean: np.ndarray
    vectors: np.ndarray
    fit_sample_size: int
    fit_sample_seed: int
    svd_seed: int

    def __post_init__(self) -> None:
        mu = np.asarray(self.passage_mean)
        vectors = np.asarray(self.vectors)
        if mu.ndim == 1:
            mu = mu.reshape(1, -1)
            object.__setattr__(self, "passage_mean", mu)
        if mu.ndim != 2 or mu.shape[0] != 1:
            raise ValueError("passage_mean must have shape (1, dimension)")
        if vectors.ndim != 2 or vectors.shape[0] != mu.shape[1]:
            raise ValueError("basis vectors must have shape (dimension, k)")


def _require_faiss() -> Any:
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "FAISS is required for PQ baselines; use the torchgpu environment"
        ) from exc
    return faiss


def load_memmap(path: Path) -> np.memmap:
    """Open an NPY array read-only without silently materialising it."""

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise TypeError(f"Expected an NPY memmap at {path}")
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional array at {path}")
    if array.dtype != np.float32:
        raise ValueError(f"Expected float32 embeddings at {path}, got {array.dtype}")
    return array


def deterministic_self_qrels(
    n_docs: int, n_queries: int, seed: int = 42
) -> np.ndarray:
    """Reproduce the cached self-query passage ids used by the source scripts."""

    if n_queries > n_docs:
        raise ValueError("Cannot sample more unique qrels than documents")
    rng = np.random.default_rng(seed)
    return rng.choice(n_docs, size=n_queries, replace=False).astype(np.int64)


def deterministic_query_split(
    n_queries: int, validation_size: int = 100, seed: int = 2026
) -> dict[str, np.ndarray]:
    """Return disjoint, stable validation/test query-index arrays.

    The permutation order is retained.  Therefore ``--max-queries`` selects a
    deterministic random prefix rather than the numerically lowest query ids.
    """

    if not 1 <= validation_size < n_queries:
        raise ValueError("validation_size must be in [1, n_queries - 1]")
    order = np.random.default_rng(seed).permutation(n_queries).astype(np.int64)
    return {
        "validation": order[:validation_size],
        "test": order[validation_size:],
    }


def select_query_indices(
    n_queries: int,
    split: str,
    validation_size: int,
    split_seed: int,
    max_queries: int | None = None,
) -> np.ndarray:
    if split == "all":
        selected = np.arange(n_queries, dtype=np.int64)
    else:
        partitions = deterministic_query_split(n_queries, validation_size, split_seed)
        if split not in partitions:
            raise ValueError("split must be validation, test, or all")
        selected = partitions[split]
    if max_queries is not None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        selected = selected[:max_queries]
    return selected


def deterministic_corpus_subset(
    n_docs: int,
    required_ids: np.ndarray,
    subset_size: int,
    seed: int,
) -> np.ndarray:
    """Sample a sorted corpus subset that always contains all required ids."""

    required = set(int(x) for x in np.asarray(required_ids, dtype=np.int64))
    if any(x < 0 or x >= n_docs for x in required):
        raise ValueError("required document id is outside the corpus")
    if len(required) > subset_size or subset_size > n_docs:
        raise ValueError("subset_size must cover required ids and fit the corpus")
    rng = np.random.default_rng(seed)
    while len(required) < subset_size:
        required.add(int(rng.integers(0, n_docs)))
    return np.asarray(sorted(required), dtype=np.int64)


def fit_projection_basis(
    docs: np.ndarray,
    projection_dim: int,
    sample_size: int = 200_000,
    sample_seed: int = 0,
    svd_seed: int = 42,
) -> ProjectionBasis:
    """Fit the deterministic legacy-compatible SVD without loading all docs.

    Keeping ``docs.mean`` and ``RandomState.choice`` is intentional: it
    reproduces the recipe that produced the existing public PQ artifacts.
    """

    from sklearn.utils.extmath import randomized_svd

    n_docs, dimension = docs.shape
    sample_size = min(int(sample_size), n_docs)
    if projection_dim > min(sample_size, dimension):
        raise ValueError("projection_dim exceeds sampled matrix rank")
    t0 = time.perf_counter()
    passage_mean = docs.mean(axis=0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(sample_seed)
    sample_ids = rng.choice(n_docs, size=sample_size, replace=False)
    sample = np.asarray(docs[sample_ids], dtype=np.float32)
    sample -= passage_mean
    _, _, vt = randomized_svd(
        sample, n_components=projection_dim, random_state=svd_seed
    )
    del sample
    basis = ProjectionBasis(
        passage_mean=passage_mean,
        vectors=vt.T.astype(np.float32),
        fit_sample_size=sample_size,
        fit_sample_seed=sample_seed,
        svd_seed=svd_seed,
    )
    # Kept as an attribute-free diagnostic to avoid polluting the immutable
    # scientific artifact with machine-dependent timing.
    _ = time.perf_counter() - t0
    return basis


def load_projection_basis(path: Path) -> ProjectionBasis:
    data = np.load(path, allow_pickle=False)
    files = set(data.files)
    mu_key = "passage_mean" if "passage_mean" in files else "mu"
    vectors_key = "vectors" if "vectors" in files else "Vk"
    if mu_key not in files or vectors_key not in files:
        raise ValueError(f"Basis {path} must contain mu/Vk or passage_mean/vectors")

    def scalar(name: str, default: int) -> int:
        return int(np.asarray(data[name]).item()) if name in files else default

    return ProjectionBasis(
        passage_mean=np.asarray(data[mu_key], dtype=np.float32),
        vectors=np.asarray(data[vectors_key], dtype=np.float32),
        fit_sample_size=scalar("fit_sample_size", -1),
        fit_sample_seed=scalar("fit_sample_seed", 0),
        svd_seed=scalar("svd_seed", 42),
    )


def save_projection_basis(path: Path, basis: ProjectionBasis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        passage_mean=np.asarray(basis.passage_mean, dtype=np.float32),
        vectors=np.asarray(basis.vectors, dtype=np.float32),
        fit_sample_size=np.int64(basis.fit_sample_size),
        fit_sample_seed=np.int64(basis.fit_sample_seed),
        svd_seed=np.int64(basis.svd_seed),
    )


def project_queries(
    queries: np.ndarray,
    basis: ProjectionBasis,
    center_online_queries: bool = False,
) -> np.ndarray:
    """Project queries under the deployable protocol (uncentred by default)."""

    q = np.asarray(queries, dtype=np.float32)
    if center_online_queries:
        q = q - basis.passage_mean
    return np.asarray(q @ basis.vectors, dtype=np.float32)


def project_doc_chunk(docs: np.ndarray, basis: ProjectionBasis) -> np.ndarray:
    docs32 = np.asarray(docs, dtype=np.float32)
    return np.asarray((docs32 - basis.passage_mean) @ basis.vectors, dtype=np.float32)


def basis_fingerprint(basis: ProjectionBasis) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(basis.passage_mean).view(np.uint8))
    digest.update(np.ascontiguousarray(basis.vectors).view(np.uint8))
    return digest.hexdigest()


def _chunk_topk(scores: np.ndarray, k: int, id_offset: int) -> tuple[np.ndarray, np.ndarray]:
    keep = min(k, scores.shape[1])
    part = np.argpartition(-scores, keep - 1, axis=1)[:, :keep]
    values = np.take_along_axis(scores, part, axis=1)
    ids = part.astype(np.int64) + id_offset
    order = np.argsort(-values, axis=1, kind="stable")
    return (
        np.take_along_axis(values, order, axis=1),
        np.take_along_axis(ids, order, axis=1),
    )


def _merge_topk(
    old_scores: np.ndarray,
    old_ids: np.ndarray,
    new_scores: np.ndarray,
    new_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.concatenate((old_scores, new_scores), axis=1)
    ids = np.concatenate((old_ids, new_ids), axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return (
        np.take_along_axis(scores, order, axis=1),
        np.take_along_axis(ids, order, axis=1),
    )


def _cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_exact_backend(requested: str) -> str:
    if requested == "auto":
        return "cuda" if _cuda_is_available() else "numpy"
    if requested == "cuda" and not _cuda_is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda is unavailable")
    return requested


def exact_projected_search_numpy(
    docs: np.ndarray,
    projected_queries: np.ndarray,
    basis: ProjectionBasis,
    top_k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Memory-bounded exhaustive projected search using NumPy BLAS."""

    nq = len(projected_queries)
    best_scores = np.full((nq, top_k), -np.inf, dtype=np.float32)
    best_ids = np.full((nq, top_k), -1, dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, len(docs), chunk_size):
        projected_docs = project_doc_chunk(docs[start : start + chunk_size], basis)
        scores = np.asarray(projected_queries @ projected_docs.T, dtype=np.float32)
        chunk_scores, chunk_ids = _chunk_topk(scores, top_k, start)
        best_scores, best_ids = _merge_topk(
            best_scores, best_ids, chunk_scores, chunk_ids, top_k
        )
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    return best_scores, best_ids, elapsed_ms


def exact_projected_search_cuda(
    docs: np.ndarray,
    projected_queries: np.ndarray,
    basis: ProjectionBasis,
    top_k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Memory-bounded exhaustive projected search on CUDA.

    TF32 is disabled because retrieval ranks can change at near ties.  "Exact"
    here means exhaustive rather than approximate indexing; arithmetic remains
    float32, matching the cached embeddings and FAISS reference.
    """

    import torch

    device = torch.device("cuda")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q = torch.as_tensor(
            np.ascontiguousarray(projected_queries), device=device, dtype=torch.float32
        )
        vectors = torch.as_tensor(
            np.ascontiguousarray(basis.vectors), device=device, dtype=torch.float32
        )
        mean = torch.as_tensor(
            np.ascontiguousarray(basis.passage_mean),
            device=device,
            dtype=torch.float32,
        )
        best_scores = torch.full(
            (len(q), top_k), -torch.inf, device=device, dtype=torch.float32
        )
        best_ids = torch.full(
            (len(q), top_k), -1, device=device, dtype=torch.int64
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, len(docs), chunk_size):
            host = np.ascontiguousarray(docs[start : start + chunk_size])
            d = torch.as_tensor(host, device=device, dtype=torch.float32)
            projected_docs = (d - mean) @ vectors
            scores = q @ projected_docs.T
            keep = min(top_k, scores.shape[1])
            values, local_ids = torch.topk(scores, keep, dim=1, largest=True)
            global_ids = local_ids.to(torch.int64) + start
            merged_scores = torch.cat((best_scores, values), dim=1)
            merged_ids = torch.cat((best_ids, global_ids), dim=1)
            best_scores, positions = torch.topk(
                merged_scores, top_k, dim=1, largest=True
            )
            best_ids = torch.gather(merged_ids, 1, positions)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        return (
            best_scores.cpu().numpy(),
            best_ids.cpu().numpy(),
            elapsed_ms,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def exact_projected_search(
    docs: np.ndarray,
    projected_queries: np.ndarray,
    basis: ProjectionBasis,
    top_k: int,
    chunk_size: int,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, float, str]:
    resolved = resolve_exact_backend(backend)
    if resolved == "cuda":
        scores, ids, elapsed = exact_projected_search_cuda(
            docs, projected_queries, basis, top_k, chunk_size
        )
    else:
        scores, ids, elapsed = exact_projected_search_numpy(
            docs, projected_queries, basis, top_k, chunk_size
        )
    return scores, ids, elapsed, resolved


def configure_faiss_threads(n_threads: int) -> None:
    if n_threads > 0:
        _require_faiss().omp_set_num_threads(n_threads)


def load_pq_index(path: Path, dimension: int, n_docs: int) -> Any:
    faiss = _require_faiss()
    index = faiss.read_index(str(path))
    if int(index.d) != dimension:
        raise ValueError(f"PQ index d={index.d}, expected {dimension}: {path}")
    if int(index.ntotal) != n_docs:
        raise ValueError(f"PQ index ntotal={index.ntotal}, expected {n_docs}: {path}")
    return index


def build_pq_index_chunked(
    docs: np.ndarray,
    basis: ProjectionBasis,
    m: int,
    nbits: int,
    train_size: int,
    train_seed: int,
    chunk_size: int,
) -> Any:
    """Train/add an IndexPQ while keeping the full projected corpus off-heap."""

    faiss = _require_faiss()
    if basis.vectors.shape[1] % m:
        raise ValueError("PQ M must divide the projected dimension")
    n_centroids = 1 << nbits
    train_size = min(int(train_size), len(docs))
    if train_size < n_centroids:
        raise ValueError(
            f"PQ training needs at least {n_centroids} vectors, got {train_size}"
        )
    rng = np.random.RandomState(train_seed)
    train_ids = rng.choice(len(docs), size=train_size, replace=False)
    training = project_doc_chunk(docs[train_ids], basis)
    index = faiss.IndexPQ(
        basis.vectors.shape[1], m, nbits, faiss.METRIC_INNER_PRODUCT
    )
    index.train(np.ascontiguousarray(training))
    del training
    for start in range(0, len(docs), chunk_size):
        projected = project_doc_chunk(docs[start : start + chunk_size], basis)
        index.add(np.ascontiguousarray(projected))
    return index


def save_faiss_index(path: Path, index: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_faiss().write_index(index, str(path))


def timed_faiss_search(
    index: Any, projected_queries: np.ndarray, shortlist_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Search one query at a time so p50/p95 are real per-query latencies."""

    nq = len(projected_queries)
    scores = np.empty((nq, shortlist_k), dtype=np.float32)
    ids = np.empty((nq, shortlist_k), dtype=np.int64)
    latency_ms = np.empty(nq, dtype=np.float64)
    if nq:
        # Untimed warm-up faults in index pages and initialises FAISS workers.
        index.search(np.ascontiguousarray(projected_queries[:1]), shortlist_k)
    for i, query in enumerate(projected_queries):
        started = time.perf_counter_ns()
        row_scores, row_ids = index.search(
            np.ascontiguousarray(query.reshape(1, -1)), shortlist_k
        )
        latency_ms[i] = (time.perf_counter_ns() - started) / 1_000_000.0
        scores[i] = row_scores[0]
        ids[i] = row_ids[0]
    if np.any(ids < 0):
        raise RuntimeError("FAISS returned missing candidates; shortlist is too large")
    return scores, ids, latency_ms


def _transport_latency_ms(payload_bytes: int, config: BenchmarkConfig) -> float:
    if config.network_mbps is None:
        return 0.0
    serialization_ms = payload_bytes * 8.0 / (config.network_mbps * 1_000.0)
    return config.network_rtt_ms + serialization_ms


def rerank_returned_full_vectors(
    docs: np.ndarray,
    queries: np.ndarray,
    candidate_ids: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Simulate server gather/serialization and client-side exact reranking."""

    nq, shortlist_k = candidate_ids.shape
    predictions = np.empty((nq, config.top_k), dtype=np.int64)
    request_bytes = np.empty(nq, dtype=np.int64)
    response_bytes = np.empty(nq, dtype=np.int64)
    fetch_serialize_ms = np.empty(nq, dtype=np.float64)
    local_rerank_ms = np.empty(nq, dtype=np.float64)
    transport_ms = np.empty(nq, dtype=np.float64)

    for i in range(nq):
        ids = np.asarray(candidate_ids[i], dtype=np.int64)
        request = ids.astype("<u8", copy=False).tobytes(order="C")
        started = time.perf_counter_ns()
        vectors = np.ascontiguousarray(docs[ids], dtype=np.float32)
        response = vectors.astype("<f4", copy=False).tobytes(order="C")
        fetch_serialize_ms[i] = (
            time.perf_counter_ns() - started
        ) / 1_000_000.0

        started = time.perf_counter_ns()
        received = np.frombuffer(response, dtype="<f4").reshape(shortlist_k, -1)
        raw_scores = received @ np.asarray(queries[i], dtype=np.float32)
        order = np.argsort(-raw_scores, kind="stable")[: config.top_k]
        predictions[i] = ids[order]
        local_rerank_ms[i] = (time.perf_counter_ns() - started) / 1_000_000.0

        request_bytes[i] = len(request)
        response_bytes[i] = len(response)
        transport_ms[i] = _transport_latency_ms(
            len(request) + len(response), config
        )

    return predictions, {
        "server_fetch_serialize_ms": fetch_serialize_ms,
        "local_deserialize_rerank_ms": local_rerank_ms,
        "estimated_transport_ms": transport_ms,
    }, {
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
    }


def rerank_projected_shortlist(
    docs: np.ndarray,
    projected_queries: np.ndarray,
    candidate_ids: np.ndarray,
    basis: ProjectionBasis,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Plaintext reference for the exact projected CKKS rerank kernel.

    The timed computation is server-style gather -> projection -> plaintext
    dot-product/rank.  In addition, projected vectors are serialized solely to
    quantify the counterfactual baseline in which they are returned to the
    client.  Those bytes and their estimated transport time are *not* included
    in ``computational_reference_end_to_end_ms`` and do not describe the
    encrypted protocol.
    """

    nq, shortlist_k = candidate_ids.shape
    predictions = np.empty((nq, config.top_k), dtype=np.int64)
    request_bytes = np.empty(nq, dtype=np.int64)
    response_bytes = np.empty(nq, dtype=np.int64)
    gather_ms = np.empty(nq, dtype=np.float64)
    project_ms = np.empty(nq, dtype=np.float64)
    rerank_ms = np.empty(nq, dtype=np.float64)
    baseline_serialize_ms = np.empty(nq, dtype=np.float64)
    baseline_transport_ms = np.empty(nq, dtype=np.float64)

    for i in range(nq):
        ids = np.asarray(candidate_ids[i], dtype=np.int64)
        request = ids.astype("<u8", copy=False).tobytes(order="C")

        started = time.perf_counter_ns()
        gathered = np.ascontiguousarray(docs[ids], dtype=np.float32)
        gather_ms[i] = (time.perf_counter_ns() - started) / 1_000_000.0

        started = time.perf_counter_ns()
        projected_docs = project_doc_chunk(gathered, basis)
        project_ms[i] = (time.perf_counter_ns() - started) / 1_000_000.0

        started = time.perf_counter_ns()
        scores = projected_docs @ np.asarray(projected_queries[i], dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[: config.top_k]
        predictions[i] = ids[order]
        rerank_ms[i] = (time.perf_counter_ns() - started) / 1_000_000.0

        # Counterfactual payload only: a plaintext return-vectors baseline.
        # The plaintext kernel above scores projected_docs directly and does
        # not rely on this byte string.
        started = time.perf_counter_ns()
        response = projected_docs.astype("<f4", copy=False).tobytes(order="C")
        baseline_serialize_ms[i] = (
            time.perf_counter_ns() - started
        ) / 1_000_000.0
        request_bytes[i] = len(request)
        response_bytes[i] = len(response)
        baseline_transport_ms[i] = _transport_latency_ms(
            len(request) + len(response), config
        )

    computational_end_to_end = gather_ms + project_ms + rerank_ms
    return predictions, {
        "server_gather_ms": gather_ms,
        "server_project_ms": project_ms,
        "plaintext_projected_rerank_ms": rerank_ms,
        "computational_reference_end_to_end_ms": computational_end_to_end,
        "counterfactual_vector_serialize_ms": baseline_serialize_ms,
        "counterfactual_vector_transport_ms": baseline_transport_ms,
    }, {
        "request_candidate_ids_bytes": request_bytes,
        "counterfactual_projected_vectors_response_bytes": response_bytes,
    }


def local_to_global_ids(local_ids: np.ndarray, corpus_ids: np.ndarray | None) -> np.ndarray:
    local = np.asarray(local_ids, dtype=np.int64)
    if corpus_ids is None:
        return local.copy()
    return np.asarray(corpus_ids, dtype=np.int64)[local]


def metric_vectors(predictions: np.ndarray, ground_truth: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query binary-relevance metrics for the cached self-query benchmark."""

    predictions = np.asarray(predictions, dtype=np.int64)
    gt = np.asarray(ground_truth, dtype=np.int64)
    if predictions.ndim != 2 or predictions.shape[0] != len(gt):
        raise ValueError("predictions/ground_truth shape mismatch")
    matches = predictions == gt[:, None]
    hit1 = matches[:, 0].astype(np.float64)
    hitk = matches.any(axis=1).astype(np.float64)
    reciprocal_rank = np.zeros(len(gt), dtype=np.float64)
    ndcg = np.zeros(len(gt), dtype=np.float64)
    for i in range(len(gt)):
        positions = np.flatnonzero(matches[i])
        if len(positions):
            rank = int(positions[0]) + 1
            reciprocal_rank[i] = 1.0 / rank
            ndcg[i] = 1.0 / np.log2(rank + 1.0)
    return {
        "hit_at_1": hit1,
        f"hit_at_{predictions.shape[1]}": hitk,
        f"mrr_at_{predictions.shape[1]}": reciprocal_rank,
        f"ndcg_at_{predictions.shape[1]}": ndcg,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 2026,
    confidence: float = 0.95,
    sample_indices: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap values must be a non-empty vector")
    if sample_indices is None:
        sample_indices = np.random.default_rng(seed).integers(
            0, len(values), size=(n_bootstrap, len(values)), dtype=np.int64
        )
    boot = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": int(len(sample_indices)),
    }


def summarize_latency(
    values: np.ndarray, sample_indices: np.ndarray
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    mean_ci = bootstrap_mean_ci(values, sample_indices=sample_indices)
    return {
        **mean_ci,
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _method_summary(
    metrics: Mapping[str, np.ndarray],
    latency: Mapping[str, np.ndarray],
    payload: Mapping[str, np.ndarray],
    sample_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "metrics": {
            key: bootstrap_mean_ci(value, sample_indices=sample_indices)
            for key, value in metrics.items()
        },
        "latency_ms": {
            key: summarize_latency(value, sample_indices)
            for key, value in latency.items()
        },
        "payload_bytes": {
            key: {
                "mean": float(np.mean(value)),
                "min": int(np.min(value)),
                "max": int(np.max(value)),
            }
            for key, value in payload.items()
        },
    }


def _hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        metadata["cuda_available"] = False
    try:
        faiss = _require_faiss()
        metadata["faiss"] = getattr(faiss, "__version__", "unknown")
        metadata["faiss_threads"] = int(faiss.omp_get_max_threads())
    except RuntimeError:
        metadata["faiss"] = None
    return metadata


def run_protocol_benchmarks(
    docs: np.ndarray,
    queries: np.ndarray,
    ground_truth_ids: np.ndarray,
    query_indices: np.ndarray,
    basis: ProjectionBasis,
    pq_index: Any,
    config: BenchmarkConfig,
    *,
    corpus_ids: np.ndarray | None = None,
    split_name: str = "validation",
    source_is_memmap: bool = False,
    pq_artifact: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run all paired baselines and return a summary plus per-query records."""

    docs = np.asanyarray(docs)
    queries = np.asarray(queries, dtype=np.float32)
    ground_truth_ids = np.asarray(ground_truth_ids, dtype=np.int64)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    if docs.ndim != 2 or queries.ndim != 2 or docs.shape[1] != queries.shape[1]:
        raise ValueError("Document/query embedding dimensions do not match")
    if len(queries) != len(ground_truth_ids) or len(queries) != len(query_indices):
        raise ValueError("queries, qrels, and query_indices must have equal length")
    if corpus_ids is not None and len(corpus_ids) != len(docs):
        raise ValueError("corpus_ids must map every local document row")
    config.validate(docs.shape[1], len(docs))
    if basis.vectors.shape != (docs.shape[1], config.projection_dim):
        raise ValueError("Projection basis shape does not match benchmark config")
    if int(pq_index.d) != config.projection_dim or int(pq_index.ntotal) != len(docs):
        raise ValueError("PQ index shape/count does not match benchmark corpus")

    projected_queries = project_queries(
        queries, basis, config.center_online_queries
    )

    _, exact_local, exact_elapsed_ms, exact_backend = exact_projected_search(
        docs,
        projected_queries,
        basis,
        config.top_k,
        config.doc_chunk_size,
        config.exact_backend,
    )
    exact_predictions = local_to_global_ids(exact_local, corpus_ids)
    exact_amortized = np.full(
        len(queries), exact_elapsed_ms / len(queries), dtype=np.float64
    )

    _, candidate_local, pq_latency = timed_faiss_search(
        pq_index, projected_queries, config.shortlist_k
    )
    candidate_global = local_to_global_ids(candidate_local, corpus_ids)
    pq_predictions = candidate_global[:, : config.top_k]

    reranked_local, rerank_parts, rerank_payload = rerank_returned_full_vectors(
        docs, queries, candidate_local, config
    )
    rerank_predictions = local_to_global_ids(reranked_local, corpus_ids)

    projected_reranked_local, projected_rerank_parts, projected_rerank_payload = (
        rerank_projected_shortlist(
            docs, projected_queries, candidate_local, basis, config
        )
    )
    projected_rerank_predictions = local_to_global_ids(
        projected_reranked_local, corpus_ids
    )

    shortlist_recall = np.any(
        candidate_global == ground_truth_ids[:, None], axis=1
    ).astype(np.float64)
    metrics_by_method = {
        "exact_projected": metric_vectors(exact_predictions, ground_truth_ids),
        "pq_only": {
            **metric_vectors(pq_predictions, ground_truth_ids),
            f"shortlist_recall_at_{config.shortlist_k}": shortlist_recall,
        },
        "return_full_vectors": {
            **metric_vectors(rerank_predictions, ground_truth_ids),
            f"shortlist_recall_at_{config.shortlist_k}": shortlist_recall,
        },
        "projected_shortlist_rerank": {
            **metric_vectors(projected_rerank_predictions, ground_truth_ids),
            f"shortlist_recall_at_{config.shortlist_k}": shortlist_recall,
        },
    }

    zero_i64 = np.zeros(len(queries), dtype=np.int64)
    latency_by_method = {
        "exact_projected": {"batch_amortized_search_ms": exact_amortized},
        "pq_only": {"client_pq_search_ms": pq_latency, "end_to_end_ms": pq_latency},
        "return_full_vectors": {
            "client_pq_search_ms": pq_latency,
            **rerank_parts,
            "end_to_end_ms": (
                pq_latency
                + rerank_parts["server_fetch_serialize_ms"]
                + rerank_parts["local_deserialize_rerank_ms"]
                + rerank_parts["estimated_transport_ms"]
            ),
        },
        "projected_shortlist_rerank": {
            "client_pq_search_ms": pq_latency,
            **projected_rerank_parts,
            "computational_reference_total_ms": (
                pq_latency
                + projected_rerank_parts[
                    "computational_reference_end_to_end_ms"
                ]
            ),
        },
    }
    payload_by_method = {
        "exact_projected": {
            "request_bytes": zero_i64,
            "response_bytes": zero_i64,
        },
        "pq_only": {
            "request_bytes": zero_i64,
            "response_bytes": zero_i64,
        },
        "return_full_vectors": rerank_payload,
        "projected_shortlist_rerank": projected_rerank_payload,
    }

    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap_indices = rng.integers(
        0,
        len(queries),
        size=(config.bootstrap_samples, len(queries)),
        dtype=np.int64,
    )
    method_summaries = {
        method: _method_summary(
            metrics_by_method[method],
            latency_by_method[method],
            payload_by_method[method],
            bootstrap_indices,
        )
        for method in metrics_by_method
    }

    primary_keys = list(metrics_by_method["exact_projected"])
    paired_deltas: dict[str, Any] = {}
    for method in (
        "pq_only",
        "return_full_vectors",
        "projected_shortlist_rerank",
    ):
        paired_deltas[method + "_minus_exact_projected"] = {
            key: bootstrap_mean_ci(
                metrics_by_method[method][key]
                - metrics_by_method["exact_projected"][key],
                sample_indices=bootstrap_indices,
            )
            for key in primary_keys
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "document_projection": "(document - passage_mean) @ V_k",
            "online_query_projection": (
                "(query - passage_mean) @ V_k [explicit ablation]"
                if config.center_online_queries
                else "query @ V_k [deployable default]"
            ),
            "query_centered": bool(config.center_online_queries),
            "return_vectors_dtype": "float32",
            "payload_scope": "application payload; headers/TLS excluded",
            "latency_scope": (
                "compute + configured transport estimate"
                if config.network_mbps is not None
                else "local compute/serialization only; network transport excluded"
            ),
            "method_semantics": {
                "exact_projected": "local exhaustive quality reference",
                "pq_only": "client-side public PQ artifact",
                "return_full_vectors": (
                    "candidate IDs request; raw float32 document vectors response; "
                    "local exact raw-space rerank"
                ),
                "projected_shortlist_rerank": (
                    "plaintext computational reference for the CKKS projected "
                    "dot-product kernel; not a claimed online protocol"
                ),
            },
            "projected_shortlist_payload_model": (
                "counterfactual baseline only: candidate IDs request and exact "
                "projected float32 vectors response; excluded from computational "
                "reference total"
            ),
        },
        "config": {
            **dataclasses.asdict(config),
            "split": split_name,
            "n_queries": len(queries),
            "n_docs": len(docs),
            "embedding_dim": docs.shape[1],
            "source_is_memmap": bool(source_is_memmap),
            "exact_backend_resolved": exact_backend,
        },
        "basis": {
            "sha256": basis_fingerprint(basis),
            "shape": list(basis.vectors.shape),
            "fit_sample_size": basis.fit_sample_size,
            "fit_sample_seed": basis.fit_sample_seed,
            "svd_seed": basis.svd_seed,
        },
        "pq_artifact": dict(pq_artifact or {}),
        "hardware": _hardware_metadata(),
        "methods": method_summaries,
        "paired_query_bootstrap_deltas": paired_deltas,
        "exact_projected_batch_elapsed_ms": exact_elapsed_ms,
    }

    predictions_by_method = {
        "exact_projected": exact_predictions,
        "pq_only": pq_predictions,
        "return_full_vectors": rerank_predictions,
        "projected_shortlist_rerank": projected_rerank_predictions,
    }
    records: list[dict[str, Any]] = []
    for i in range(len(queries)):
        for method in predictions_by_method:
            metrics = metrics_by_method[method]
            record: dict[str, Any] = {
                "query_index": int(query_indices[i]),
                "split": split_name,
                "ground_truth_id": int(ground_truth_ids[i]),
                "method": method,
                "method_role": summary["protocol"]["method_semantics"][method],
                "predictions": [int(x) for x in predictions_by_method[method][i]],
                "metrics": {key: float(value[i]) for key, value in metrics.items()},
                "latency_ms": {
                    key: float(value[i])
                    for key, value in latency_by_method[method].items()
                },
                "payload_bytes": {
                    key: int(value[i])
                    for key, value in payload_by_method[method].items()
                },
            }
            if method != "exact_projected":
                record["candidate_ids"] = [int(x) for x in candidate_global[i]]
            if method == "projected_shortlist_rerank":
                record["computational_reference"] = True
                record["payload_is_counterfactual_return_vectors_baseline"] = True
            records.append(record)
    return summary, records


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _legacy_index_is_compatible(config: BenchmarkConfig, spec: ModelSpec) -> bool:
    return bool(
        spec.legacy_pq_file
        and config.projection_dim == spec.projection_dim
        and config.pq_m == spec.pq_m
        and config.pq_nbits == spec.pq_nbits
        and config.fit_sample_size == 200_000
        and config.fit_sample_seed == 0
        and config.svd_seed == 42
    )


def run_cached_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache_dir = Path(args.cache_dir)
    spec = MODEL_SPECS[args.model]
    docs_path = cache_dir / spec.docs_file
    queries_path = cache_dir / spec.queries_file
    docs = load_memmap(docs_path)
    all_queries = load_memmap(queries_path)
    if docs.shape[1] != spec.dimension or all_queries.shape[1] != spec.dimension:
        raise ValueError("Cached array dimension does not match model specification")

    query_indices = select_query_indices(
        len(all_queries),
        args.split,
        args.validation_size,
        args.split_seed,
        args.max_queries,
    )
    all_qrels = deterministic_self_qrels(len(docs), len(all_queries), args.qrel_seed)
    queries = np.asarray(all_queries[query_indices], dtype=np.float32)
    qrels = all_qrels[query_indices]

    config = BenchmarkConfig(
        model=args.model,
        projection_dim=args.projection_dim,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits,
        shortlist_k=args.shortlist_k,
        top_k=args.top_k,
        center_online_queries=args.center_online_queries,
        exact_backend=args.exact_backend,
        doc_chunk_size=args.doc_chunk_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        fit_sample_size=args.fit_sample_size,
        fit_sample_seed=args.fit_sample_seed,
        svd_seed=args.svd_seed,
        pq_train_size=args.pq_train_size,
        pq_train_seed=args.pq_train_seed,
        faiss_threads=args.faiss_threads,
        network_mbps=args.network_mbps,
        network_rtt_ms=args.network_rtt_ms,
    )
    config.validate(docs.shape[1], len(docs))
    configure_faiss_threads(config.faiss_threads)

    if args.basis_path:
        basis = load_projection_basis(Path(args.basis_path))
        basis_origin = {"source": "loaded", **_file_metadata(Path(args.basis_path))}
    else:
        print(
            f"Fitting centred passage SVD: n={min(config.fit_sample_size, len(docs))}, "
            f"d={docs.shape[1]}, k={config.projection_dim}",
            flush=True,
        )
        basis = fit_projection_basis(
            docs,
            config.projection_dim,
            config.fit_sample_size,
            config.fit_sample_seed,
            config.svd_seed,
        )
        basis_origin = {"source": "computed"}
        if args.basis_out:
            save_projection_basis(Path(args.basis_out), basis)
            basis_origin.update(_file_metadata(Path(args.basis_out)))

    explicit_pq = Path(args.pq_index_path) if args.pq_index_path else None
    legacy_path = (
        cache_dir / "index_cache" / spec.legacy_pq_file
        if spec.legacy_pq_file
        else None
    )
    if explicit_pq is not None:
        index_path = explicit_pq
    elif _legacy_index_is_compatible(config, spec) and legacy_path and legacy_path.exists():
        index_path = legacy_path
    else:
        index_path = None

    if index_path is not None:
        print(f"Loading PQ artifact: {index_path}", flush=True)
        pq_index = load_pq_index(index_path, config.projection_dim, len(docs))
        pq_artifact = {"source": "loaded", **_file_metadata(index_path)}
    else:
        if not args.allow_build_pq:
            raise RuntimeError(
                "No compatible PQ artifact selected. Pass --pq-index-path or "
                "explicitly opt into training with --allow-build-pq."
            )
        print("Training and adding the PQ index in projected chunks", flush=True)
        pq_index = build_pq_index_chunked(
            docs,
            basis,
            config.pq_m,
            config.pq_nbits,
            config.pq_train_size,
            config.pq_train_seed,
            config.doc_chunk_size,
        )
        pq_artifact = {"source": "computed", "serialized_path": None}
        if args.pq_index_out:
            save_faiss_index(Path(args.pq_index_out), pq_index)
            pq_artifact.update(_file_metadata(Path(args.pq_index_out)))

    print(
        "Online query protocol: "
        + ("CENTERED ABLATION" if config.center_online_queries else "UNCENTERED DEPLOYABLE"),
        flush=True,
    )
    summary, records = run_protocol_benchmarks(
        docs,
        queries,
        qrels,
        query_indices,
        basis,
        pq_index,
        config,
        split_name=args.split,
        source_is_memmap=True,
        pq_artifact=pq_artifact,
    )
    summary["source"] = {
        "documents": {**_file_metadata(docs_path), "shape": list(docs.shape)},
        "queries": {**_file_metadata(queries_path), "shape": list(all_queries.shape)},
        "basis_origin": basis_origin,
        "qrel_recipe": {
            "type": "self-query deterministic sampled passage id",
            "seed": args.qrel_seed,
        },
        "split_recipe": {
            "validation_size": args.validation_size,
            "seed": args.split_seed,
        },
    }
    return summary, records


def run_cache_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Short cache-backed run that cannot accidentally scan the full million."""

    cache_dir = Path(args.cache_dir)
    spec = MODEL_SPECS[args.model]
    docs_path = cache_dir / spec.docs_file
    queries_path = cache_dir / spec.queries_file
    docs_mmap = load_memmap(docs_path)
    queries_mmap = load_memmap(queries_path)
    partitions = deterministic_query_split(
        len(queries_mmap), args.validation_size, args.split_seed
    )
    query_indices = partitions["validation"][: args.smoke_queries]
    all_qrels = deterministic_self_qrels(
        len(docs_mmap), len(queries_mmap), args.qrel_seed
    )
    qrels = all_qrels[query_indices]

    n_subset = min(args.smoke_docs, len(docs_mmap))
    if n_subset < len(qrels) + (1 << args.smoke_pq_nbits):
        raise ValueError("smoke-docs is too small for qrels and PQ centroids")
    corpus_ids = deterministic_corpus_subset(
        len(docs_mmap), qrels, n_subset, args.smoke_seed
    )
    docs = np.asarray(docs_mmap[corpus_ids], dtype=np.float32)
    queries = np.asarray(queries_mmap[query_indices], dtype=np.float32)

    projection_dim = min(args.smoke_projection_dim, docs.shape[1], len(docs) - 1)
    if projection_dim % args.smoke_pq_m:
        raise ValueError("smoke projection dimension must be divisible by smoke PQ M")
    config = BenchmarkConfig(
        model=args.model + "-smoke",
        projection_dim=projection_dim,
        pq_m=args.smoke_pq_m,
        pq_nbits=args.smoke_pq_nbits,
        shortlist_k=min(args.shortlist_k, len(docs)),
        top_k=min(args.top_k, args.shortlist_k, len(docs)),
        center_online_queries=args.center_online_queries,
        exact_backend=args.exact_backend,
        doc_chunk_size=min(args.doc_chunk_size, len(docs)),
        bootstrap_samples=min(args.bootstrap_samples, 500),
        bootstrap_seed=args.bootstrap_seed,
        fit_sample_size=len(docs),
        fit_sample_seed=args.fit_sample_seed,
        svd_seed=args.svd_seed,
        pq_train_size=len(docs),
        pq_train_seed=args.pq_train_seed,
        faiss_threads=args.faiss_threads,
        network_mbps=args.network_mbps,
        network_rtt_ms=args.network_rtt_ms,
    )
    config.validate(docs.shape[1], len(docs))
    configure_faiss_threads(config.faiss_threads)
    basis = fit_projection_basis(
        docs,
        projection_dim,
        len(docs),
        config.fit_sample_seed,
        config.svd_seed,
    )
    pq_index = build_pq_index_chunked(
        docs,
        basis,
        config.pq_m,
        config.pq_nbits,
        len(docs),
        config.pq_train_seed,
        config.doc_chunk_size,
    )
    summary, records = run_protocol_benchmarks(
        docs,
        queries,
        qrels,
        query_indices,
        basis,
        pq_index,
        config,
        corpus_ids=corpus_ids,
        split_name="validation-smoke",
        source_is_memmap=True,
        pq_artifact={"source": "ephemeral smoke IndexPQ"},
    )
    summary["smoke"] = {
        "full_corpus_was_not_scanned": True,
        "sampled_docs": len(docs),
        "source_docs": len(docs_mmap),
        "source_documents": _file_metadata(docs_path),
        "source_queries": _file_metadata(queries_path),
    }
    return summary, records


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _metric_mean(summary: Mapping[str, Any], method: str, key: str) -> float:
    return float(summary["methods"][method]["metrics"][key]["mean"])


def print_compact_summary(summary: Mapping[str, Any]) -> None:
    top_k = int(summary["config"]["top_k"])
    shortlist_k = int(summary["config"]["shortlist_k"])
    print(
        f"n_docs={summary['config']['n_docs']:,} "
        f"n_queries={summary['config']['n_queries']} "
        f"backend={summary['config']['exact_backend_resolved']}",
        flush=True,
    )
    print(summary["protocol"]["online_query_projection"], flush=True)
    for method in (
        "exact_projected",
        "pq_only",
        "return_full_vectors",
        "projected_shortlist_rerank",
    ):
        hit1 = _metric_mean(summary, method, "hit_at_1")
        hitk = _metric_mean(summary, method, f"hit_at_{top_k}")
        mrr = _metric_mean(summary, method, f"mrr_at_{top_k}")
        line = f"{method:22s} Hit@1={hit1:.3f} Hit@{top_k}={hitk:.3f} MRR={mrr:.3f}"
        if method != "exact_projected":
            recall = _metric_mean(
                summary, method, f"shortlist_recall_at_{shortlist_k}"
            )
            line += f" shortlist-recall={recall:.3f}"
        print(line, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified exact/PQ/return-vector retrieval baselines"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="e5-base")
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--pq-m", type=int)
    parser.add_argument("--pq-nbits", type=int)
    parser.add_argument("--shortlist-k", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--split", choices=("validation", "test", "all"), default="validation")
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--qrel-seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--center-online-queries",
        action="store_true",
        help="Explicit legacy ablation; deployable default leaves queries uncentred",
    )
    parser.add_argument("--exact-backend", choices=("auto", "numpy", "cuda"), default="auto")
    parser.add_argument("--doc-chunk-size", type=int, default=16_384)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--fit-sample-size", type=int, default=200_000)
    parser.add_argument("--fit-sample-seed", type=int, default=0)
    parser.add_argument("--svd-seed", type=int, default=42)
    parser.add_argument("--basis-path", type=Path)
    parser.add_argument("--basis-out", type=Path)
    parser.add_argument("--pq-index-path", type=Path)
    parser.add_argument("--pq-index-out", type=Path)
    parser.add_argument("--allow-build-pq", action="store_true")
    parser.add_argument("--pq-train-size", type=int, default=200_000)
    parser.add_argument("--pq-train-seed", type=int, default=42)
    parser.add_argument("--faiss-threads", type=int, default=0)
    parser.add_argument("--network-mbps", type=float)
    parser.add_argument("--network-rtt-ms", type=float, default=0.0)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--query-log-jsonl", type=Path)

    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-docs", type=int, default=2_048)
    parser.add_argument("--smoke-queries", type=int, default=6)
    parser.add_argument("--smoke-projection-dim", type=int, default=64)
    parser.add_argument("--smoke-pq-m", type=int, default=8)
    parser.add_argument("--smoke-pq-nbits", type=int, default=4)
    parser.add_argument("--smoke-seed", type=int, default=731)
    return parser


def _apply_model_defaults(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.model]
    if args.projection_dim is None:
        args.projection_dim = spec.projection_dim
    if args.pq_m is None:
        args.pq_m = spec.pq_m
    if args.pq_nbits is None:
        args.pq_nbits = spec.pq_nbits


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_model_defaults(args)
    if args.smoke:
        print("Running bounded cache-backed smoke test (not the full benchmark)", flush=True)
        summary, records = run_cache_smoke(args)
    else:
        print("Running full cache-backed benchmark", flush=True)
        summary, records = run_cached_benchmark(args)
    print_compact_summary(summary)
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
        print(f"summary: {Path(args.summary_json).resolve()}", flush=True)
    if args.query_log_jsonl:
        write_jsonl(Path(args.query_log_jsonl), records)
        print(f"query log: {Path(args.query_log_jsonl).resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
