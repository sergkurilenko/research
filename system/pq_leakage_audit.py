"""Quantify what the public PQ artefact reveals about provider vectors.

This audit deliberately treats product quantisation as a lossy disclosure,
not as a privacy mechanism.  It reconstructs approximate projected vectors
from the public FAISS index, lifts them through the public SVD basis, and
compares them with the provider's exact projected and original embeddings.
Large arrays are memory mapped and only a deterministic sample is gathered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA = "public_pq_leakage_audit.v1"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("distribution input must be non-empty and finite")
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity for corresponding rows, with a stable zero rule."""

    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("inputs must be two matrices with the same shape")
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def sample_ids(n_docs: int, sample_size: int, seed: int) -> np.ndarray:
    if n_docs <= 0 or sample_size <= 0:
        raise ValueError("n_docs and sample_size must be positive")
    rng = np.random.default_rng(seed)
    count = min(n_docs, sample_size)
    return np.sort(rng.choice(n_docs, size=count, replace=False).astype(np.int64))


def compute_vector_disclosure(
    raw: np.ndarray,
    projected: np.ndarray,
    reconstructed_projected: np.ndarray,
    passage_mean: np.ndarray,
    basis_vectors: np.ndarray,
) -> dict[str, Any]:
    """Compute projected- and original-space reconstruction diagnostics."""

    raw = np.asarray(raw, dtype=np.float32)
    projected = np.asarray(projected, dtype=np.float32)
    reconstructed_projected = np.asarray(reconstructed_projected, dtype=np.float32)
    mean = np.asarray(passage_mean, dtype=np.float32).reshape(1, -1)
    basis = np.asarray(basis_vectors, dtype=np.float32)
    if projected.shape != reconstructed_projected.shape:
        raise ValueError("projected matrices must have the same shape")
    if raw.shape[0] != projected.shape[0]:
        raise ValueError("raw and projected samples must have the same row count")
    if basis.shape != (raw.shape[1], projected.shape[1]):
        raise ValueError("basis shape is inconsistent with sampled vectors")

    lifted = reconstructed_projected @ basis.T + mean
    projected_error = reconstructed_projected - projected
    raw_error = lifted - raw
    projected_relative_l2 = np.linalg.norm(projected_error, axis=1) / np.maximum(
        np.linalg.norm(projected, axis=1), 1e-12
    )
    raw_relative_l2 = np.linalg.norm(raw_error, axis=1) / np.maximum(
        np.linalg.norm(raw, axis=1), 1e-12
    )
    return {
        "projected_space": {
            "cosine": _distribution(row_cosine(projected, reconstructed_projected)),
            "relative_l2_error": _distribution(projected_relative_l2),
            "coordinate_rmse": float(np.sqrt(np.mean(projected_error**2))),
        },
        "lifted_original_space": {
            "cosine": _distribution(row_cosine(raw, lifted)),
            "relative_l2_error": _distribution(raw_relative_l2),
            "coordinate_rmse": float(np.sqrt(np.mean(raw_error**2))),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FAISS is required for the PQ audit") from exc

    raw = np.load(args.raw_docs, mmap_mode="r", allow_pickle=False)
    projected = np.load(args.projected_docs, mmap_mode="r", allow_pickle=False)
    basis_archive = np.load(args.basis, allow_pickle=False)
    passage_mean = basis_archive["passage_mean"]
    basis_vectors = basis_archive["vectors"]
    index = faiss.read_index(str(args.pq_index))
    if raw.ndim != 2 or projected.ndim != 2:
        raise ValueError("embedding arrays must be matrices")
    if raw.shape[0] != projected.shape[0] or index.ntotal != raw.shape[0]:
        raise ValueError("raw, projected, and PQ artefacts disagree on n_docs")
    if index.d != projected.shape[1]:
        raise ValueError("PQ and projected dimensions differ")

    ids = sample_ids(raw.shape[0], args.sample_size, args.seed)
    reconstructed = index.reconstruct_batch(ids)
    metrics = compute_vector_disclosure(
        np.asarray(raw[ids], dtype=np.float32),
        np.asarray(projected[ids], dtype=np.float32),
        reconstructed,
        passage_mean,
        basis_vectors,
    )
    raw_bytes = int(raw.shape[0] * raw.shape[1] * raw.dtype.itemsize)
    projected_bytes = int(
        projected.shape[0] * projected.shape[1] * projected.dtype.itemsize
    )
    pq_bytes = args.pq_index.stat().st_size
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": {
            "claim": (
                "The public PQ artefact is a lossy view of provider geometry; "
                "it is not cryptographic document privacy."
            ),
            "non_claim": (
                "No resistance to reconstruction, inversion, membership "
                "inference, or adaptive score-oracle extraction is asserted."
            ),
        },
        "sample": {
            "n_docs": int(raw.shape[0]),
            "sample_size": int(ids.size),
            "seed": int(args.seed),
            "id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        },
        "artefacts": {
            "raw_docs": {
                "path": str(args.raw_docs.resolve()),
                "shape": list(raw.shape),
                "logical_float32_bytes": raw_bytes,
            },
            "projected_docs": {
                "path": str(args.projected_docs.resolve()),
                "shape": list(projected.shape),
                "logical_float32_bytes": projected_bytes,
            },
            "public_pq_index": {
                "path": str(args.pq_index.resolve()),
                "dimension": int(index.d),
                "ntotal": int(index.ntotal),
                "serialized_bytes": int(pq_bytes),
                "sha256": _sha256(args.pq_index),
                "compression_vs_raw_float32": float(raw_bytes / pq_bytes),
                "compression_vs_projected_float32": float(projected_bytes / pq_bytes),
            },
            "basis": {
                "path": str(args.basis.resolve()),
                "shape": list(basis_vectors.shape),
                "public_in_protocol": True,
            },
        },
        "reconstruction": metrics,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "platform": platform.platform(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-docs", type=Path, required=True)
    parser.add_argument("--projected-docs", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--pq-index", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
