"""Build a read-only-friendly projected provider matrix in bounded chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def load_basis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        mean_key = "passage_mean" if "passage_mean" in data else "mu"
        vectors_key = "vectors" if "vectors" in data else "Vk"
        mean = np.asarray(data[mean_key], dtype=np.float32)
        vectors = np.asarray(data[vectors_key], dtype=np.float32)
    if mean.ndim == 1:
        mean = mean[None, :]
    if mean.shape != (1, vectors.shape[0]):
        raise ValueError("Basis mean/vectors dimensions do not match")
    return mean, vectors


def project_numpy(
    source: np.ndarray,
    destination: np.memmap,
    mean: np.ndarray,
    vectors: np.ndarray,
    chunk_size: int,
) -> list[float]:
    timings: list[float] = []
    for start in range(0, len(source), chunk_size):
        end = min(start + chunk_size, len(source))
        chunk = np.array(source[start:end], dtype=np.float32, copy=True)
        started = time.perf_counter()
        destination[start:end] = (chunk - mean) @ vectors
        destination.flush()
        timings.append((time.perf_counter() - started) * 1_000.0)
        print(f"{end:,}/{len(source):,}", flush=True)
    return timings


def project_cuda(
    source: np.ndarray,
    destination: np.memmap,
    mean: np.ndarray,
    vectors: np.ndarray,
    chunk_size: int,
) -> list[float]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but no CUDA device is available")
    device = torch.device("cuda")
    mean_device = torch.from_numpy(np.ascontiguousarray(mean)).to(device)
    vectors_device = torch.from_numpy(np.ascontiguousarray(vectors)).to(device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    timings: list[float] = []
    try:
        for start in range(0, len(source), chunk_size):
            end = min(start + chunk_size, len(source))
            host = np.array(source[start:end], dtype=np.float32, copy=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            chunk = torch.from_numpy(host).to(device)
            projected = (chunk - mean_device) @ vectors_device
            destination[start:end] = projected.cpu().numpy()
            destination.flush()
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1_000.0)
            print(f"{end:,}/{len(source):,}", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="cuda")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --force")
    source = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.dtype != np.float32:
        raise ValueError("Input must be a two-dimensional float32 NPY array")
    mean, vectors = load_basis(args.basis)
    if source.shape[1] != vectors.shape[0]:
        raise ValueError("Input dimension does not match basis")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.float32,
        shape=(len(source), vectors.shape[1]),
    )
    started = time.perf_counter()
    if args.backend == "cuda":
        timings = project_cuda(
            source, destination, mean, vectors, args.chunk_size
        )
    else:
        timings = project_numpy(
            source, destination, mean, vectors, args.chunk_size
        )
    elapsed = time.perf_counter() - started
    del destination
    output = np.load(args.output, mmap_mode="r", allow_pickle=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(args.input.resolve()),
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "size_bytes": args.input.stat().st_size,
        },
        "basis": {
            "path": str(args.basis.resolve()),
            "vectors_shape": list(vectors.shape),
            "mean_sha256": sha256_array(mean),
            "vectors_sha256": sha256_array(vectors),
        },
        "output": {
            "path": str(args.output.resolve()),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "size_bytes": args.output.stat().st_size,
        },
        "backend": args.backend,
        "chunk_size": args.chunk_size,
        "elapsed_seconds": elapsed,
        "chunk_latency_ms": {
            "mean": float(np.mean(timings)),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
            "count": len(timings),
        },
        "platform": platform.platform(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["output"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
