"""Capture the exact software and hardware state used by an experiment.

The output is intentionally JSON rather than prose so every result bundle can
embed or hash it.  Missing optional packages are reported instead of silently
omitted.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "faiss-cpu",
    "tenseal",
    "torch",
    "transformers",
    "datasets",
    "sentence-transformers",
    "fastapi",
    "uvicorn",
    "httpx",
    "psutil",
)


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = (completed.stdout or completed.stderr).strip()
    return value or None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def torch_hardware() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {"available": False}
    cuda_available = bool(torch.cuda.is_available())
    payload: dict[str, Any] = {
        "available": True,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
    }
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": int(props.total_memory),
                "multiprocessor_count": int(props.multi_processor_count),
            }
        )
    return payload


def git_state(repo: Path) -> dict[str, Any]:
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"], repo),
        "branch": command_output(["git", "branch", "--show-current"], repo),
        "status_porcelain": command_output(
            ["git", "status", "--porcelain=v1"], repo
        ),
    }


def capture(repo: Path) -> dict[str, Any]:
    selected_env = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "SHARD_DATA",
        )
    }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": package_versions(),
        "torch_hardware": torch_hardware(),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ]
        ),
        "git": git_state(repo),
        "environment": selected_env,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = capture(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
