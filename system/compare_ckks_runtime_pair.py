"""Compare two identical packed-CKKS microbenchmark JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


METHODS = (
    "naive_per_candidate",
    "block_packed",
    "segmented_score_packed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty distribution")
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing(values: Sequence[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "median": float(statistics.median(samples)),
        "p95": _percentile(samples, 95),
        "p99": _percentile(samples, 99),
        "min": min(samples),
        "max": max(samples),
    }


def _method_summary(method: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "server_ms": _timing(method["server_ms"]["samples"]),
        "client_decrypt_ms": _timing(method["client_decrypt_ms"]["samples"]),
        "max_abs_error": float(method["correctness"]["max_abs_error"]),
        "mean_abs_error": float(method["correctness"]["mean_abs_error"]),
        "rmse": float(method["correctness"]["rmse"]),
        "passes_reported_tolerance": bool(
            method["correctness"]["allclose_rtol_3e-4_atol_3e-5"]
        ),
        "wire_bytes": int(method["response"]["wire_bytes"]),
        "seal_ciphertext_count": int(
            method["response"]["seal_ciphertext_count"]
        ),
    }


def compare(
    windows_path: Path, wsl_path: Path, wheel_dir: Path | None = None
) -> dict[str, Any]:
    artifacts = {
        "windows": (windows_path, json.loads(windows_path.read_text(encoding="utf-8"))),
        "wsl_linux": (wsl_path, json.loads(wsl_path.read_text(encoding="utf-8"))),
    }
    windows = artifacts["windows"][1]
    wsl = artifacts["wsl_linux"][1]
    errors: list[str] = []
    if windows["schema"] != "packed_ckks_microbenchmark.v1":
        errors.append("unexpected Windows schema")
    if wsl["schema"] != "packed_ckks_microbenchmark.v1":
        errors.append("unexpected WSL schema")
    if windows["config"] != wsl["config"]:
        errors.append("benchmark configurations differ")
    if windows["software"]["tenseal"] != wsl["software"]["tenseal"]:
        errors.append("TenSEAL versions differ")

    runtime_summaries: dict[str, Any] = {}
    for runtime, (path, payload) in artifacts.items():
        methods = {
            method: _method_summary(payload["methods"][method])
            for method in METHODS
        }
        runtime_summaries[runtime] = {
            "artifact": {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            },
            "software": payload["software"],
            "setup": payload["setup"],
            "implementation": payload["implementation"],
            "methods": methods,
            "packed_speedup_naive_over_one_response_server_median": float(
                payload["comparison"][
                    "server_speedup_naive_over_single_response_median"
                ]
            ),
        }
        if payload["implementation"]["server_has_secret_key"]:
            errors.append(f"{runtime} server has a secret key")
        for method, summary in methods.items():
            if not summary["passes_reported_tolerance"]:
                errors.append(f"{runtime} {method} failed reported tolerance")

    method_ratios: dict[str, Any] = {}
    for method in METHODS:
        win = runtime_summaries["windows"]["methods"][method]
        lin = runtime_summaries["wsl_linux"]["methods"][method]
        if win["wire_bytes"] != lin["wire_bytes"]:
            errors.append(f"{method} wire sizes differ")
        method_ratios[method] = {
            "wsl_over_windows_server_median": lin["server_ms"]["median"]
            / win["server_ms"]["median"],
            "wsl_over_windows_server_p95": lin["server_ms"]["p95"]
            / win["server_ms"]["p95"],
            "wsl_over_windows_client_decrypt_median": lin["client_decrypt_ms"]
            ["median"]
            / win["client_decrypt_ms"]["median"],
        }

    wheels: list[dict[str, Any]] = []
    if wheel_dir is not None:
        for path in sorted(Path(wheel_dir).glob("*.whl")):
            wheels.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        if not wheels:
            errors.append("no archived WSL binary wheels found")

    return {
        "schema": "ckks_runtime_pair.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": not errors,
        "errors": errors,
        "scope": {
            "hardware": "same physical Intel Core i5-14400F host",
            "comparison": (
                "Windows native versus Ubuntu 22.04 under WSL2; same TenSEAL "
                "0.3.16, source, seed, CKKS parameters, d=672, K=100, and repeats"
            ),
            "not_independent_hardware_replication": True,
            "not_exact_python_numpy_parity": True,
            "important_runtime_difference": (
                "Windows Python 3.11/NumPy 2.4 and WSL Python 3.10/NumPy 2.2; "
                "SEAL serialization uses runtime-local temporary files, so filesystem "
                "and binding overhead differ substantially"
            ),
            "gpu_used": False,
        },
        "config": windows["config"],
        "runtimes": runtime_summaries,
        "wsl_binary_wheels": wheels,
        "ratios": method_ratios,
    }


def write_csv(path: Path, comparison: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "runtime",
        "method",
        "server_p50_ms",
        "server_p95_ms",
        "server_p99_ms",
        "client_decrypt_p50_ms",
        "client_decrypt_p95_ms",
        "max_abs_error",
        "wire_bytes",
        "seal_ciphertext_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for runtime, runtime_payload in comparison["runtimes"].items():
            for method, summary in runtime_payload["methods"].items():
                writer.writerow(
                    {
                        "runtime": runtime,
                        "method": method,
                        "server_p50_ms": summary["server_ms"]["median"],
                        "server_p95_ms": summary["server_ms"]["p95"],
                        "server_p99_ms": summary["server_ms"]["p99"],
                        "client_decrypt_p50_ms": summary["client_decrypt_ms"][
                            "median"
                        ],
                        "client_decrypt_p95_ms": summary["client_decrypt_ms"][
                            "p95"
                        ],
                        "max_abs_error": summary["max_abs_error"],
                        "wire_bytes": summary["wire_bytes"],
                        "seal_ciphertext_count": summary["seal_ciphertext_count"],
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--wsl", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare(args.windows, args.wsl, args.wheel_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.csv, result)
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "output": str(args.output.resolve()),
                "csv": str(args.csv.resolve()),
                "errors": result["errors"],
            }
        ),
        flush=True,
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
