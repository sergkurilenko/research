"""Aggregate repeated unified-CKKS process runs without hiding run variance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA = "unified_ckks_restart_aggregate.v1"


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("values must be non-empty and finite")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row at {path}:{line_number}")
                rows.append(value)
    return rows


def aggregate(summary_paths: Sequence[Path]) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in summary_paths]
    if len(paths) < 2:
        raise ValueError("at least two restart summaries are required")
    summaries = [_read_json(path) for path in paths]
    reference = summaries[0]
    invariant_config_keys = (
        "method",
        "shortlist_k",
        "top_k",
        "split",
        "n_docs",
        "n_queries",
        "projected_dimension",
        "ckks",
    )
    expected = {key: reference["config"][key] for key in invariant_config_keys}
    expected_artifact = reference["artifacts"]["projected_documents"]["path"]
    all_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    pids: set[int] = set()
    for ordinal, (path, summary) in enumerate(zip(paths, summaries), 1):
        actual = {key: summary["config"][key] for key in invariant_config_keys}
        if actual != expected:
            raise ValueError(f"restart config mismatch: {path}")
        if summary["artifacts"]["projected_documents"]["path"] != expected_artifact:
            raise ValueError(f"projected corpus mismatch: {path}")
        attestation = summary["server_attestation"]
        if (
            not attestation["context_is_public"]
            or attestation["has_secret_key"]
            or attestation["has_decryptor_attribute"]
        ):
            raise ValueError(f"server attestation failed: {path}")
        pid = int(attestation["pid"])
        if pid in pids:
            raise ValueError("restart server PIDs are not unique")
        pids.add(pid)

        log_path = path.with_suffix(".jsonl")
        rows = _read_jsonl(log_path)
        if len(rows) != int(summary["config"]["n_queries"]):
            raise ValueError(f"JSONL count mismatch: {log_path}")
        all_rows.extend(rows)
        latency = summary["methods"]["ckks_projected_shortlist"]["latency_ms"]
        run_rows.append(
            {
                "restart": ordinal,
                "summary_path": str(path),
                "query_log_path": str(log_path),
                "server_pid": pid,
                "startup_ms": float(attestation["startup_ms"]),
                "online_end_to_end": {
                    key: float(latency["online_end_to_end"][key])
                    for key in ("mean", "p50", "p95", "p99")
                },
                "server_total": {
                    key: float(latency["server_total"][key])
                    for key in ("mean", "p50", "p95", "p99")
                },
            }
        )

    phase_names = sorted(all_rows[0]["latency_ms"])
    combined_latency = {
        phase: distribution([float(row["latency_ms"][phase]) for row in all_rows])
        for phase in phase_names
    }
    aggregate_restart_quantiles = {
        metric: distribution(
            [run["online_end_to_end"][metric] for run in run_rows]
        )
        for metric in ("mean", "p50", "p95", "p99")
    }
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "five independent spawned-server restarts over the same frozen "
            "validation queries; query duplicates are retained only for systems latency"
        ),
        "restart_count": len(run_rows),
        "request_count": len(all_rows),
        "config": expected,
        "projected_documents": expected_artifact,
        "runs": run_rows,
        "combined_request_latency_ms": combined_latency,
        "across_restart_online_summary_ms": aggregate_restart_quantiles,
        "payload_bytes": {
            "request": distribution(
                [float(row["payload_bytes"]["request"]) for row in all_rows]
            ),
            "response": distribution(
                [float(row["payload_bytes"]["response"]) for row in all_rows]
            ),
            "total": distribution(
                [float(row["payload_bytes"]["total"]) for row in all_rows]
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate(args.summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    online = result["combined_request_latency_ms"]["online_end_to_end"]
    print(json.dumps({key: online[key] for key in ("p50", "p95", "p99")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
