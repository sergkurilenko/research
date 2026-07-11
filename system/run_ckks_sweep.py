"""Reproducible, resumable sweep runner for the packed CKKS microbenchmark.

The runner executes two intersecting sweeps and writes one atomic JSON
checkpoint after every unique point:

* dimension sweep: d in {192, 384, 672, 768}, K=100;
* shortlist sweep: K in {20, 40, 64, 80, 100, 200}, d=672.

The shared d=672/K=100 point is stored once and tagged as belonging to both
sweeps.  By default it is imported from the existing 20-repeat benchmark in
``results/system_revision/ckks_micro_d672_K100.json``; its SHA-256 digest and
actual measurement settings are recorded in provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .packed_ckks import CKKSParameters, run_microbenchmark
except ImportError:  # Direct execution: python system/run_ckks_sweep.py
    from packed_ckks import CKKSParameters, run_microbenchmark


SWEEP_SCHEMA = "packed_ckks_sweep.v1"
DIMENSION_VALUES = (192, 384, 672, 768)
CANDIDATE_VALUES = (20, 40, 64, 80, 100, 200)
DIMENSION_SWEEP_CANDIDATES = 100
CANDIDATE_SWEEP_DIMENSION = 672
DEFAULT_SEED = 20260710

_WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = _WORKSPACE / "results" / "system_revision" / "ckks_sweep.json"
DEFAULT_REUSE_POINT = (
    _WORKSPACE / "results" / "system_revision" / "ckks_micro_d672_K100.json"
)


@dataclass(frozen=True)
class SweepPoint:
    dimension: int
    candidate_count: int
    sweep_membership: tuple[str, ...]

    @property
    def point_id(self) -> str:
        return f"d{self.dimension}_K{self.candidate_count}"

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.point_id,
            "dimension": self.dimension,
            "candidate_count": self.candidate_count,
            "sweep_membership": list(self.sweep_membership),
        }


BenchmarkCallable = Callable[..., dict[str, Any]]


def build_point_plan() -> tuple[SweepPoint, ...]:
    """Build the stable nine-point plan, merging the intersecting point."""

    ordered_keys: list[tuple[int, int]] = []
    membership: dict[tuple[int, int], list[str]] = {}

    def add(dimension: int, candidates: int, sweep: str) -> None:
        key = (dimension, candidates)
        if key not in membership:
            ordered_keys.append(key)
            membership[key] = []
        membership[key].append(sweep)

    for dimension in DIMENSION_VALUES:
        add(dimension, DIMENSION_SWEEP_CANDIDATES, "dimension")
    for candidates in CANDIDATE_VALUES:
        add(CANDIDATE_SWEEP_DIMENSION, candidates, "candidate_count")

    return tuple(
        SweepPoint(dimension, candidates, tuple(membership[(dimension, candidates)]))
        for dimension, candidates in ordered_keys
    )


def run_sweep(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    repeats: int = 7,
    warmups: int = 2,
    seed: int = DEFAULT_SEED,
    reuse_point_path: Path | None = DEFAULT_REUSE_POINT,
    force: bool = False,
    max_new_points: int | None = None,
    benchmark: BenchmarkCallable = run_microbenchmark,
) -> dict[str, Any]:
    """Run or resume the sweep, checkpointing atomically after every point.

    ``max_new_points`` is mainly useful for checkpoint/resume tests and manual
    staged runs.  A stopped staged run has status ``partial`` and can be resumed
    with the same configuration.
    """

    output_path = Path(output_path).resolve()
    reuse_point_path = (
        None if reuse_point_path is None else Path(reuse_point_path).resolve()
    )
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if max_new_points is not None and max_new_points <= 0:
        raise ValueError("max_new_points must be positive when provided")

    plan = build_point_plan()
    expected_config = _make_sweep_config(
        plan=plan,
        repeats=repeats,
        warmups=warmups,
        seed=seed,
        reuse_point_path=reuse_point_path,
    )
    if output_path.exists() and not force:
        state = _load_checkpoint(output_path)
        _validate_resume_state(state, expected_config, plan)
    else:
        state = _new_state(expected_config, plan)
        _checkpoint(output_path, state)

    completed = {str(point["id"]): point for point in state["points"]}
    new_points = 0
    state["status"] = "running"
    state.pop("last_error", None)
    _update_progress(state, plan)
    _checkpoint(output_path, state)

    try:
        for specification in plan:
            if specification.point_id in completed:
                continue
            if max_new_points is not None and new_points >= max_new_points:
                state["status"] = "partial"
                break

            print(
                f"[ckks-sweep] starting {specification.point_id}",
                file=sys.stderr,
                flush=True,
            )
            result, provenance = _obtain_result(
                specification,
                repeats=repeats,
                warmups=warmups,
                seed=seed,
                reuse_point_path=reuse_point_path,
                benchmark=benchmark,
            )
            _validate_microbenchmark_result(result, specification)
            _adopt_or_validate_environment(state, result)
            point = _make_point_record(specification, result, provenance)
            state["points"].append(point)
            completed[specification.point_id] = point
            new_points += 1
            _update_progress(state, plan)
            state["status"] = (
                "complete"
                if state["progress"]["remaining_points"] == 0
                else "running"
            )
            _checkpoint(output_path, state)
            headline = point["headline"]
            print(
                "[ckks-sweep] completed "
                f"{specification.point_id}: packed median="
                f"{headline['single_response_server_median_ms']:.3f} ms, "
                f"p95={headline['single_response_server_p95_ms']:.3f} ms, "
                f"speedup={headline['server_speedup_naive_over_single_response']:.3f}x",
                file=sys.stderr,
                flush=True,
            )
        else:
            state["status"] = "complete"

        _update_progress(state, plan)
        if state["progress"]["remaining_points"] == 0:
            state["status"] = "complete"
        elif state["status"] == "running":
            state["status"] = "partial"
        _checkpoint(output_path, state)
        return state
    except BaseException as exc:
        state["status"] = "interrupted"
        state["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
            "recorded_utc": _utc_now(),
        }
        _update_progress(state, plan)
        _checkpoint(output_path, state)
        raise


def _make_sweep_config(
    *,
    plan: Sequence[SweepPoint],
    repeats: int,
    warmups: int,
    seed: int,
    reuse_point_path: Path | None,
) -> dict[str, Any]:
    return {
        "dimension_sweep": {
            "dimensions": list(DIMENSION_VALUES),
            "candidate_count": DIMENSION_SWEEP_CANDIDATES,
        },
        "candidate_count_sweep": {
            "dimension": CANDIDATE_SWEEP_DIMENSION,
            "candidate_counts": list(CANDIDATE_VALUES),
        },
        "unique_point_count": len(plan),
        "point_order": [point.point_id for point in plan],
        "requested_repeats": repeats,
        "requested_warmups": warmups,
        "seed": seed,
        "ckks": CKKSParameters().as_json(),
        "reused_point_policy": {
            "point_id": "d672_K100",
            "source_path": None if reuse_point_path is None else str(reuse_point_path),
            "use_when_present": reuse_point_path is not None,
        },
        "checkpoint_policy": "atomic rewrite after every completed point",
    }


def _new_state(config: Mapping[str, Any], plan: Sequence[SweepPoint]) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema": SWEEP_SCHEMA,
        "created_utc": now,
        "updated_utc": now,
        "status": "initialized",
        "environment": None,
        "config": dict(config),
        "plan": [point.as_json() for point in plan],
        "progress": {
            "completed_points": 0,
            "total_points": len(plan),
            "remaining_points": len(plan),
        },
        "points": [],
    }


def _obtain_result(
    specification: SweepPoint,
    *,
    repeats: int,
    warmups: int,
    seed: int,
    reuse_point_path: Path | None,
    benchmark: BenchmarkCallable,
) -> tuple[dict[str, Any], dict[str, Any]]:
    can_reuse = (
        specification.dimension == 672
        and specification.candidate_count == 100
        and reuse_point_path is not None
        and reuse_point_path.is_file()
    )
    if can_reuse:
        raw = reuse_point_path.read_bytes()
        result = json.loads(raw.decode("utf-8"))
        provenance = {
            "kind": "reused_existing_microbenchmark",
            "source_path": str(reuse_point_path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_created_utc": result.get("created_utc"),
            "requested_sweep_repeats": repeats,
            "requested_sweep_warmups": warmups,
            "actual_repeats": result.get("config", {}).get("repeats"),
            "actual_warmups": result.get("config", {}).get("warmups"),
            "reuse_reason": (
                "same d=672,K=100 point already measured with 20 repeats; "
                "higher-repeat source retained"
            ),
        }
        return result, provenance

    result = benchmark(
        dimension=specification.dimension,
        candidate_count=specification.candidate_count,
        repeats=repeats,
        warmups=warmups,
        seed=seed,
    )
    provenance = {
        "kind": "measured_by_sweep_runner",
        "source_path": None,
        "source_sha256": None,
        "actual_repeats": result.get("config", {}).get("repeats"),
        "actual_warmups": result.get("config", {}).get("warmups"),
        "seed": seed,
    }
    return result, provenance


def _validate_microbenchmark_result(
    result: Mapping[str, Any], specification: SweepPoint
) -> None:
    if result.get("schema") != "packed_ckks_microbenchmark.v1":
        raise ValueError(
            f"{specification.point_id}: unsupported microbenchmark schema "
            f"{result.get('schema')!r}"
        )
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{specification.point_id}: missing benchmark config")
    if int(config.get("dimension", -1)) != specification.dimension:
        raise ValueError(f"{specification.point_id}: source dimension mismatch")
    if int(config.get("candidate_count", -1)) != specification.candidate_count:
        raise ValueError(f"{specification.point_id}: source candidate-count mismatch")
    methods = result.get("methods")
    required_methods = {
        "naive_per_candidate",
        "block_packed",
        "segmented_score_packed",
    }
    if not isinstance(methods, Mapping) or not required_methods.issubset(methods):
        raise ValueError(f"{specification.point_id}: incomplete benchmark methods")
    for method in required_methods:
        details = methods[method]
        for timing_name in ("server_ms", "client_decrypt_ms"):
            timing = details.get(timing_name, {})
            if "median" not in timing or "p95" not in timing:
                raise ValueError(
                    f"{specification.point_id}/{method}: missing {timing_name} median/p95"
                )
        phases = details.get("server_phase_ms", {})
        for phase in (
            "request_deserialize",
            "plaintext_encode",
            "he_evaluate",
            "score_pack",
            "ciphertext_serialize",
        ):
            if "median" not in phases.get(phase, {}) or "p95" not in phases.get(
                phase, {}
            ):
                raise ValueError(
                    f"{specification.point_id}/{method}: missing phase {phase}"
                )
        response = details.get("response", {})
        for field_name in (
            "wire_bytes",
            "ckks_payload_bytes",
            "payload_count",
            "seal_ciphertext_count",
        ):
            if field_name not in response:
                raise ValueError(
                    f"{specification.point_id}/{method}: missing response {field_name}"
                )
        correctness = details.get("correctness", {})
        if "max_abs_error" not in correctness or "rmse" not in correctness:
            raise ValueError(f"{specification.point_id}/{method}: missing error metrics")


def _adopt_or_validate_environment(
    state: dict[str, Any], result: Mapping[str, Any]
) -> None:
    candidate = {
        "software": result.get("software"),
        "implementation": result.get("implementation"),
    }
    if state.get("environment") is None:
        state["environment"] = candidate
        return
    current = state["environment"]
    for section, keys in (
        ("software", ("python", "tenseal", "numpy", "he_backend")),
        (
            "implementation",
            ("algorithm", "server_has_secret_key", "decryption_location"),
        ),
    ):
        for key in keys:
            if current.get(section, {}).get(key) != candidate.get(section, {}).get(key):
                raise ValueError(
                    f"environment mismatch for {section}.{key}: "
                    f"{current.get(section, {}).get(key)!r} != "
                    f"{candidate.get(section, {}).get(key)!r}"
                )


def _make_point_record(
    specification: SweepPoint,
    result: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    single = result["methods"]["segmented_score_packed"]
    comparison = result["comparison"]
    return {
        **specification.as_json(),
        "status": "complete",
        "completed_utc": _utc_now(),
        "provenance": dict(provenance),
        "measurement_schema": result["schema"],
        "measurement_created_utc": result.get("created_utc"),
        "measurement_config": result["config"],
        "setup": result["setup"],
        # Raw samples, median/p95 summaries, all server phases, response sizes,
        # ciphertext counts, and correctness metrics are retained verbatim.
        "methods": result["methods"],
        "comparison": comparison,
        "headline": {
            "single_response_server_median_ms": single["server_ms"]["median"],
            "single_response_server_p95_ms": single["server_ms"]["p95"],
            "single_response_client_median_ms": single["client_decrypt_ms"][
                "median"
            ],
            "single_response_wire_bytes": single["response"]["wire_bytes"],
            "single_response_ciphertext_count": single["response"][
                "seal_ciphertext_count"
            ],
            "single_response_max_abs_error": single["correctness"][
                "max_abs_error"
            ],
            "server_speedup_naive_over_single_response": comparison[
                "server_speedup_naive_over_single_response_median"
            ],
            "wire_reduction_naive_over_single_response": comparison[
                "wire_size_reduction_naive_over_single_response_ratio"
            ],
        },
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load sweep checkpoint {path}") from exc
    if state.get("schema") != SWEEP_SCHEMA:
        raise ValueError(f"unsupported sweep checkpoint schema: {state.get('schema')!r}")
    if not isinstance(state.get("points"), list):
        raise ValueError("sweep checkpoint has no point list")
    return state


def _validate_resume_state(
    state: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    plan: Sequence[SweepPoint],
) -> None:
    current_config = state.get("config")
    if not isinstance(current_config, Mapping):
        raise ValueError("sweep checkpoint has no config")
    # Source paths can move while their content provenance remains embedded;
    # all measurement-affecting settings must match exactly.
    for key in (
        "dimension_sweep",
        "candidate_count_sweep",
        "unique_point_count",
        "point_order",
        "requested_repeats",
        "requested_warmups",
        "seed",
        "ckks",
    ):
        if current_config.get(key) != expected_config.get(key):
            raise ValueError(
                f"cannot resume: sweep config field {key!r} differs; use --force"
            )
    known_ids = {point.point_id for point in plan}
    seen: set[str] = set()
    for point in state["points"]:
        point_id = str(point.get("id"))
        if point_id not in known_ids:
            raise ValueError(f"checkpoint contains unknown point {point_id!r}")
        if point_id in seen:
            raise ValueError(f"checkpoint contains duplicate point {point_id!r}")
        if point.get("status") != "complete":
            raise ValueError(f"checkpoint point {point_id!r} is not complete")
        seen.add(point_id)


def _update_progress(state: dict[str, Any], plan: Sequence[SweepPoint]) -> None:
    completed = len(state["points"])
    state["progress"] = {
        "completed_points": completed,
        "total_points": len(plan),
        "remaining_points": len(plan) - completed,
    }


def _checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = _utc_now()
    rendered = json.dumps(
        state,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run/resume the packed CKKS dimension and shortlist sweeps."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reuse-point", type=Path, default=DEFAULT_REUSE_POINT)
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="measure d=672,K=100 again instead of importing the 20-repeat point",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard an existing checkpoint and start a new sweep",
    )
    parser.add_argument(
        "--max-new-points",
        type=int,
        default=None,
        help="optional staged-run limit; resume later without this option",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state = run_sweep(
        output_path=args.output,
        repeats=args.repeats,
        warmups=args.warmups,
        seed=args.seed,
        reuse_point_path=None if args.no_reuse_existing else args.reuse_point,
        force=args.force,
        max_new_points=args.max_new_points,
    )
    summary = {
        "schema": state["schema"],
        "status": state["status"],
        "output": str(Path(args.output).resolve()),
        "progress": state["progress"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
