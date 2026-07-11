"""Individually instrument the lattice-estimator extended/default-cost attacks.

Unlike the ADPS16/GSA ``rough`` evidence, these points use the estimator's
default MATZOV reduction-cost model and GSA shape model.  Each attack/model is
a separate Sage process with atomic, resume-aware checkpoints.  Arora-GB is
deliberately excluded: its exact-CBD run already has an authoritative separate
1800-second timeout record and must not be silently retried here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from system.ckks_security_individual import atomic_write_json
from system.ckks_security_transcript import (
    EXPECTED_ESTIMATOR_COMMIT,
    extract_parameters,
    git_commit,
    parse_rop_exponents,
    windows_to_wsl,
    wsl_runtime_metadata,
)


SCHEMA = "ckks_lwe_extended_individual.v1"
MODELS = {
    "seal_exact_cbd21": "ND.CenteredBinomial(21, n=n)",
    "sigma_3_2_sensitivity": "ND.DiscreteGaussian(3.2, n=n)",
}
ATTACKS = {
    "usvp": (
        "LWE.primal_usvp(params, red_cost_model=RC.MATZOV, "
        "red_shape_model=Simulator.GSA)"
    ),
    "dual": "LWE.dual(params, red_cost_model=RC.MATZOV)",
    "dual_hybrid": "LWE.dual_hybrid(params, red_cost_model=RC.MATZOV)",
    "bdd": (
        "LWE.primal_bdd(params, red_cost_model=RC.MATZOV, "
        "red_shape_model=Simulator.GSA)"
    ),
    "bdd_hybrid": (
        "LWE.primal_hybrid(params, mitm=False, babai=False, "
        "red_cost_model=RC.MATZOV, red_shape_model=Simulator.GSA)"
    ),
    "bdd_mitm_hybrid": (
        "LWE.primal_hybrid(params, mitm=True, babai=True, "
        "red_cost_model=RC.MATZOV, red_shape_model=Simulator.GSA)"
    ),
    "bkw": "LWE.coded_bkw(params)",
}
DEFAULT_ORDER = (
    "usvp",
    "dual",
    "dual_hybrid",
    "bdd",
    "bdd_hybrid",
    "bdd_mitm_hybrid",
    "bkw",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def program(parameters: dict[str, Any], model: str, attack: str) -> str:
    if model not in MODELS or attack not in ATTACKS:
        raise ValueError("unknown extended model or attack")
    return f'''# Auto-generated extended/default-cost estimator point.
# Estimator commit: {EXPECTED_ESTIMATOR_COMMIT}
from estimator import *
from sage.all import oo
import time

n = 8192
q = Integer({parameters["coeff_modulus_product_q"]})
params = LWE.Parameters(
    n=n,
    q=q,
    Xs=ND.Uniform(-1, 1, n=n),
    Xe={MODELS[model]},
    m=oo,
    tag="SEAL CKKS N8192 q[60,40,60] {model}",
)
print("TRANSCRIPT_SCHEMA ckks_lwe_extended_individual.raw.v1", flush=True)
print("ESTIMATOR_COMMIT {EXPECTED_ESTIMATOR_COMMIT}", flush=True)
print("COST_MODEL MATZOV", flush=True)
print("SHAPE_MODEL GSA", flush=True)
print("MODEL_BEGIN {model}", flush=True)
print("PARAMETERS", params, flush=True)
print("ATTACK_BEGIN {attack}", flush=True)
started = time.monotonic()
try:
    result = {ATTACKS[attack]}
    print("ATTACK {attack} ::", result, flush=True)
    print("ATTACK_STATUS completed", flush=True)
except Exception as error:
    print("ATTACK_STATUS failed", type(error).__name__, repr(error), flush=True)
    raise
finally:
    print("ATTACK_ELAPSED_SECONDS", time.monotonic() - started, flush=True)
print("MODEL_END {model}", flush=True)
'''


def run_one(
    *,
    estimator_dir: Path,
    output_dir: Path,
    distro: str,
    model: str,
    attack: str,
    parameters: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    stem = f"extended_{model}_{attack}"
    sage_path = output_dir / f"{stem}.sage"
    stdout_path = output_dir / f"{stem}_stdout.txt"
    stderr_path = output_dir / f"{stem}_stderr.txt"
    source = program(parameters, model, attack)
    sage_path.write_text(source, encoding="utf-8", newline="\n")
    estimator_wsl = windows_to_wsl(estimator_dir)
    command = [
        "wsl",
        "-d",
        distro,
        "-u",
        "root",
        "--cd",
        estimator_wsl,
        "--",
        "env",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={estimator_wsl}",
        "sage",
        windows_to_wsl(sage_path),
    ]
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_stream, (
        stderr_path.open("w", encoding="utf-8", newline="\n")
    ) as stderr_stream:
        process = subprocess.Popen(
            command,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "WSL_UTF8": "1"},
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
            subprocess.run(
                ["wsl", "--terminate", distro], capture_output=True, env=os.environ
            )
            return_code = 124
    elapsed = time.perf_counter() - started
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    return {
        "model": model,
        "attack": attack,
        "cost_model": "MATZOV",
        "shape_model": "GSA",
        "status": (
            "timed_out_partial_output_archived"
            if timed_out
            else ("completed" if return_code == 0 else "failed")
        ),
        "exit_code": return_code,
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "command": command,
        "sage_input": sage_path.name,
        "sage_input_sha256": sha256(source.encode("utf-8")),
        "stdout": stdout_path.name,
        "stdout_sha256": sha256(stdout.encode("utf-8")),
        "stderr": stderr_path.name,
        "stderr_sha256": sha256(stderr.encode("utf-8")),
        "parsed_costs": parse_rop_exponents(stdout),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    estimator_dir = args.estimator_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit(estimator_dir)
    if commit != EXPECTED_ESTIMATOR_COMMIT:
        raise RuntimeError(f"unexpected estimator commit: {commit}")
    parameters = extract_parameters()
    all_points = [
        (model, attack) for attack in DEFAULT_ORDER for model in MODELS
    ]
    valid_points = {f"{model}:{attack}" for model, attack in all_points}
    if args.point:
        unknown = set(args.point) - valid_points
        if unknown:
            raise ValueError(f"unknown --point values: {sorted(unknown)}")
        points = [tuple(value.split(":", maxsplit=1)) for value in args.point]
    else:
        points = all_points

    record_path = output_dir / "extended_security_estimates.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("estimator_commit") != commit:
            raise RuntimeError("refusing to resume an incompatible extended record")
    else:
        record = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "estimator_commit": commit,
            "cost_model": "MATZOV (estimator default)",
            "shape_model": "GSA (estimator default)",
            "distinction_from_rough": (
                "Do not compare these exponents as if they used the rough "
                "ADPS16/GSA cost model."
            ),
            "arora_gb_policy": (
                "excluded; authoritative exact-CBD individual timeout is stored "
                "in individual_security_estimates.json"
            ),
            "parameters": parameters,
            "runtime": wsl_runtime_metadata(args.wsl_distro),
            "overall_budget_seconds": args.overall_timeout_seconds,
            "timeout_termination_semantics": (
                "kill the Windows WSL client then terminate only the dedicated "
                "CodexSageEstimatorUbuntu distro"
            ),
            "attacks": [],
        }
    existing = {
        f"{value['model']}:{value['attack']}": value
        for value in record.get("attacks", [])
    }
    session_started = time.perf_counter()
    for model, attack in points:
        point = f"{model}:{attack}"
        if point in existing:
            continue
        remaining = args.overall_timeout_seconds - (time.perf_counter() - session_started)
        if remaining <= 0:
            record["overall_budget_exhausted_before_point"] = point
            atomic_write_json(record_path, record)
            break
        point_timeout = min(args.timeout_per_attack_seconds, remaining)
        record["active_point"] = point
        atomic_write_json(record_path, record)
        result = run_one(
            estimator_dir=estimator_dir,
            output_dir=output_dir,
            distro=args.wsl_distro,
            model=model,
            attack=attack,
            parameters=parameters,
            timeout_seconds=point_timeout,
        )
        record["attacks"].append(result)
        existing[point] = result
        record.pop("active_point", None)
        record["last_session_elapsed_seconds"] = time.perf_counter() - session_started
        atomic_write_json(record_path, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wsl-distro", default="CodexSageEstimatorUbuntu")
    parser.add_argument("--timeout-per-attack-seconds", type=float, default=900.0)
    parser.add_argument("--overall-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--point", action="append")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    record = run(build_parser().parse_args(argv))
    print(
        json.dumps(
            [
                {
                    "model": value["model"],
                    "attack": value["attack"],
                    "status": value["status"],
                    "elapsed_seconds": value["elapsed_seconds"],
                    "parsed_costs": value["parsed_costs"],
                }
                for value in record["attacks"]
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
