"""Run CKKS LWE attacks one-by-one so partial results are immediately durable.

This is the fallback companion to the upstream ``LWE.estimate.rough`` batch.
It uses the same pinned parameters and ADPS16/GSA rough cost model, but starts
each estimator attack in a separate Sage process with its own timeout and raw
stdout/stderr.  It must not be used to hide a failed batch: the batch status
and transcript remain the primary record.
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

from system.ckks_security_transcript import (
    EXPECTED_ESTIMATOR_COMMIT,
    extract_parameters,
    git_commit,
    parse_rop_exponents,
    windows_to_wsl,
    wsl_runtime_metadata,
)


SCHEMA = "ckks_lwe_individual_attacks.v1"
MODELS = {
    "seal_exact_cbd21": "ND.CenteredBinomial(21, n=n)",
    "sigma_3_2_sensitivity": "ND.DiscreteGaussian(3.2, n=n)",
}
ATTACKS = {
    "usvp": "LWE.primal_usvp(params, red_cost_model=RC.ADPS16, red_shape_model='gsa')",
    "dual": "LWE.dual(params, red_cost_model=RC.ADPS16)",
    "dual_hybrid": "LWE.dual_hybrid(params, red_cost_model=RC.ADPS16)",
    "arora_gb": "LWE.arora_gb.cost_bounded(params)",
}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def program(parameters: dict[str, Any], model: str, attack: str) -> str:
    if model not in MODELS or attack not in ATTACKS:
        raise ValueError("unknown model or attack")
    if model != "seal_exact_cbd21" and attack == "arora_gb":
        raise ValueError("Arora-GB bounded-noise model does not apply to Gaussian Xe")
    return f'''# Auto-generated individually instrumented estimator input.
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
print("TRANSCRIPT_SCHEMA ckks_lwe_individual_attack.raw.v1", flush=True)
print("ESTIMATOR_COMMIT {EXPECTED_ESTIMATOR_COMMIT}", flush=True)
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
    stem = f"individual_{model}_{attack}"
    sage_path = output_dir / f"{stem}.sage"
    stdout_path = output_dir / f"{stem}_stdout.txt"
    stderr_path = output_dir / f"{stem}_stderr.txt"
    source = program(parameters, model, attack)
    sage_path.write_text(source, encoding="utf-8", newline="\n")
    command = [
        "wsl",
        "-d",
        distro,
        "-u",
        "root",
        "--cd",
        windows_to_wsl(estimator_dir),
        "--",
        "env",
        "PYTHONUNBUFFERED=1",
        f"PYTHONPATH={windows_to_wsl(estimator_dir)}",
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
    all_jobs = [
        (model, attack)
        for model in MODELS
        for attack in ATTACKS
        if not (model != "seal_exact_cbd21" and attack == "arora_gb")
    ]
    valid_points = {f"{model}:{attack}" for model, attack in all_jobs}
    if args.point:
        unknown_points = set(args.point) - valid_points
        if unknown_points:
            raise ValueError(f"unknown --point values: {sorted(unknown_points)}")
        jobs = [tuple(point.split(":", maxsplit=1)) for point in args.point]
    else:
        jobs = all_jobs
    record_path = output_dir / "individual_security_estimates.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("estimator_commit") != commit:
            raise RuntimeError("refusing to resume an incompatible individual record")
    else:
        record = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "relationship_to_primary": (
                "fallback instrumentation; retain the upstream rough batch status "
                "and do not select only completed/favourable attacks"
            ),
            "estimator_commit": commit,
            "parameters": parameters,
            "runtime": wsl_runtime_metadata(args.wsl_distro),
            "timeout_termination_semantics": (
                "kill the Windows WSL client process, then terminate only the "
                "dedicated CodexSageEstimatorUbuntu distro so the Linux Sage child "
                "cannot continue after a recorded timeout"
            ),
            "attacks": [],
            "superseded_attempts": [],
        }
    rerun_points = set(args.rerun_point or [])
    unknown = rerun_points - valid_points
    if unknown:
        raise ValueError(f"unknown --rerun-point values: {sorted(unknown)}")
    existing = {
        f"{value['model']}:{value['attack']}": value
        for value in record.get("attacks", [])
    }
    for model, attack in jobs:
        point = f"{model}:{attack}"
        if point in existing and point not in rerun_points:
            # Completed, failed, and timed-out points are all durable.  A caller
            # must opt in explicitly before replacing any recorded attempt.
            continue
        if point in existing:
            record.setdefault("superseded_attempts", []).append(existing[point])
            record["attacks"] = [
                value
                for value in record["attacks"]
                if f"{value['model']}:{value['attack']}" != point
            ]
        record["active_point"] = point
        atomic_write_json(record_path, record)
        result = run_one(
            estimator_dir=estimator_dir,
            output_dir=output_dir,
            distro=args.wsl_distro,
            model=model,
            attack=attack,
            parameters=parameters,
            timeout_seconds=args.timeout_per_attack_seconds,
        )
        record["attacks"].append(result)
        record.pop("active_point", None)
        # Make every completed/partial attack durable before the next starts.
        atomic_write_json(record_path, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wsl-distro", default="CodexSageEstimatorUbuntu")
    parser.add_argument("--timeout-per-attack-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--point",
        action="append",
        help="Run only this MODEL:ATTACK point; repeat to define an ordered subset",
    )
    parser.add_argument(
        "--rerun-point",
        action="append",
        help="Explicitly rerun MODEL:ATTACK; otherwise every recorded status is preserved",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            [
                {
                    "model": value["model"],
                    "attack": value["attack"],
                    "status": value["status"],
                    "elapsed_seconds": value["elapsed_seconds"],
                }
                for value in result["attacks"]
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
