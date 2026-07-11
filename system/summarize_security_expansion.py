"""Build a concise, machine-readable summary of the security expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "security_expansion_summary.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attack_table(record: Mapping[str, Any]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for value in record["attacks"]:
        model = value["model"]
        parsed = value.get("parsed_costs", {}).get(model, {})
        models.setdefault(model, {})[value["attack"]] = {
            "status": value["status"],
            "rop_log2": parsed.get("minimum_rop_log2"),
            "elapsed_seconds": value["elapsed_seconds"],
            "stdout_sha256": value["stdout_sha256"],
            "stderr_sha256": value["stderr_sha256"],
        }
    for attacks in models.values():
        completed = [
            value["rop_log2"]
            for value in attacks.values()
            if value["status"] == "completed" and value["rop_log2"] is not None
        ]
        attacks["minimum_completed_rop_log2"] = min(completed) if completed else None
        attacks["all_requested_attacks_completed"] = all(
            value["status"] == "completed"
            for name, value in attacks.items()
            if isinstance(value, dict) and name not in {"minimum_completed_rop_log2"}
        )
    return models


def summarize(
    *,
    parameters_record: Mapping[str, Any],
    rough_record: Mapping[str, Any],
    extended_record: Mapping[str, Any],
    circuit_record: Mapping[str, Any],
    million_record: Mapping[str, Any],
    beir_record: Mapping[str, Any],
) -> dict[str, Any]:
    params = parameters_record["parameters"]
    million = million_record["collections"]["million_vector_heldout"]
    million_k100 = {
        method: {
            "direction_cosine_mean": million["methods"][f"K100:{method}"][
                "direction_cosine"
            ]["mean"],
            "exact_top10_overlap_mean": million["methods"][f"K100:{method}"][
                "exact_top10_overlap_fraction"
            ]["mean"],
            "linkability_auc": million["methods"][f"K100:{method}"][
                "linkability"
            ]["roc_auc"],
        }
        for method in ("centroid", "log_rank", "ridge_rank_ls")
    }
    beir_points = [
        collection["methods"]["K20:log_rank"]
        for collection in beir_record["collections"].values()
    ]
    fixed = circuit_record["identical_encrypted_request_replay"]
    fresh = circuit_record["fresh_client_encryption"]
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ckks_parameters": {
            "poly_modulus_degree": params["poly_modulus_degree"],
            "coeff_mod_bit_sizes": params["coeff_mod_bit_sizes_observed"],
            "coeff_modulus_primes": params["coeff_modulus_primes"],
            "log2_q": params["log2_q"],
            "scale_bits": params["scale_bits"],
            "seal_tc128_max_coeff_modulus_bits": params[
                "seal_tc128_max_coeff_modulus_bits"
            ],
            "seal_tc128_headroom_bits": params["seal_tc128_headroom_bits"],
            "secret_distribution": params["secret_distribution"],
            "error_distribution": params["error_distribution"],
        },
        "standard_rough_batch": {
            "status": parameters_record["estimator"]["status"],
            "elapsed_seconds": parameters_record["estimator"].get("elapsed_seconds"),
            "timeout_seconds": parameters_record["estimator"].get("timeout_seconds"),
            "numerical_estimate_available": any(
                value.get("minimum_rop_log2") is not None
                for value in parameters_record["estimator"]
                .get("parsed_costs", {})
                .values()
            ),
            "stdout_sha256": parameters_record["estimator"].get("stdout_sha256"),
            "interpretation": (
                "The upstream buffered batch timed out in bounded-CBD Arora-GB; "
                "it is retained and is not presented as a completed estimate."
            ),
        },
        "rough_individual": {
            "cost_model": "ADPS16",
            "shape_model": "GSA",
            "models": _attack_table(rough_record),
            "interpretation": (
                "Completed uSVP/dual/dual-hybrid costs are heuristic rough-model "
                "estimates. Exact-CBD Arora-GB timed out and remains incomplete."
            ),
        },
        "extended_individual": {
            "cost_model": "MATZOV (estimator default)",
            "shape_model": "GSA (estimator default)",
            "models": _attack_table(extended_record),
            "interpretation": (
                "Do not numerically combine these exponents with ADPS16 rough "
                "exponents. Both BKW points timed out and remain incomplete."
            ),
        },
        "candidate_id_leakage": {
            "million_query_count": million["query_count"],
            "million_k100": million_k100,
            "beir_confirmatory_query_count": sum(
                value["query_count"] for value in beir_record["collections"].values()
            ),
            "beir_k20_log_rank_ranges": {
                "direction_cosine_mean": [
                    min(value["direction_cosine"]["mean"] for value in beir_points),
                    max(value["direction_cosine"]["mean"] for value in beir_points),
                ],
                "exact_top10_overlap_mean": [
                    min(
                        value["exact_top10_overlap_fraction"]["mean"]
                        for value in beir_points
                    ),
                    max(
                        value["exact_top10_overlap_fraction"]["mean"]
                        for value in beir_points
                    ),
                ],
                "linkability_auc": [
                    min(value["linkability"]["roc_auc"] for value in beir_points),
                    max(value["linkability"]["roc_auc"] for value in beir_points),
                ],
            },
        },
        "circuit_privacy_boundary": {
            "fixed_request_unique_response_hashes": fixed["unique_response_hashes"],
            "fresh_encryption_unique_response_hashes": fresh["unique_response_hashes"],
            "explicit_output_sanitisation_present": circuit_record[
                "implementation_trace"
            ]["explicit_output_sanitisation_present"],
            "maximum_observed_abs_error": max(
                max(fixed["max_abs_decryption_error"]),
                max(fresh["max_abs_decryption_error"]),
            ),
            "formal_circuit_privacy_established": False,
        },
        "claim_boundary": {
            "supported": [
                "the exact parameter set passes Microsoft SEAL's tc128 modulus guard",
                "completed lattice-estimator core attacks exceed 128 bits under both stated cost models",
                "candidate identifiers retain substantial semantic/linkability information",
            ],
            "not_supported": [
                "an exhaustive all-attack estimator minimum (Arora-GB/BKW timed out)",
                "circular/KDM security of Galois keys",
                "circuit privacy or q-IND-CPA-D security",
                "document, database, or access-pattern privacy",
            ],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--security-dir", type=Path, required=True)
    parser.add_argument("--expansion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    security = args.security_dir
    expansion = args.expansion_dir
    inputs = {
        "security_parameters": security / "security_parameters.json",
        "rough_individual": security / "individual_security_estimates.json",
        "extended_individual": security / "extended_security_estimates.json",
        "circuit_privacy": expansion / "circuit_privacy_boundary.json",
        "candidate_million": expansion / "candidate_id_million.json",
        "candidate_beir": expansion / "candidate_id_beir_confirmatory.json",
    }
    result = summarize(
        parameters_record=load(inputs["security_parameters"]),
        rough_record=load(inputs["rough_individual"]),
        extended_record=load(inputs["extended_individual"]),
        circuit_record=load(inputs["circuit_privacy"]),
        million_record=load(inputs["candidate_million"]),
        beir_record=load(inputs["candidate_beir"]),
    )
    result["evidence_files"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in inputs.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rough_min": {
                    model: value["minimum_completed_rop_log2"]
                    for model, value in result["rough_individual"]["models"].items()
                },
                "extended_min": {
                    model: value["minimum_completed_rop_log2"]
                    for model, value in result["extended_individual"]["models"].items()
                },
                "standard_batch": result["standard_rough_batch"]["status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
