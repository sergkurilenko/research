"""Audit the prototype's circuit-privacy boundary without an extraction attack.

The experiment is intentionally narrow.  It checks whether replaying the same
encrypted request through the deterministic CKKS evaluation path returns the
same serialized ciphertext, and whether fresh client encryption changes that
response.  It also records the server operation trace from the implementation.
Neither outcome proves or disproves a formal circuit-privacy notion; the useful
fact is whether an explicit sanitisation/noise-flooding step exists before the
evaluated ciphertext is returned to the key owner.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from system.packed_ckks import (
    CKKSParameters,
    _server_scores,
    client_decrypt_scores,
    create_context_pair,
    encrypt_query,
    plaintext_scores,
    server_segmented_score_packed,
)


SCHEMA = "ckks_circuit_privacy_boundary.v1"
DISCLAIMER = (
    "This diagnostic is not a circuit-privacy proof and is not a key-recovery "
    "experiment. Standard CKKS IND-CPA query confidentiality does not by itself "
    "guarantee that an evaluated ciphertext hides the server circuit/plaintext "
    "operands from the decrypting client. The prototype claims no circuit privacy, "
    "q-IND-CPA-D/IND-CPA-D security, malicious-client security, or output "
    "sanitisation."
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def implementation_trace() -> dict[str, Any]:
    source = inspect.getsource(_server_scores)
    lowered = source.lower()
    checks = {
        "multiply_plain": "multiply_plain" in lowered,
        "rescale": "rescale_to_next" in lowered,
        "rotation": "rotate_vector" in lowered,
        "ciphertext_addition": "add_inplace" in lowered,
        "fresh_encrypt_zero": "encrypt_zero" in lowered,
        "noise_flooding_named": "noise_flood" in lowered or "smudg" in lowered,
        "rerandomization_named": "rerandom" in lowered or "re-random" in lowered,
    }
    return {
        "server_score_source_sha256": sha256(source.encode("utf-8")),
        "operation_presence": checks,
        "explicit_output_sanitisation_present": bool(
            checks["fresh_encrypt_zero"]
            or checks["noise_flooding_named"]
            or checks["rerandomization_named"]
        ),
    }


def run(*, dimension: int, candidates: int, repeats: int, seed: int) -> dict[str, Any]:
    if not 1 <= dimension <= 4096 or candidates < 1 or repeats < 2:
        raise ValueError("invalid dimension/candidates/repeats")
    rng = np.random.default_rng(seed)
    query = rng.normal(size=dimension)
    query /= np.linalg.norm(query)
    docs = rng.normal(size=(candidates, dimension))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    expected = plaintext_scores(query, docs)
    parameters = CKKSParameters()
    pair = create_context_pair(parameters, max_query_dimension=dimension)

    fixed_request = encrypt_query(pair.client, query)
    fixed_wires: list[bytes] = []
    fixed_errors: list[float] = []
    for _ in range(repeats):
        response = server_segmented_score_packed(
            pair.server, fixed_request, docs
        )
        wire = response.to_bytes()
        fixed_wires.append(wire)
        fixed_errors.append(
            float(np.max(np.abs(client_decrypt_scores(pair.client, wire) - expected)))
        )

    fresh_wires: list[bytes] = []
    fresh_errors: list[float] = []
    for _ in range(repeats):
        request = encrypt_query(pair.client, query)
        response = server_segmented_score_packed(pair.server, request, docs)
        wire = response.to_bytes()
        fresh_wires.append(wire)
        fresh_errors.append(
            float(np.max(np.abs(client_decrypt_scores(pair.client, wire) - expected)))
        )

    fixed_hashes = [sha256(value) for value in fixed_wires]
    fresh_hashes = [sha256(value) for value in fresh_wires]
    trace = implementation_trace()
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "config": {
            "dimension": dimension,
            "candidate_count": candidates,
            "repeats": repeats,
            "seed": seed,
            "ckks": parameters.as_json(),
        },
        "implementation_trace": trace,
        "identical_encrypted_request_replay": {
            "unique_response_hashes": len(set(fixed_hashes)),
            "all_serialized_responses_identical": len(set(fixed_hashes)) == 1,
            "response_sha256": fixed_hashes,
            "max_abs_decryption_error": fixed_errors,
        },
        "fresh_client_encryption": {
            "unique_response_hashes": len(set(fresh_hashes)),
            "all_serialized_responses_unique": len(set(fresh_hashes)) == repeats,
            "response_sha256": fresh_hashes,
            "max_abs_decryption_error": fresh_errors,
        },
        "interpretation": {
            "observed": (
                "The evaluator is deterministic for a fixed input ciphertext and "
                "contains no explicit output sanitisation/noise-flooding operation."
            ),
            "not_established": (
                "Fresh-input ciphertext diversity and small numerical error do not "
                "establish circuit privacy or decryption-oracle security."
            ),
            "safe_protocol_boundary": (
                "Keep the restricted honest-client model; do not claim provider "
                "operand/circuit privacy. A formally analysed sanitisation protocol "
                "would require a separate parameter/noise/correctness study and is "
                "not retrofitted here."
            ),
        },
        "primary_references": [
            {
                "title": "On the Security of Homomorphic Encryption on Approximate Numbers",
                "url": "https://eprint.iacr.org/2020/1533",
                "scope": "CKKS decryption-oracle/approximate-output security boundary",
            },
            {
                "title": "Circuit Privacy for FHEW/TFHE-Style FHE in Practice",
                "url": "https://eprint.iacr.org/2022/1459",
                "scope": "circuit-privacy definition and sanitisation/noise-flooding discussion",
            },
            {
                "title": "Maliciously Circuit-Private FHE from Information-Theoretic Principles",
                "url": "https://eprint.iacr.org/2022/495",
                "scope": "formal circuit-privacy distinction from ordinary FHE input privacy",
            },
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "backend": "Microsoft SEAL through tenseal.sealapi",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        dimension=args.dimension,
        candidates=args.candidates,
        repeats=args.repeats,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fixed_unique": result["identical_encrypted_request_replay"][
                    "unique_response_hashes"
                ],
                "fresh_unique": result["fresh_client_encryption"][
                    "unique_response_hashes"
                ],
                "sanitisation": result["implementation_trace"][
                    "explicit_output_sanitisation_present"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
