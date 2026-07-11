"""Internal consistency validator for ``systems_expansion.v1`` artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != "systems_expansion.v1":
        errors.append("unexpected schema")

    conditions = payload.get("scaling", {}).get("conditions", [])
    aggregates = payload.get("scaling", {}).get("aggregate", [])
    if not conditions or not aggregates:
        errors.append("scaling conditions/aggregates are missing")
    condition_attempted = sum(int(row["attempted_requests"]) for row in conditions)
    condition_success = sum(int(row["successful_requests"]) for row in conditions)
    condition_failed = sum(int(row["failed_requests"]) for row in conditions)
    if condition_attempted != condition_success + condition_failed:
        errors.append("condition request accounting is inconsistent")
    aggregate_attempted = sum(int(row["attempted_requests"]) for row in aggregates)
    aggregate_failed = sum(int(row["failed_requests"]) for row in aggregates)
    if aggregate_attempted != condition_attempted:
        errors.append("aggregate attempted count differs from conditions")
    if aggregate_failed != condition_failed:
        errors.append("aggregate failed count differs from conditions")

    for condition in conditions:
        concurrency = int(condition["concurrency"])
        if len(condition["server_attestations"]) != concurrency:
            errors.append(f"concurrency {concurrency} has the wrong attestation count")
        if int(condition["workers"]) != concurrency:
            errors.append(f"condition concurrency {concurrency} differs from workers")
        for attestation in condition["server_attestations"]:
            if not attestation["context_is_public"]:
                errors.append("a worker did not attest a public context")
            if attestation["has_secret_key"]:
                errors.append("a worker attested possession of a secret key")
            if attestation["has_secret_key_attribute"]:
                errors.append("a worker exposed a secret-key attribute")
            if attestation["has_decryptor_attribute"]:
                errors.append("a worker exposed a decryptor attribute")

    key_material = payload.get("sessions", {}).get("key_material", {})
    components = key_material.get("public_components_bytes", {})
    if components:
        recomputed = sum(int(value) for value in components.values()) + int(
            key_material["context_envelope_overhead_bytes"]
        )
        if recomputed != int(key_material["serialized_public_server_context_bytes"]):
            errors.append("public-context component sizes do not sum to envelope size")
    if key_material.get("contains_secret_key_in_server_context") is not False:
        errors.append("server-context secret-key flag is not false")
    if key_material.get("contains_relinearization_keys") is not False:
        errors.append("unexpected relinearization keys")

    payload_reference = payload.get("sessions", {}).get("payload_reference", {})
    if payload_reference:
        expected_request = int(payload_reference["query_wire_bytes"]) + int(
            payload_reference["candidate_id_bytes"]
        )
        if expected_request != int(payload_reference["request_application_bytes"]):
            errors.append("session request-byte accounting is inconsistent")
        per_request = expected_request + int(payload_reference["response_wire_bytes"])
        context_bytes = int(key_material["serialized_public_server_context_bytes"])
        for count_text, row in payload["sessions"]["amortization"].items():
            count = int(count_text)
            expected = context_bytes / count + per_request
            if abs(float(row["effective_total_bytes_per_request"]) - expected) > 1e-6:
                errors.append(f"amortization bytes are inconsistent for {count} requests")

    transports = payload.get("loopback_transport", {}).get("transports", {})
    for name in ("tcp", "tls13"):
        if name not in transports:
            errors.append(f"missing {name} transport")
            continue
        attestation = transports[name]["server_attestation"]
        if not attestation["loopback_only"] or attestation["bound_address"] != "127.0.0.1":
            errors.append(f"{name} was not confined to IPv4 loopback")
        if attestation["has_secret_key"] or attestation["has_decryptor_attribute"]:
            errors.append(f"{name} server role separation failed")
    if "tls13" in transports:
        startup = transports["tls13"]["startup"]
        if startup["tls_version"] != "TLSv1.3":
            errors.append("TLS transport did not negotiate TLS 1.3")

    for name, value in payload.get("environment", {}).get(
        "thread_environment", {}
    ).items():
        if value != "1":
            errors.append(f"thread environment {name} was not fixed to one")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    errors = validate(payload)
    report = {
        "schema": "systems_expansion_validation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(args.input.resolve()),
            "bytes": args.input.stat().st_size,
            "sha256": _sha256(args.input),
        },
        "valid": not errors,
        "errors": errors,
        "checks": {
            "condition_count": len(payload["scaling"]["conditions"]),
            "aggregate_count": len(payload["scaling"]["aggregate"]),
            "attempted_requests": sum(
                int(row["attempted_requests"])
                for row in payload["scaling"]["conditions"]
            ),
            "failed_requests": sum(
                int(row["failed_requests"])
                for row in payload["scaling"]["conditions"]
            ),
            "role_attestations": sum(
                len(row["server_attestations"])
                for row in payload["scaling"]["conditions"]
            ),
            "key_component_sum_checked": True,
            "amortization_checked": True,
            "loopback_and_tls_checked": True,
            "thread_environment_checked": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
