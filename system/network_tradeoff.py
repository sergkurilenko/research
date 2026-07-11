"""Deterministic link model for measured CKKS and vector-return payloads.

This is not a WAN benchmark.  It adds one configured RTT and ideal payload
serialization time to measured local compute, explicitly excluding TCP/TLS
headers, congestion, retransmissions, and payload fetching after ranking.
"""

from __future__ import annotations

import argparse

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "retrieval_link_tradeoff.v1"


def ideal_serialization_ms(payload_bytes: float, link_mbps: float) -> float:
    if payload_bytes < 0 or link_mbps <= 0:
        raise ValueError("payload must be non-negative and link_mbps positive")
    return float(payload_bytes * 8.0 / (link_mbps * 1_000_000.0) * 1_000.0)


def model_latency(
    local_ms: float, payload_bytes: float, link_mbps: float, rtt_ms: float
) -> float:
    if local_ms < 0 or rtt_ms < 0:
        raise ValueError("latencies cannot be negative")
    return float(
        local_ms + ideal_serialization_ms(payload_bytes, link_mbps) + rtt_ms
    )


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def build_tradeoff(
    unified: dict[str, Any],
    vector: dict[str, Any],
    retrieval: dict[str, Any],
    links_mbps: Sequence[float],
    rtts_ms: Sequence[float],
) -> dict[str, Any]:
    ckks = unified["methods"]["ckks_projected_shortlist"]
    pq_latency = retrieval["methods"]["projected_shortlist_rerank"]["latency_ms"][
        "client_pq_search_ms"
    ]
    methods: dict[str, dict[str, Any]] = {
        "packed_ckks": {
            "local_p50_ms": float(ckks["latency_ms"]["online_end_to_end"]["p50"]),
            "local_p95_ms": float(ckks["latency_ms"]["online_end_to_end"]["p95"]),
            "payload_bytes": float(ckks["payload_bytes"]["total"]["mean"]),
            "disclosure": (
                "query numerical values and scores encrypted; candidate IDs, public "
                "PQ geometry, sizes/timing exposed; exact vectors not directly returned"
            ),
        },
        "pq_only": {
            "local_p50_ms": float(pq_latency["p50"]),
            "local_p95_ms": float(pq_latency["p95"]),
            "payload_bytes": 0.0,
            "disclosure": "public PQ geometry; approximate ordering; no rerank request",
        },
    }
    for source_name in (
        "projected_float32_return",
        "projected_float16_return",
        "projected_int8_symmetric_return",
        "raw_float32_return",
        "raw_float16_return",
    ):
        source = vector["methods"][source_name]
        compute = source["latency_ms"]["end_to_end_compute_ms"]
        methods[source_name] = {
            "local_p50_ms": float(pq_latency["p50"] + compute["p50"]),
            "local_p95_ms": float(pq_latency["p95"] + compute["p95"]),
            "payload_bytes": float(
                source["payload_bytes"]["total_application_bytes"]["mean"]
            ),
            "disclosure": source["disclosure_class"],
        }

    scenarios: list[dict[str, Any]] = []
    for link in links_mbps:
        for rtt in rtts_ms:
            rows: dict[str, Any] = {}
            for name, method in methods.items():
                payload = method["payload_bytes"]
                rows[name] = {
                    **method,
                    "ideal_payload_serialization_ms": ideal_serialization_ms(
                        payload, link
                    ),
                    "modeled_p50_ms": model_latency(
                        method["local_p50_ms"], payload, link, rtt
                    ),
                    "modeled_p95_ms": model_latency(
                        method["local_p95_ms"], payload, link, rtt
                    ),
                }
            scenarios.append(
                {"link_mbps": float(link), "rtt_ms": float(rtt), "methods": rows}
            )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "formula": "local measured compute + one RTT + 8*payload_bytes/link_bps",
            "not_measured": (
                "No WAN sockets were used; TCP/TLS framing, handshake, congestion, "
                "retransmission, queueing, and final document-payload fetch are excluded."
            ),
            "pq_only_payload_note": (
                "Zero refers only to ranking after the public index is onboarded; "
                "fetching final document content is outside every method row."
            ),
        },
        "scenarios": scenarios,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified", type=Path, required=True)
    parser.add_argument("--vector-baselines", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--links-mbps", type=float, nargs="+", default=[100.0, 1000.0])
    parser.add_argument("--rtts-ms", type=float, nargs="+", default=[1.0, 20.0, 80.0])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_tradeoff(
        _load(args.unified),
        _load(args.vector_baselines),
        _load(args.retrieval),
        args.links_mbps,
        args.rtts_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["scenarios"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
