"""Tests for the concurrency/session/socket benchmark harness."""

from __future__ import annotations

import socket
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from system.packed_ckks import CKKSParameters, create_context_pair
from system.systems_expansion import (
    _MSG_SCORE,
    _build_request_pool,
    distribution,
    inspect_key_material,
    recv_frame,
    run_loopback_benchmark,
    run_scaling_condition,
    send_frame,
)
from system.tls_test_material import generate_loopback_tls_material


class SystemsExpansionTests(unittest.TestCase):
    def test_distribution_includes_tail_percentiles(self) -> None:
        summary = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["p50"], 2.5)
        self.assertGreater(summary["p99"], summary["p95"])

    def test_binary_frame_round_trip_and_exact_byte_count(self) -> None:
        left, right = socket.socketpair()
        try:
            parts = (b"encrypted-query", b"\x00\x01\x02")
            received: list[object] = []

            def reader() -> None:
                received.extend(recv_frame(right))

            thread = threading.Thread(target=reader)
            thread.start()
            sent_bytes = send_frame(left, _MSG_SCORE, parts)
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(received[0], _MSG_SCORE)
            self.assertEqual(received[1], parts)
            self.assertEqual(received[2], sent_bytes)
        finally:
            left.close()
            right.close()

    def test_key_components_and_real_loopback_tcp_tls(self) -> None:
        rng = np.random.default_rng(20260711)
        docs = rng.normal(size=(12, 9)).astype(np.float32)
        pair = create_context_pair(CKKSParameters(), max_query_dimension=9)
        keys = inspect_key_material(pair)
        components = keys["public_components_bytes"]
        self.assertGreater(components["public_key"], 0)
        self.assertGreater(components["galois_keys"], components["public_key"])
        self.assertFalse(keys["contains_secret_key_in_server_context"])

        with tempfile.TemporaryDirectory(prefix="systems_expansion_test_") as directory:
            docs_path = Path(directory) / "docs.npy"
            np.save(docs_path, docs, allow_pickle=False)
            pool = _build_request_pool(
                pair,
                dimension=9,
                n_docs=len(docs),
                candidate_count=5,
                pool_size=2,
                seed=17,
            )
            certificate_path, private_key_path = generate_loopback_tls_material(
                Path(directory) / "tls"
            )
            result = run_loopback_benchmark(
                pair=pair,
                projected_docs_path=docs_path,
                request_pool=pool,
                warmups=0,
                repeats=1,
                certificate_path=certificate_path,
                private_key_path=private_key_path,
                response_timeout_s=60.0,
            )
            scaling = run_scaling_condition(
                public_context=pair.serialized_server_context,
                client_context=pair.client,
                projected_docs_path=docs_path,
                method="segmented_score_packed",
                concurrency=2,
                requests_per_worker=1,
                warmup_requests_per_worker=0,
                request_pool=pool,
                restart=0,
                logical_cpus=2,
                response_timeout_s=60.0,
            )

        self.assertTrue(result["transports"]["tcp"]["server_attestation"]["loopback_only"])
        self.assertEqual(result["transports"]["tls13"]["startup"]["tls_version"], "TLSv1.3")
        self.assertEqual(result["transports"]["tls13"]["records"][0]["score_count"], 5)
        self.assertEqual(scaling["successful_requests"], 2)
        self.assertEqual(scaling["failed_requests"], 0)
        self.assertGreater(scaling["throughput_qps"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
