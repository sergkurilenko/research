"""Unit tests for the CKKS sweep planner, provenance, and resume logic."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parents[1] / "system"
sys.path.insert(0, str(MODULE_DIR))

import run_ckks_sweep as sweep  # noqa: E402


def fake_microbenchmark(
    *, dimension: int, candidate_count: int, repeats: int, warmups: int, seed: int
) -> dict[str, Any]:
    def timing(base: float) -> dict[str, Any]:
        samples = [base + index for index in range(repeats)]
        return {
            "samples": samples,
            "count": repeats,
            "min": min(samples),
            "median": samples[len(samples) // 2],
            "mean": sum(samples) / len(samples),
            "p95": max(samples),
            "max": max(samples),
        }

    methods: dict[str, Any] = {}
    method_bases = {
        "naive_per_candidate": 100.0,
        "block_packed": 30.0,
        "segmented_score_packed": 20.0,
    }
    for method, base in method_bases.items():
        methods[method] = {
            "server_ms": timing(base),
            "client_decrypt_ms": timing(base / 10),
            "server_phase_ms": {
                phase: timing(base / 5)
                for phase in (
                    "request_deserialize",
                    "plaintext_encode",
                    "he_evaluate",
                    "score_pack",
                    "ciphertext_serialize",
                )
            },
            "response": {
                "wire_bytes": int(base * 1000),
                "ckks_payload_bytes": int(base * 900),
                "payload_count": 1,
                "seal_ciphertext_count": 1,
                "score_counts_per_payload": [candidate_count],
                "score_slot_indices": [list(range(candidate_count))],
            },
            "correctness": {
                "max_abs_error": 1e-6,
                "mean_abs_error": 5e-7,
                "rmse": 6e-7,
                "allclose_rtol_3e-4_atol_3e-5": True,
            },
        }
    return {
        "schema": "packed_ckks_microbenchmark.v1",
        "created_utc": "2026-07-10T00:00:00+00:00",
        "software": {
            "python": "3.11.15",
            "tenseal": "0.3.16",
            "numpy": "2.4.3",
            "he_backend": "fake-test-backend",
        },
        "implementation": {
            "algorithm": "power_of_two_segmented_reduction",
            "server_has_secret_key": False,
            "decryption_location": "client_only",
        },
        "config": {
            "dimension": dimension,
            "candidate_count": candidate_count,
            "repeats": repeats,
            "warmups": warmups,
            "seed": seed,
            "ckks": sweep.CKKSParameters().as_json(),
        },
        "setup": {
            "context_and_key_generation_ms": 1.0,
            "serialized_public_server_context_bytes": 100,
            "query_encrypt_serialize_ms": 1.0,
            "serialized_query_bytes": 10,
        },
        "methods": methods,
        "comparison": {
            "server_speedup_naive_over_single_response_median": 5.0,
            "wire_size_reduction_naive_over_single_response_ratio": 5.0,
        },
    }


class SweepRunnerTest(unittest.TestCase):
    def test_plan_merges_intersection(self) -> None:
        plan = sweep.build_point_plan()
        self.assertEqual(len(plan), 9)
        ids = [point.point_id for point in plan]
        self.assertEqual(len(ids), len(set(ids)))
        shared = next(point for point in plan if point.point_id == "d672_K100")
        self.assertEqual(shared.sweep_membership, ("dimension", "candidate_count"))

    def test_checkpoint_resume_skips_completed_points(self) -> None:
        calls: list[tuple[int, int]] = []

        def recording_benchmark(**kwargs: Any) -> dict[str, Any]:
            calls.append((kwargs["dimension"], kwargs["candidate_count"]))
            return fake_microbenchmark(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep.json"
            first = sweep.run_sweep(
                output_path=output,
                repeats=3,
                warmups=1,
                reuse_point_path=None,
                max_new_points=2,
                benchmark=recording_benchmark,
            )
            self.assertEqual(first["status"], "partial")
            self.assertEqual(first["progress"]["completed_points"], 2)
            self.assertEqual(len(calls), 2)

            calls.clear()
            resumed = sweep.run_sweep(
                output_path=output,
                repeats=3,
                warmups=1,
                reuse_point_path=None,
                benchmark=recording_benchmark,
            )
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["progress"]["completed_points"], 9)
            self.assertEqual(len(calls), 7)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["status"], "complete")
            self.assertEqual(len(loaded["points"]), 9)

    def test_reused_point_has_digest_and_actual_repeat_provenance(self) -> None:
        calls: list[tuple[int, int]] = []

        def recording_benchmark(**kwargs: Any) -> dict[str, Any]:
            calls.append((kwargs["dimension"], kwargs["candidate_count"]))
            return fake_microbenchmark(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.json"
            source.write_text(
                json.dumps(
                    fake_microbenchmark(
                        dimension=672,
                        candidate_count=100,
                        repeats=20,
                        warmups=2,
                        seed=sweep.DEFAULT_SEED,
                    )
                ),
                encoding="utf-8",
            )
            state = sweep.run_sweep(
                output_path=root / "sweep.json",
                repeats=7,
                warmups=2,
                reuse_point_path=source,
                max_new_points=3,
                benchmark=recording_benchmark,
            )
            reused = next(point for point in state["points"] if point["id"] == "d672_K100")
            provenance = reused["provenance"]
            self.assertEqual(provenance["kind"], "reused_existing_microbenchmark")
            self.assertEqual(provenance["actual_repeats"], 20)
            self.assertEqual(provenance["requested_sweep_repeats"], 7)
            self.assertEqual(len(provenance["source_sha256"]), 64)
            self.assertEqual(len(calls), 2)
            self.assertIn("median", reused["methods"]["segmented_score_packed"]["server_ms"])
            self.assertIn(
                "p95",
                reused["methods"]["segmented_score_packed"]["server_phase_ms"][
                    "he_evaluate"
                ],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
