from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from system.compare_ckks_runtime_pair import METHODS, compare


def _method(server_scale: float) -> dict:
    return {
        "server_ms": {"samples": [server_scale, 2 * server_scale, 3 * server_scale]},
        "client_decrypt_ms": {"samples": [1.0, 2.0, 3.0]},
        "correctness": {
            "max_abs_error": 1e-6,
            "mean_abs_error": 5e-7,
            "rmse": 6e-7,
            "allclose_rtol_3e-4_atol_3e-5": True,
        },
        "response": {"wire_bytes": 100, "seal_ciphertext_count": 1},
    }


def _payload(platform_name: str, server_scale: float) -> dict:
    return {
        "schema": "packed_ckks_microbenchmark.v1",
        "config": {"dimension": 672, "candidate_count": 100, "seed": 7},
        "software": {
            "platform": platform_name,
            "python": "3.11",
            "numpy": "2.4",
            "tenseal": "0.3.16",
        },
        "setup": {"context_and_key_generation_ms": 10.0},
        "implementation": {"server_has_secret_key": False},
        "methods": {method: _method(server_scale) for method in METHODS},
        "comparison": {"server_speedup_naive_over_single_response_median": 4.0},
    }


class RuntimePairComparisonTests(unittest.TestCase):
    def test_valid_pair_and_ratio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="runtime_pair_test_") as directory:
            root = Path(directory)
            windows = root / "windows.json"
            wsl = root / "wsl.json"
            windows.write_text(json.dumps(_payload("Windows", 2.0)), encoding="utf-8")
            wsl.write_text(json.dumps(_payload("Linux", 1.0)), encoding="utf-8")
            result = compare(windows, wsl)
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            result["ratios"]["segmented_score_packed"]
            ["wsl_over_windows_server_median"],
            0.5,
        )
        self.assertEqual(
            result["runtimes"]["windows"]["methods"]["naive_per_candidate"]
            ["server_ms"]["p95"],
            5.8,
        )

    def test_configuration_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="runtime_pair_test_") as directory:
            root = Path(directory)
            windows = root / "windows.json"
            wsl = root / "wsl.json"
            win_payload = _payload("Windows", 1.0)
            wsl_payload = _payload("Linux", 1.0)
            wsl_payload["config"]["dimension"] = 384
            windows.write_text(json.dumps(win_payload), encoding="utf-8")
            wsl.write_text(json.dumps(wsl_payload), encoding="utf-8")
            result = compare(windows, wsl)
        self.assertFalse(result["valid"])
        self.assertIn("benchmark configurations differ", result["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
