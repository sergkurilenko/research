from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from system.packed_ckks import CKKSParameters
from system.score_oracle_extraction import run_score_oracle_extraction


class ScoreOracleExtractionTests(unittest.TestCase):
    def test_toy_d8_actual_spawn_recovers_document(self) -> None:
        rng = np.random.default_rng(20260710)
        docs = rng.normal(size=(7, 8)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="score_oracle_test_") as directory:
            path = Path(directory) / "projected.npy"
            np.save(path, docs, allow_pickle=False)
            result = run_score_oracle_extraction(
                projected_docs_path=path,
                candidate_ids=[2],
                max_dimension=8,
                warmup_queries=1,
                ckks_parameters=CKKSParameters(),
            )

        self.assertEqual(result["experiment"]["actual_online_query_count"], 8)
        self.assertEqual(result["experiment"]["queries_per_document"], 8)
        self.assertFalse(result["protocol"]["online_client_has_exact_projected_rows"])
        self.assertIn("outside the prescribed-client model", result["experiment"]["disclaimer"])
        self.assertIn("Rate limiting", result["experiment"]["disclaimer"])

        attestation = result["server_attestation"]
        self.assertNotEqual(attestation["pid"], os.getpid())
        self.assertEqual(attestation["start_method"], "spawn")
        self.assertTrue(attestation["context_is_public"])
        self.assertFalse(attestation["has_secret_key"])
        self.assertFalse(attestation["has_secret_key_attribute"])
        self.assertFalse(attestation["has_decryptor_attribute"])

        metrics = result["documents"][0]["metrics"]
        self.assertGreater(metrics["cosine_similarity"], 0.999999)
        self.assertLess(metrics["relative_l2_error"], 1e-4)
        self.assertLess(metrics["rmse"], 1e-4)
        self.assertLess(metrics["max_absolute_error"], 1e-3)
        self.assertEqual(result["seal_ciphertext_count"]["min"], 1)
        self.assertEqual(result["seal_ciphertext_count"]["max"], 1)
        self.assertEqual(result["application_payload_bytes"]["request"]["count"], 8)

    def test_rejects_duplicate_or_out_of_range_ids(self) -> None:
        docs = np.zeros((3, 8), dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix="score_oracle_test_") as directory:
            path = Path(directory) / "projected.npy"
            np.save(path, docs, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                run_score_oracle_extraction(
                    projected_docs_path=path,
                    candidate_ids=[1, 1],
                )
            with self.assertRaisesRegex(ValueError, "outside"):
                run_score_oracle_extraction(
                    projected_docs_path=path,
                    candidate_ids=[3],
                )


if __name__ == "__main__":
    unittest.main()
