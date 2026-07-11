"""Correctness and role-separation tests for system/packed_ckks.py.

These tests use only the standard-library ``unittest`` runner so they can be
executed in the supplied Python 3.11 environment without installing pytest:

    python -m unittest discover -s revision_workspace/tests -p test_packed_ckks.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1] / "system"
sys.path.insert(0, str(MODULE_DIR))

import packed_ckks as pckks  # noqa: E402


class PackedCKKSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = pckks.CKKSParameters()
        cls.contexts = pckks.create_context_pair(cls.parameters)

    def synthetic_case(
        self, *, dimension: int = 13, candidates: int = 9, seed: int = 7
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes]:
        rng = np.random.default_rng(seed)
        query = rng.normal(size=dimension)
        query /= np.linalg.norm(query)
        matrix = rng.normal(size=(candidates, dimension))
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        expected = matrix @ query
        serialized_query = pckks.encrypt_query(self.contexts.client, query)
        return query, matrix, expected, serialized_query

    def assert_scores_close(self, actual: np.ndarray, expected: np.ndarray) -> None:
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_server_context_is_public_and_cannot_decrypt(self) -> None:
        self.assertTrue(self.contexts.client.has_secret_key())
        self.assertFalse(self.contexts.server.has_secret_key())
        self.assertTrue(self.contexts.server.has_public_key())
        self.assertTrue(self.contexts.server.has_galois_keys())
        self.assertTrue(self.contexts.server.is_public())
        self.assertFalse(hasattr(self.contexts.server, "secret_key"))
        self.assertFalse(hasattr(self.contexts.server, "decryptor"))

        reconstructed = pckks.deserialize_server_context(
            self.contexts.serialized_server_context
        )
        self.assertFalse(reconstructed.has_secret_key())
        self.assertFalse(hasattr(reconstructed, "decryptor"))

        _, matrix, _, serialized_query = self.synthetic_case(candidates=2)
        with self.assertRaisesRegex(ValueError, "secret key"):
            pckks.client_decrypt_scores(
                self.contexts.server,
                pckks.server_block_packed_scores(
                    self.contexts.server, serialized_query, matrix
                ),
            )

    def test_naive_and_single_block_packed_match_plaintext(self) -> None:
        _, matrix, expected, serialized_query = self.synthetic_case(
            dimension=17, candidates=11, seed=11
        )
        naive = pckks.server_naive_scores(
            self.contexts.server, serialized_query, matrix
        )
        packed = pckks.server_block_packed_scores(
            self.contexts.server, serialized_query, matrix
        )

        naive_scores = pckks.client_decrypt_scores(
            self.contexts.client, naive.to_bytes()
        )
        packed_scores = pckks.client_decrypt_scores(
            self.contexts.client, packed.to_bytes()
        )
        self.assert_scores_close(naive_scores, expected)
        self.assert_scores_close(packed_scores, expected)
        self.assert_scores_close(packed_scores, naive_scores)

        self.assertEqual(naive.score_count, matrix.shape[0])
        self.assertEqual(naive.seal_ciphertext_count, matrix.shape[0])
        self.assertEqual(len(naive.payloads), matrix.shape[0])
        self.assertEqual(packed.score_count, matrix.shape[0])
        self.assertEqual(packed.seal_ciphertext_count, 1)
        self.assertEqual(len(packed.payloads), 1)
        self.assertLess(len(packed.to_bytes()), len(naive.to_bytes()))

    def test_multi_block_response_roundtrip_and_score_order(self) -> None:
        _, matrix, expected, serialized_query = self.synthetic_case(
            dimension=19, candidates=10, seed=23
        )
        packed = pckks.server_block_packed_scores(
            self.contexts.server,
            serialized_query,
            matrix,
            block_size=4,
        )
        self.assertEqual(packed.score_counts, (4, 4, 2))
        self.assertEqual(packed.seal_ciphertext_counts, (1, 1, 1))
        self.assertEqual(
            packed.score_indices,
            ((0, 32, 64, 96), (0, 32, 64, 96), (0, 32)),
        )

        wire = packed.to_bytes()
        parsed = pckks.EncryptedScoreResponse.from_bytes(wire)
        self.assertEqual(parsed.method, "block_packed")
        self.assertEqual(parsed.score_counts, packed.score_counts)
        self.assertEqual(parsed.payloads, packed.payloads)
        self.assert_scores_close(
            pckks.client_decrypt_scores(self.contexts.client, wire), expected
        )

    def test_invalid_dimensions_and_server_role_are_rejected(self) -> None:
        _, matrix, _, serialized_query = self.synthetic_case(
            dimension=8, candidates=3
        )
        with self.assertRaisesRegex(ValueError, "dimension"):
            pckks.server_block_packed_scores(
                self.contexts.server, serialized_query, matrix[:, :-1]
            )
        with self.assertRaisesRegex(ValueError, "must not contain a secret key"):
            pckks.server_naive_scores(
                self.contexts.client, serialized_query, matrix
            )
        with self.assertRaisesRegex(ValueError, "block_size"):
            pckks.server_block_packed_scores(
                self.contexts.server, serialized_query, matrix, block_size=0
            )

    def test_microbenchmark_json_schema(self) -> None:
        result = pckks.run_microbenchmark(
            dimension=9,
            candidate_count=5,
            block_size=3,
            repeats=2,
            warmups=0,
            seed=101,
        )
        # It must be a portable JSON object, not a structure containing numpy
        # scalars or ciphertext objects.
        roundtripped = json.loads(json.dumps(result))
        self.assertEqual(roundtripped["schema"], pckks.MICROBENCHMARK_SCHEMA)
        self.assertFalse(roundtripped["implementation"]["server_has_secret_key"])
        self.assertTrue(roundtripped["implementation"]["client_has_secret_key"])
        self.assertEqual(
            roundtripped["methods"]["naive_per_candidate"]["response"][
                "seal_ciphertext_count"
            ],
            5,
        )
        self.assertEqual(
            roundtripped["methods"]["block_packed"]["response"][
                "seal_ciphertext_count"
            ],
            2,
        )
        self.assertEqual(
            roundtripped["methods"]["segmented_score_packed"]["response"][
                "seal_ciphertext_count"
            ],
            1,
        )
        for method in (
            "naive_per_candidate",
            "block_packed",
            "segmented_score_packed",
        ):
            details = roundtripped["methods"][method]
            self.assertEqual(details["server_ms"]["count"], 2)
            self.assertEqual(details["client_decrypt_ms"]["count"], 2)
            tolerance_key = (
                "allclose_rtol_3e-4_atol_3e-5"
                if method == "segmented_score_packed"
                else "allclose_rtol_1e-4_atol_1e-5"
            )
            self.assertTrue(details["correctness"][tolerance_key])

    def test_single_response_packs_multiple_groups(self) -> None:
        _, matrix, expected, serialized_query = self.synthetic_case(
            dimension=257, candidates=20, seed=313
        )
        response = pckks.server_segmented_score_packed(
            self.contexts.server, serialized_query, matrix
        )
        self.assertEqual(response.seal_ciphertext_count, 1)
        self.assertEqual(response.score_counts, (20,))
        self.assertEqual(len(set(response.score_indices[0])), 20)
        actual = pckks.client_decrypt_scores(self.contexts.client, response.to_bytes())
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
