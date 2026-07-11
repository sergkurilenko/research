from __future__ import annotations

import unittest

import numpy as np

from system.pq_leakage_audit import (
    compute_vector_disclosure,
    row_cosine,
    sample_ids,
)


class PQLeakageAuditTests(unittest.TestCase):
    def test_row_cosine_identity_and_orthogonal(self) -> None:
        left = np.array([[1.0, 0.0], [0.0, 1.0]])
        right = np.array([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(row_cosine(left, right), [1.0, 0.0])

    def test_sample_ids_are_deterministic_sorted_and_unique(self) -> None:
        first = sample_ids(100, 20, 9)
        second = sample_ids(100, 20, 9)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:-1] < first[1:]))

    def test_exact_reconstruction_has_unit_cosine_and_zero_error(self) -> None:
        raw = np.array([[1.0, 2.0, 0.0], [-1.0, 0.5, 1.0]], dtype=np.float32)
        mean = np.zeros((1, 3), dtype=np.float32)
        basis = np.eye(3, dtype=np.float32)
        result = compute_vector_disclosure(raw, raw, raw, mean, basis)
        self.assertAlmostEqual(result["projected_space"]["cosine"]["mean"], 1.0)
        self.assertAlmostEqual(result["projected_space"]["coordinate_rmse"], 0.0)
        self.assertAlmostEqual(
            result["lifted_original_space"]["relative_l2_error"]["max"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
