from __future__ import annotations

import unittest

import numpy as np

from system.pq_scaling_bench import prefix_index, timing_summary


class PQScalingBenchTests(unittest.TestCase):
    def test_timing_summary_quantiles(self) -> None:
        result = timing_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["p50"], 2.5)
        self.assertGreaterEqual(result["p99"], result["p95"])

    def test_prefix_index_truncates_codes_and_search_ids(self) -> None:
        import faiss

        rng = np.random.default_rng(3)
        train = rng.normal(size=(128, 8)).astype(np.float32)
        source = faiss.IndexPQ(8, 2, 4)
        source.train(train)
        source.add(train)
        prefix = prefix_index(source, 31, faiss)
        self.assertEqual(prefix.ntotal, 31)
        self.assertEqual(faiss.vector_to_array(prefix.codes).size, 31 * prefix.code_size)
        _, ids = prefix.search(train[:2], 5)
        self.assertTrue(np.all(ids >= 0))
        self.assertTrue(np.all(ids < 31))


if __name__ == "__main__":
    unittest.main()
