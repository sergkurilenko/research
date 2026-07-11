from __future__ import annotations

import unittest

from system.aggregate_restarts import distribution


class AggregateRestartsTests(unittest.TestCase):
    def test_distribution(self) -> None:
        result = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["p50"], 2.5)
        self.assertGreater(result["p99"], result["p95"])

    def test_distribution_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            distribution([])


if __name__ == "__main__":
    unittest.main()
