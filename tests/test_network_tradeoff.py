from __future__ import annotations

import unittest

from system.network_tradeoff import ideal_serialization_ms, model_latency


class NetworkTradeoffTests(unittest.TestCase):
    def test_one_megabit_at_one_mbps_is_one_second(self) -> None:
        self.assertAlmostEqual(ideal_serialization_ms(125_000, 1.0), 1000.0)

    def test_latency_adds_one_rtt(self) -> None:
        self.assertAlmostEqual(model_latency(5.0, 125_000, 100.0, 20.0), 35.0)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            ideal_serialization_ms(1, 0)


if __name__ == "__main__":
    unittest.main()
