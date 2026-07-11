from __future__ import annotations

import numpy as np

from system.ir_expansion import (
    append_zeros,
    bootstrap_delta,
    canonical_metric_summary,
    subset_query_assets,
    variable_candidate_metrics,
)


def test_bootstrap_requires_complete_interval_for_equivalence() -> None:
    reference = np.zeros(20, dtype=np.float64)
    identical = bootstrap_delta(
        reference, reference, samples=500, seed=7, margin=0.002
    )
    assert identical["non_inferior"] is True
    assert identical["equivalent_within_symmetric_margin"] is True

    degraded = bootstrap_delta(
        np.full(20, -0.003), reference, samples=500, seed=7, margin=0.002
    )
    assert degraded["non_inferior"] is False
    assert degraded["equivalent_within_symmetric_margin"] is False


def test_canonical_zero_rows_change_mean_without_changing_evaluable() -> None:
    evaluable, canonical = canonical_metric_summary(
        {"ndcg_at_10": np.asarray([1.0, 0.5])}, zero_count=1
    )
    assert evaluable["ndcg_at_10"] == 0.75
    assert canonical["ndcg_at_10"] == 0.5
    np.testing.assert_array_equal(append_zeros([1.0, 0.5], 1), [1.0, 0.5, 0.0])


def test_subset_query_assets_preserves_requested_order_and_reports_absent() -> None:
    assets = {
        "query_ids": ["q1", "q2", "q3"],
        "queries": np.arange(6, dtype=np.float32).reshape(3, 2),
        "proj672_queries": np.arange(9, dtype=np.float32).reshape(3, 3),
        "qrels": {"q1": {}, "q2": {}, "q3": {}},
    }
    subset, absent = subset_query_assets(assets, ["q3", "missing", "q1"])
    assert subset["query_ids"] == ["q3", "q1"]
    assert absent == ["missing"]
    np.testing.assert_array_equal(subset["queries"], [[4, 5], [0, 1]])


def test_variable_candidate_metrics_supports_k_below_100() -> None:
    assets = {
        "corpus_ids": ["d1", "d2", "d3", "d4"],
        "query_ids": ["q1"],
        "qrels": {"q1": {"d2": 1.0, "d4": 1.0}},
    }
    candidates = np.asarray([[0, 1]], dtype=np.int64)
    reranked = np.asarray([[1, 0]], dtype=np.int64)
    summary, arrays = variable_candidate_metrics(candidates, reranked, assets)
    assert summary["candidate_recall_at_k"] == 0.5
    assert summary["recall_at_100_with_at_most_k_returned"] == 0.5
    assert summary["ndcg_at_10"] > 0.0
    assert summary["mrr_at_10"] == 1.0
    assert arrays["reranked_recall_at_100"][0] == 0.5
