import numpy as np

from system.candidate_id_leakage import (
    METHODS,
    cosine_rows,
    evaluate_collection,
    linkability_metrics,
    ranking_overlap,
    reconstruct_from_candidates,
)


class ExactIndex:
    def __init__(self, docs: np.ndarray):
        self.docs = np.asarray(docs, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int):
        scores = np.asarray(queries @ self.docs.T, dtype=np.float32)
        ids = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        values = np.take_along_axis(scores, ids, axis=1)
        return values, ids.astype(np.int64)


def _synthetic(seed: int = 9):
    rng = np.random.default_rng(seed)
    docs = rng.normal(size=(320, 24)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    queries = rng.normal(size=(8, 24)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return docs, queries


def test_candidate_estimators_are_finite_unit_directions():
    docs, queries = _synthetic()
    ids = np.argsort(-(docs @ queries[0]))[:40]
    for method in METHODS:
        estimate = reconstruct_from_candidates(docs[ids], method)
        assert estimate.shape == (docs.shape[1],)
        assert np.all(np.isfinite(estimate))
        assert np.isclose(np.linalg.norm(estimate), 1.0)
        assert float(estimate @ queries[0]) > 0.25


def test_overlap_and_linkability_helpers():
    reference = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    candidate = np.asarray([[1, 7, 3], [8, 5, 6]], dtype=np.int64)
    assert np.allclose(ranking_overlap(reference, candidate, 3), [2 / 3, 2 / 3])

    rng = np.random.default_rng(4)
    view_a = rng.normal(size=(20, 12))
    view_b = view_a + 0.05 * rng.normal(size=view_a.shape)
    metrics = linkability_metrics(view_a, view_b, seed=3)
    assert metrics["roc_auc"] > 0.95
    assert metrics["positive_count"] == 20
    assert metrics["negative_count"] > metrics["positive_count"]


def test_small_end_to_end_collection_is_candidate_id_only():
    docs, queries = _synthetic()
    index = ExactIndex(docs)
    true_ids = np.argmax(queries @ docs.T, axis=1)
    relevant = [{int(value)} for value in true_ids]
    result, rows = evaluate_collection(
        name="synthetic",
        projected_docs=docs,
        projected_queries=queries,
        pq_index=index,
        ks=[8, 16],
        relevant_local_ids=relevant,
        exact_backend="numpy",
        exact_query_batch_size=8,
        exact_doc_chunk_size=64,
        ridge_fraction=0.1,
        link_seed=2026,
    )
    assert result["query_count"] == len(queries)
    assert len(result["methods"]) == 2 * len(METHODS)
    assert len(rows) == len(queries) * 2 * len(METHODS)
    assert "no PQ distances" in result["threat_observation"]
    for method in result["methods"].values():
        assert -1.0 <= method["direction_cosine"]["mean"] <= 1.0
        assert 0.0 <= method["exact_top10_overlap_fraction"]["mean"] <= 1.0
        assert 0.0 <= method["linkability"]["roc_auc"] <= 1.0


def test_cosine_rows_handles_nonunit_vectors():
    left = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    right = np.asarray([[2.0, 0.0], [1.0, -1.0]])
    assert np.allclose(cosine_rows(left, right), [1.0, 0.0])
