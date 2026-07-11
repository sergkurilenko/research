from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from system import graded_ir_bench as gib


def _normalised_random(n: int, d: int, seed: int = 7) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return gib.l2_normalize_rows(values)


def _qrels(qids: list[str], docid: str = "d0") -> dict[str, dict[str, float]]:
    return {qid: {docid: 1.0} for qid in qids}


def test_required_beir_collection_registry_is_complete_and_canonical() -> None:
    assert set(gib.REQUIRED_DATASETS) == {
        "scifact",
        "nfcorpus",
        "arguana",
        "scidocs",
        "fiqa",
        "trec-covid",
    }
    assert set(gib.REQUIRED_DATASETS).issubset(gib.DATASET_SPECS)
    assert gib.canonical_dataset_name("TREC_COVID") == "trec-covid"
    assert gib.canonical_dataset_name("FiQA-2018") == "fiqa"
    with pytest.raises(ValueError, match="Unknown dataset"):
        gib.canonical_dataset_name("not-beir")


def test_e5_prefixes_and_title_passage_join_are_exact() -> None:
    assert gib.format_e5_query("  What is IR?  ") == "query: What is IR?"
    assert gib.format_e5_passage(" body ", " title ") == "passage: title\nbody"
    assert gib.format_e5_passage("body", "") == "passage: body"
    assert gib.format_e5_passage("", "title") == "passage: title"
    # A user string that happens to start similarly is still data; the harness
    # applies the literal model-card prefix exactly once at encoding time.
    assert gib.format_e5_query("query: literal") == "query: query: literal"


def test_l2_normalisation_is_float32_unit_length_and_rejects_zero() -> None:
    values = np.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=np.float64)
    normalised = gib.l2_normalize_rows(values)
    assert normalised.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalised, axis=1), 1.0, atol=1e-7)
    with pytest.raises(ValueError, match="zero embedding"):
        gib.l2_normalize_rows(np.zeros((1, 3), dtype=np.float32))


def test_masked_mean_pool_excludes_padding() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]], [[2.0, 6.0], [8.0, 0.0], [0.0, 0.0]]]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = gib.masked_mean_pool(hidden, mask)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0], [2.0, 6.0]]))


def test_official_dev_or_train_is_validation_and_test_remains_full() -> None:
    qrels = {
        "train": _qrels(["tr2", "tr1"]),
        "dev": _qrels(["dv2", "dv1"]),
        "test": _qrels(["te3", "te1", "te2"]),
    }
    validation, validation_meta = gib.select_evaluation_qids(qrels, "validation")
    test, test_meta = gib.select_evaluation_qids(qrels, "test")
    official, official_meta = gib.select_evaluation_qids(qrels, "official-test")

    assert validation == ["dv1", "dv2"]
    assert test == ["te1", "te2", "te3"]
    assert official == test
    assert validation_meta["source_qrels"] == "dev"
    assert test_meta["beir_comparable_full_test"] is True
    assert official_meta["policy"] == "full_official_test_frozen_only"


def test_official_train_is_used_when_dev_is_absent() -> None:
    qrels = {"train": _qrels(["tr2", "tr1"]), "test": _qrels(["te1"])}
    validation, meta = gib.select_evaluation_qids(qrels, "validation")
    assert validation == ["tr1", "tr2"]
    assert meta["source_qrels"] == "train"


def test_missing_positive_documents_are_explicitly_excluded() -> None:
    retained, qrels, audit = gib.exclude_queries_with_missing_positive_documents(
        ["q1", "q2"],
        {"q1": {"d1": 1.0}, "q2": {"missing": 1.0, "d2": 0.0}},
        {"d1", "d2"},
    )
    assert retained == ["q1"]
    assert qrels == {"q1": {"d1": 1.0}}
    assert audit["excluded_query_count"] == 1
    assert audit["missing_positive_document_count"] == 1


def test_test_only_hash_partition_is_stable_disjoint_and_exhaustive() -> None:
    qids = [f"q{i:03d}" for i in range(101)]
    forward = {"test": _qrels(qids)}
    reverse = {"test": _qrels(list(reversed(qids)))}
    validation, validation_meta = gib.select_evaluation_qids(
        forward, "validation", validation_fraction=0.2, seed=91
    )
    test, test_meta = gib.select_evaluation_qids(
        reverse, "test", validation_fraction=0.2, seed=91
    )
    validation_again, _ = gib.select_evaluation_qids(
        reverse, "validation", validation_fraction=0.2, seed=91
    )

    assert validation == validation_again
    assert len(validation) == math.ceil(101 * 0.2)
    assert not set(validation) & set(test)
    assert set(validation) | set(test) == set(qids)
    assert validation_meta["beir_comparable_full_test"] is False
    assert test_meta["policy"] == "deterministic_hash_partition_of_official_test"


def test_qrels_tsv_preserves_string_ids_grades_and_max_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\n"
        "001\tdoc-A\t1\n"
        "001\tdoc-A\t3\n"
        "001\tdoc-B\t0\n"
        "q2\tdoc-C\t2.5\n",
        encoding="utf-8",
    )
    loaded = gib.load_qrels_tsv(path)
    assert set(loaded) == {"001", "q2"}
    assert loaded["001"] == {"doc-A": 3.0, "doc-B": 0.0}
    assert loaded["q2"]["doc-C"] == 2.5


def test_projection_is_corpus_only_deterministic_orthonormal_and_deployable() -> None:
    corpus = _normalised_random(240, 12, seed=3)
    query = _normalised_random(4, 12, seed=8)
    first = gib.fit_projection_basis(corpus, 5, 180, 11, 13, svd_iterations=3)
    second = gib.fit_projection_basis(corpus, 5, 180, 11, 13, svd_iterations=3)

    np.testing.assert_array_equal(first.passage_mean, second.passage_mean)
    np.testing.assert_allclose(first.vectors, second.vectors, atol=1e-6)
    np.testing.assert_allclose(first.vectors.T @ first.vectors, np.eye(5), atol=2e-5)
    deployed = gib.project_queries(query, first)
    np.testing.assert_allclose(deployed, query @ first.vectors, atol=1e-6)
    assert not np.allclose(deployed, (query - first.passage_mean) @ first.vectors)
    assert gib.projection_fingerprint(first) == gib.projection_fingerprint(second)


def test_document_centering_changes_scores_only_by_one_query_constant() -> None:
    corpus = _normalised_random(91, 10, seed=14)
    query = _normalised_random(1, 10, seed=15)
    basis = gib.fit_projection_basis(corpus, 6, 80, 1, 2, svd_iterations=3)
    q_projected = gib.project_queries(query, basis)
    uncentred = q_projected @ (corpus @ basis.vectors).T
    centred = q_projected @ gib.project_corpus_chunk(corpus, basis).T
    difference = uncentred - centred
    np.testing.assert_allclose(difference, np.full_like(difference, difference[0, 0]), atol=2e-6)
    np.testing.assert_array_equal(np.argsort(-uncentred), np.argsort(-centred))


def test_projection_basis_and_projected_matrix_round_trip(tmp_path: Path) -> None:
    corpus = _normalised_random(73, 8, seed=4)
    queries = _normalised_random(5, 8, seed=5)
    basis = gib.fit_projection_basis(corpus, 4, 60, 2, 3, svd_iterations=2)
    basis_path = tmp_path / "basis.npz"
    gib.save_projection_basis(basis_path, basis)
    loaded = gib.load_projection_basis(basis_path)
    np.testing.assert_array_equal(loaded.passage_mean, basis.passage_mean)
    np.testing.assert_array_equal(loaded.vectors, basis.vectors)

    corpus_path = tmp_path / "corpus.npy"
    query_path = tmp_path / "queries.npy"
    projected_corpus, corpus_hit = gib.project_matrix_to_npy(
        corpus, corpus_path, loaded, "corpus", chunk_size=11
    )
    projected_queries, query_hit = gib.project_matrix_to_npy(
        queries, query_path, loaded, "query", chunk_size=2
    )
    assert corpus_hit is False and query_hit is False
    np.testing.assert_allclose(
        projected_corpus, gib.project_corpus_chunk(corpus, basis), atol=1e-7
    )
    np.testing.assert_allclose(
        projected_queries, gib.project_queries(queries, basis), atol=1e-7
    )
    _, corpus_hit = gib.project_matrix_to_npy(corpus, corpus_path, loaded, "corpus", 11)
    assert corpus_hit is True


def test_numpy_exact_search_is_exhaustive_and_uses_docid_tie_break() -> None:
    corpus = np.zeros((120, 4), dtype=np.float32)
    corpus[0] = [1.0, 0.0, 0.0, 0.0]
    corpus[1] = [1.0, 0.0, 0.0, 0.0]
    corpus[2] = [0.5, 0.0, 0.0, 0.0]
    queries = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    scores, ids, elapsed, backend = gib.exact_topk_search(
        corpus,
        queries,
        100,
        backend="numpy",
        query_batch_size=1,
        doc_chunk_size=17,
    )
    np.testing.assert_array_equal(ids[0, :3], [0, 1, 2])
    np.testing.assert_allclose(scores[0, :3], [1.0, 1.0, 0.5])
    assert elapsed >= 0.0
    assert backend == "numpy"


def test_beir_identical_query_document_is_removed_with_spare_result() -> None:
    corpus_ids = ["q1", "d1", "d2", "q2", "d3"]
    rankings = np.asarray([[0, 1, 2], [4, 3, 2]], dtype=np.int64)
    filtered = gib.remove_identical_query_documents(
        rankings, ["q1", "q2"], corpus_ids, target_k=2
    )
    np.testing.assert_array_equal(filtered, [[1, 2], [4, 2]])
    with pytest.raises(ValueError, match="lacks a spare"):
        gib.remove_identical_query_documents(
            np.asarray([[0, 1]], dtype=np.int64), ["q1"], corpus_ids, target_k=2
        )


def test_cuda_exact_search_matches_numpy_on_distinct_random_scores() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    corpus = _normalised_random(211, 16, seed=31)
    queries = _normalised_random(7, 16, seed=32)
    numpy_scores, numpy_ids, _, _ = gib.exact_topk_search(
        corpus, queries, 100, backend="numpy", query_batch_size=3, doc_chunk_size=37
    )
    cuda_scores, cuda_ids, _, _ = gib.exact_topk_search(
        corpus, queries, 100, backend="cuda", query_batch_size=3, doc_chunk_size=37
    )
    np.testing.assert_array_equal(cuda_ids, numpy_ids)
    np.testing.assert_allclose(cuda_scores, numpy_scores, atol=2e-6)


def test_metrics_use_linear_graded_ndcg_and_binary_recall() -> None:
    corpus_ids = [f"d{i}" for i in range(100)]
    first = np.arange(100, dtype=np.int64)
    second = np.asarray([0, 50] + [i for i in range(1, 100) if i != 50], dtype=np.int64)
    rankings = np.stack((first, second))
    qrels = {
        "q1": {"d0": 3.0, "d2": 1.0},
        "q2": {"d50": 2.0, "d77": 0.0},
    }
    metrics = gib.evaluate_rankings(rankings, ["q1", "q2"], corpus_ids, qrels)
    expected_q1_ndcg = (3.0 + 1.0 / math.log2(4)) / (
        3.0 + 1.0 / math.log2(3)
    )
    assert metrics["ndcg_at_10"][0] == pytest.approx(expected_q1_ndcg)
    assert metrics["recall_at_10"][0] == 1.0
    assert metrics["recall_at_100"][0] == 1.0
    assert metrics["mrr_at_10"][0] == 1.0
    assert metrics["ndcg_at_10"][1] == pytest.approx(1.0 / math.log2(3))
    assert metrics["recall_at_10"][1] == 1.0
    assert metrics["mrr_at_10"][1] == 0.5


def test_metrics_require_full_recall100_ranking_and_positive_qrels() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        gib.evaluate_rankings(
            np.zeros((1, 10), dtype=np.int64), ["q"], ["d"], {"q": {"d": 1.0}}
        )
    with pytest.raises(ValueError, match="no positive"):
        gib.evaluate_rankings(
            np.zeros((1, 100), dtype=np.int64), ["q"], ["d"], {"q": {"d": 0.0}}
        )


def test_exact_projected_shortlist_reranks_only_supplied_candidates() -> None:
    corpus = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidates = np.asarray([[2, 1, 3], [0, 2, 1]], dtype=np.int64)
    scores, ids, latency = gib.rerank_projected_shortlist(
        corpus, queries, candidates, top_k=2
    )
    np.testing.assert_array_equal(ids, [[1, 2], [2, 1]])
    np.testing.assert_allclose(scores, [[0.9, 0.0], [1.0, 0.1]])
    assert np.all(latency >= 0.0)
    assert 0 not in ids[0]  # Best global document was not a PQ candidate.


def test_bootstrap_is_seeded_and_pairing_operates_on_query_differences() -> None:
    reference = np.asarray([0.0, 0.4, 0.8, 1.0])
    candidate = reference + np.asarray([0.1, 0.1, 0.1, 0.1])
    first = gib.bootstrap_mean_ci(reference, n_bootstrap=500, seed=4)
    second = gib.bootstrap_mean_ci(reference, n_bootstrap=500, seed=4)
    delta = gib.paired_bootstrap_delta(candidate, reference, 500, seed=4)
    assert first == second
    assert first["mean"] == pytest.approx(0.55)
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    assert delta["mean_delta"] == pytest.approx(0.1)
    assert delta["ci_low"] == pytest.approx(0.1)
    assert delta["ci_high"] == pytest.approx(0.1)


def test_method_summary_reports_aligned_deltas_from_raw_exact() -> None:
    raw = {name: np.asarray([0.0, 0.5, 1.0]) for name in gib.METRIC_NAMES}
    better = {name: values + 0.1 for name, values in raw.items()}
    summary, paired = gib.summarise_method_metrics(
        {"raw_exact": raw, "candidate": better}, n_bootstrap=100, seed=9
    )
    assert set(summary) == {"raw_exact", "candidate"}
    for metric in gib.METRIC_NAMES:
        assert paired["candidate_minus_raw_exact"][metric]["mean_delta"] == pytest.approx(
            0.1
        )


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def encode_to_npy(self, texts: list[str], kind: str, output: Path) -> None:
        self.calls.append((list(texts), kind))
        values = np.zeros((len(texts), 4), dtype=np.float32)
        for row in range(len(texts)):
            values[row, row % 4] = 1.0
        np.save(output, values)


def test_embedding_cache_persists_ids_metadata_and_avoids_reencoding(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npy"
    ids = ["a", "b", "c"]
    texts = ["A", "B", "C"]
    fake = _FakeEncoder()
    matrix, hit = gib.load_or_encode_embeddings(
        path, ids, texts, "query", lambda: fake, {"revision": "abc"}
    )
    assert hit is False
    assert fake.calls == [(texts, "query")]
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert json.loads(path.with_suffix(".ids.json").read_text()) == ids
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    assert metadata["revision"] == "abc"
    assert metadata["prefix_recipe"] == gib.PREFIX_RECIPE

    def fail_getter() -> _FakeEncoder:
        raise AssertionError("cache hit must not instantiate the model")

    matrix_again, hit = gib.load_or_encode_embeddings(
        path, ids, texts, "query", fail_getter, {"revision": "abc"}
    )
    assert hit is True
    np.testing.assert_array_equal(matrix_again, matrix)


def test_embedding_cache_with_different_ids_is_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npy"
    first = _FakeEncoder()
    gib.load_or_encode_embeddings(
        path, ["a", "b"], ["A", "B"], "passage", lambda: first, {}
    )
    second = _FakeEncoder()
    matrix_second, hit = gib.load_or_encode_embeddings(
        path, ["b", "a"], ["B", "A"], "passage", lambda: second, {}
    )
    assert hit is False
    assert len(second.calls) == 1
    del matrix_second

    third = _FakeEncoder()
    _, hit = gib.load_or_encode_embeddings(
        path,
        ["b", "a"],
        ["B", "A"],
        "passage",
        lambda: third,
        {"dataset_revision": "new-revision"},
    )
    assert hit is False
    assert len(third.calls) == 1


def test_pq_build_and_projected_rerank_synthetic() -> None:
    pytest.importorskip("faiss")
    corpus = _normalised_random(700, 16, seed=43)
    queries = corpus[[3, 99, 511]]
    index = gib.build_pq_index(
        corpus,
        m=4,
        nbits=4,
        train_size=640,
        train_seed=17,
        chunk_size=113,
        faiss_threads=1,
        iterations=5,
    )
    _, candidates = index.search(np.ascontiguousarray(queries), 100)
    assert int(index.ntotal) == len(corpus)
    assert candidates.shape == (3, 100)
    scores, reranked, _ = gib.rerank_projected_shortlist(
        corpus, queries, candidates, top_k=100
    )
    assert scores.shape == reranked.shape == (3, 100)
    # Each self document should survive this generously sized shortlist and
    # become rank one after exact scoring.
    np.testing.assert_array_equal(reranked[:, 0], [3, 99, 511])


def test_configuration_guards_recall100_and_deployable_pq_geometry() -> None:
    valid = gib.BenchmarkConfig(
        projection_dim=8, pq_m=2, retrieve_k=100, candidate_k=100
    )
    valid.validate(embedding_dim=12, n_docs=120, n_queries=2)
    with pytest.raises(ValueError, match="at least 100"):
        gib.BenchmarkConfig(
            projection_dim=8, pq_m=2, retrieve_k=10, candidate_k=20
        ).validate(12, 120, 2)
    with pytest.raises(ValueError, match="must divide"):
        gib.BenchmarkConfig(
            projection_dim=7, pq_m=2, retrieve_k=100, candidate_k=100
        ).validate(12, 120, 2)


def test_smoke_cli_keeps_real_model_but_scales_system_components() -> None:
    args = gib.build_parser().parse_args(["--smoke"])
    config = gib._config_from_args(args)
    assert config.model_name == gib.DEFAULT_MODEL
    assert config.split == "test"
    assert config.max_queries == 8
    assert config.max_corpus == 1_024
    assert config.projection_dim == 64
    assert config.pq_m == 8
    assert config.retrieve_k == config.candidate_k == 100
    assert config.smoke is True


def test_offline_revision_is_explicit_or_marked_unresolved() -> None:
    assert gib.resolve_hf_revision("unused", "model", "abc123", offline=True) == "abc123"
    assert (
        gib.resolve_hf_revision("unused", "dataset", None, offline=True)
        == "local-cache-unresolved"
    )


def test_dataset_sidecars_cache_ids_qrels_and_revisions(tmp_path: Path) -> None:
    prepared = gib.PreparedDataset(
        name="scifact",
        corpus_ids=["d1", "d2"],
        corpus_texts=["one", "two"],
        query_ids=["q1"],
        query_texts=["query"],
        qrels={"q1": {"d1": 2.0}},
        metadata={
            "dataset_revision": "data-sha",
            "qrels_revision": "qrels-sha",
            "corpus_ids_sha256": gib.ids_fingerprint(["d1", "d2"]),
            "query_ids_sha256": gib.ids_fingerprint(["q1"]),
            "split_policy": {"policy": "synthetic"},
        },
    )
    directory = gib._cache_sidecars(prepared, tmp_path)
    assert json.loads((directory / "corpus_ids.json").read_text()) == ["d1", "d2"]
    assert json.loads((directory / "query_ids.json").read_text()) == ["q1"]
    assert json.loads((directory / "qrels.json").read_text()) == {
        "q1": {"d1": 2.0}
    }
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["dataset_revision"] == "data-sha"
    assert metadata["qrels_revision"] == "qrels-sha"
