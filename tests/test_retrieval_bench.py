from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from system import retrieval_bench as rb


def _normalised_random(n: int, d: int, seed: int = 7) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def _identity_basis(docs: np.ndarray) -> rb.ProjectionBasis:
    return rb.ProjectionBasis(
        passage_mean=docs.mean(axis=0, keepdims=True).astype(np.float32),
        vectors=np.eye(docs.shape[1], dtype=np.float32),
        fit_sample_size=len(docs),
        fit_sample_seed=0,
        svd_seed=42,
    )


def test_deterministic_query_split_is_stable_disjoint_and_exhaustive() -> None:
    first = rb.deterministic_query_split(500, validation_size=100, seed=2026)
    second = rb.deterministic_query_split(500, validation_size=100, seed=2026)

    np.testing.assert_array_equal(first["validation"], second["validation"])
    np.testing.assert_array_equal(first["test"], second["test"])
    assert len(first["validation"]) == 100
    assert len(first["test"]) == 400
    assert not set(first["validation"]) & set(first["test"])
    assert set(np.concatenate([first["validation"], first["test"]])) == set(range(500))


def test_select_query_indices_uses_deterministic_random_prefix() -> None:
    selected = rb.select_query_indices(50, "validation", 10, 99, max_queries=4)
    expected = np.random.default_rng(99).permutation(50)[:4]
    np.testing.assert_array_equal(selected, expected)


def test_deterministic_self_qrels_matches_source_recipe() -> None:
    actual = rb.deterministic_self_qrels(1_000_000, 500, seed=42)
    expected = np.random.default_rng(42).choice(
        1_000_000, size=500, replace=False
    )
    np.testing.assert_array_equal(actual, expected)
    assert len(np.unique(actual)) == 500


def test_smoke_subset_has_exact_size_and_keeps_every_qrel() -> None:
    required = np.asarray([7, 91, 403, 998], dtype=np.int64)
    first = rb.deterministic_corpus_subset(1_000, required, 127, seed=31)
    second = rb.deterministic_corpus_subset(1_000, required, 127, seed=31)

    assert len(first) == 127
    assert set(required).issubset(first)
    np.testing.assert_array_equal(first, second)


def test_online_query_is_uncentred_by_default_and_centering_is_explicit() -> None:
    basis = rb.ProjectionBasis(
        passage_mean=np.asarray([[10.0, -3.0, 2.0]], dtype=np.float32),
        vectors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32
        ),
        fit_sample_size=10,
        fit_sample_seed=0,
        svd_seed=42,
    )
    query = np.asarray([[11.0, -1.0, 9.0]], dtype=np.float32)

    deployable = rb.project_queries(query, basis)
    legacy_ablation = rb.project_queries(query, basis, center_online_queries=True)

    np.testing.assert_allclose(deployable, [[11.0, -1.0]])
    np.testing.assert_allclose(legacy_ablation, [[1.0, 2.0]])
    assert not np.array_equal(deployable, legacy_ablation)


def test_document_centering_only_adds_query_constant_to_projected_scores() -> None:
    docs = _normalised_random(31, 8)
    query = _normalised_random(1, 8, seed=19)
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(8, 5)))
    basis = rb.ProjectionBasis(
        passage_mean=docs.mean(axis=0, keepdims=True),
        vectors=q.astype(np.float32),
        fit_sample_size=len(docs),
        fit_sample_seed=0,
        svd_seed=42,
    )

    projected_query = rb.project_queries(query, basis)
    uncentred_doc_scores = projected_query @ (docs @ basis.vectors).T
    centred_doc_scores = projected_query @ rb.project_doc_chunk(docs, basis).T
    score_shift = uncentred_doc_scores - centred_doc_scores

    np.testing.assert_allclose(
        score_shift, np.full_like(score_shift, score_shift[0, 0]), atol=2e-6
    )
    np.testing.assert_array_equal(
        np.argsort(-uncentred_doc_scores), np.argsort(-centred_doc_scores)
    )


def test_load_memmap_keeps_npy_out_of_ram(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npy"
    expected = _normalised_random(23, 6)
    np.save(path, expected)

    loaded = rb.load_memmap(path)

    assert isinstance(loaded, np.memmap)
    assert loaded.mode == "r"
    np.testing.assert_array_equal(loaded, expected)


def test_fit_projection_basis_is_deterministic_and_orthonormal() -> None:
    docs = _normalised_random(180, 12)
    first = rb.fit_projection_basis(docs, 5, 120, sample_seed=2, svd_seed=11)
    second = rb.fit_projection_basis(docs, 5, 120, sample_seed=2, svd_seed=11)

    np.testing.assert_array_equal(first.passage_mean, second.passage_mean)
    np.testing.assert_allclose(first.vectors, second.vectors, atol=1e-6)
    np.testing.assert_allclose(
        first.vectors.T @ first.vectors, np.eye(5), atol=2e-5
    )
    assert rb.basis_fingerprint(first) == rb.basis_fingerprint(second)


def test_exact_projected_numpy_search_accepts_memmap(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs.npy"
    docs = _normalised_random(91, 10)
    np.save(docs_path, docs)
    docs_mmap = rb.load_memmap(docs_path)
    selected = np.asarray([3, 22, 71], dtype=np.int64)
    queries = docs[selected]
    basis = _identity_basis(docs)
    projected_queries = rb.project_queries(queries, basis)

    _, ids, elapsed_ms, backend = rb.exact_projected_search(
        docs_mmap,
        projected_queries,
        basis,
        top_k=5,
        chunk_size=13,
        backend="numpy",
    )

    np.testing.assert_array_equal(ids[:, 0], selected)
    assert elapsed_ms >= 0.0
    assert backend == "numpy"


def test_build_pq_index_chunked_trains_and_adds_all_rows() -> None:
    pytest.importorskip("faiss")
    docs = _normalised_random(256, 8)
    basis = _identity_basis(docs)

    index = rb.build_pq_index_chunked(
        docs,
        basis,
        m=2,
        nbits=2,
        train_size=128,
        train_seed=42,
        chunk_size=37,
    )

    assert int(index.d) == 8
    assert int(index.ntotal) == len(docs)
    projected_query = rb.project_queries(docs[:1], basis)
    scores, ids = index.search(projected_query, 10)
    assert scores.shape == (1, 10)
    assert ids.shape == (1, 10)
    assert np.all(ids >= 0)


def test_bootstrap_ci_is_seeded_and_contains_observed_mean() -> None:
    values = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    first = rb.bootstrap_mean_ci(values, n_bootstrap=500, seed=8)
    second = rb.bootstrap_mean_ci(values, n_bootstrap=500, seed=8)

    assert first == second
    assert first["mean"] == pytest.approx(0.6)
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_projected_shortlist_rerank_is_exact_plaintext_kernel_reference() -> None:
    docs = np.asarray(
        [
            [2.0, 0.0, 99.0, -8.0],
            [1.0, 3.0, -50.0, 2.0],
            [-1.0, 0.0, 500.0, 7.0],
            [0.0, 2.0, -12.0, 1.0],
        ],
        dtype=np.float32,
    )
    basis = rb.ProjectionBasis(
        passage_mean=np.asarray([[0.5, -0.5, 10.0, 4.0]], dtype=np.float32),
        vectors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
        fit_sample_size=4,
        fit_sample_seed=0,
        svd_seed=42,
    )
    projected_queries = np.asarray([[1.0, 2.0]], dtype=np.float32)
    candidates = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    config = rb.BenchmarkConfig(
        model="synthetic-projected",
        projection_dim=2,
        pq_m=1,
        shortlist_k=4,
        top_k=2,
        exact_backend="numpy",
        bootstrap_samples=10,
    )

    predictions, latency, payload = rb.rerank_projected_shortlist(
        docs, projected_queries, candidates, basis, config
    )

    # Scores for (d - mu)[:2] dot [1, 2] rank document 1, then document 3.
    np.testing.assert_array_equal(predictions, [[1, 3]])
    assert payload["request_candidate_ids_bytes"][0] == 4 * 8
    assert payload["counterfactual_projected_vectors_response_bytes"][0] == 4 * 2 * 4
    for key in (
        "server_gather_ms",
        "server_project_ms",
        "plaintext_projected_rerank_ms",
        "computational_reference_end_to_end_ms",
    ):
        assert latency[key][0] >= 0.0


def test_unified_protocols_share_queries_and_return_vector_payload_is_exact() -> None:
    faiss = pytest.importorskip("faiss")
    docs = _normalised_random(96, 8)
    basis = _identity_basis(docs)
    projected_docs = rb.project_doc_chunk(docs, basis)
    index = faiss.IndexFlatIP(projected_docs.shape[1])
    index.add(np.ascontiguousarray(projected_docs))

    local_query_rows = np.asarray([2, 17, 41, 88], dtype=np.int64)
    corpus_ids = np.arange(1_000, 1_000 + len(docs), dtype=np.int64)
    queries = docs[local_query_rows]
    qrels = corpus_ids[local_query_rows]
    query_indices = np.asarray([101, 102, 103, 104], dtype=np.int64)
    config = rb.BenchmarkConfig(
        model="synthetic",
        projection_dim=8,
        pq_m=2,
        pq_nbits=2,
        shortlist_k=20,
        top_k=5,
        exact_backend="numpy",
        doc_chunk_size=17,
        bootstrap_samples=200,
        bootstrap_seed=55,
        fit_sample_size=len(docs),
    )

    summary, records = rb.run_protocol_benchmarks(
        docs,
        queries,
        qrels,
        query_indices,
        basis,
        index,
        config,
        corpus_ids=corpus_ids,
        split_name="validation",
    )

    for method in (
        "exact_projected",
        "pq_only",
        "return_full_vectors",
        "projected_shortlist_rerank",
    ):
        assert summary["methods"][method]["metrics"]["hit_at_1"]["mean"] == 1.0
    assert summary["protocol"]["query_centered"] is False
    assert summary["protocol"]["online_query_projection"].startswith("query @")
    assert len(records) == len(queries) * 4

    returned = [r for r in records if r["method"] == "return_full_vectors"]
    assert {r["query_index"] for r in returned} == set(query_indices)
    for record in returned:
        assert record["payload_bytes"]["request_bytes"] == 20 * 8
        assert record["payload_bytes"]["response_bytes"] == 20 * 8 * 4
        assert len(record["candidate_ids"]) == 20
        assert record["metrics"]["hit_at_1"] == 1.0

    paired = summary["paired_query_bootstrap_deltas"]
    assert (
        paired["return_full_vectors_minus_exact_projected"]["hit_at_1"]["mean"]
        == 0.0
    )
    assert (
        paired["projected_shortlist_rerank_minus_exact_projected"]["hit_at_1"][
            "mean"
        ]
        == 0.0
    )

    projected = [
        r for r in records if r["method"] == "projected_shortlist_rerank"
    ]
    assert len(projected) == len(queries)
    for record in projected:
        assert len(record["candidate_ids"]) == 20
        assert record["computational_reference"] is True
        assert record["payload_is_counterfactual_return_vectors_baseline"] is True
        assert record["payload_bytes"]["request_candidate_ids_bytes"] == 20 * 8
        assert (
            record["payload_bytes"][
                "counterfactual_projected_vectors_response_bytes"
            ]
            == 20 * 8 * 4
        )
        assert record["latency_ms"]["server_gather_ms"] >= 0.0
        assert record["latency_ms"]["server_project_ms"] >= 0.0
        assert record["latency_ms"]["plaintext_projected_rerank_ms"] >= 0.0

    assert "computational reference" in summary["protocol"]["method_semantics"][
        "projected_shortlist_rerank"
    ]
    assert summary["protocol"]["projected_shortlist_payload_model"].startswith(
        "counterfactual baseline only"
    )


def test_transport_estimate_is_zero_unless_network_is_configured() -> None:
    local = rb.BenchmarkConfig(network_mbps=None)
    wan = rb.BenchmarkConfig(network_mbps=100.0, network_rtt_ms=10.0)

    assert rb._transport_latency_ms(125_000, local) == 0.0
    assert rb._transport_latency_ms(125_000, wan) == pytest.approx(20.0)


def test_primary_model_defaults_to_e5_base_uncentred_protocol() -> None:
    parser = rb.build_parser()
    args = parser.parse_args([])
    rb._apply_model_defaults(args)

    assert args.model == "e5-base"
    assert args.projection_dim == 672
    assert args.pq_m == 96
    assert args.center_online_queries is False
    assert args.split == "validation"
