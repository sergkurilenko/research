from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from system import vector_return_baselines as vrb


def _normalised_random(n: int, d: int, seed: int = 9) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def test_float_payloads_are_really_serialized_and_have_exact_wire_sizes() -> None:
    vectors = np.asarray(
        [[0.1, -0.2, 0.3], [1.25, -2.5, 0.0]], dtype=np.float32
    )

    payload32 = vrb.encode_vector_payload(vectors, "float32")
    decoded32 = vrb.decode_vector_payload(payload32, "float32", 2, 3)
    payload16 = vrb.encode_vector_payload(vectors, "float16")
    decoded16 = vrb.decode_vector_payload(payload16, "float16", 2, 3)

    assert isinstance(payload32, bytes)
    assert isinstance(payload16, bytes)
    assert len(payload32) == 2 * 3 * 4
    assert len(payload16) == 2 * 3 * 2
    np.testing.assert_array_equal(decoded32, vectors)
    np.testing.assert_array_equal(decoded16, vectors.astype(np.float16).astype(np.float32))


def test_per_vector_symmetric_int8_layout_scales_and_error_bound() -> None:
    vectors = np.asarray(
        [
            [-2.0, -0.3, 0.1, 1.5],
            [0.01, -0.02, 0.03, -0.04],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    payload = vrb.encode_vector_payload(vectors, "symmetric_int8_per_vector")
    decoded = vrb.decode_vector_payload(
        payload, "symmetric_int8_per_vector", 3, 4
    )

    assert len(payload) == 3 * 4 + 3 * 4  # int8 codes + float32 scales
    expected_scales = np.asarray([2.0 / 127.0, 0.04 / 127.0, 1.0])
    error = np.max(np.abs(vectors - decoded), axis=1)
    assert error[0] <= expected_scales[0] / 2 + 1e-6
    assert error[1] <= expected_scales[1] / 2 + 1e-6
    assert error[2] == 0.0


def test_decoder_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError, match="expected"):
        vrb.decode_vector_payload(b"\x00" * 7, "float32", 1, 2)
    with pytest.raises(ValueError, match="expected"):
        vrb.decode_vector_payload(b"\x00" * 7, "symmetric_int8_per_vector", 1, 4)


def _write_frozen_log(path: Path, *, mismatch: bool = False) -> None:
    rows = []
    for query_index, gt, candidates in (
        (3, 13, [13, 4, 9, 2]),
        (7, 27, [27, 1, 8, 5]),
    ):
        rows.append(
            {
                "query_index": query_index,
                "split": "test",
                "ground_truth_id": gt,
                "method": "exact_projected",
                "predictions": candidates[:2],
            }
        )
        rows.append(
            {
                "query_index": query_index,
                "split": "test",
                "ground_truth_id": gt,
                "method": "pq_only",
                "candidate_ids": candidates,
            }
        )
        repeated = list(candidates)
        if mismatch and query_index == 7:
            repeated[-1] = 6
        rows.append(
            {
                "query_index": query_index,
                "split": "test",
                "ground_truth_id": gt,
                "method": "return_full_vectors",
                "candidate_ids": repeated,
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_frozen_loader_preserves_order_and_cross_checks_candidate_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frozen.jsonl"
    _write_frozen_log(path)

    frozen = vrb.load_frozen_queries(path, shortlist_k=4, expected_queries=2)

    assert [query.query_index for query in frozen] == [3, 7]
    assert [query.ground_truth_id for query in frozen] == [13, 27]
    np.testing.assert_array_equal(frozen[0].candidate_ids, [13, 4, 9, 2])
    assert vrb.frozen_candidates_sha256(frozen) == vrb.frozen_candidates_sha256(
        frozen
    )

    bad = tmp_path / "bad.jsonl"
    _write_frozen_log(bad, mismatch=True)
    with pytest.raises(ValueError, match="differ across frozen methods"):
        vrb.load_frozen_queries(bad, shortlist_k=4, expected_queries=2)


def test_execute_method_measures_real_payload_and_ranks_decoded_vectors() -> None:
    docs = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [-1.0, 0.0]], dtype=np.float32
    )
    candidates = np.asarray([1, 2, 0, 3], dtype=np.int64)
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    execution = vrb.execute_method(
        "raw_float16_return", candidates, query, docs, top_k=2
    )

    np.testing.assert_array_equal(execution.predictions, [0, 2])
    assert execution.payload_bytes["request_candidate_ids_bytes"] == 4 * 8
    assert execution.payload_bytes["response_vector_bytes"] == 4 * 2 * 2
    assert execution.payload_bytes["total_application_bytes"] == 48
    for value in execution.latency_ms.values():
        assert value >= 0.0


def test_numerical_comparison_detects_exact_order_and_score_error() -> None:
    reference = vrb.MethodExecution(
        method="projected_float32_return",
        scores=np.asarray([1.0, 0.8, 0.2], dtype=np.float32),
        predictions=np.asarray([10, 11], dtype=np.int64),
        latency_ms={},
        payload_bytes={},
    )
    approximate = vrb.MethodExecution(
        method="projected_float16_return",
        scores=np.asarray([1.0, 0.79, 0.21], dtype=np.float32),
        predictions=np.asarray([10, 11], dtype=np.int64),
        latency_ms={},
        payload_bytes={},
    )

    comparison = vrb.numerical_comparison(approximate, reference)

    assert comparison["reference_method"] == "projected_float32_return"
    assert comparison["score_mae"] > 0.0
    assert comparison["top1_match"] is True
    assert comparison["top10_set_match"] is True
    assert comparison["top10_exact_order_match"] is True


def _small_audit_fixture() -> tuple[
    list[vrb.FrozenQuery], np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    raw_docs = _normalised_random(40, 6)
    mean = raw_docs.mean(axis=0, keepdims=True).astype(np.float32)
    rng = np.random.default_rng(44)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 4)))
    basis = basis.astype(np.float32)
    projected_docs = ((raw_docs - mean) @ basis).astype(np.float32)
    raw_queries = np.stack([raw_docs[3], raw_docs[19]]).astype(np.float32)
    frozen = [
        vrb.FrozenQuery(0, 3, np.asarray([3, 1, 9, 12, 20, 7], dtype=np.int64)),
        vrb.FrozenQuery(1, 19, np.asarray([19, 5, 6, 8, 13, 31], dtype=np.int64)),
    ]
    return frozen, raw_queries, raw_docs, projected_docs, basis


def test_full_small_audit_has_five_disclosure_methods_and_paired_bootstrap() -> None:
    frozen, raw_queries, raw_docs, projected_docs, basis = _small_audit_fixture()
    config = vrb.AuditConfig(
        shortlist_k=6,
        top_k=3,
        expected_queries=2,
        bootstrap_samples=100,
        bootstrap_seed=4,
        validate_projection_vectors=2,
    )

    records = vrb.run_audit(
        frozen, raw_queries, raw_docs, projected_docs, basis, config
    )

    assert len(records) == 2 * 5
    assert {record["method"] for record in records} == set(vrb.METHODS)
    assert all(record["is_privacy_protocol"] is False for record in records)
    projected32 = [
        record for record in records if record["method"] == "projected_float32_return"
    ]
    for record in projected32:
        assert record["numerical_vs_float32"]["score_max_abs"] == 0.0
        assert record["numerical_vs_float32"]["top10_exact_order_match"] is True
        assert record["payload_bytes"]["response_vector_bytes"] == 6 * 4 * 4

    int8 = [
        record
        for record in records
        if record["method"] == "projected_int8_symmetric_return"
    ]
    assert all(
        record["payload_bytes"]["response_vector_bytes"] == 6 * 4 + 6 * 4
        for record in int8
    )

    summary = vrb.build_summary(
        records,
        config,
        frozen,
        input_metadata={},
        projection_validation={"checked_vectors": 2},
        basis_metadata={"raw_dimension": 6, "projected_dimension": 4},
    )
    assert "not privacy protocols" in summary["privacy_statement"]
    assert set(summary["methods"]) == set(vrb.METHODS)
    paired = summary["paired_query_bootstrap_metric_deltas"]
    assert "projected_int8_minus_projected_float32" in paired
    assert "raw_float16_minus_raw_float32" in paired


def test_projected_cache_validation_uses_centered_docs_and_basis() -> None:
    frozen, _, raw_docs, projected_docs, basis = _small_audit_fixture()
    mean = raw_docs.mean(axis=0, keepdims=True).astype(np.float32)

    result = vrb.validate_projected_cache(
        raw_docs, projected_docs, mean, basis, frozen, n_vectors=4, atol=1e-6
    )

    assert result["checked_vectors"] == 4
    assert result["max_abs_error"] == pytest.approx(0.0, abs=1e-6)
    corrupted = projected_docs.copy()
    corrupted[frozen[0].candidate_ids[0], 0] += 0.1
    with pytest.raises(ValueError, match="does not match"):
        vrb.validate_projected_cache(
            raw_docs, corrupted, mean, basis, frozen, n_vectors=4, atol=1e-6
        )


def test_cli_defaults_target_frozen_k100_test_outputs() -> None:
    args = vrb.build_parser().parse_args([])

    assert args.shortlist_k == 100
    assert args.top_k == 10
    assert args.expected_queries == 400
    assert args.frozen_query_log.name == "retrieval_e5base_k672_K100_test.jsonl"
    assert args.summary_json.name == "vector_return_baselines_e5base_K100_test.json"
    assert args.query_log_jsonl.name == "vector_return_baselines_e5base_K100_test.jsonl"
