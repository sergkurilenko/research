from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from system.packed_ckks import CKKSParameters
from system.retrieval_bench import ProjectionBasis
from system.unified_ckks_bench import (
    UnifiedBenchmarkConfig,
    run_unified_ckks_benchmark,
)


class ExactInnerProductIndex:
    """Tiny in-process client-side PQ stand-in used only by unit tests."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.d = int(self.vectors.shape[1])
        self.ntotal = int(self.vectors.shape[0])

    def search(self, queries: np.ndarray, k: int):
        scores = np.asarray(queries, dtype=np.float32) @ self.vectors.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return (
            np.take_along_axis(scores, order, axis=1).astype(np.float32),
            order.astype(np.int64),
        )


class UnifiedCKKSBenchmarkTests(unittest.TestCase):
    def test_spawned_public_server_and_exact_toy_ranking(self) -> None:
        rng = np.random.default_rng(20260710)
        docs = rng.normal(size=(12, 8)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        qrel_ids = np.asarray([1, 5, 9], dtype=np.int64)
        queries = docs[qrel_ids].copy()
        basis = ProjectionBasis(
            passage_mean=np.zeros((1, 8), dtype=np.float32),
            vectors=np.eye(8, dtype=np.float32),
            fit_sample_size=len(docs),
            fit_sample_seed=0,
            svd_seed=42,
        )
        index = ExactInnerProductIndex(docs)

        with tempfile.TemporaryDirectory(prefix="unified_ckks_test_") as directory:
            docs_path = Path(directory) / "projected_docs.npy"
            np.save(docs_path, docs, allow_pickle=False)
            summary, records = run_unified_ckks_benchmark(
                projected_docs_path=docs_path,
                queries=queries,
                query_indices=np.asarray([10, 11, 12], dtype=np.int64),
                ground_truth_ids=qrel_ids,
                basis=basis,
                pq_index=index,
                config=UnifiedBenchmarkConfig(
                    method="segmented_score_packed",
                    shortlist_k=4,
                    top_k=3,
                    split="toy",
                    warmup_queries=1,
                    bootstrap_samples=200,
                    bootstrap_seed=7,
                ),
                ckks_parameters=CKKSParameters(),
            )

        attestation = summary["server_attestation"]
        self.assertNotEqual(attestation["pid"], os.getpid())
        self.assertEqual(attestation["start_method"], "spawn")
        self.assertTrue(attestation["context_is_public"])
        self.assertFalse(attestation["has_secret_key"])
        self.assertFalse(attestation["has_secret_key_attribute"])
        self.assertFalse(attestation["has_decryptor_attribute"])
        self.assertTrue(attestation["projected_docs_read_only"])

        ckks = summary["methods"]["ckks_projected_shortlist"]
        self.assertEqual(ckks["metrics"]["hit_at_1"]["mean"], 1.0)
        self.assertEqual(
            summary["numerical_audit"]["top_k_exact_order_match"]["mean"],
            1.0,
        )
        self.assertLess(
            summary["numerical_audit"]["max_abs_error"]["max"], 1e-3
        )
        self.assertTrue(
            all(
                row["predictions"]["ckks_projected_shortlist"]
                == row["predictions"]["plaintext_projected_shortlist"]
                for row in records
            )
        )
        self.assertTrue(
            all(row["seal_ciphertext_count"] == 1 for row in records)
        )

    def test_config_rejects_invalid_shortlist(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 <= top_k"):
            UnifiedBenchmarkConfig(shortlist_k=2, top_k=3).validate(
                n_docs=10, n_queries=1, dimension=8
            )


if __name__ == "__main__":
    unittest.main()
