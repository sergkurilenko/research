from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from system.graded_ckks_replay import (
    ReplayConfig,
    load_replay_artifacts,
    replay_from_files,
)
from system.graded_ir_bench import ProjectionBasis, save_projection_basis
from system.packed_ckks import CKKSParameters


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class GradedCKKSReplayTests(unittest.TestCase):
    def _build_toy_artifacts(self, root: Path) -> tuple[Path, Path]:
        corpus_ids = [f"d{index:03d}" for index in range(120)]
        query_ids = ["q-plus", "q-minus"]
        qrels = {
            "q-plus": {corpus_ids[-1]: 2.0},
            "q-minus": {corpus_ids[0]: 1.0},
        }
        sidecars = root / "cache" / "data" / "scifact" / "toy"
        _write_json(sidecars / "corpus_ids.json", corpus_ids)
        _write_json(sidecars / "query_ids.json", query_ids)
        _write_json(sidecars / "qrels.json", qrels)

        projection = root / "cache" / "projection" / "scifact" / "toy"
        projection.mkdir(parents=True, exist_ok=True)
        docs = np.zeros((120, 4), dtype=np.float32)
        docs[:, 0] = np.linspace(-2.0, 2.0, len(docs), dtype=np.float32)
        queries = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.save(projection / "corpus.npy", docs, allow_pickle=False)
        np.save(projection / "queries-toy.npy", queries, allow_pickle=False)
        basis = ProjectionBasis(
            passage_mean=np.zeros((1, 4), dtype=np.float32),
            vectors=np.eye(4, dtype=np.float32),
            fit_sample_size=120,
            fit_sample_seed=17,
            svd_seed=42,
            svd_iterations=5,
        )
        save_projection_basis(projection / "basis.npz", basis)

        aggregate = {
            "schema_version": 1,
            "datasets": {
                "scifact": {
                    "dataset": {"name": "scifact"},
                    "config": {"candidate_k": 100, "retrieve_k": 100},
                    "cache": {
                        "data_sidecars": str(sidecars),
                        "projection_basis": str(projection / "basis.npz"),
                        "query_embeddings": str(
                            root / "cache" / "embeddings" / "queries-toy.npy"
                        ),
                    },
                }
            },
        }
        aggregate_path = root / "graded.json"
        _write_json(aggregate_path, aggregate)

        plus = list(reversed(corpus_ids[-100:]))
        minus = list(corpus_ids[:100])
        records = []
        for qid, ranking in (("q-plus", plus), ("q-minus", minus)):
            for method in ("pq_only", "pq_projected_rerank"):
                records.append(
                    {
                        "dataset": "scifact",
                        "query_id": qid,
                        "method": method,
                        "ranked_doc_ids": ranking,
                    }
                )
        jsonl_path = root / "graded.jsonl"
        jsonl_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return aggregate_path, jsonl_path

    def test_actual_spawn_public_only_and_exact_toy_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="graded_ckks_replay_") as directory:
            aggregate, jsonl = self._build_toy_artifacts(Path(directory))
            summary, records = replay_from_files(
                aggregate,
                jsonl,
                ReplayConfig(
                    dataset="scifact",
                    warmup_queries=1,
                    bootstrap_samples=100,
                    bootstrap_seed=9,
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

        self.assertEqual(
            summary["metrics"]["ckks_segmented_score_packed"]["ndcg_at_10"][
                "mean"
            ],
            1.0,
        )
        self.assertEqual(
            summary["numerical_audit"]["top100_exact_order_match"]["mean"],
            1.0,
        )
        self.assertEqual(
            summary["numerical_audit"]["top10_exact_order_match"]["mean"],
            1.0,
        )
        self.assertLess(
            summary["numerical_audit"]["max_abs_error"]["max"], 1e-3
        )
        self.assertTrue(all(row["seal_ciphertext_count"] == 1 for row in records))
        self.assertTrue(
            all(
                row["ranked_doc_ids"] == row["plaintext_reference_doc_ids"]
                for row in records
            )
        )

    def test_loader_rejects_non_frozen_candidate_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="graded_ckks_replay_") as directory:
            aggregate, jsonl = self._build_toy_artifacts(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            payload["datasets"]["scifact"]["config"]["candidate_k"] = 101
            _write_json(aggregate, payload)
            with self.assertRaisesRegex(ValueError, "candidate_k=retrieve_k=100"):
                load_replay_artifacts(aggregate, jsonl, "scifact")


if __name__ == "__main__":
    unittest.main()
