"""Expanded IR experiments for the journal revision.

This module deliberately reuses the immutable per-query records and embedding
caches produced by :mod:`system.graded_ir_bench`.  It adds four auditable
stages that can be resumed independently:

``splits``
    Materialise the confirmatory query identifiers.  Official dev/train qrels
    are validation whenever they exist; test-only collections use the frozen
    20/80 SHA-256 partition.  Existing actual-CKKS records are subset without
    replay because each record is a complete, query-independent request.

``pareto``
    Evaluate nested corpus-only SVD prefixes at 192, 256, 384, 512, 672 and
    the full-rank 768-dimensional centred geometry.  The first 672 components
    are exactly the frozen basis used by the paper.  A PQ point with eight
    dimensions per subquantizer is evaluated at each dimension.

``controls``
    Compare SVD with a seeded orthogonal random projection and coordinate
    truncation at representative dimensions.

``pq``
    Evaluate K, M/code-byte and PQ-training-seed sensitivity at d=672.

The script never changes the manuscript.  All scientific outputs are written
under ``results/system_revision/ir_expansion`` and all large reusable indices
under the external graded-IR cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from system import graded_ir_bench as gib
except ImportError:  # pragma: no cover - direct ``python system/foo.py`` use
    import graded_ir_bench as gib  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "system_revision" / "ir_expansion"
DEFAULT_MASTER = ROOT / "results" / "system_revision" / "graded_ir_official_test.json"
DEFAULT_VALIDATION_MASTER = (
    ROOT / "results" / "system_revision" / "graded_ir_validation.json"
)
DEFAULT_CACHE = Path(os.environ.get("GRADED_IR_CACHE", ROOT / "cache" / "graded_ir"))
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", ROOT / "cache" / "huggingface"))
DIMENSIONS = (192, 256, 384, 512, 672, 768)
CONTROL_DIMENSIONS = (384, 672)
K_VALUES = (20, 50, 100, 200)
PQ_SEEDS = (17, 42, 2026)
PQ_TIMING_REPEATS = 5
NI_MARGIN_NDCG = 0.002
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def fingerprint(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()[:length]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{line_number}") from exc


def percentile_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bootstrap_delta(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
    margin: float = NI_MARGIN_NDCG,
) -> dict[str, Any]:
    """Paired bootstrap interval and revision-plan decision rules.

    The frozen revision plan defines non-inferiority from the lower endpoint
    of the *two-sided* ``1-alpha`` interval.  Equivalence is reported only if
    that complete interval lies inside ``[-margin, margin]``.  Merely covering
    zero is never interpreted as evidence of equivalence.
    """

    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape or candidate_array.ndim != 1:
        raise ValueError("paired samples must be aligned one-dimensional arrays")
    if not len(candidate_array):
        raise ValueError("paired samples cannot be empty")
    differences = candidate_array - reference_array
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    block = 256
    for start in range(0, samples, block):
        stop = min(start + block, samples)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indices].mean(axis=1)
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "count": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "paired_two_sided_confidence_level": float(1.0 - alpha),
        "paired_two_sided_ci": [lower, upper],
        "non_inferiority_margin": float(margin),
        "non_inferior": bool(lower > -margin),
        "equivalent_within_symmetric_margin": bool(lower > -margin and upper < margin),
        "interpretation": (
            "Non-inferiority uses the lower endpoint; equivalence requires the complete "
            "interval inside the symmetric margin. An interval containing zero alone "
            "is inconclusive about equivalence and no rule establishes superiority."
        ),
    }


def read_qrels_files(dataset_summary: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    files = dataset_summary["dataset"]["qrels_files"]
    return {str(split): gib.load_qrels_tsv(Path(path)) for split, path in files.items()}


def canonical_arguana_missing(dataset_summary: Mapping[str, Any]) -> list[str]:
    policy = dataset_summary["dataset"]["split_policy"]
    audit = policy.get("missing_positive_qrel_audit", {})
    if not audit or not audit.get("excluded_query_count"):
        return []
    qrels = read_qrels_files(dataset_summary)["test"]
    corpus_ids = set(load_json(Path(dataset_summary["cache"]["data_sidecars"]) / "corpus_ids.json"))
    return sorted(
        qid
        for qid, judgments in qrels.items()
        if any(float(grade) > 0 and docid not in corpus_ids for docid, grade in judgments.items())
    )


def method_metric_records(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for row in iter_jsonl(path):
        result.setdefault(str(row["method"]), {})[str(row["query_id"])] = {
            str(metric): float(value) for metric, value in row["metrics"].items()
        }
    return result


def summarise_selected_records(
    records: Mapping[str, Mapping[str, Mapping[str, float]]],
    selected_qids: Sequence[str],
    *,
    zero_qids: Sequence[str] = (),
) -> dict[str, Any]:
    selected = list(map(str, selected_qids))
    zero = set(map(str, zero_qids))
    output: dict[str, Any] = {}
    for method, rows in records.items():
        metric_names = sorted(next(iter(rows.values())))
        output[method] = {}
        for metric in metric_names:
            values = []
            for qid in selected:
                if qid in zero:
                    values.append(0.0)
                elif qid in rows:
                    values.append(float(rows[qid][metric]))
                else:
                    raise ValueError(f"selected query {qid} absent from {method} records")
            output[method][metric] = {
                "mean": float(np.mean(values)),
                "count": len(values),
            }
    return output


def summarise_ckks_subset(
    path: Path,
    selected_qids: Sequence[str],
    *,
    canonical_zero_qids: Sequence[str] = (),
) -> dict[str, Any]:
    wanted = set(map(str, selected_qids))
    rows = [row for row in iter_jsonl(path) if str(row["query_id"]) in wanted]
    found = {str(row["query_id"]) for row in rows}
    if found != wanted:
        raise ValueError(f"CKKS subset mismatch: missing={sorted(wanted - found)[:5]}")
    metric_methods = sorted(rows[0]["metrics"])
    zero_count = len(canonical_zero_qids)
    metrics_evaluable: dict[str, Any] = {}
    metrics_canonical: dict[str, Any] = {}
    for method in metric_methods:
        metrics_evaluable[method] = {}
        metrics_canonical[method] = {}
        for metric in sorted(rows[0]["metrics"][method]):
            values = [float(row["metrics"][method][metric]) for row in rows]
            metrics_evaluable[method][metric] = {
                "mean": float(np.mean(values)), "count": len(values)
            }
            canonical_values = values + [0.0] * zero_count
            metrics_canonical[method][metric] = {
                "mean": float(np.mean(canonical_values)),
                "count": len(canonical_values),
            }
    ckks = [float(row["metrics"]["ckks_segmented_score_packed"]["ndcg_at_10"]) for row in rows]
    plain = [float(row["metrics"]["pq_projected_rerank"]["ndcg_at_10"]) for row in rows]
    canonical_ckks = ckks + [0.0] * zero_count
    canonical_plain = plain + [0.0] * zero_count
    latency_keys = sorted(rows[0]["latency_ms"])
    return {
        "selected_query_count": len(rows),
        "canonical_query_count": len(rows) + zero_count,
        "canonical_zero_query_ids": list(canonical_zero_qids),
        "selected_query_ids_sha256": gib.ids_fingerprint(sorted(wanted)),
        "metrics_canonical_primary": metrics_canonical,
        "metrics_evaluable_sensitivity": metrics_evaluable,
        "ckks_minus_plaintext_ndcg_at_10_canonical_primary": bootstrap_delta(
            canonical_ckks, canonical_plain
        ),
        "ckks_minus_plaintext_ndcg_at_10_evaluable_sensitivity": bootstrap_delta(
            ckks, plain
        ),
        "latency_ms": {
            key: percentile_summary([float(row["latency_ms"][key]) for row in rows])
            for key in latency_keys
        },
        "subset_validity": (
            "Exact subset of independent per-query CKKS requests; no cross-query state, "
            "training, aggregation, or cryptographic replay is required."
        ),
    }


def run_splits(master_path: Path, results_dir: Path) -> dict[str, Any]:
    master = load_json(master_path)
    output: dict[str, Any] = {
        "schema": "ir_expansion.confirmatory_splits.v1",
        "created_utc": utc_now(),
        "split_seed": 2026,
        "validation_fraction": 0.2,
        "datasets": {},
    }
    splits_dir = results_dir / "splits"
    for dataset in gib.REQUIRED_DATASETS:
        summary = master["datasets"][dataset]
        qrels_by_split = read_qrels_files(summary)
        validation_ids, validation_policy = gib.select_evaluation_qids(
            qrels_by_split, "validation", validation_fraction=0.2, seed=2026
        )
        test_ids, test_policy = gib.select_evaluation_qids(
            qrels_by_split, "test", validation_fraction=0.2, seed=2026
        )
        missing = canonical_arguana_missing(summary)
        atomic_text(splits_dir / f"{dataset}_validation_qids.txt", "\n".join(validation_ids) + "\n")
        atomic_text(splits_dir / f"{dataset}_confirmatory_qids.txt", "\n".join(test_ids) + "\n")
        records_path = ROOT / "results" / "system_revision" / f"graded_ir_official_test_{dataset}.jsonl"
        records = method_metric_records(records_path)
        selected_missing = sorted(set(test_ids) & set(missing))
        metrics = summarise_selected_records(records, test_ids, zero_qids=selected_missing)
        ckks_path = ROOT / "results" / "system_revision" / f"graded_ckks_replay_{dataset}_full.jsonl"
        evaluable_test_ids = [qid for qid in test_ids if qid not in set(missing)]
        ckks = summarise_ckks_subset(
            ckks_path,
            evaluable_test_ids,
            canonical_zero_qids=selected_missing,
        )
        output["datasets"][dataset] = {
            "validation": {
                "count": len(validation_ids),
                "ids_sha256": gib.ids_fingerprint(validation_ids),
                "policy": validation_policy,
                "ids_file": str((splits_dir / f"{dataset}_validation_qids.txt").resolve()),
            },
            "confirmatory": {
                "canonical_count": len(test_ids),
                "evaluable_count": len(evaluable_test_ids),
                "ids_sha256": gib.ids_fingerprint(test_ids),
                "policy": test_policy,
                "ids_file": str((splits_dir / f"{dataset}_confirmatory_qids.txt").resolve()),
                "canonical_missing_positive_qids": selected_missing,
                "canonical_metric_policy": (
                    "Queries whose sole positive document is absent from the official corpus "
                    "are retained with zero metrics."
                    if selected_missing else "All selected positive judgments exist in the corpus."
                ),
                "ir_metrics": metrics,
                "actual_ckks_evaluable_subset": ckks,
            },
        }
    atomic_json(results_dir / "confirmatory_splits.json", output)
    return output


def load_assets(summary: Mapping[str, Any]) -> dict[str, Any]:
    cache = summary["cache"]
    sidecars = Path(cache["data_sidecars"])
    basis = gib.load_projection_basis(Path(cache["projection_basis"]))
    return {
        "corpus": np.load(cache["corpus_embeddings"], mmap_mode="r", allow_pickle=False),
        "queries": np.load(cache["query_embeddings"], mmap_mode="r", allow_pickle=False),
        "proj672_corpus": np.load(
            str(Path(cache["projection_basis"]).parent / "corpus.npy"), mmap_mode="r", allow_pickle=False
        ),
        "proj672_queries": np.load(
            str(Path(cache["projection_basis"]).parent / f"queries-{summary['dataset']['query_ids_sha256'][:16]}.npy"),
            mmap_mode="r",
            allow_pickle=False,
        ),
        "basis": basis,
        "corpus_ids": load_json(sidecars / "corpus_ids.json"),
        "query_ids": load_json(sidecars / "query_ids.json"),
        "qrels": load_json(sidecars / "qrels.json"),
    }


def subset_query_assets(
    assets: Mapping[str, Any], requested_qids: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    """Return an aligned query-only view and IDs absent from the cache.

    Absent IDs are expected only for the five canonical ArguAna qrels whose
    positive document is absent upstream.  They are carried as deterministic
    zero rows by the caller rather than silently discarded.
    """

    positions = {str(qid): row for row, qid in enumerate(assets["query_ids"])}
    present = [str(qid) for qid in requested_qids if str(qid) in positions]
    missing = [str(qid) for qid in requested_qids if str(qid) not in positions]
    indices = np.asarray([positions[qid] for qid in present], dtype=np.int64)
    if not len(indices):
        raise ValueError("requested split has no evaluable query embeddings")
    result = dict(assets)
    result["query_ids"] = present
    result["queries"] = np.asarray(assets["queries"][indices], dtype=np.float32)
    result["proj672_queries"] = np.asarray(
        assets["proj672_queries"][indices], dtype=np.float32
    )
    result["qrels"] = {qid: assets["qrels"][qid] for qid in present}
    return result, missing


def split_query_ids(results_dir: Path, dataset: str, split: str) -> list[str]:
    suffix = "validation" if split == "validation" else "confirmatory"
    path = results_dir / "splits" / f"{dataset}_{suffix}_qids.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"run the splits stage before {split}: missing {path}"
        )
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_assets(
    dataset: str,
    split: str,
    official_master: Mapping[str, Any],
    validation_master: Mapping[str, Any],
    results_dir: Path,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    requested = split_query_ids(results_dir, dataset, split)
    source_master = validation_master if split == "validation" else official_master
    assets = load_assets(source_master["datasets"][dataset])
    subset, absent = subset_query_assets(assets, requested)
    # Guard against accidentally treating an arbitrary cache miss as an
    # evaluable zero.  Every absent ID must be one of the audited ArguAna IDs.
    audited_missing = set(canonical_arguana_missing(official_master["datasets"][dataset]))
    if set(absent) - audited_missing:
        raise ValueError(f"unaudited query IDs absent from cache: {sorted(set(absent)-audited_missing)[:5]}")
    metadata = {
        "requested_count": len(requested),
        "evaluable_count": len(subset["query_ids"]),
        "zero_count": len(absent),
        "zero_query_ids": absent,
        "requested_ids_sha256": gib.ids_fingerprint(requested),
        "evaluable_ids_sha256": gib.ids_fingerprint(subset["query_ids"]),
    }
    return subset, absent, metadata


def append_zeros(values: Sequence[float], count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if count:
        array = np.concatenate((array, np.zeros(count, dtype=np.float64)))
    return array


def canonical_metric_summary(
    per_query: Mapping[str, Sequence[float]], zero_count: int
) -> tuple[dict[str, float], dict[str, float]]:
    evaluable: dict[str, float] = {}
    canonical: dict[str, float] = {}
    for metric, values in per_query.items():
        array = np.asarray(values, dtype=np.float64)
        if np.any(np.isfinite(array)):
            evaluable[metric] = float(np.nanmean(array))
            canonical_values = append_zeros(array[np.isfinite(array)], zero_count)
            canonical[metric] = float(canonical_values.mean())
        else:
            evaluable[metric] = math.nan
            canonical[metric] = math.nan
    return evaluable, canonical


def method_metrics_for_split(
    dataset: str,
    split: str,
    qids: Sequence[str],
    method: str,
) -> dict[str, np.ndarray]:
    stem = "graded_ir_validation" if split == "validation" else "graded_ir_official_test"
    path = ROOT / "results" / "system_revision" / f"{stem}_{dataset}.jsonl"
    by_qid: dict[str, Mapping[str, float]] = {}
    for row in iter_jsonl(path):
        if row["method"] == method:
            by_qid[str(row["query_id"])] = row["metrics"]
    if any(qid not in by_qid for qid in qids):
        missing = [qid for qid in qids if qid not in by_qid]
        raise ValueError(f"{method} split records missing qids: {missing[:5]}")
    metric_names = sorted(next(iter(by_qid.values())))
    return {
        metric: np.asarray([float(by_qid[qid][metric]) for qid in qids], dtype=np.float64)
        for metric in metric_names
    }


def projected_views(assets: Mapping[str, Any], dimension: int) -> tuple[np.ndarray, np.ndarray]:
    if dimension <= 672:
        return assets["proj672_corpus"][:, :dimension], assets["proj672_queries"][:, :dimension]
    if dimension != 768:
        raise ValueError("only the declared Pareto dimensions are supported")
    corpus = np.asarray(assets["corpus"], dtype=np.float32) - assets["basis"].passage_mean
    return corpus, np.asarray(assets["queries"], dtype=np.float32)


def explained_variance_ratio(assets: Mapping[str, Any], dimension: int) -> float:
    if dimension == 768:
        return 1.0
    corpus = assets["corpus"]
    basis = assets["basis"]
    rng = np.random.default_rng(basis.fit_sample_seed)
    sample_count = min(int(basis.fit_sample_size), len(corpus))
    sample_ids = rng.choice(len(corpus), size=sample_count, replace=False)
    total = 0.0
    retained = 0.0
    for start in range(0, sample_count, 4096):
        ids = sample_ids[start : start + 4096]
        raw = np.asarray(corpus[ids], dtype=np.float64) - np.asarray(basis.passage_mean, dtype=np.float64)
        projected = np.asarray(assets["proj672_corpus"][ids, :dimension], dtype=np.float64)
        total += float(np.square(raw).sum())
        retained += float(np.square(projected).sum())
    return retained / total


def exact_metrics(
    corpus: np.ndarray,
    queries: np.ndarray,
    assets: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, Any]]:
    corpus_set = set(assets["corpus_ids"])
    spare = 1 if any(qid in corpus_set for qid in assets["query_ids"]) else 0
    _, ids, elapsed_ms, backend = gib.exact_topk_search(
        corpus,
        queries,
        100 + spare,
        backend="cuda",
        query_batch_size=128,
        doc_chunk_size=16_384,
    )
    ranking = gib.remove_identical_query_documents(
        ids, assets["query_ids"], assets["corpus_ids"], 100
    )
    per_query = gib.evaluate_rankings(
        ranking, assets["query_ids"], assets["corpus_ids"], assets["qrels"]
    )
    return (
        {metric: float(values.mean()) for metric, values in per_query.items()},
        per_query,
        {
            "backend": backend,
            "total_ms": elapsed_ms,
            "mean_per_query_ms": elapsed_ms / len(queries),
        },
    )


def variable_candidate_metrics(
    candidates: np.ndarray,
    reranked: np.ndarray,
    assets: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    corpus_ids = np.asarray(assets["corpus_ids"], dtype=object)
    qids = assets["query_ids"]
    ndcg = np.zeros(len(qids), dtype=np.float64)
    mrr = np.zeros(len(qids), dtype=np.float64)
    candidate_recall = np.zeros(len(qids), dtype=np.float64)
    rerank_recall_100 = np.zeros(len(qids), dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    for row, qid in enumerate(qids):
        judgments = {str(k): float(v) for k, v in assets["qrels"][qid].items()}
        positives = {docid for docid, grade in judgments.items() if grade > 0}
        candidate_docs = [str(corpus_ids[int(index)]) for index in candidates[row]]
        candidate_recall[row] = len(positives.intersection(candidate_docs)) / len(positives)
        top10 = [str(corpus_ids[int(index)]) for index in reranked[row, :10]]
        gains = np.asarray([judgments.get(docid, 0.0) for docid in top10], dtype=np.float64)
        ideal = np.asarray(
            sorted((grade for grade in judgments.values() if grade > 0), reverse=True)[:10],
            dtype=np.float64,
        )
        ndcg[row] = float(np.sum(gains * discounts[: len(gains)])) / float(
            np.sum(ideal * discounts[: len(ideal)])
        )
        for rank, docid in enumerate(top10, start=1):
            if docid in positives:
                mrr[row] = 1.0 / rank
                break
        # Standard Recall@100 remains well-defined when a system returns fewer
        # than 100 results: the unfilled ranks contribute no additional hits.
        top100 = [str(corpus_ids[int(index)]) for index in reranked[row, :100]]
        rerank_recall_100[row] = len(positives.intersection(top100)) / len(positives)
    arrays = {
        "ndcg_at_10": ndcg,
        "mrr_at_10": mrr,
        "candidate_recall_at_k": candidate_recall,
        "recall_at_100_with_at_most_k_returned": candidate_recall,
        "reranked_recall_at_100": rerank_recall_100,
    }
    summary = {
        name: float(np.nanmean(values)) if np.any(np.isfinite(values)) else math.nan
        for name, values in arrays.items()
    }
    return summary, arrays


def pq_index_path(
    cache_root: Path,
    dataset: str,
    dimension: int,
    m: int,
    seed: int,
    corpus_fingerprint: str,
    basis_sha256: str,
    train_size: int,
) -> tuple[Path, dict[str, Any]]:
    recipe = {
        "schema": "ir_expansion.pq.v1",
        "dataset": dataset,
        "dimension": dimension,
        "m": m,
        "nbits": 8,
        "train_size": int(train_size),
        "train_seed": seed,
        "iterations": 25,
        "corpus_ids_sha256": corpus_fingerprint,
        "basis_sha256": basis_sha256,
    }
    return cache_root / "ir_expansion" / "pq" / dataset / fingerprint(recipe) / "index.faiss", recipe


def get_or_build_pq(
    projected_corpus: np.ndarray,
    *,
    cache_root: Path,
    dataset: str,
    dimension: int,
    m: int,
    seed: int,
    corpus_fingerprint: str,
    basis_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    import faiss

    path, recipe = pq_index_path(
        cache_root,
        dataset,
        dimension,
        m,
        seed,
        corpus_fingerprint,
        basis_sha256,
        min(50_000, len(projected_corpus)),
    )
    metadata_path = path.with_name("metadata.json")
    if path.exists() and metadata_path.exists():
        index = faiss.read_index(str(path))
        cache_hit = True
        progress(f"PQ cache hit dataset={dataset} d={dimension} M={m} seed={seed}")
    else:
        progress(f"PQ build start dataset={dataset} d={dimension} M={m} seed={seed}")
        started = time.perf_counter()
        index = gib.build_pq_index(
            projected_corpus,
            m=m,
            nbits=8,
            train_size=50_000,
            train_seed=seed,
            chunk_size=16_384,
            faiss_threads=1,
            iterations=25,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        gib.save_faiss_index_atomic(path, index)
        recipe["build_seconds"] = time.perf_counter() - started
        atomic_json(metadata_path, recipe)
        cache_hit = False
        progress(
            f"PQ build done dataset={dataset} d={dimension} M={m} seed={seed} "
            f"seconds={recipe['build_seconds']:.3f}"
        )
    if int(index.d) != dimension or int(index.ntotal) != len(projected_corpus):
        raise ValueError(f"invalid cached PQ index: {path}")
    return index, {
        **recipe,
        "path": str(path.resolve()),
        "cache_hit": cache_hit,
        "serialized_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "code_bytes_per_document": m,
        "float32_projected_bytes_per_document": 4 * dimension,
    }


def evaluate_pq(
    index: Any,
    projected_corpus: np.ndarray,
    projected_queries: np.ndarray,
    assets: Mapping[str, Any],
    k_values: Sequence[int],
) -> dict[str, Any]:
    query_array = np.ascontiguousarray(projected_queries, dtype=np.float32)
    corpus_set = set(assets["corpus_ids"])
    spare = 1 if any(qid in corpus_set for qid in assets["query_ids"]) else 0
    output: dict[str, Any] = {}
    for k in k_values:
        index.search(query_array[:1], k + spare)
        search_samples: list[float] = []
        raw_candidates: np.ndarray | None = None
        for _ in range(PQ_TIMING_REPEATS):
            started = time.perf_counter()
            _, raw_candidates = index.search(query_array, k + spare)
            search_samples.append((time.perf_counter() - started) * 1000.0)
        assert raw_candidates is not None
        search_ms = float(np.median(search_samples))
        if np.any(raw_candidates < 0):
            raise RuntimeError("PQ returned an incomplete candidate set")
        candidates = gib.remove_identical_query_documents(
            raw_candidates, assets["query_ids"], assets["corpus_ids"], k
        )
        _, reranked, latency = gib.rerank_projected_shortlist(
            projected_corpus, projected_queries, candidates, min(k, 100)
        )
        summary, arrays = variable_candidate_metrics(candidates, reranked, assets)
        output[str(k)] = {
            "metrics": summary,
            "rerank_latency_ms": percentile_summary(latency),
            "pq_search_total_ms": search_ms,
            "pq_search_mean_per_query_ms": search_ms / len(query_array),
            "pq_search_repeats": PQ_TIMING_REPEATS,
            "pq_search_total_ms_samples": search_samples,
            "per_query": {
                metric: [None if not np.isfinite(v) else float(v) for v in values]
                for metric, values in arrays.items()
            },
        }
    return output


def frozen_method_metrics(dataset: str, method: str) -> dict[str, np.ndarray]:
    path = ROOT / "results" / "system_revision" / f"graded_ir_official_test_{dataset}.jsonl"
    values: dict[str, list[float]] = {}
    for row in iter_jsonl(path):
        if row["method"] == method:
            for metric, value in row["metrics"].items():
                values.setdefault(metric, []).append(float(value))
    return {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}


def frozen_raw_metrics(dataset: str) -> dict[str, np.ndarray]:
    return frozen_method_metrics(dataset, "raw_exact")


def finalise_pq_split(
    point: dict[str, Any],
    projected_exact: Mapping[str, Sequence[float]],
    raw_exact: Mapping[str, Sequence[float]],
    zero_count: int,
) -> dict[str, Any]:
    arrays = {
        metric: np.asarray(
            [math.nan if value is None else float(value) for value in values],
            dtype=np.float64,
        )
        for metric, values in point.pop("per_query").items()
    }
    evaluable, canonical = canonical_metric_summary(arrays, zero_count)
    pq_ndcg = append_zeros(arrays["ndcg_at_10"], zero_count)
    projected_ndcg = append_zeros(projected_exact["ndcg_at_10"], zero_count)
    raw_ndcg = append_zeros(raw_exact["ndcg_at_10"], zero_count)
    point["metrics_evaluable_sensitivity"] = evaluable
    point["metrics_canonical_primary"] = canonical
    point["rerank_minus_projected_exact_ndcg_at_10"] = bootstrap_delta(
        pq_ndcg, projected_ndcg
    )
    point["rerank_minus_raw_exact_ndcg_at_10"] = bootstrap_delta(pq_ndcg, raw_ndcg)
    return point


def run_pareto(
    master_path: Path,
    validation_master_path: Path,
    results_dir: Path,
    cache_root: Path,
) -> dict[str, Any]:
    master = load_json(master_path)
    validation_master = load_json(validation_master_path)
    manifest = {
        "schema": "ir_expansion.svd_pareto.v2",
        "created_utc": utc_now(),
        "dimensions": list(DIMENSIONS),
        "pq_rule": (
            "M=d/8, 8-bit subquantizers, K=100, seed=42. This fixes an "
            "eight-dimensional subvector width and is a joint SVD/PQ storage "
            "co-design curve, not a pure projection-dimension ablation."
        ),
        "pure_svd_ablation": (
            "projected_exact rows change only the nested SVD dimension and are "
            "the unconfounded projection analysis"
        ),
        "selection_policy": (
            "Validation rows may guide operating-point discussion. Strict-confirmatory "
            "rows are reported separately and never used to choose a dimension."
        ),
        "non_inferiority": {
            "metric": "nDCG@10",
            "smallest_effect_size_of_interest": NI_MARGIN_NDCG,
            "confidence_interval": "paired two-sided 95% percentile bootstrap",
            "status": (
                "Protocol-declared revision analysis, not a prospectively registered trial. "
                "Non-inferiority uses the lower endpoint; equivalence requires the entire "
                "interval inside [-0.002, 0.002]."
            ),
        },
        "datasets": {},
    }
    output_path = results_dir / "svd_pareto.json"
    if output_path.exists():
        previous = load_json(output_path)
        if previous.get("schema") == manifest["schema"]:
            manifest["datasets"] = previous.get("datasets", {})
            manifest["created_utc"] = previous.get("created_utc", manifest["created_utc"])
    for dataset in gib.REQUIRED_DATASETS:
        progress(f"Pareto dataset start: {dataset}")
        dataset_result = manifest["datasets"].setdefault(dataset, {"dimensions": {}})
        summary = master["datasets"][dataset]
        validation_assets, validation_zeros, validation_meta = split_assets(
            dataset, "validation", master, validation_master, results_dir
        )
        confirmatory_assets, confirmatory_zeros, confirmatory_meta = split_assets(
            dataset, "confirmatory", master, validation_master, results_dir
        )
        split_contexts = {
            "validation": (validation_assets, validation_zeros, validation_meta),
            "strict_confirmatory": (
                confirmatory_assets,
                confirmatory_zeros,
                confirmatory_meta,
            ),
        }
        dataset_result["splits"] = {
            name: metadata for name, (_, _, metadata) in split_contexts.items()
        }
        basis_hash = summary["cache"]["projection_basis_sha256"]
        for dimension in DIMENSIONS:
            key = str(dimension)
            if key in dataset_result["dimensions"]:
                continue
            progress(f"Pareto point start dataset={dataset} d={dimension}")
            if dimension <= 672:
                projected_corpus = confirmatory_assets["proj672_corpus"][:, :dimension]
            else:
                projected_corpus = (
                    np.asarray(confirmatory_assets["corpus"], dtype=np.float32)
                    - confirmatory_assets["basis"].passage_mean
                )
            m = dimension // 8
            index, pq_metadata = get_or_build_pq(
                projected_corpus,
                cache_root=cache_root,
                dataset=dataset,
                dimension=dimension,
                m=m,
                seed=42,
                corpus_fingerprint=summary["dataset"]["corpus_ids_sha256"],
                basis_sha256=basis_hash if dimension <= 672 else "full-rank-centred-identity",
            )
            split_results: dict[str, Any] = {}
            for split_name, (assets, zero_ids, metadata) in split_contexts.items():
                projected_queries = (
                    assets["proj672_queries"][:, :dimension]
                    if dimension <= 672
                    else assets["queries"]
                )
                exact_evaluable, exact_per_query, timing = exact_metrics(
                    projected_corpus, projected_queries, assets
                )
                raw = method_metrics_for_split(
                    dataset,
                    "validation" if split_name == "validation" else "confirmatory",
                    assets["query_ids"],
                    "raw_exact",
                )
                _, exact_canonical = canonical_metric_summary(
                    exact_per_query, len(zero_ids)
                )
                projected_ni = bootstrap_delta(
                    append_zeros(exact_per_query["ndcg_at_10"], len(zero_ids)),
                    append_zeros(raw["ndcg_at_10"], len(zero_ids)),
                )
                pq = evaluate_pq(
                    index, projected_corpus, projected_queries, assets, (100,)
                )["100"]
                pq = finalise_pq_split(
                    pq, exact_per_query, raw, len(zero_ids)
                )
                split_results[split_name] = {
                    "split_metadata": metadata,
                    "projected_exact_metrics_evaluable_sensitivity": exact_evaluable,
                    "projected_exact_metrics_canonical_primary": exact_canonical,
                    "projected_exact_timing": timing,
                    "projected_exact_minus_raw_exact_ndcg_at_10": projected_ni,
                    "pq_projected_rerank": pq,
                }
            dataset_result["dimensions"][key] = {
                "explained_variance_ratio_on_frozen_fit_sample": explained_variance_ratio(
                    confirmatory_assets, dimension
                ),
                "splits": split_results,
                "pq_index": pq_metadata,
            }
            atomic_json(output_path, manifest)
            progress(f"Pareto point done dataset={dataset} d={dimension}")
            del index, projected_corpus
    manifest["completed_utc"] = utc_now()
    atomic_json(output_path, manifest)
    return manifest


def project_control(
    assets: Mapping[str, Any],
    dimension: int,
    method: str,
    random_seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None]:
    corpus = np.asarray(assets["corpus"], dtype=np.float32)
    queries = np.asarray(assets["queries"], dtype=np.float32)
    mean = np.asarray(assets["basis"].passage_mean, dtype=np.float32)
    if method == "coordinate":
        return corpus[:, :dimension] - mean[:, :dimension], queries[:, :dimension], {
            "method": "first-d coordinate truncation",
            "seed": None,
        }, None
    if method != "random":
        raise ValueError("control method must be coordinate or random")
    rng = np.random.default_rng(random_seed)
    gaussian = rng.standard_normal(
        (corpus.shape[1], max(CONTROL_DIMENSIONS)), dtype=np.float32
    )
    projector_full, _ = np.linalg.qr(gaussian, mode="reduced")
    projector = np.asarray(projector_full[:, :dimension], dtype=np.float32)
    # Torch keeps the large corpus projection on the available GPU while the
    # final arrays stay in host memory for the common exact-search routine.
    import torch

    device = torch.device("cuda")
    p = torch.from_numpy(projector).to(device)
    q_out = (torch.from_numpy(queries).to(device) @ p).cpu().numpy().astype(np.float32)
    d_out = np.empty((len(corpus), dimension), dtype=np.float32)
    for start in range(0, len(corpus), 8192):
        chunk = torch.from_numpy(np.asarray(corpus[start : start + 8192] - mean)).to(device)
        d_out[start : start + len(chunk)] = (chunk @ p).cpu().numpy()
    del p
    torch.cuda.empty_cache()
    return d_out, q_out, {
        "method": "seeded Gaussian orthogonal projection",
        "seed": random_seed,
        "projector_sha256": hashlib.sha256(np.ascontiguousarray(projector).view(np.uint8)).hexdigest(),
    }, projector


def run_controls(
    master_path: Path,
    validation_master_path: Path,
    results_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    master = load_json(master_path)
    validation_master = load_json(validation_master_path)
    output_path = results_dir / "projection_controls.json"
    output: dict[str, Any] = {
        "schema": "ir_expansion.projection_controls.v2",
        "created_utc": utc_now(),
        "dimensions": list(CONTROL_DIMENSIONS),
        "methods": ["svd", "random", "coordinate"],
        "selection_policy": (
            "Validation and strict-confirmatory metrics are disjoint; controls "
            "are interpreted on validation without choosing from confirmatory rows."
        ),
        "datasets": {},
    }
    if output_path.exists() and not force:
        previous = load_json(output_path)
        if previous.get("schema") == output["schema"]:
            output = previous
    for dataset in gib.REQUIRED_DATASETS:
        progress(f"Projection controls dataset start: {dataset}")
        data_result = output["datasets"].setdefault(dataset, {})
        validation_assets, validation_zeros, validation_meta = split_assets(
            dataset, "validation", master, validation_master, results_dir
        )
        confirmatory_assets, confirmatory_zeros, confirmatory_meta = split_assets(
            dataset, "confirmatory", master, validation_master, results_dir
        )
        contexts = {
            "validation": (validation_assets, validation_zeros, validation_meta),
            "strict_confirmatory": (
                confirmatory_assets,
                confirmatory_zeros,
                confirmatory_meta,
            ),
        }
        for dimension in CONTROL_DIMENSIONS:
            dimension_result = data_result.setdefault(str(dimension), {})
            for method in ("svd", "random", "coordinate"):
                if method in dimension_result:
                    continue
                progress(
                    f"Projection control start dataset={dataset} d={dimension} method={method}"
                )
                projector: np.ndarray | None = None
                if method == "svd":
                    docs = confirmatory_assets["proj672_corpus"][:, :dimension]
                    metadata = {
                        "method": "nested prefix of frozen corpus-only SVD basis",
                        "basis_sha256": master["datasets"][dataset]["cache"][
                            "projection_basis_sha256"
                        ],
                    }
                    confirm_queries = confirmatory_assets["proj672_queries"][:, :dimension]
                else:
                    docs, confirm_queries, metadata, projector = project_control(
                        confirmatory_assets, dimension, method
                    )
                split_results: dict[str, Any] = {}
                for split_name, (assets, zero_ids, split_meta) in contexts.items():
                    if split_name == "strict_confirmatory":
                        queries = confirm_queries
                    elif method == "svd":
                        queries = assets["proj672_queries"][:, :dimension]
                    elif method == "coordinate":
                        queries = assets["queries"][:, :dimension]
                    else:
                        assert projector is not None
                        queries = np.asarray(assets["queries"] @ projector, dtype=np.float32)
                    metrics, per_query, timing = exact_metrics(docs, queries, assets)
                    _, canonical_metrics = canonical_metric_summary(
                        per_query, len(zero_ids)
                    )
                    raw = method_metrics_for_split(
                        dataset,
                        "validation" if split_name == "validation" else "confirmatory",
                        assets["query_ids"],
                        "raw_exact",
                    )
                    delta = bootstrap_delta(
                        append_zeros(per_query["ndcg_at_10"], len(zero_ids)),
                        append_zeros(raw["ndcg_at_10"], len(zero_ids)),
                    )
                    split_results[split_name] = {
                        "split_metadata": split_meta,
                        "metrics_evaluable_sensitivity": metrics,
                        "metrics_canonical_primary": canonical_metrics,
                        "timing": timing,
                        "minus_raw_exact_ndcg_at_10": delta,
                    }
                dimension_result[method] = {
                    "metadata": metadata,
                    "splits": split_results,
                }
                atomic_json(output_path, output)
                progress(
                    f"Projection control done dataset={dataset} d={dimension} method={method}"
                )
                del docs
    output["completed_utc"] = utc_now()
    atomic_json(output_path, output)
    return output


def run_pq_sensitivity(
    master_path: Path,
    validation_master_path: Path,
    results_dir: Path,
    cache_root: Path,
    *,
    force_evaluation: bool = False,
) -> dict[str, Any]:
    master = load_json(master_path)
    validation_master = load_json(validation_master_path)
    output_path = results_dir / "pq_sensitivity.json"
    output: dict[str, Any] = {
        "schema": "ir_expansion.pq_sensitivity.v2",
        "created_utc": utc_now(),
        "dimension": 672,
        "k_values": list(K_VALUES),
        "m_values": [32, 48, 84, 96],
        "seeds": list(PQ_SEEDS),
        "design": (
            "All datasets: seed 42 for every M and K. FiQA and TREC-COVID: "
            "additional seeds 17 and 2026 for M=84 and deployment M=96."
        ),
        "selection_policy": (
            "Validation and strict-confirmatory query sets are disjoint. No M, K, "
            "or seed is selected from strict-confirmatory results."
        ),
        "datasets": {},
    }
    if output_path.exists() and not force_evaluation:
        previous = load_json(output_path)
        if previous.get("schema") == output["schema"]:
            output = previous
    for dataset in gib.REQUIRED_DATASETS:
        progress(f"PQ sensitivity dataset start: {dataset}")
        summary = master["datasets"][dataset]
        validation_assets, validation_zeros, validation_meta = split_assets(
            dataset, "validation", master, validation_master, results_dir
        )
        confirmatory_assets, confirmatory_zeros, confirmatory_meta = split_assets(
            dataset, "confirmatory", master, validation_master, results_dir
        )
        contexts = {
            "validation": (validation_assets, validation_zeros, validation_meta),
            "strict_confirmatory": (
                confirmatory_assets,
                confirmatory_zeros,
                confirmatory_meta,
            ),
        }
        docs = confirmatory_assets["proj672_corpus"]
        dataset_result = output["datasets"].setdefault(dataset, {})
        recipes = [(m, 42) for m in (32, 48, 84, 96)]
        if dataset in {"fiqa", "trec-covid"}:
            recipes.extend((m, seed) for m in (84, 96) for seed in (17, 2026))
        for m, seed in recipes:
            key = f"M{m}_seed{seed}"
            if key in dataset_result:
                continue
            progress(f"PQ sensitivity point start dataset={dataset} {key}")
            # The frozen deployment index is reused byte-for-byte for M=96,
            # seed=42 instead of training an equivalent duplicate.
            if m == 96 and seed == 42:
                import faiss

                frozen_path = Path(summary["cache"]["pq_index"])
                index = faiss.read_index(str(frozen_path))
                metadata = {
                    "path": str(frozen_path.resolve()),
                    "cache_hit": True,
                    "source": "frozen paper deployment index",
                    "serialized_bytes": frozen_path.stat().st_size,
                    "sha256": sha256_file(frozen_path),
                    "code_bytes_per_document": 96,
                    "float32_projected_bytes_per_document": 4 * 672,
                }
            else:
                index, metadata = get_or_build_pq(
                    docs,
                    cache_root=cache_root,
                    dataset=dataset,
                    dimension=672,
                    m=m,
                    seed=seed,
                    corpus_fingerprint=summary["dataset"]["corpus_ids_sha256"],
                    basis_sha256=summary["cache"]["projection_basis_sha256"],
                )
            split_results: dict[str, Any] = {}
            for split_name, (assets, zero_ids, split_meta) in contexts.items():
                queries = assets["proj672_queries"]
                evaluated = evaluate_pq(index, docs, queries, assets, K_VALUES)
                projected_exact = method_metrics_for_split(
                    dataset,
                    "validation" if split_name == "validation" else "confirmatory",
                    assets["query_ids"],
                    "projected_exact",
                )
                raw_exact = method_metrics_for_split(
                    dataset,
                    "validation" if split_name == "validation" else "confirmatory",
                    assets["query_ids"],
                    "raw_exact",
                )
                evaluated = {
                    k: finalise_pq_split(
                        point, projected_exact, raw_exact, len(zero_ids)
                    )
                    for k, point in evaluated.items()
                }
                split_results[split_name] = {
                    "split_metadata": split_meta,
                    "k": evaluated,
                }
            dataset_result[key] = {"index": metadata, "splits": split_results}
            atomic_json(output_path, output)
            progress(f"PQ sensitivity point done dataset={dataset} {key}")
            del index
        # Seed dispersion is descriptive robustness evidence; no seed is
        # selected based on these values.  It is available only where the
        # predeclared three-seed design was run.
        dispersion: dict[str, Any] = {}
        for m in (84, 96):
            keys = [f"M{m}_seed{seed}" for seed in PQ_SEEDS]
            if not all(key in dataset_result for key in keys):
                continue
            dispersion[str(m)] = {}
            for split_name in ("validation", "strict_confirmatory"):
                dispersion[str(m)][split_name] = {}
                for k in K_VALUES:
                    dispersion[str(m)][split_name][str(k)] = {}
                    for metric in (
                        "ndcg_at_10",
                        "mrr_at_10",
                        "candidate_recall_at_k",
                        "recall_at_100_with_at_most_k_returned",
                    ):
                        series = [
                            float(
                                dataset_result[key]["splits"][split_name]["k"][str(k)][
                                    "metrics_canonical_primary"
                                ][metric]
                            )
                            for key in keys
                        ]
                        dispersion[str(m)][split_name][str(k)][metric] = {
                            "values_by_seed": dict(
                                zip(map(str, PQ_SEEDS), series, strict=True)
                            ),
                            "mean": float(np.mean(series)),
                            "sample_std": float(np.std(series, ddof=1)),
                            "range": [float(np.min(series)), float(np.max(series))],
                            "selection_note": (
                                "descriptive only; no seed is selected from this split"
                            ),
                        }
        if dispersion:
            dataset_result["seed_dispersion"] = dispersion
            atomic_json(output_path, output)
    output["completed_utc"] = utc_now()
    atomic_json(output_path, output)
    return output


def write_summary_csv(results_dir: Path) -> None:
    pareto_path = results_dir / "svd_pareto.json"
    if pareto_path.exists():
        data = load_json(pareto_path)
        rows: list[dict[str, Any]] = []
        for dataset, dataset_result in data["datasets"].items():
            for dimension, point in dataset_result["dimensions"].items():
                for split, split_point in point["splits"].items():
                    exact = split_point[
                        "projected_exact_metrics_canonical_primary"
                    ]
                    pq = split_point["pq_projected_rerank"][
                        "metrics_canonical_primary"
                    ]
                    rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "dimension": dimension,
                            "explained_variance": point[
                                "explained_variance_ratio_on_frozen_fit_sample"
                            ],
                            "projected_ndcg10": exact["ndcg_at_10"],
                            "projected_recall100": exact["recall_at_100"],
                            "pq_ndcg10": pq["ndcg_at_10"],
                            "pq_candidate_recall100": pq[
                                "recall_at_100_with_at_most_k_returned"
                            ],
                            "pq_reranked_recall100": pq[
                                "reranked_recall_at_100"
                            ],
                            "pq_m": point["pq_index"]["m"],
                            "pq_bytes_per_doc": point["pq_index"][
                                "code_bytes_per_document"
                            ],
                            "pq_index_bytes": point["pq_index"]["serialized_bytes"],
                        }
                    )
        path = results_dir / "svd_pareto.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def environment_manifest() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "torch", "faiss", "sklearn"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - diagnostic only
            packages[name] = f"unavailable: {exc}"
    return {
        "created_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "command": sys.argv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("splits", "pareto", "controls", "pq", "all")
    )
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument(
        "--validation-master", type=Path, default=DEFAULT_VALIDATION_MASTER
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--force-controls",
        action="store_true",
        help="rerun projection controls instead of resuming their checkpoint",
    )
    parser.add_argument(
        "--force-pq-eval",
        action="store_true",
        help="re-evaluate every PQ point while reusing cached trained indices",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(results_dir / "environment.json", environment_manifest())
    stages = (
        ("splits", "pareto", "controls", "pq") if args.stage == "all" else (args.stage,)
    )
    for stage in stages:
        if stage == "splits":
            run_splits(args.master.resolve(), results_dir)
        elif stage == "pareto":
            run_pareto(
                args.master.resolve(),
                args.validation_master.resolve(),
                results_dir,
                args.cache_root.resolve(),
            )
        elif stage == "controls":
            run_controls(
                args.master.resolve(),
                args.validation_master.resolve(),
                results_dir,
                force=args.force_controls,
            )
        elif stage == "pq":
            run_pq_sensitivity(
                args.master.resolve(),
                args.validation_master.resolve(),
                results_dir,
                args.cache_root.resolve(),
                force_evaluation=args.force_pq_eval,
            )
    write_summary_csv(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
