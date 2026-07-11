"""Reproducible graded-IR benchmark for the revised systems paper.

The harness evaluates ``intfloat/multilingual-e5-base`` on the canonical
Hugging Face mirrors of six BEIR collections: SciFact, NFCorpus, ArguAna,
SciDocs, FiQA-2018, and TREC-COVID.  It deliberately keeps the scientific
protocol small and auditable:

* E5 inputs use the literal ``query: `` and ``passage: `` prefixes, mean
  pooling over non-padding tokens, and row-wise L2 normalisation.
* ``raw_exact`` is exhaustive inner-product retrieval over the original
  embeddings and is never derived from an approximate candidate set.
* An optional corpus-only SVD learns ``V`` and a passage mean.  Documents are
  represented by ``(d - mean) @ V`` while the deployable online query is
  exactly ``q @ V``.  No query/test mean is fitted or subtracted.
* FAISS PQ selects a shortlist.  ``pq_projected_rerank`` scores only those
  candidates with exact projected dot products; ``pq_only`` exposes the
  approximate ordering as a separate baseline.
* Metrics follow the BEIR/trec_eval conventions: linear graded gains for
  nDCG@10 and binary relevance (grade > 0) for Recall and reciprocal rank.
  Confidence intervals for method differences use paired query bootstrap.

All downloads and generated artifacts are rooted at configurable paths.  A
safe GPU smoke run is::

    python system/graded_ir_bench.py --smoke \
      --cache-root cache/graded_ir \
      --hf-home cache/huggingface \
      --output-json results/graded_ir_scifact_smoke.json

For paper results, tune only on ``--split validation``.  Freeze every setting
before running ``--split test``.  Collections with no official development or
training qrels use a deterministic 20/80 hash partition of the official test
queries.  ``--split official-test`` evaluates the full official test set and
is intended only for the final, frozen BEIR-comparable report.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "GRADED_IR_CACHE",
        str(Path(__file__).resolve().parents[1] / "cache" / "graded_ir"),
    )
)
REQUIRED_DATASETS = (
    "scifact",
    "nfcorpus",
    "arguana",
    "scidocs",
    "fiqa",
    "trec-covid",
)
PREFIX_RECIPE = "e5-query-passage-v1"
POOLING_RECIPE = "masked-mean+l2-v1"
METRIC_NAMES = ("ndcg_at_10", "recall_at_100", "recall_at_10", "mrr_at_10")


@dataclass(frozen=True)
class DatasetSpec:
    """Pinned repository layout and the available official qrel files."""

    name: str
    dataset_repo: str
    qrels_repo: str
    qrel_splits: tuple[str, ...]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "scifact": DatasetSpec(
        "scifact", "BeIR/scifact", "BeIR/scifact-qrels", ("train", "test")
    ),
    "nfcorpus": DatasetSpec(
        "nfcorpus",
        "BeIR/nfcorpus",
        "BeIR/nfcorpus-qrels",
        ("train", "dev", "test"),
    ),
    "arguana": DatasetSpec(
        "arguana", "BeIR/arguana", "BeIR/arguana-qrels", ("test",)
    ),
    "scidocs": DatasetSpec(
        "scidocs", "BeIR/scidocs", "BeIR/scidocs-qrels", ("test",)
    ),
    "fiqa": DatasetSpec(
        "fiqa", "BeIR/fiqa", "BeIR/fiqa-qrels", ("train", "dev", "test")
    ),
    "trec-covid": DatasetSpec(
        "trec-covid",
        "BeIR/trec-covid",
        "BeIR/trec-covid-qrels",
        ("test",),
    ),
}

_DATASET_ALIASES = {
    "treccovid": "trec-covid",
    "trec_covid": "trec-covid",
    "fiqa-2018": "fiqa",
    "fiqa2018": "fiqa",
    "sci-fact": "scifact",
    "nf-corpus": "nfcorpus",
    "sci-docs": "scidocs",
}


@dataclass(frozen=True)
class ProjectionBasis:
    """Corpus-only projection and complete deterministic fit metadata."""

    passage_mean: np.ndarray
    vectors: np.ndarray
    fit_sample_size: int
    fit_sample_seed: int
    svd_seed: int
    svd_iterations: int = 5

    def __post_init__(self) -> None:
        mean = np.asarray(self.passage_mean, dtype=np.float32)
        vectors = np.asarray(self.vectors, dtype=np.float32)
        if mean.ndim == 1:
            mean = mean.reshape(1, -1)
            object.__setattr__(self, "passage_mean", mean)
        if mean.ndim != 2 or mean.shape[0] != 1:
            raise ValueError("passage_mean must have shape (1, embedding_dim)")
        if vectors.ndim != 2 or vectors.shape[0] != mean.shape[1]:
            raise ValueError("vectors must have shape (embedding_dim, projection_dim)")


@dataclass
class BenchmarkConfig:
    """Scientific and systems settings for one dataset run."""

    split: str = "validation"
    validation_fraction: float = 0.2
    split_seed: int = 2026
    max_queries: int | None = None
    max_corpus: int | None = None
    model_name: str = DEFAULT_MODEL
    max_length: int = 512
    encode_batch_size: int = 32
    device: str = "cuda"
    use_amp: bool = True
    projection_dim: int = 672
    projection_fit_size: int = 50_000
    projection_fit_seed: int = 17
    svd_seed: int = 42
    svd_iterations: int = 5
    pq_m: int = 96
    pq_nbits: int = 8
    pq_train_size: int = 50_000
    pq_train_seed: int = 42
    pq_iterations: int = 25
    candidate_k: int = 1_000
    retrieve_k: int = 100
    exact_backend: str = "auto"
    exact_query_batch_size: int = 128
    exact_doc_chunk_size: int = 16_384
    projection_chunk_size: int = 16_384
    faiss_threads: int = 1
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force_reencode: bool = False
    force_rebuild: bool = False
    offline: bool = False
    smoke: bool = False

    def validate(self, embedding_dim: int, n_docs: int, n_queries: int) -> None:
        if self.split not in {"validation", "test", "official-test"}:
            raise ValueError("split must be validation, test, or official-test")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly between 0 and 1")
        if n_docs < 1 or n_queries < 1:
            raise ValueError("the selected corpus and query set must be non-empty")
        if self.max_length < 1 or self.encode_batch_size < 1:
            raise ValueError("encoding lengths and batch sizes must be positive")
        if self.projection_dim < 0 or self.projection_dim > embedding_dim:
            raise ValueError(
                f"projection_dim must be 0 or in [1, {embedding_dim}], "
                f"got {self.projection_dim}"
            )
        active_dim = self.projection_dim or embedding_dim
        if active_dim % self.pq_m:
            raise ValueError(f"PQ M={self.pq_m} must divide active dimension={active_dim}")
        if not 1 <= self.retrieve_k <= n_docs:
            raise ValueError("retrieve_k must be in [1, number of documents]")
        if not self.retrieve_k <= self.candidate_k <= n_docs:
            raise ValueError("candidate_k must be >= retrieve_k and <= corpus size")
        if self.retrieve_k < 100:
            raise ValueError("retrieve_k must be at least 100 for Recall@100")
        if self.exact_backend not in {"auto", "numpy", "cuda"}:
            raise ValueError("exact_backend must be auto, numpy, or cuda")
        for value, label in (
            (self.exact_query_batch_size, "exact_query_batch_size"),
            (self.exact_doc_chunk_size, "exact_doc_chunk_size"),
            (self.projection_chunk_size, "projection_chunk_size"),
            (self.bootstrap_samples, "bootstrap_samples"),
            (self.pq_m, "pq_m"),
        ):
            if value < 1:
                raise ValueError(f"{label} must be positive")
        if self.pq_nbits < 1 or self.pq_nbits > 16:
            raise ValueError("pq_nbits must be in [1, 16]")


@dataclass
class PreparedDataset:
    name: str
    corpus_ids: list[str]
    corpus_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, float]]
    metadata: dict[str, Any]


def canonical_dataset_name(name: str) -> str:
    canonical = name.strip().lower().replace(" ", "-")
    canonical = _DATASET_ALIASES.get(canonical, canonical)
    if canonical not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset {name!r}; choose one of {sorted(DATASET_SPECS)}"
        )
    return canonical


def format_e5_query(text: str) -> str:
    """Apply the model-card query prefix exactly once."""

    return "query: " + str(text).strip()


def format_e5_passage(text: str, title: str | None = None) -> str:
    """Join title/body deterministically and apply the E5 passage prefix."""

    body = str(text or "").strip()
    heading = str(title or "").strip()
    joined = f"{heading}\n{body}" if heading and body else (heading or body)
    return "passage: " + joined


def l2_normalize_rows(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional embedding matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("cannot L2-normalise a zero embedding")
    return np.asarray(array / norms, dtype=np.float32)


def masked_mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool non-padding tokens; kept public for a small torch unit test."""

    import torch

    mask = attention_mask[..., None].bool()
    hidden = last_hidden_state.masked_fill(~mask, 0.0)
    denominator = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return hidden.sum(dim=1) / denominator


def configure_hf_cache(hf_home: Path) -> dict[str, str]:
    """Configure every Hugging Face cache before importing HF libraries."""

    root = Path(hf_home).expanduser().resolve()
    hub = root / "hub"
    datasets = root / "datasets"
    for directory in (root, hub, datasets):
        directory.mkdir(parents=True, exist_ok=True)
    values = {
        "HF_HOME": str(root),
        "HF_HUB_CACHE": str(hub),
        "HF_DATASETS_CACHE": str(datasets),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    os.environ.update(values)
    return values


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _fingerprint(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()[:length]


def ids_fingerprint(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in ids:
        encoded = str(item).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stable_hash_order(ids: Iterable[str], seed: int) -> list[str]:
    def key(item: str) -> tuple[bytes, str]:
        payload = f"{seed}\0{item}".encode("utf-8")
        return hashlib.sha256(payload).digest(), item

    return sorted((str(item) for item in ids), key=key)


def select_evaluation_qids(
    qrels_by_split: Mapping[str, Mapping[str, Mapping[str, float]]],
    requested_split: str,
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[list[str], dict[str, Any]]:
    """Choose validation/test queries without silently tuning on test.

    If official dev qrels exist they are validation.  Otherwise official train
    qrels are validation.  Test remains the complete official test set in both
    cases.  A test-only collection receives a deterministic hash partition;
    the explicit ``official-test`` mode always returns the complete test set.
    """

    if requested_split not in {"validation", "test", "official-test"}:
        raise ValueError("invalid requested split")
    available = set(qrels_by_split)
    if "test" not in available:
        raise ValueError("BEIR dataset must provide official test qrels")

    separate_validation = "dev" if "dev" in available else (
        "train" if "train" in available else None
    )
    if requested_split == "official-test":
        qids = sorted(str(qid) for qid in qrels_by_split["test"])
        return qids, {
            "requested_split": requested_split,
            "source_qrels": "test",
            "policy": "full_official_test_frozen_only",
            "beir_comparable_full_test": True,
        }

    if separate_validation is not None:
        source = separate_validation if requested_split == "validation" else "test"
        qids = sorted(str(qid) for qid in qrels_by_split[source])
        return qids, {
            "requested_split": requested_split,
            "source_qrels": source,
            "policy": f"official_{separate_validation}_for_validation_official_test_for_test",
            "beir_comparable_full_test": requested_split == "test",
        }

    official = _stable_hash_order(qrels_by_split["test"], seed)
    if len(official) < 2:
        raise ValueError("test-only split needs at least two queries to partition")
    validation_size = int(math.ceil(len(official) * validation_fraction))
    validation_size = min(max(validation_size, 1), len(official) - 1)
    if requested_split == "validation":
        selected = official[:validation_size]
    else:
        selected = official[validation_size:]
    return selected, {
        "requested_split": requested_split,
        "source_qrels": "test",
        "policy": "deterministic_hash_partition_of_official_test",
        "partition_seed": seed,
        "validation_fraction": validation_fraction,
        "validation_queries": validation_size,
        "heldout_test_queries": len(official) - validation_size,
        "beir_comparable_full_test": False,
    }


def resolve_hf_revision(
    repo_id: str,
    repo_type: str,
    requested_revision: str | None,
    offline: bool = False,
) -> str:
    """Resolve a branch/tag to a commit SHA for the run manifest and cache key."""

    if offline:
        if requested_revision is None:
            return "local-cache-unresolved"
        return requested_revision
    from huggingface_hub import HfApi

    api = HfApi()
    if repo_type == "model":
        info = api.model_info(repo_id, revision=requested_revision)
    elif repo_type == "dataset":
        info = api.dataset_info(repo_id, revision=requested_revision)
    else:
        raise ValueError("repo_type must be model or dataset")
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}")
    return str(info.sha)


def _normalise_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_qrels_tsv(path: Path) -> dict[str, dict[str, float]]:
    """Load the canonical BEIR TSV format without coercing string identifiers."""

    result: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"qrels file has no header: {path}")
        lookup = {_normalise_header(name): name for name in reader.fieldnames}

        def column(*aliases: str) -> str:
            for alias in aliases:
                if _normalise_header(alias) in lookup:
                    return lookup[_normalise_header(alias)]
            raise ValueError(f"qrels file {path} lacks one of columns {aliases}")

        query_col = column("query-id", "query_id", "qid")
        corpus_col = column("corpus-id", "corpus_id", "docid", "doc_id")
        score_col = column("score", "relevance", "rel")
        for row in reader:
            qid = str(row[query_col]).strip()
            docid = str(row[corpus_col]).strip()
            if not qid or not docid:
                raise ValueError(f"empty identifier in qrels file {path}")
            score = float(row[score_col])
            previous = result.setdefault(qid, {}).get(docid, -math.inf)
            result[qid][docid] = max(previous, score)
    if not result:
        raise ValueError(f"qrels file is empty: {path}")
    return result


def _row_id(row: Mapping[str, Any]) -> str:
    for key in ("_id", "id", "query_id", "doc_id", "corpus_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise ValueError(f"BEIR row has no recognised id column: {sorted(row)}")


def _load_beir_rows(
    repo_id: str,
    config_name: str,
    split_name: str,
    revision: str,
    hf_datasets_cache: Path,
    offline: bool,
) -> Any:
    from datasets import DownloadConfig, load_dataset

    kwargs: dict[str, Any] = {
        "path": repo_id,
        "name": config_name,
        "split": split_name,
        "cache_dir": str(hf_datasets_cache),
        "download_config": DownloadConfig(local_files_only=offline),
    }
    if revision != "local-cache-unresolved":
        kwargs["revision"] = revision
    return load_dataset(**kwargs)


def exclude_queries_with_missing_positive_documents(
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    corpus_ids: set[str],
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, Any]]:
    """Drop only queries whose positive qrels cannot be evaluated.

    The released upstream ArguAna archive and its Hub mirror contain five
    qrels whose judged positive document is absent from the corpus. Keeping
    such a query creates an artificial retrieval failure. The exclusion is
    deterministic and carried in the result manifest rather than silently
    changing relevance labels.
    """

    missing_by_query: dict[str, list[str]] = {}
    for query_id in query_ids:
        missing = sorted(
            doc_id
            for doc_id, grade in qrels[query_id].items()
            if float(grade) > 0.0 and doc_id not in corpus_ids
        )
        if missing:
            missing_by_query[query_id] = missing
    if not missing_by_query:
        return list(query_ids), {qid: dict(qrels[qid]) for qid in query_ids}, {
            "excluded_query_count": 0,
            "missing_positive_document_count": 0,
        }

    excluded = sorted(missing_by_query)
    retained = [qid for qid in query_ids if qid not in missing_by_query]
    if not retained:
        raise ValueError("all selected queries have missing positive documents")
    missing_docs = sorted(
        {doc_id for values in missing_by_query.values() for doc_id in values}
    )
    return retained, {qid: dict(qrels[qid]) for qid in retained}, {
        "excluded_query_count": len(excluded),
        "excluded_query_ids_sha256": ids_fingerprint(excluded),
        "missing_positive_document_count": len(missing_docs),
        "missing_positive_document_ids_sha256": ids_fingerprint(missing_docs),
        "policy": "exclude_queries_with_missing_positive_documents",
    }


def prepare_beir_dataset(
    dataset_name: str,
    config: BenchmarkConfig,
    dataset_revision: str,
    qrels_revision: str,
    hf_home: Path,
) -> PreparedDataset:
    """Download pinned BEIR rows/qrels and apply the declared split policy."""

    from huggingface_hub import hf_hub_download

    name = canonical_dataset_name(dataset_name)
    spec = DATASET_SPECS[name]
    hf_home = Path(hf_home).resolve()
    datasets_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"

    qrels_by_split: dict[str, dict[str, dict[str, float]]] = {}
    qrels_files: dict[str, str] = {}
    for split in spec.qrel_splits:
        downloaded = hf_hub_download(
            repo_id=spec.qrels_repo,
            filename=f"{split}.tsv",
            repo_type="dataset",
            revision=None if qrels_revision == "local-cache-unresolved" else qrels_revision,
            cache_dir=str(hub_cache),
            local_files_only=config.offline,
        )
        qrels_files[split] = str(downloaded)
        qrels_by_split[split] = load_qrels_tsv(Path(downloaded))

    selected_qids, policy = select_evaluation_qids(
        qrels_by_split,
        config.split,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )
    if config.max_queries is not None:
        if config.max_queries < 1:
            raise ValueError("max_queries must be positive")
        selected_qids = _stable_hash_order(selected_qids, config.split_seed)[
            : config.max_queries
        ]
        policy["max_queries_debug_limit"] = config.max_queries
        policy["beir_comparable_full_test"] = False

    source_qrels = qrels_by_split[str(policy["source_qrels"])]
    selected_set = set(selected_qids)
    selected_qrels = {
        qid: dict(source_qrels[qid]) for qid in selected_qids if qid in source_qrels
    }
    if set(selected_qrels) != selected_set:
        missing = sorted(selected_set - set(selected_qrels))[:5]
        raise ValueError(f"selected qids missing from qrels: {missing}")

    query_rows = _load_beir_rows(
        spec.dataset_repo,
        "queries",
        "queries",
        dataset_revision,
        datasets_cache,
        config.offline,
    )
    query_by_id: dict[str, str] = {}
    for row in query_rows:
        qid = _row_id(row)
        if qid in selected_set:
            query_by_id[qid] = str(row.get("text", row.get("query", "")) or "")
    missing_queries = sorted(selected_set - set(query_by_id))
    if missing_queries:
        raise ValueError(f"qrels reference missing query rows: {missing_queries[:5]}")
    # Source row order is stable and makes cache arrays easy to audit.
    query_ids = [qid for qid in query_by_id]
    query_texts = [query_by_id[qid] for qid in query_ids]
    selected_qrels = {qid: selected_qrels[qid] for qid in query_ids}

    corpus_rows = _load_beir_rows(
        spec.dataset_repo,
        "corpus",
        "corpus",
        dataset_revision,
        datasets_cache,
        config.offline,
    )
    corpus: list[tuple[str, str]] = []
    seen_docids: set[str] = set()
    for row in corpus_rows:
        docid = _row_id(row)
        if docid in seen_docids:
            raise ValueError(f"duplicate corpus id: {docid}")
        seen_docids.add(docid)
        text = format_e5_passage(
            str(row.get("text", "") or ""),
            None if row.get("title") is None else str(row.get("title")),
        )[len("passage: ") :]
        corpus.append((docid, text))

    query_ids, selected_qrels, missing_qrel_audit = (
        exclude_queries_with_missing_positive_documents(
            query_ids, selected_qrels, seen_docids
        )
    )
    query_texts = [query_by_id[qid] for qid in query_ids]
    if missing_qrel_audit["excluded_query_count"]:
        policy["missing_positive_qrel_audit"] = missing_qrel_audit
        policy["beir_comparable_full_test"] = False

    relevant = {
        docid
        for judgments in selected_qrels.values()
        for docid, grade in judgments.items()
        if float(grade) > 0.0
    }

    original_corpus_size = len(corpus)
    if config.max_corpus is not None and config.max_corpus < original_corpus_size:
        if config.max_corpus < len(relevant):
            raise ValueError(
                f"max_corpus={config.max_corpus} cannot retain {len(relevant)} "
                "relevant documents"
            )
        fillers = [docid for docid, _ in corpus if docid not in relevant]
        keep = set(relevant)
        keep.update(
            _stable_hash_order(fillers, config.split_seed)[
                : config.max_corpus - len(relevant)
            ]
        )
        corpus = [(docid, text) for docid, text in corpus if docid in keep]
        policy["max_corpus_debug_limit"] = config.max_corpus
        policy["original_corpus_size"] = original_corpus_size
        policy["all_positive_qrels_retained"] = True
        policy["beir_comparable_full_test"] = False

    corpus_ids = [docid for docid, _ in corpus]
    corpus_texts = [text for _, text in corpus]
    corpus_set = set(corpus_ids)
    selected_qrels = {
        qid: {docid: grade for docid, grade in rels.items() if docid in corpus_set}
        for qid, rels in selected_qrels.items()
    }
    if any(not any(grade > 0 for grade in rels.values()) for rels in selected_qrels.values()):
        raise ValueError("every evaluated query must retain at least one positive qrel")

    return PreparedDataset(
        name=name,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=selected_qrels,
        metadata={
            "dataset": name,
            "dataset_repo": spec.dataset_repo,
            "dataset_revision": dataset_revision,
            "qrels_repo": spec.qrels_repo,
            "qrels_revision": qrels_revision,
            "qrels_files": qrels_files,
            "split_policy": policy,
            "corpus_count": len(corpus_ids),
            "query_count": len(query_ids),
            "positive_judgments": sum(
                grade > 0 for rels in selected_qrels.values() for grade in rels.values()
            ),
            "corpus_ids_sha256": ids_fingerprint(corpus_ids),
            "query_ids_sha256": ids_fingerprint(query_ids),
        },
    )


class E5Encoder:
    """Lazy Hugging Face encoder that writes float32 NPY files atomically."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        hf_home: Path,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 32,
        use_amp: bool = True,
        offline: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA encoding requested but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.model_name = model_name
        self.revision = revision
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        cache_dir = str(Path(hf_home).resolve() / "hub")
        revision_arg = None if revision == "local-cache-unresolved" else revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision_arg,
            cache_dir=cache_dir,
            local_files_only=offline,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision_arg,
            cache_dir=cache_dir,
            local_files_only=offline,
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    def encode_to_npy(self, texts: Sequence[str], kind: str, output: Path) -> None:
        import torch
        import torch.nn.functional as functional

        if kind not in {"query", "passage"}:
            raise ValueError("kind must be query or passage")
        if not texts:
            raise ValueError("cannot encode an empty text collection")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
        matrix: np.memmap | None = None
        try:
            matrix = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(len(texts), self.embedding_dim),
            )
            started = time.perf_counter()
            for start in range(0, len(texts), self.batch_size):
                raw_batch = texts[start : start + self.batch_size]
                if kind == "query":
                    batch = [format_e5_query(text) for text in raw_batch]
                else:
                    batch = [format_e5_passage(text) for text in raw_batch]
                tokens = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                autocast_enabled = self.use_amp
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                ):
                    output_state = self.model(**tokens)
                    pooled = masked_mean_pool(
                        output_state.last_hidden_state, tokens["attention_mask"]
                    )
                pooled = functional.normalize(pooled.float(), p=2, dim=1)
                matrix[start : start + len(batch)] = pooled.cpu().numpy()
                if start == 0 or start + len(batch) == len(texts) or (
                    start // self.batch_size
                ) % 50 == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    done = start + len(batch)
                    print(
                        f"[{kind}] encoded {done}/{len(texts)} "
                        f"({done / elapsed:.1f} rows/s)",
                        file=sys.stderr,
                        flush=True,
                    )
            matrix.flush()
            del matrix
            matrix = None
            os.replace(temporary, output)
        finally:
            if matrix is not None:
                del matrix
            if temporary.exists():
                temporary.unlink()


def _validate_embedding_cache(
    path: Path,
    ids_path: Path,
    expected_ids: Sequence[str],
    metadata_path: Path | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> np.memmap:
    if not path.exists() or not ids_path.exists():
        raise FileNotFoundError(path)
    cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if cached_ids != list(expected_ids):
        raise ValueError(f"embedding id sidecar does not match requested rows: {path}")
    if metadata_path is not None:
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, expected in (expected_metadata or {}).items():
            if cached_metadata.get(str(key)) != _jsonable(expected):
                raise ValueError(
                    f"embedding metadata field {key!r} does not match: {path}"
                )
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(matrix, np.memmap) or matrix.ndim != 2:
        raise ValueError(f"embedding cache must be a two-dimensional NPY memmap: {path}")
    if matrix.dtype != np.float32 or len(matrix) != len(expected_ids):
        raise ValueError(f"embedding cache has wrong dtype/shape: {path}")
    sample_step = max(len(matrix) // 1024, 1)
    norms = np.linalg.norm(np.asarray(matrix[::sample_step]), axis=1)
    if not np.all(np.isfinite(norms)) or not np.allclose(norms, 1.0, atol=2e-4):
        raise ValueError(f"embedding cache is not finite and L2-normalised: {path}")
    return matrix


def load_or_encode_embeddings(
    path: Path,
    ids: Sequence[str],
    texts: Sequence[str],
    kind: str,
    encoder_getter: Callable[[], Any],
    metadata: Mapping[str, Any],
    force: bool = False,
) -> tuple[np.memmap, bool]:
    """Load a validated embedding cache or atomically build it once."""

    ids_path = path.with_suffix(".ids.json")
    metadata_path = path.with_suffix(".meta.json")
    if not force:
        try:
            return _validate_embedding_cache(
                path, ids_path, ids, metadata_path, metadata
            ), True
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
    encoder = encoder_getter()
    encoder.encode_to_npy(texts, kind, path)
    _atomic_write_json(ids_path, list(ids))
    full_metadata = dict(metadata)
    full_metadata.update(
        {
            "kind": kind,
            "rows": len(ids),
            "ids_sha256": ids_fingerprint(ids),
            "prefix_recipe": PREFIX_RECIPE,
            "pooling_recipe": POOLING_RECIPE,
        }
    )
    _atomic_write_json(metadata_path, full_metadata)
    return _validate_embedding_cache(
        path, ids_path, ids, metadata_path, metadata
    ), False


def _chunked_mean(matrix: np.ndarray, chunk_size: int = 16_384) -> np.ndarray:
    total = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, len(matrix), chunk_size):
        total += np.asarray(matrix[start : start + chunk_size], dtype=np.float64).sum(
            axis=0
        )
    return (total / len(matrix)).astype(np.float32, copy=False).reshape(1, -1)


def fit_projection_basis(
    corpus_embeddings: np.ndarray,
    projection_dim: int,
    sample_size: int,
    sample_seed: int,
    svd_seed: int,
    svd_iterations: int = 5,
) -> ProjectionBasis:
    """Fit a deterministic randomized SVD using corpus rows only."""

    from sklearn.utils.extmath import randomized_svd

    n_docs, embedding_dim = corpus_embeddings.shape
    sample_size = min(int(sample_size), n_docs)
    if not 1 <= projection_dim <= min(sample_size, embedding_dim):
        raise ValueError("projection_dim exceeds the sampled corpus matrix rank")
    mean = _chunked_mean(corpus_embeddings)
    rng = np.random.default_rng(sample_seed)
    sample_ids = rng.choice(n_docs, size=sample_size, replace=False)
    sample = np.asarray(corpus_embeddings[sample_ids], dtype=np.float32)
    sample -= mean
    _, _, vt = randomized_svd(
        sample,
        n_components=projection_dim,
        n_iter=svd_iterations,
        random_state=svd_seed,
        flip_sign=True,
    )
    return ProjectionBasis(
        passage_mean=mean,
        vectors=np.asarray(vt.T, dtype=np.float32),
        fit_sample_size=sample_size,
        fit_sample_seed=sample_seed,
        svd_seed=svd_seed,
        svd_iterations=svd_iterations,
    )


def projection_fingerprint(basis: ProjectionBasis) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(basis.passage_mean).view(np.uint8))
    digest.update(np.ascontiguousarray(basis.vectors).view(np.uint8))
    digest.update(
        _canonical_json_bytes(
            {
                "fit_sample_size": basis.fit_sample_size,
                "fit_sample_seed": basis.fit_sample_seed,
                "svd_seed": basis.svd_seed,
                "svd_iterations": basis.svd_iterations,
            }
        )
    )
    return digest.hexdigest()


def save_projection_basis(path: Path, basis: ProjectionBasis) -> None:
    _atomic_save_npz(
        path,
        passage_mean=np.asarray(basis.passage_mean, dtype=np.float32),
        vectors=np.asarray(basis.vectors, dtype=np.float32),
        fit_sample_size=np.int64(basis.fit_sample_size),
        fit_sample_seed=np.int64(basis.fit_sample_seed),
        svd_seed=np.int64(basis.svd_seed),
        svd_iterations=np.int64(basis.svd_iterations),
    )


def load_projection_basis(path: Path) -> ProjectionBasis:
    with np.load(path, allow_pickle=False) as data:
        return ProjectionBasis(
            passage_mean=np.asarray(data["passage_mean"], dtype=np.float32),
            vectors=np.asarray(data["vectors"], dtype=np.float32),
            fit_sample_size=int(np.asarray(data["fit_sample_size"]).item()),
            fit_sample_seed=int(np.asarray(data["fit_sample_seed"]).item()),
            svd_seed=int(np.asarray(data["svd_seed"]).item()),
            svd_iterations=int(np.asarray(data["svd_iterations"]).item()),
        )


def project_queries(queries: np.ndarray, basis: ProjectionBasis) -> np.ndarray:
    """Deployable online projection: intentionally no passage/query centering."""

    return np.asarray(np.asarray(queries, dtype=np.float32) @ basis.vectors, dtype=np.float32)


def project_corpus_chunk(corpus: np.ndarray, basis: ProjectionBasis) -> np.ndarray:
    values = np.asarray(corpus, dtype=np.float32)
    return np.asarray((values - basis.passage_mean) @ basis.vectors, dtype=np.float32)


def project_matrix_to_npy(
    source: np.ndarray,
    output: Path,
    basis: ProjectionBasis,
    kind: str,
    chunk_size: int,
    force: bool = False,
) -> tuple[np.memmap, bool]:
    expected_shape = (len(source), basis.vectors.shape[1])
    if not force and output.exists():
        cached = np.load(output, mmap_mode="r", allow_pickle=False)
        if cached.shape == expected_shape and cached.dtype == np.float32:
            return cached, True
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    matrix: np.memmap | None = None
    try:
        matrix = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=expected_shape
        )
        for start in range(0, len(source), chunk_size):
            chunk = source[start : start + chunk_size]
            if kind == "corpus":
                matrix[start : start + len(chunk)] = project_corpus_chunk(chunk, basis)
            elif kind == "query":
                matrix[start : start + len(chunk)] = project_queries(chunk, basis)
            else:
                raise ValueError("kind must be corpus or query")
        matrix.flush()
        del matrix
        matrix = None
        os.replace(temporary, output)
    finally:
        if matrix is not None:
            del matrix
        if temporary.exists():
            temporary.unlink()
    return np.load(output, mmap_mode="r", allow_pickle=False), False


def _topk_rows(
    scores: np.ndarray, ids: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape != ids.shape or scores.ndim != 2:
        raise ValueError("scores and ids must be equally shaped matrices")
    keep = min(k, scores.shape[1])
    partition = np.argpartition(-scores, keep - 1, axis=1)[:, :keep]
    values = np.take_along_axis(scores, partition, axis=1)
    chosen_ids = np.take_along_axis(ids, partition, axis=1)
    ordered_scores = np.empty_like(values)
    ordered_ids = np.empty_like(chosen_ids)
    for row in range(len(values)):
        order = np.lexsort((chosen_ids[row], -values[row]))
        ordered_scores[row] = values[row, order]
        ordered_ids[row] = chosen_ids[row, order]
    return ordered_scores, ordered_ids


def _merge_topk(
    old_scores: np.ndarray,
    old_ids: np.ndarray,
    new_scores: np.ndarray,
    new_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _topk_rows(
        np.concatenate((old_scores, new_scores), axis=1),
        np.concatenate((old_ids, new_ids), axis=1),
        k,
    )


def resolve_exact_backend(requested: str) -> str:
    if requested not in {"auto", "numpy", "cuda"}:
        raise ValueError("backend must be auto, numpy, or cuda")
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    if requested == "auto":
        return "cuda" if cuda else "numpy"
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA exact search requested but unavailable")
    return requested


def exact_topk_search(
    corpus: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    backend: str = "auto",
    query_batch_size: int = 128,
    doc_chunk_size: int = 16_384,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Memory-bounded exhaustive inner-product search on CPU or CUDA."""

    if corpus.ndim != 2 or queries.ndim != 2 or corpus.shape[1] != queries.shape[1]:
        raise ValueError("corpus/query embeddings must be compatible matrices")
    if not 1 <= top_k <= len(corpus):
        raise ValueError("top_k must fit the corpus")
    resolved = resolve_exact_backend(backend)
    all_scores = np.empty((len(queries), top_k), dtype=np.float32)
    all_ids = np.empty((len(queries), top_k), dtype=np.int64)
    started = time.perf_counter()

    if resolved == "numpy":
        for q_start in range(0, len(queries), query_batch_size):
            q = np.ascontiguousarray(queries[q_start : q_start + query_batch_size])
            best_scores = np.full((len(q), top_k), -np.inf, dtype=np.float32)
            best_ids = np.full((len(q), top_k), -1, dtype=np.int64)
            for d_start in range(0, len(corpus), doc_chunk_size):
                docs = np.ascontiguousarray(corpus[d_start : d_start + doc_chunk_size])
                scores = np.asarray(q @ docs.T, dtype=np.float32)
                ids = np.broadcast_to(
                    np.arange(d_start, d_start + len(docs), dtype=np.int64),
                    scores.shape,
                )
                new_scores, new_ids = _topk_rows(scores, ids, top_k)
                best_scores, best_ids = _merge_topk(
                    best_scores, best_ids, new_scores, new_ids, top_k
                )
            all_scores[q_start : q_start + len(q)] = best_scores
            all_ids[q_start : q_start + len(q)] = best_ids
    else:
        import torch

        device = torch.device("cuda")
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.cuda.synchronize()
            for q_start in range(0, len(queries), query_batch_size):
                # ``np.ascontiguousarray`` may return the original read-only
                # memmap.  Torch warns even though search never mutates it, so
                # make ownership/writability explicit at the CUDA boundary.
                q_host = np.array(
                    queries[q_start : q_start + query_batch_size],
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )
                q = torch.as_tensor(q_host, dtype=torch.float32, device=device)
                best_scores = torch.full(
                    (len(q), top_k), -torch.inf, dtype=torch.float32, device=device
                )
                best_ids = torch.full(
                    (len(q), top_k), -1, dtype=torch.int64, device=device
                )
                for d_start in range(0, len(corpus), doc_chunk_size):
                    docs_host = np.array(
                        corpus[d_start : d_start + doc_chunk_size],
                        dtype=np.float32,
                        order="C",
                        copy=True,
                    )
                    docs = torch.as_tensor(docs_host, dtype=torch.float32, device=device)
                    scores = q @ docs.T
                    keep = min(top_k, len(docs))
                    values, local = torch.topk(scores, keep, dim=1, largest=True)
                    global_ids = local.to(torch.int64) + d_start
                    merged_scores = torch.cat((best_scores, values), dim=1)
                    merged_ids = torch.cat((best_ids, global_ids), dim=1)
                    best_scores, positions = torch.topk(
                        merged_scores, top_k, dim=1, largest=True
                    )
                    best_ids = torch.gather(merged_ids, 1, positions)
                host_scores = best_scores.cpu().numpy()
                host_ids = best_ids.cpu().numpy()
                # Make the final tie rule explicit and platform-independent.
                host_scores, host_ids = _topk_rows(host_scores, host_ids, top_k)
                all_scores[q_start : q_start + len(q)] = host_scores
                all_ids[q_start : q_start + len(q)] = host_ids
            torch.cuda.synchronize()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    return all_scores, all_ids, elapsed_ms, resolved


def remove_identical_query_documents(
    rankings: np.ndarray,
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    target_k: int,
) -> np.ndarray:
    """Apply BEIR's default ``query_id != document_id`` evaluation rule.

    Callers must retrieve one spare result whenever any query id occurs in the
    corpus.  Filtering before metric computation preserves a true top-100
    list after removal instead of silently evaluating only 99 documents.
    """

    if rankings.ndim != 2 or len(rankings) != len(query_ids):
        raise ValueError("rankings must align one row per query")
    if not 1 <= target_k <= rankings.shape[1]:
        raise ValueError("target_k must fit the supplied ranking width")
    corpus = np.asarray([str(docid) for docid in corpus_ids], dtype=object)
    if np.any(rankings < 0) or np.any(rankings >= len(corpus)):
        raise ValueError("ranking contains an invalid local corpus index")
    filtered = np.empty((len(rankings), target_k), dtype=np.int64)
    for row, (qid, local_ids) in enumerate(zip(query_ids, rankings, strict=True)):
        keep = [int(index) for index in local_ids if str(corpus[index]) != str(qid)]
        if len(keep) < target_k:
            raise ValueError(
                "ranking lacks a spare result after BEIR identical-id filtering"
            )
        filtered[row] = keep[:target_k]
    return filtered


def _require_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required for the PQ protocols") from exc
    return faiss


def build_pq_index(
    corpus: np.ndarray,
    m: int,
    nbits: int,
    train_size: int,
    train_seed: int,
    chunk_size: int,
    faiss_threads: int = 1,
    iterations: int = 25,
) -> Any:
    """Train/add a deterministic CPU IndexPQ with inner-product scoring."""

    faiss = _require_faiss()
    if corpus.shape[1] % m:
        raise ValueError("PQ M must divide the embedding dimension")
    train_size = min(int(train_size), len(corpus))
    centroids = 1 << nbits
    if train_size < centroids:
        raise ValueError(f"PQ training needs at least {centroids} rows")
    if faiss_threads > 0:
        faiss.omp_set_num_threads(faiss_threads)
    sample_ids = np.random.default_rng(train_seed).choice(
        len(corpus), size=train_size, replace=False
    )
    training = np.ascontiguousarray(corpus[sample_ids], dtype=np.float32)
    index = faiss.IndexPQ(corpus.shape[1], m, nbits, faiss.METRIC_INNER_PRODUCT)
    if hasattr(index.pq, "cp"):
        index.pq.cp.seed = int(train_seed)
        index.pq.cp.niter = int(iterations)
    index.train(training)
    for start in range(0, len(corpus), chunk_size):
        index.add(np.ascontiguousarray(corpus[start : start + chunk_size]))
    if int(index.ntotal) != len(corpus):
        raise RuntimeError("FAISS index did not receive every corpus row")
    return index


def save_faiss_index_atomic(path: Path, index: Any) -> None:
    faiss = _require_faiss()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        faiss.write_index(index, str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rerank_projected_shortlist(
    projected_corpus: np.ndarray,
    projected_queries: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rerank each PQ shortlist with exact projected dot products."""

    if candidate_ids.ndim != 2 or len(candidate_ids) != len(projected_queries):
        raise ValueError("candidate_ids must align one row per query")
    if np.any(candidate_ids < 0) or np.any(candidate_ids >= len(projected_corpus)):
        raise ValueError("candidate id lies outside the corpus")
    keep = min(top_k, candidate_ids.shape[1])
    scores_out = np.empty((len(candidate_ids), keep), dtype=np.float32)
    ids_out = np.empty((len(candidate_ids), keep), dtype=np.int64)
    latency_ms = np.empty(len(candidate_ids), dtype=np.float64)
    for row, candidates in enumerate(candidate_ids):
        started = time.perf_counter_ns()
        docs = np.asarray(projected_corpus[candidates], dtype=np.float32)
        scores = np.asarray(docs @ projected_queries[row], dtype=np.float32)
        order = np.lexsort((candidates, -scores))[:keep]
        scores_out[row] = scores[order]
        ids_out[row] = candidates[order]
        latency_ms[row] = (time.perf_counter_ns() - started) / 1_000_000.0
    return scores_out, ids_out, latency_ms


def evaluate_rankings(
    rankings: np.ndarray,
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
) -> dict[str, np.ndarray]:
    """Return aligned per-query BEIR-style graded metrics."""

    if rankings.ndim != 2 or len(rankings) != len(query_ids):
        raise ValueError("rankings must have one row per query")
    if rankings.shape[1] < 100:
        raise ValueError("rankings need at least 100 columns for Recall@100")
    corpus_ids_array = np.asarray([str(docid) for docid in corpus_ids], dtype=object)
    if np.any(rankings < 0) or np.any(rankings >= len(corpus_ids_array)):
        raise ValueError("ranking contains an invalid local corpus index")
    result = {name: np.zeros(len(query_ids), dtype=np.float64) for name in METRIC_NAMES}
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    for row, qid in enumerate(query_ids):
        judgments = {str(k): float(v) for k, v in qrels[str(qid)].items()}
        positives = {docid for docid, grade in judgments.items() if grade > 0.0}
        if not positives:
            raise ValueError(f"query {qid} has no positive qrels")
        retrieved = corpus_ids_array[rankings[row]]
        top10 = retrieved[:10]
        gains = np.asarray([judgments.get(str(docid), 0.0) for docid in top10])
        dcg = float(np.sum(gains * discounts))
        ideal_gains = np.asarray(
            sorted((grade for grade in judgments.values() if grade > 0.0), reverse=True)[
                :10
            ],
            dtype=np.float64,
        )
        idcg = float(np.sum(ideal_gains * discounts[: len(ideal_gains)]))
        result["ndcg_at_10"][row] = dcg / idcg if idcg else 0.0
        hits10 = sum(str(docid) in positives for docid in retrieved[:10])
        hits100 = sum(str(docid) in positives for docid in retrieved[:100])
        result["recall_at_10"][row] = hits10 / len(positives)
        result["recall_at_100"][row] = hits100 / len(positives)
        reciprocal = 0.0
        for rank, docid in enumerate(top10, start=1):
            if str(docid) in positives:
                reciprocal = 1.0 / rank
                break
        result["mrr_at_10"][row] = reciprocal
    return result


def _bootstrap_means(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
    chunk_size: int = 256,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    output = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, chunk_size):
        count = min(chunk_size, n_bootstrap - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        output[start : start + count] = values[indices].mean(axis=1)
    return output


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    distribution = _bootstrap_means(values, n_bootstrap, seed)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_samples": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }


def paired_bootstrap_delta(
    candidate: np.ndarray,
    reference: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("paired samples must be equally shaped vectors")
    differences = candidate - reference
    distribution = _bootstrap_means(differences, n_bootstrap, seed)
    nonpositive = (np.count_nonzero(distribution <= 0.0) + 1) / (n_bootstrap + 1)
    nonnegative = (np.count_nonzero(distribution >= 0.0) + 1) / (n_bootstrap + 1)
    return {
        "mean_delta": float(np.mean(differences)),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "two_sided_bootstrap_p": float(min(1.0, 2.0 * min(nonpositive, nonnegative))),
        "bootstrap_samples": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }


def summarise_method_metrics(
    metrics_by_method: Mapping[str, Mapping[str, np.ndarray]],
    n_bootstrap: int,
    seed: int,
    reference: str = "raw_exact",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if reference not in metrics_by_method:
        raise ValueError(f"reference method {reference} is missing")
    summary: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for method, metrics in metrics_by_method.items():
        summary[method] = {}
        for metric_index, metric in enumerate(METRIC_NAMES):
            metric_seed = seed + 10_000 * metric_index
            summary[method][metric] = bootstrap_mean_ci(
                metrics[metric], n_bootstrap, metric_seed
            )
        if method != reference:
            paired[f"{method}_minus_{reference}"] = {}
            for metric_index, metric in enumerate(METRIC_NAMES):
                metric_seed = seed + 10_000 * metric_index
                paired[f"{method}_minus_{reference}"][metric] = paired_bootstrap_delta(
                    metrics[metric],
                    metrics_by_method[reference][metric],
                    n_bootstrap,
                    metric_seed,
                )
    return summary, paired


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def _software_hardware_metadata() -> dict[str, Any]:
    packages: dict[str, str] = {"numpy": np.__version__}
    gpu: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        packages["torch"] = torch.__version__
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            gpu["name"] = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            gpu["total_memory_bytes"] = int(properties.total_memory)
    except ImportError:
        pass
    for module_name in ("transformers", "datasets", "faiss", "sklearn"):
        try:
            module = __import__(module_name)
            packages[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "gpu": gpu,
    }


def _cache_sidecars(prepared: PreparedDataset, cache_root: Path) -> Path:
    data_key = _fingerprint(
        {
            "dataset_revision": prepared.metadata["dataset_revision"],
            "qrels_revision": prepared.metadata["qrels_revision"],
            "corpus_ids": prepared.metadata["corpus_ids_sha256"],
            "query_ids": prepared.metadata["query_ids_sha256"],
            "split": prepared.metadata["split_policy"],
        }
    )
    directory = cache_root / "data" / prepared.name / data_key
    _atomic_write_json(directory / "corpus_ids.json", prepared.corpus_ids)
    _atomic_write_json(directory / "query_ids.json", prepared.query_ids)
    _atomic_write_json(directory / "qrels.json", prepared.qrels)
    _atomic_write_json(directory / "metadata.json", prepared.metadata)
    return directory


def run_dataset_benchmark(
    dataset_name: str,
    config: BenchmarkConfig,
    cache_root: Path,
    hf_home: Path,
    model_revision: str,
    dataset_revision: str,
    qrels_revision: str,
    encoder_getter: Callable[[], Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute every method on one aligned set of corpus/query rows."""

    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    prepared = prepare_beir_dataset(
        dataset_name, config, dataset_revision, qrels_revision, hf_home
    )
    data_cache = _cache_sidecars(prepared, cache_root)
    model_recipe = {
        "model": config.model_name,
        "revision": model_revision,
        "dataset_revision": dataset_revision,
        "max_length": config.max_length,
        "prefix": PREFIX_RECIPE,
        "pooling": POOLING_RECIPE,
        "amp": config.use_amp,
        "encoding_device": config.device,
    }
    model_key = _slug(config.model_name) + "-" + _fingerprint(model_recipe)
    embeddings_dir = cache_root / "embeddings" / prepared.name / model_key
    corpus_key = prepared.metadata["corpus_ids_sha256"][:16]
    query_key = prepared.metadata["query_ids_sha256"][:16]
    corpus_path = embeddings_dir / f"corpus-{corpus_key}.npy"
    query_path = embeddings_dir / f"queries-{query_key}.npy"
    embedding_metadata = {
        **model_recipe,
        "dataset_revision": dataset_revision,
        "qrels_revision": qrels_revision,
    }
    corpus_embeddings, corpus_cache_hit = load_or_encode_embeddings(
        corpus_path,
        prepared.corpus_ids,
        prepared.corpus_texts,
        "passage",
        encoder_getter,
        embedding_metadata,
        config.force_reencode,
    )
    query_embeddings, query_cache_hit = load_or_encode_embeddings(
        query_path,
        prepared.query_ids,
        prepared.query_texts,
        "query",
        encoder_getter,
        embedding_metadata,
        config.force_reencode,
    )
    config.validate(
        corpus_embeddings.shape[1], len(corpus_embeddings), len(query_embeddings)
    )

    corpus_id_set = set(prepared.corpus_ids)
    has_identical_ids = any(qid in corpus_id_set for qid in prepared.query_ids)
    spare = 1 if has_identical_ids else 0
    if config.retrieve_k + spare > len(corpus_embeddings):
        raise ValueError("corpus is too small for BEIR identical-id filtering")
    if config.candidate_k + spare > len(corpus_embeddings):
        raise ValueError("candidate_k leaves no spare for BEIR identical-id filtering")
    exact_search_k = config.retrieve_k + spare
    candidate_search_k = config.candidate_k + spare

    raw_scores, raw_ids, raw_ms, raw_backend = exact_topk_search(
        corpus_embeddings,
        query_embeddings,
        exact_search_k,
        backend=config.exact_backend,
        query_batch_size=config.exact_query_batch_size,
        doc_chunk_size=config.exact_doc_chunk_size,
    )
    del raw_scores
    rankings: dict[str, np.ndarray] = {
        "raw_exact": remove_identical_query_documents(
            raw_ids,
            prepared.query_ids,
            prepared.corpus_ids,
            config.retrieve_k,
        )
    }
    timings: dict[str, Any] = {
        "raw_exact": {
            "total_ms": raw_ms,
            "mean_per_query_ms": raw_ms / len(query_embeddings),
            "backend": raw_backend,
        }
    }
    protocol: dict[str, Any] = {
        "raw_exact": "exhaustive inner product over original L2-normalised E5 vectors",
        "identical_id_policy": (
            "BEIR default: exclude result document when document_id == query_id; "
            "retrieve one spare result before filtering"
        ),
    }
    cache_info: dict[str, Any] = {
        "data_sidecars": str(data_cache),
        "corpus_embeddings": str(corpus_path),
        "query_embeddings": str(query_path),
        "corpus_embedding_cache_hit": corpus_cache_hit,
        "query_embedding_cache_hit": query_cache_hit,
    }

    if config.projection_dim > 0:
        projection_recipe = {
            "corpus_ids": prepared.metadata["corpus_ids_sha256"],
            "model_recipe": model_recipe,
            "projection_dim": config.projection_dim,
            "fit_size": min(config.projection_fit_size, len(corpus_embeddings)),
            "fit_seed": config.projection_fit_seed,
            "svd_seed": config.svd_seed,
            "svd_iterations": config.svd_iterations,
            "online_query": "q @ V (no centering)",
            "document": "(d - corpus_mean) @ V",
        }
        projection_dir = cache_root / "projection" / prepared.name / _fingerprint(
            projection_recipe
        )
        basis_path = projection_dir / "basis.npz"
        basis_cache_hit = basis_path.exists() and not config.force_rebuild
        if basis_cache_hit:
            basis = load_projection_basis(basis_path)
        else:
            basis = fit_projection_basis(
                corpus_embeddings,
                config.projection_dim,
                config.projection_fit_size,
                config.projection_fit_seed,
                config.svd_seed,
                config.svd_iterations,
            )
            save_projection_basis(basis_path, basis)
        basis_hash = projection_fingerprint(basis)
        projected_corpus, projected_corpus_hit = project_matrix_to_npy(
            corpus_embeddings,
            projection_dir / "corpus.npy",
            basis,
            "corpus",
            config.projection_chunk_size,
            config.force_rebuild,
        )
        projected_queries, projected_query_hit = project_matrix_to_npy(
            query_embeddings,
            projection_dir / f"queries-{query_key}.npy",
            basis,
            "query",
            config.projection_chunk_size,
            config.force_rebuild,
        )
        _, projected_ids, projected_ms, projected_backend = exact_topk_search(
            projected_corpus,
            projected_queries,
            exact_search_k,
            backend=config.exact_backend,
            query_batch_size=config.exact_query_batch_size,
            doc_chunk_size=config.exact_doc_chunk_size,
        )
        rankings["projected_exact"] = remove_identical_query_documents(
            projected_ids,
            prepared.query_ids,
            prepared.corpus_ids,
            config.retrieve_k,
        )
        timings["projected_exact"] = {
            "total_ms": projected_ms,
            "mean_per_query_ms": projected_ms / len(query_embeddings),
            "backend": projected_backend,
        }
        protocol["projected_exact"] = (
            "exhaustive projected IP; corpus=(d-mean)@V; online query=q@V"
        )

        pq_recipe = {
            "basis_sha256": basis_hash,
            "corpus_ids": prepared.metadata["corpus_ids_sha256"],
            "m": config.pq_m,
            "nbits": config.pq_nbits,
            "train_size": min(config.pq_train_size, len(projected_corpus)),
            "train_seed": config.pq_train_seed,
            "iterations": config.pq_iterations,
            "metric": "inner_product",
            "faiss_threads": config.faiss_threads,
        }
        pq_dir = cache_root / "pq" / prepared.name / _fingerprint(pq_recipe)
        pq_path = pq_dir / "index.faiss"
        pq_meta_path = pq_dir / "metadata.json"
        faiss = _require_faiss()
        if config.faiss_threads > 0:
            faiss.omp_set_num_threads(config.faiss_threads)
        pq_cache_hit = pq_path.exists() and pq_meta_path.exists() and not config.force_rebuild
        if pq_cache_hit:
            index = faiss.read_index(str(pq_path))
            if int(index.d) != config.projection_dim or int(index.ntotal) != len(
                projected_corpus
            ):
                raise ValueError("cached PQ index dimension/count mismatch")
        else:
            pq_started = time.perf_counter()
            index = build_pq_index(
                projected_corpus,
                config.pq_m,
                config.pq_nbits,
                config.pq_train_size,
                config.pq_train_seed,
                config.projection_chunk_size,
                config.faiss_threads,
                config.pq_iterations,
            )
            pq_recipe["build_seconds"] = time.perf_counter() - pq_started
            save_faiss_index_atomic(pq_path, index)
            _atomic_write_json(pq_meta_path, pq_recipe)

        query_array = np.ascontiguousarray(projected_queries, dtype=np.float32)
        # Warm-up removes one-time FAISS page/codebook initialisation from timing.
        index.search(query_array[:1], candidate_search_k)
        pq_started = time.perf_counter()
        pq_scores, unfiltered_candidates = index.search(
            query_array, candidate_search_k
        )
        pq_total_ms = (time.perf_counter() - pq_started) * 1_000.0
        if np.any(unfiltered_candidates < 0):
            raise RuntimeError("FAISS PQ returned an incomplete shortlist")
        candidates = remove_identical_query_documents(
            unfiltered_candidates,
            prepared.query_ids,
            prepared.corpus_ids,
            config.candidate_k,
        )
        rankings["pq_only"] = candidates[:, : config.retrieve_k].copy()
        _, reranked, rerank_latency = rerank_projected_shortlist(
            projected_corpus, projected_queries, candidates, config.retrieve_k
        )
        rankings["pq_projected_rerank"] = reranked
        timings["pq_only"] = {
            "total_ms": pq_total_ms,
            "mean_per_query_ms": pq_total_ms / len(query_embeddings),
            "candidate_k": config.candidate_k,
        }
        timings["pq_projected_rerank"] = {
            "candidate_selection_total_ms": pq_total_ms,
            "exact_rerank": _percentiles(rerank_latency),
            "mean_end_to_end_per_query_ms": pq_total_ms / len(query_embeddings)
            + float(rerank_latency.mean()),
        }
        protocol["pq_only"] = "FAISS IndexPQ approximate ordering"
        protocol["pq_projected_rerank"] = (
            "same PQ candidates, reranked by exact projected dot products"
        )
        cache_info.update(
            {
                "projection_basis": str(basis_path),
                "projection_basis_sha256": basis_hash,
                "projection_basis_cache_hit": basis_cache_hit,
                "projected_corpus_cache_hit": projected_corpus_hit,
                "projected_queries_cache_hit": projected_query_hit,
                "pq_index": str(pq_path),
                "pq_index_cache_hit": pq_cache_hit,
            }
        )
        projection_metadata: dict[str, Any] | None = {
            **projection_recipe,
            "basis_sha256": basis_hash,
            "passage_mean_source": "corpus embeddings only",
            "test_query_statistics_used": False,
        }
        pq_metadata: dict[str, Any] | None = pq_recipe
        del pq_scores
    else:
        projection_metadata = None
        pq_metadata = None

    metrics_by_method = {
        method: evaluate_rankings(
            method_rankings,
            prepared.query_ids,
            prepared.corpus_ids,
            prepared.qrels,
        )
        for method, method_rankings in rankings.items()
    }
    metrics_summary, paired = summarise_method_metrics(
        metrics_by_method,
        config.bootstrap_samples,
        config.bootstrap_seed,
        reference="raw_exact",
    )
    records: list[dict[str, Any]] = []
    for row, qid in enumerate(prepared.query_ids):
        for method, method_rankings in rankings.items():
            records.append(
                {
                    "dataset": prepared.name,
                    "split": config.split,
                    "query_id": qid,
                    "method": method,
                    "metrics": {
                        metric: float(values[row])
                        for metric, values in metrics_by_method[method].items()
                    },
                    "ranked_doc_ids": [
                        prepared.corpus_ids[int(index)] for index in method_rankings[row]
                    ],
                }
            )

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": prepared.metadata,
        "model": {
            "name": config.model_name,
            "revision": model_revision,
            "embedding_dimension": int(corpus_embeddings.shape[1]),
            "max_length": config.max_length,
            "prefix_recipe": PREFIX_RECIPE,
            "pooling_recipe": POOLING_RECIPE,
            "l2_normalized": True,
            "encoding_device_requested": config.device,
            "mixed_precision": config.use_amp,
        },
        "config": dataclasses.asdict(config),
        "protocol": protocol,
        "projection": projection_metadata,
        "pq": pq_metadata,
        "metrics": metrics_summary,
        "paired_query_bootstrap_deltas": paired,
        "timings": timings,
        "cache": cache_info,
        "environment": _software_hardware_metadata(),
        "warning": (
            "Smoke/debug subsets are not BEIR-comparable."
            if config.smoke or config.max_queries or config.max_corpus
            else None
        ),
    }
    return summary, records


def _parse_revision_overrides(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"revision override must be DATASET=REVISION, got {value!r}")
        name, revision = value.split("=", 1)
        result[canonical_dataset_name(name)] = revision.strip()
    return result


def _output_for_dataset(path: Path, dataset: str, multiple: bool) -> Path:
    text = str(path)
    if "{dataset}" in text:
        return Path(text.format(dataset=dataset))
    if not multiple:
        return path
    return path.with_name(f"{path.stem}-{dataset}{path.suffix or '.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--all-required", action="store_true")
    parser.add_argument("--split", choices=("validation", "test", "official-test"), default="validation")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--max-corpus", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision", action="append", default=[], metavar="DATASET=REV")
    parser.add_argument("--qrels-revision", action="append", default=[], metavar="DATASET=REV")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=672)
    parser.add_argument("--no-projection", action="store_true")
    parser.add_argument("--projection-fit-size", type=int, default=50_000)
    parser.add_argument("--projection-fit-seed", type=int, default=17)
    parser.add_argument("--svd-seed", type=int, default=42)
    parser.add_argument("--svd-iterations", type=int, default=5)
    parser.add_argument("--pq-m", type=int, default=96)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--pq-train-size", type=int, default=50_000)
    parser.add_argument("--pq-train-seed", type=int, default=42)
    parser.add_argument("--pq-iterations", type=int, default=25)
    parser.add_argument("--candidate-k", type=int, default=1_000)
    parser.add_argument("--retrieve-k", type=int, default=100)
    parser.add_argument("--exact-backend", choices=("auto", "numpy", "cuda"), default="auto")
    parser.add_argument("--exact-query-batch-size", type=int, default=128)
    parser.add_argument("--exact-doc-chunk-size", type=int, default=16_384)
    parser.add_argument("--projection-chunk-size", type=int, default=16_384)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--per-query-jsonl", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    projection_dim = 0 if args.no_projection else args.projection_dim
    config = BenchmarkConfig(
        split=args.split,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        max_queries=args.max_queries,
        max_corpus=args.max_corpus,
        model_name=args.model,
        max_length=args.max_length,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
        use_amp=not args.no_amp,
        projection_dim=projection_dim,
        projection_fit_size=args.projection_fit_size,
        projection_fit_seed=args.projection_fit_seed,
        svd_seed=args.svd_seed,
        svd_iterations=args.svd_iterations,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits,
        pq_train_size=args.pq_train_size,
        pq_train_seed=args.pq_train_seed,
        pq_iterations=args.pq_iterations,
        candidate_k=args.candidate_k,
        retrieve_k=args.retrieve_k,
        exact_backend=args.exact_backend,
        exact_query_batch_size=args.exact_query_batch_size,
        exact_doc_chunk_size=args.exact_doc_chunk_size,
        projection_chunk_size=args.projection_chunk_size,
        faiss_threads=args.faiss_threads,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        force_reencode=args.force_reencode,
        force_rebuild=args.force_rebuild,
        offline=args.offline,
        smoke=args.smoke,
    )
    if args.smoke:
        # Keep the real E5 model and real SciFact qrels, but make every systems
        # component finish quickly.  All positive qrels for selected queries
        # are retained in the deterministic corpus subset.
        config.split = "test"
        config.max_queries = min(config.max_queries or 8, 8)
        config.max_corpus = min(config.max_corpus or 1_024, 1_024)
        config.projection_dim = 64
        config.projection_fit_size = 1_024
        config.pq_m = 8
        config.pq_nbits = 4
        config.pq_train_size = 1_024
        config.candidate_k = 100
        config.retrieve_k = 100
        config.bootstrap_samples = min(config.bootstrap_samples, 500)
        config.encode_batch_size = min(config.encode_batch_size, 16)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    datasets = list(REQUIRED_DATASETS) if args.all_required else (
        [canonical_dataset_name(name) for name in args.dataset] or ["scifact"]
    )
    if args.smoke:
        datasets = ["scifact"]
    if len(set(datasets)) != len(datasets):
        parser.error("duplicate --dataset values")
    cache_root = args.cache_root.expanduser().resolve()
    hf_home = (args.hf_home or (cache_root / "huggingface")).expanduser().resolve()
    configure_hf_cache(hf_home)
    seed_everything(args.split_seed)
    config = _config_from_args(args)
    try:
        dataset_overrides = _parse_revision_overrides(args.dataset_revision)
        qrels_overrides = _parse_revision_overrides(args.qrels_revision)
    except ValueError as exc:
        parser.error(str(exc))

    model_revision = resolve_hf_revision(
        config.model_name, "model", args.model_revision, config.offline
    )
    encoder_holder: list[E5Encoder] = []

    def encoder_getter() -> E5Encoder:
        if not encoder_holder:
            encoder_holder.append(
                E5Encoder(
                    config.model_name,
                    model_revision,
                    hf_home,
                    device=config.device,
                    max_length=config.max_length,
                    batch_size=config.encode_batch_size,
                    use_amp=config.use_amp,
                    offline=config.offline,
                )
            )
        return encoder_holder[0]

    summaries: dict[str, Any] = {}
    all_records: dict[str, list[dict[str, Any]]] = {}
    for name in datasets:
        spec = DATASET_SPECS[name]
        dataset_revision = resolve_hf_revision(
            spec.dataset_repo,
            "dataset",
            dataset_overrides.get(name),
            config.offline,
        )
        qrels_revision = resolve_hf_revision(
            spec.qrels_repo,
            "dataset",
            qrels_overrides.get(name),
            config.offline,
        )
        print(
            f"Running {name}: data={dataset_revision}, qrels={qrels_revision}, "
            f"model={model_revision}",
            file=sys.stderr,
            flush=True,
        )
        summary, records = run_dataset_benchmark(
            name,
            config,
            cache_root,
            hf_home,
            model_revision,
            dataset_revision,
            qrels_revision,
            encoder_getter,
        )
        summaries[name] = summary
        all_records[name] = records

    aggregate = {
        "schema_version": 1,
        "datasets": summaries,
        "required_dataset_coverage": {
            name: name in summaries for name in REQUIRED_DATASETS
        },
        "frozen_revisions": {
            "model": {config.model_name: model_revision},
            "datasets": {
                name: summaries[name]["dataset"]["dataset_revision"] for name in datasets
            },
            "qrels": {
                name: summaries[name]["dataset"]["qrels_revision"] for name in datasets
            },
        },
    }
    if args.output_json:
        output = args.output_json.expanduser().resolve()
        _atomic_write_json(output, aggregate)
        print(f"Wrote {output}", file=sys.stderr)
    else:
        print(json.dumps(_jsonable(aggregate), indent=2, ensure_ascii=False))

    if args.per_query_jsonl:
        base = args.per_query_jsonl.expanduser().resolve()
        multiple = len(datasets) > 1
        for name, records in all_records.items():
            path = _output_for_dataset(base, name, multiple)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
            try:
                with temporary.open("w", encoding="utf-8") as stream:
                    for record in records:
                        stream.write(
                            json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            print(f"Wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
