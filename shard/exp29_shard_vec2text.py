"""Experiment 29: outcome-level Vec2Text attacks against actual SHARD views.

This is deliberately *not* the old global-linear inversion baseline.  It
encodes cached AG News records and controlled synthetic PII with the exact
GTR-base embedding path used by the official Vec2Text ``gtr-base`` corrector,
fits a full PCA basis, exposes the true SHARD public prefix and a cell-keyed
residual, and attacks the globally largest cell.  If that cell contains no
controlled PII, a separately labelled secondary cohort attacks the cell with
the largest PII population so that PII recall has a non-empty denominator.
The reconstruction is deliberately attacker-favourable: the observer is given
the true full PCA basis V and document mean mu.  Thus the only unknown in the
``unknown_key`` and fitted-alignment cases is the target cell key H.
The evaluated attacker views are:

* the raw native GTR embedding (exact-geometry reference control);
* public prefix only;
* public prefix plus the keyed residual treated as if the key were identity;
* an oracle that knows the cell key;
* minimum-norm OLS fitted on nested m=8/16/32/64 in-cell known pairs;
* orthogonal Procrustes at selected known-pair budgets.

Every reconstructed native GTR embedding is passed to the official pretrained
Vec2Text corrector.  The script reports text token-F1, sentence BLEU, exact
match, controlled-PII item recall, pre-decoding geometry cosine, and cosine
between the target embedding and the embedding of the decoded text.

Important scope: the official public corrector supports GTR-base.  Therefore
this experiment is an outcome-level GTR case study; it is not direct evidence
about e5 inversion and must not be generalized to arbitrary encoders.  The
corrector requires its native, unnormalized mean-pooled GTR vectors; this is a
checkpoint-compatible SHARD split stress test, not the paper's exact
L2-normalized retrieval configuration.

Windows compatibility: vec2text imports ``resource`` and vec2text 0.0.13 uses
an old custom-transformers initialization pattern.  Before importing vec2text
we install the required no-op resource shim.  At model load time we construct
the nested T5/GTR modules from config and then let the *outer official jxm
checkpoint* load every weight.  This avoids the Transformers 5.x prohibition
on nested ``from_pretrained`` inside a meta-device context.  Official checkpoint
parameters are cast to FP16 for compatible inference on the 8-GiB GPU; there is
no training or fine-tuning.  The raw-control sanity check prevents a silent
embedding-path bug.

Examples (PowerShell, from the repository root)::

    # Memory-capped validation first.
    python shard/exp29_shard_vec2text.py --profile smoke \
        --out results/exp29_shard_vec2text/smoke

    # Primary run.  It is checkpointed after every method and can be resumed.
    python shard/exp29_shard_vec2text.py --profile full \
        --out results/exp29_shard_vec2text
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import types
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


# Must precede ``import vec2text`` on Windows.
if "resource" not in sys.modules:
    _resource = types.ModuleType("resource")
    _resource.RLIMIT_CORE = 4
    _resource.RLIM_INFINITY = -1
    _resource.setrlimit = lambda *args, **kwargs: None
    _resource.getrlimit = lambda *args, **kwargs: (_resource.RLIM_INFINITY,) * 2
    sys.modules["resource"] = _resource

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import sacrebleu
import torch
from datasets import Dataset
from sklearn.cluster import KMeans
from unittest.mock import patch

import transformers
import vec2text


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "n_ag": 240,
        "n_pii": 80,
        "n_test": 2,
        "geometries": "192:4:17:101",
        "known_pairs": "8,16",
        "procrustes_pairs": "16",
        "encode_batch_size": 32,
        "invert_batch_size": 1,
        "num_steps": 2,
        "bootstrap_reps": 100,
    },
    "full": {
        "n_ag": 2400,
        "n_pii": 600,
        "n_test": 12,
        # d_pub:C:cell_seed:key_seed.  Two geometries and two independent keys.
        "geometries": (
            "192:16:17:101;192:16:17:103;"
            "384:32:29:107;384:32:29:109"
        ),
        "known_pairs": "8,16,32,64",
        "procrustes_pairs": "32,64",
        "encode_batch_size": 64,
        "invert_batch_size": 2,
        "num_steps": 8,
        "bootstrap_reps": 2000,
    },
}


@dataclass(frozen=True)
class Geometry:
    d_pub: int
    cells: int
    cell_seed: int
    key_seed: int

    @property
    def base_id(self) -> str:
        return f"p{self.d_pub}_c{self.cells}_cs{self.cell_seed}"

    @property
    def run_id(self) -> str:
        return f"{self.base_id}_ks{self.key_seed}"


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_geometries(value: str) -> list[Geometry]:
    geometries: list[Geometry] = []
    for item in value.split(";"):
        if not item.strip():
            continue
        fields = [int(part.strip()) for part in item.split(":")]
        if len(fields) != 4:
            raise ValueError(
                "Each geometry must be d_pub:C:cell_seed:key_seed; got " + item
            )
        geometries.append(Geometry(*fields))
    if not geometries:
        raise ValueError("At least one geometry is required")
    if len({g.run_id for g in geometries}) != len(geometries):
        raise ValueError("Geometry run identifiers must be unique")
    return geometries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "exp29_shard_vec2text",
    )
    parser.add_argument("--n-ag", type=int)
    parser.add_argument("--n-pii", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--geometries")
    parser.add_argument("--known-pairs")
    parser.add_argument("--procrustes-pairs")
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--invert-batch-size", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--beam-width", type=int, default=0)
    parser.add_argument("--max-embed-tokens", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=20260712)
    parser.add_argument("--pii-seed", type=int, default=290029)
    parser.add_argument("--selection-seed", type=int, default=290030)
    parser.add_argument("--bootstrap-reps", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=290031)
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-raw-token-f1",
        type=float,
        default=0.10,
        help="Abort before SHARD cases if the official raw control falls below this.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Rebuild CSV/JSON summaries and README from case JSON files.",
    )
    args = parser.parse_args()
    for key, value in PROFILES[args.profile].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def setup_logging(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("exp29")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(out / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info(0)
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        "total_bytes": int(total),
        "free_bytes": int(free),
        "total_gib": total / 2**30,
        "free_gib": free / 2**30,
        "multi_processor_count": props.multi_processor_count,
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Geometry):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def vec2text_transformers5_compat():
    """Build nested modules from config while outer official weights are loaded.

    Transformers 5.x constructs a custom PreTrainedModel under a meta-device
    context and rejects nested ``from_pretrained`` calls.  Both jxm checkpoints
    include all T5 and GTR weights, so config-only nested skeletons are exactly
    what the outer checkpoint loader needs.
    """

    vec2text.models.InversionModel.all_tied_weights_keys = {}
    vec2text.models.CorrectorEncoderModel.all_tied_weights_keys = {}

    def seq2seq_from_config(cls, model_name: str, *args, **kwargs):
        config = transformers.AutoConfig.from_pretrained(model_name)
        return transformers.AutoModelForSeq2SeqLM.from_config(config)

    def auto_from_config(cls, model_name: str, *args, **kwargs):
        config = transformers.AutoConfig.from_pretrained(model_name)
        return transformers.AutoModel.from_config(config)

    with patch.object(
        transformers.AutoModelForSeq2SeqLM,
        "from_pretrained",
        new=classmethod(seq2seq_from_config),
    ), patch.object(
        transformers.AutoModel,
        "from_pretrained",
        new=classmethod(auto_from_config),
    ):
        yield


def load_official_corrector(logger: logging.Logger):
    logger.info(
        "Loading official GTR checkpoints jxm/gtr__nq__32 and "
        "jxm/gtr__nq__32__correct"
    )
    with vec2text_transformers5_compat():
        inversion_model = vec2text.models.InversionModel.from_pretrained(
            "jxm/gtr__nq__32"
        )
        corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
            "jxm/gtr__nq__32__correct"
        )
    corrector = vec2text.api.load_corrector(inversion_model, corrector_model)
    # The legacy Trainer reads mixed-precision flags from the two checkpoint
    # configs independently.  On current PyTorch it may move the inversion
    # model/embedder as fp16 while leaving the corrector fp32, which fails on a
    # recursive correction step when a hypothesis embedding crosses modules.
    # Use the embedder's inference dtype consistently (fp16 on this RTX host).
    embedder_dtype = next(corrector.embedder.parameters()).dtype
    corrector.model.to(dtype=embedder_dtype)
    corrector.inversion_trainer.model.to(dtype=embedder_dtype)
    # Transformers 5.x permits a ``None`` length penalty, whereas vec2text
    # 0.0.13 performs arithmetic with it during recursive reranking.
    for module in (corrector.model, corrector.inversion_trainer.model):
        encoder_decoder = getattr(module, "encoder_decoder", None)
        generation_config = getattr(encoder_decoder, "generation_config", None)
        if generation_config is not None and generation_config.length_penalty is None:
            generation_config.length_penalty = 1.0
    corrector.inversion_trainer.model.eval()
    corrector.model.eval()
    logger.info(
        "Corrector loaded in consistent dtype=%s; GPU snapshot: %s",
        embedder_dtype,
        gpu_snapshot(),
    )
    return corrector


def cached_ag_news_arrow() -> Path:
    root = Path.home() / ".cache" / "huggingface" / "datasets" / "ag_news"
    candidates = sorted(root.glob("**/ag_news-test.arrow"))
    if not candidates:
        candidates = sorted(root.glob("**/ag_news-train.arrow"))
    if not candidates:
        raise FileNotFoundError(
            "Cached AG News Arrow data not found under " + str(root)
        )
    return candidates[0]


def normalize_text(text: str) -> str:
    text = html.unescape(str(text))
    text = text.replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_ag_news(n: int, seed: int) -> tuple[list[str], Path]:
    source_path = cached_ag_news_arrow()
    dataset = Dataset.from_file(str(source_path))
    candidates = [i for i, value in enumerate(dataset["text"]) if normalize_text(value)]
    if n > len(candidates):
        raise ValueError(f"Requested {n} AG News rows but only {len(candidates)} are cached")
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.asarray(candidates), size=n, replace=False)
    return [normalize_text(dataset[int(i)]["text"]) for i in selected], source_path


def make_synthetic_pii(n: int, seed: int) -> tuple[list[str], list[list[dict[str, str]]]]:
    """Generate controlled PII strings with exact item annotations.

    Email usernames include the record index and act as unique identifiers;
    names intentionally come from finite dictionaries and can repeat.
    """

    rng = np.random.default_rng(seed)
    first_names = [
        "Alice", "Bruno", "Carla", "Daniel", "Elena", "Farah", "George",
        "Hana", "Ivan", "Julia", "Kamil", "Lina", "Marco", "Nadia",
        "Owen", "Priya", "Rafael", "Sara", "Tomas", "Yuki",
    ]
    last_names = [
        "Brown", "Costa", "Diaz", "Evans", "Fischer", "Gupta", "Harris",
        "Ito", "Jones", "Khan", "Lopez", "Miller", "Novak", "Ortiz",
        "Petrov", "Quinn", "Rossi", "Singh", "Taylor", "Wang",
    ]
    domains = ["example.com", "mail.test", "demo.org", "sample.net"]
    texts: list[str] = []
    metadata: list[list[dict[str, str]]] = []
    for i in range(n):
        first = first_names[int(rng.integers(len(first_names)))]
        last = last_names[int(rng.integers(len(last_names)))]
        name = f"{first} {last}"
        user = f"{first[0].lower()}{last.lower()}{i:04d}"
        email = f"{user}@{domains[int(rng.integers(len(domains)))]}"
        phone = (
            f"{int(rng.integers(201, 990)):03d}-"
            f"{int(rng.integers(201, 990)):03d}-"
            f"{int(rng.integers(0, 10000)):04d}"
        )
        card = "-".join(f"{int(rng.integers(0, 10000)):04d}" for _ in range(4))
        text = f"PII record: {name}; email {email}; phone {phone}; card {card}."
        texts.append(text)
        metadata.append(
            [
                {"type": "name", "value": name},
                {"type": "email", "value": email},
                {"type": "phone", "value": phone},
                {"type": "card", "value": card},
            ]
        )
    return texts, metadata


def tokenizer_references(tokenizer, texts: list[str], max_tokens: int) -> list[str]:
    refs: list[str] = []
    batch_size = 256
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=False,
            truncation=True,
            max_length=max_tokens,
        )
        refs.extend(tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True))
    return [normalize_text(value) for value in refs]


def canonical(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


def filter_visible_pii(
    reference: str, items: list[dict[str, str]]
) -> list[dict[str, str]]:
    ref_canonical = canonical(reference)
    return [item for item in items if canonical(item["value"]) in ref_canonical]


def corpus_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["reference"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["source"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_or_load_corpus(
    args: argparse.Namespace, tokenizer, out: Path, logger: logging.Logger
) -> tuple[list[dict[str, Any]], str, str]:
    corpus_path = out / "corpus.jsonl"
    meta_path = out / "corpus_metadata.json"
    expected_n = args.n_ag + args.n_pii
    if args.resume and corpus_path.exists() and meta_path.exists():
        records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if len(records) == expected_n and meta.get("max_embed_tokens") == args.max_embed_tokens:
            logger.info("Reusing %d-record corpus from %s", len(records), corpus_path)
            return records, meta["corpus_sha256"], meta["ag_news_cache"]
        raise RuntimeError(
            "Existing corpus is incompatible with requested sizes/token length; "
            "use a new --out directory or remove only the exp29 output."
        )

    ag_texts, ag_path = load_ag_news(args.n_ag, args.sample_seed)
    pii_texts, pii_metadata = make_synthetic_pii(args.n_pii, args.pii_seed)
    originals = ag_texts + pii_texts
    sources = ["ag_news"] * len(ag_texts) + ["synthetic_pii"] * len(pii_texts)
    all_pii = [[] for _ in ag_texts] + pii_metadata
    references = tokenizer_references(tokenizer, originals, args.max_embed_tokens)
    records = []
    for index, (original, reference, source, pii_items) in enumerate(
        zip(originals, references, sources, all_pii)
    ):
        records.append(
            {
                "corpus_index": index,
                "source": source,
                "original": original,
                "reference": reference,
                # Do not score an item that was truncated before embedding.
                "pii_items": filter_visible_pii(reference, pii_items),
            }
        )
    rng = np.random.default_rng(args.sample_seed + 1)
    rng.shuffle(records)
    for index, record in enumerate(records):
        record["corpus_index"] = index
    digest = corpus_hash(records)
    with corpus_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    dump_json(
        meta_path,
        {
            "n_ag_news": args.n_ag,
            "n_synthetic_pii": args.n_pii,
            "max_embed_tokens": args.max_embed_tokens,
            "ag_news_cache": str(ag_path),
            "corpus_sha256": digest,
            "pii_note": "Only items visible before the 32-token embedding truncation are scored.",
        },
    )
    logger.info("Built corpus: n=%d, sha256=%s", len(records), digest)
    return records, digest, str(ag_path)


def model_device(corrector) -> torch.device:
    return next(corrector.embedder.parameters()).device


def model_embedding_dtype(corrector) -> torch.dtype:
    return next(corrector.inversion_trainer.model.parameters()).dtype


def encode_texts(
    corrector,
    texts: list[str],
    max_tokens: int,
    batch_size: int,
    logger: logging.Logger,
    events: list[dict[str, Any]],
) -> np.ndarray:
    """Use vec2text's exact GTR call: attention-mask mean pooling, no L2 norm."""

    device = model_device(corrector)
    values: list[np.ndarray] = []
    cursor = 0
    active_batch = batch_size
    started = time.time()
    while cursor < len(texts):
        batch = texts[cursor : cursor + active_batch]
        try:
            tokenized = corrector.embedder_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(device)
            # no_grad creates ordinary tensors; inference_mode tensors cannot be
            # reused by older vec2text correction code outside that context.
            with torch.no_grad():
                embedding = corrector.inversion_trainer.call_embedding_model(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                )
            values.append(embedding.detach().float().cpu().numpy())
            cursor += len(batch)
            if cursor % max(100, active_batch) == 0 or cursor == len(texts):
                logger.info("Encoded %d/%d texts", cursor, len(texts))
            del tokenized, embedding
        except BaseException as exc:
            if not is_cuda_oom(exc) or active_batch <= 1:
                raise
            clear_cuda()
            new_batch = max(1, active_batch // 2)
            event = {
                "stage": "encoding",
                "cursor": cursor,
                "old_batch": active_batch,
                "new_batch": new_batch,
                "error": str(exc),
            }
            events.append(event)
            logger.warning("Encoding OOM; reducing batch %d -> %d", active_batch, new_batch)
            active_batch = new_batch
    encoded = np.vstack(values).astype(np.float32)
    logger.info(
        "Encoding complete in %.1fs; shape=%s; norm mean=%.6f",
        time.time() - started,
        encoded.shape,
        np.linalg.norm(encoded, axis=1).mean(),
    )
    return encoded


def load_or_encode_embeddings(
    args: argparse.Namespace,
    corrector,
    records: list[dict[str, Any]],
    digest: str,
    out: Path,
    logger: logging.Logger,
    events: list[dict[str, Any]],
) -> np.ndarray:
    path = out / "embeddings_gtr_base.npy"
    meta_path = out / "embeddings_metadata.json"
    if args.resume and path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embeddings = np.load(path)
        if (
            meta.get("corpus_sha256") == digest
            and embeddings.shape == (len(records), 768)
            and meta.get("embedding_path") == "corrector.inversion_trainer.call_embedding_model"
        ):
            logger.info("Reusing cached GTR embeddings from %s", path)
            return embeddings.astype(np.float32, copy=False)
        raise RuntimeError("Existing exp29 embedding cache is incompatible with this corpus")
    embeddings = encode_texts(
        corrector,
        [record["reference"] for record in records],
        args.max_embed_tokens,
        args.encode_batch_size,
        logger,
        events,
    )
    np.save(path, embeddings)
    dump_json(
        meta_path,
        {
            "corpus_sha256": digest,
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "embedding_path": "corrector.inversion_trainer.call_embedding_model",
            "pooling": "attention-mask mean pooling in vec2text InversionModel",
            "l2_normalized": False,
            "checkpoint": "sentence-transformers/gtr-t5-base bundled in jxm/gtr__nq__32",
        },
    )
    return embeddings


def token_counts(value: str) -> Counter[str]:
    return Counter(re.findall(r"[A-Za-z0-9]+", value.lower()))


def token_f1(reference: str, hypothesis: str) -> float:
    ref = token_counts(reference)
    hyp = token_counts(hypothesis)
    n_ref, n_hyp = sum(ref.values()), sum(hyp.values())
    if n_ref == 0 or n_hyp == 0:
        return 0.0
    overlap = sum((ref & hyp).values())
    precision, recall = overlap / n_hyp, overlap / n_ref
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def pii_hits(items: list[dict[str, str]], hypothesis: str) -> tuple[int, int, dict[str, bool]]:
    hyp = canonical(hypothesis)
    detail = {item["type"]: canonical(item["value"]) in hyp for item in items}
    return sum(detail.values()), len(items), detail


def invert_embeddings(
    corrector,
    embeddings: np.ndarray,
    case: str,
    requested_batch: int,
    requested_steps: int,
    beam_width: int,
    logger: logging.Logger,
    events: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Invert with batch-size and, only if necessary, step-count fallback."""

    device = model_device(corrector)
    dtype = model_embedding_dtype(corrector)
    steps = requested_steps
    while True:
        outputs: list[str] = []
        cursor = 0
        active_batch = requested_batch
        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            while cursor < len(embeddings):
                batch_np = embeddings[cursor : cursor + active_batch]
                try:
                    tensor = torch.tensor(batch_np, device=device, dtype=dtype)
                    with torch.inference_mode():
                        recovered = vec2text.invert_embeddings(
                            tensor,
                            corrector=corrector,
                            num_steps=steps,
                            sequence_beam_width=beam_width,
                        )
                    outputs.extend(recovered)
                    cursor += len(batch_np)
                    del tensor
                    clear_cuda()
                except BaseException as exc:
                    if not is_cuda_oom(exc):
                        raise
                    clear_cuda()
                    if active_batch > 1:
                        new_batch = max(1, active_batch // 2)
                        event = {
                            "stage": "inversion",
                            "case": case,
                            "cursor": cursor,
                            "steps": steps,
                            "old_batch": active_batch,
                            "new_batch": new_batch,
                            "error": str(exc),
                        }
                        events.append(event)
                        logger.warning(
                            "%s OOM; reducing inversion batch %d -> %d",
                            case,
                            active_batch,
                            new_batch,
                        )
                        active_batch = new_batch
                        continue
                    raise
            seconds = time.time() - started
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            return outputs, {
                "seconds": seconds,
                "requested_batch_size": requested_batch,
                "effective_batch_size": active_batch,
                "requested_steps": requested_steps,
                "effective_steps": steps,
                "peak_allocated_bytes": peak,
            }
        except BaseException as exc:
            if not is_cuda_oom(exc) or steps <= 1:
                raise
            new_steps = max(1, steps // 2)
            event = {
                "stage": "inversion_steps",
                "case": case,
                "old_steps": steps,
                "new_steps": new_steps,
                "error": str(exc),
            }
            events.append(event)
            logger.warning("%s OOM at batch 1; restarting with steps %d -> %d", case, steps, new_steps)
            steps = new_steps
            clear_cuda()


def evaluate_predictions(
    records: list[dict[str, Any]],
    target_indices: np.ndarray,
    hypotheses: list[str],
    attack_embeddings: np.ndarray,
    true_embeddings: np.ndarray,
    recovered_embeddings: np.ndarray,
    run: dict[str, Any],
    method: str,
    m: int | None,
) -> list[dict[str, Any]]:
    geometry_cos = cosine_rows(attack_embeddings, true_embeddings)
    recovered_cos = cosine_rows(recovered_embeddings, true_embeddings)
    rows: list[dict[str, Any]] = []
    for position, (index, hypothesis) in enumerate(zip(target_indices, hypotheses)):
        record = records[int(index)]
        hits, total, detail = pii_hits(record["pii_items"], hypothesis)
        reference = record["reference"]
        row = {
            **run,
            "method": method,
            "m": m,
            "target_position": position,
            "corpus_index": int(index),
            "source": record["source"],
            "reference": reference,
            "hypothesis": normalize_text(hypothesis),
            "token_f1": token_f1(reference, hypothesis),
            "bleu": sacrebleu.sentence_bleu(hypothesis, [reference]).score / 100.0,
            "exact_match": float(reference.strip() == hypothesis.strip()),
            "normalized_exact_match": float(canonical(reference) == canonical(hypothesis)),
            "pii_hits": hits,
            "pii_total": total,
            "pii_item_hits": detail,
            "geometry_cosine": float(geometry_cos[position]),
            "recovered_text_embedding_cosine": float(recovered_cos[position]),
        }
        rows.append(row)
    return rows


def raw_control_sanity(
    args: argparse.Namespace,
    corrector,
    out: Path,
    logger: logging.Logger,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    path = out / "raw_control.json"
    if args.resume and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("official_embedding_path") and payload["token_f1_mean"] >= args.require_raw_token_f1:
            logger.info("Reusing passed raw-control sanity check")
            return payload
    originals = [
        "The city council approved a new public transit plan in Boston.",
        "Scientists discovered a new planet near a distant star.",
        "Alice Brown email alice.brown@example.com phone 415-555-1234.",
        "The company reported stronger revenue after expanding cloud services.",
    ]
    references = tokenizer_references(
        corrector.embedder_tokenizer, originals, args.max_embed_tokens
    )
    embeddings = encode_texts(
        corrector,
        references,
        args.max_embed_tokens,
        min(args.encode_batch_size, 4),
        logger,
        events,
    )
    hypotheses, inversion_meta = invert_embeddings(
        corrector,
        embeddings,
        "raw_control",
        min(args.invert_batch_size, 2),
        max(2, min(args.num_steps, 4)),
        args.beam_width,
        logger,
        events,
    )
    scores = [token_f1(r, h) for r, h in zip(references, hypotheses)]
    bleus = [sacrebleu.sentence_bleu(h, [r]).score / 100.0 for r, h in zip(references, hypotheses)]
    payload = {
        "official_embedding_path": "corrector.inversion_trainer.call_embedding_model",
        "pooling": "attention-mask mean pooling; no L2 normalization",
        "token_f1_mean": float(np.mean(scores)),
        "bleu_mean": float(np.mean(bleus)),
        "threshold": args.require_raw_token_f1,
        "passed": bool(np.mean(scores) >= args.require_raw_token_f1),
        "inversion": inversion_meta,
        "samples": [
            {"reference": r, "hypothesis": h, "token_f1": f1, "bleu": bleu}
            for r, h, f1, bleu in zip(references, hypotheses, scores, bleus)
        ],
    }
    dump_json(path, payload)
    if not payload["passed"]:
        raise RuntimeError(
            f"Raw official GTR control token-F1={payload['token_f1_mean']:.4f} "
            f"is below {args.require_raw_token_f1:.4f}; refusing an uninformative SHARD run"
        )
    logger.info(
        "Raw-control passed: token-F1=%.4f, BLEU=%.4f",
        payload["token_f1_mean"],
        payload["bleu_mean"],
    )
    return payload


def pca_full(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    centered = x - mu
    covariance = centered.astype(np.float64).T @ centered.astype(np.float64)
    covariance /= max(1, len(centered))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return mu, vectors[:, order].astype(np.float32), values[order]


def assign_cells(prefix: np.ndarray, centroids: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    labels = np.empty(len(prefix), dtype=np.int32)
    centroid_norm = np.sum(centroids * centroids, axis=1)
    for start in range(0, len(prefix), chunk):
        batch = prefix[start : start + chunk]
        distances = centroid_norm[None, :] - 2.0 * batch @ centroids.T
        labels[start : start + len(batch)] = distances.argmin(axis=1)
    return labels


def kmeans_cells(
    prefix: np.ndarray, cells: int, seed: int, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    if cells > len(prefix):
        raise ValueError("Number of cells exceeds corpus size")
    # k-means++ avoids the severely degenerate cells produced by a single
    # random-centroid Lloyd start on anisotropic GTR embeddings.  This is the
    # same public-prefix geometry, with a standard deterministic estimator.
    estimator = KMeans(
        n_clusters=cells,
        init="k-means++",
        n_init=10,
        max_iter=iterations,
        random_state=seed,
        algorithm="lloyd",
    )
    labels = estimator.fit_predict(prefix).astype(np.int32)
    centroids = estimator.cluster_centers_.astype(np.float32)
    return labels, centroids


def orthogonal_key(dim: int, seed: int, cell: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 1_000_003 + cell)
    matrix = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def ols_map(stored: np.ndarray, native: np.ndarray, rtol: float) -> tuple[np.ndarray, int]:
    u, singular, vt = np.linalg.svd(stored.astype(np.float64), full_matrices=False)
    if singular.size == 0 or singular[0] == 0:
        return np.zeros((stored.shape[1], native.shape[1]), np.float32), 0
    keep = singular > rtol * singular[0]
    inverse = np.zeros_like(singular)
    inverse[keep] = 1.0 / singular[keep]
    fitted = vt.T @ (inverse[:, None] * (u.T @ native.astype(np.float64)))
    return fitted.astype(np.float32), int(keep.sum())


def procrustes_map(stored: np.ndarray, native: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(
        stored.astype(np.float64).T @ native.astype(np.float64),
        full_matrices=False,
    )
    return (left @ right_t).astype(np.float32)


def stratified_targets(
    members: np.ndarray,
    records: list[dict[str, Any]],
    n_test: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pii = np.asarray([i for i in members if records[int(i)]["source"] == "synthetic_pii"])
    ag = np.asarray([i for i in members if records[int(i)]["source"] == "ag_news"])
    rng.shuffle(pii)
    rng.shuffle(ag)
    target: list[int] = []
    each = n_test // 2
    target.extend(int(i) for i in pii[:each])
    target.extend(int(i) for i in ag[: n_test - len(target)])
    used = set(target)
    remaining = np.asarray([i for i in members if int(i) not in used])
    rng.shuffle(remaining)
    target.extend(int(i) for i in remaining[: n_test - len(target)])
    if len(target) != n_test:
        raise ValueError("Largest cell cannot supply requested test cases")
    rng.shuffle(target)
    return np.asarray(target, dtype=np.int64)


def reconstruct_native(
    prefix: np.ndarray,
    residual: np.ndarray,
    basis: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    coordinates = np.concatenate([prefix, residual], axis=1)
    return (coordinates @ basis.T + mu).astype(np.float32)


def build_attack_cases(
    x: np.ndarray,
    records: list[dict[str, Any]],
    centered_coordinates: np.ndarray,
    basis: np.ndarray,
    mu: np.ndarray,
    geometry: Geometry,
    known_pairs: list[int],
    procrustes_pairs: list[int],
    n_test: int,
    selection_seed: int,
    kmeans_iters: int,
    rtol: float,
    geometry_cache: dict[str, dict[str, Any]],
    target_cohort: str,
) -> tuple[dict[str, tuple[np.ndarray, int | None, int | None]], np.ndarray, dict[str, Any]]:
    d = x.shape[1]
    if not 0 < geometry.d_pub < d:
        raise ValueError(f"d_pub={geometry.d_pub} must be between 1 and {d - 1}")
    if geometry.base_id not in geometry_cache:
        prefix_all = centered_coordinates[:, : geometry.d_pub]
        residual_all = centered_coordinates[:, geometry.d_pub :]
        labels, centroids = kmeans_cells(
            prefix_all, geometry.cells, geometry.cell_seed, kmeans_iters
        )
        counts = np.bincount(labels, minlength=geometry.cells)
        largest = int(np.argmax(counts))
        pii_mask = np.asarray(
            [record["source"] == "synthetic_pii" for record in records],
            dtype=bool,
        )
        pii_counts = np.bincount(labels[pii_mask], minlength=geometry.cells)
        largest_pii = int(np.argmax(pii_counts)) if pii_counts.max() else largest
        geometry_cache[geometry.base_id] = {
            "prefix": prefix_all,
            "residual": residual_all,
            "labels": labels,
            "centroids": centroids,
            "counts": counts,
            "pii_counts": pii_counts,
            "largest": largest,
            "largest_pii": largest_pii,
            "cohort_selections": {},
        }
    cached = geometry_cache[geometry.base_id]
    if target_cohort == "largest":
        target_cell = cached["largest"]
    elif target_cohort == "largest_pii":
        target_cell = cached["largest_pii"]
    else:
        raise ValueError(f"Unknown target cohort {target_cohort!r}")
    selection_key = f"{target_cohort}:{target_cell}"
    if selection_key not in cached["cohort_selections"]:
        members = np.flatnonzero(cached["labels"] == target_cell)
        target = stratified_targets(
            members,
            records,
            n_test,
            selection_seed
            + geometry.cell_seed
            + geometry.d_pub
            + geometry.cells
            + 97 * target_cell,
        )
        target_set = set(int(i) for i in target)
        anchors = np.asarray([i for i in members if int(i) not in target_set])
        rng = np.random.default_rng(
            selection_seed
            + 10_000
            + geometry.cell_seed
            + geometry.d_pub
            + geometry.cells
            + 193 * target_cell
        )
        rng.shuffle(anchors)
        cached["cohort_selections"][selection_key] = {
            "members": members,
            "targets": target,
            "anchors": anchors,
        }
    selected = cached["cohort_selections"][selection_key]
    prefix_all = cached["prefix"]
    residual_all = cached["residual"]
    members = selected["members"]
    target = selected["targets"]
    anchors = selected["anchors"]
    max_m = max(known_pairs + procrustes_pairs + [0])
    if len(anchors) < max_m:
        raise ValueError(
            f"Target cell {target_cell} has {len(members)} rows, only {len(anchors)} "
            f"anchors after n_test={n_test}; need m={max_m}. Increase corpus or reduce C."
        )
    residual_dim = d - geometry.d_pub
    key = orthogonal_key(residual_dim, geometry.key_seed, target_cell)
    target_prefix = prefix_all[target]
    target_residual = residual_all[target]
    target_stored = target_residual @ key.T
    cases: dict[str, tuple[np.ndarray, int | None, int | None]] = {}
    zeros = np.zeros_like(target_residual)
    cases["raw"] = (x[target].copy(), None, None)
    cases["prefix_only"] = (
        reconstruct_native(target_prefix, zeros, basis, mu),
        None,
        None,
    )
    cases["unknown_key"] = (
        reconstruct_native(target_prefix, target_stored, basis, mu),
        None,
        None,
    )
    cases["oracle_residual"] = (
        reconstruct_native(target_prefix, target_residual, basis, mu),
        None,
        residual_dim,
    )
    for m in known_pairs:
        anchor = anchors[:m]
        anchor_native = residual_all[anchor]
        anchor_stored = anchor_native @ key.T
        mapping, rank = ols_map(anchor_stored, anchor_native, rtol)
        estimate = target_stored @ mapping
        cases[f"ols_m{m}"] = (
            reconstruct_native(target_prefix, estimate, basis, mu),
            m,
            rank,
        )
    for m in procrustes_pairs:
        anchor = anchors[:m]
        anchor_native = residual_all[anchor]
        anchor_stored = anchor_native @ key.T
        mapping = procrustes_map(anchor_stored, anchor_native)
        estimate = target_stored @ mapping
        cases[f"procrustes_m{m}"] = (
            reconstruct_native(target_prefix, estimate, basis, mu),
            m,
            int(np.linalg.matrix_rank(anchor_stored)),
        )
    run_info = {
        "run_id": f"{geometry.run_id}_{target_cohort}",
        "geometry_id": geometry.base_id,
        "target_cohort": target_cohort,
        "is_global_largest_cell": target_cell == cached["largest"],
        "d": d,
        "d_pub": geometry.d_pub,
        "d_priv": residual_dim,
        "cells": geometry.cells,
        "cell_seed": geometry.cell_seed,
        "key_seed": geometry.key_seed,
        "target_cell": target_cell,
        "target_cell_size": len(members),
        "target_cell_ag_news": sum(records[int(i)]["source"] == "ag_news" for i in members),
        "target_cell_synthetic_pii": sum(records[int(i)]["source"] == "synthetic_pii" for i in members),
        "n_test": len(target),
        "n_test_ag_news": sum(records[int(i)]["source"] == "ag_news" for i in target),
        "n_test_synthetic_pii": sum(records[int(i)]["source"] == "synthetic_pii" for i in target),
        "target_indices": target.tolist(),
        "anchor_indices_first_max_m": anchors[:max_m].tolist(),
        "cell_sizes": cached["counts"].tolist(),
        "cell_pii_sizes": cached["pii_counts"].tolist(),
    }
    return cases, target, run_info


METRICS = (
    "token_f1",
    "bleu",
    "exact_match",
    "normalized_exact_match",
    "pii_recall",
    "pii_identifier_recall",
    "pii_name_recall",
    "pii_email_recall",
    "pii_phone_recall",
    "pii_card_recall",
    "geometry_cosine",
    "recovered_text_embedding_cosine",
)


def metric_value(rows: list[dict[str, Any]], metric: str) -> float:
    if metric == "pii_recall":
        total = sum(int(row["pii_total"]) for row in rows)
        return (
            sum(int(row["pii_hits"]) for row in rows) / total
            if total
            else float("nan")
        )
    if metric == "pii_identifier_recall":
        values = [
            bool(hit)
            for row in rows
            for item_type, hit in row.get("pii_item_hits", {}).items()
            if item_type in {"email", "phone", "card"}
        ]
        return float(np.mean(values)) if values else float("nan")
    if metric.startswith("pii_") and metric.endswith("_recall"):
        item_type = metric.removeprefix("pii_").removesuffix("_recall")
        eligible = [
            bool(row["pii_item_hits"][item_type])
            for row in rows
            if item_type in row.get("pii_item_hits", {})
        ]
        return float(np.mean(eligible)) if eligible else float("nan")
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return float(np.mean(values)) if len(values) else float("nan")


def bootstrap_rows(
    rows: list[dict[str, Any]], metric: str, reps: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(rows) < 2 or reps <= 0:
        value = metric_value(rows, metric)
        return value, value
    estimates = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        sample = [rows[int(i)] for i in rng.integers(0, len(rows), size=len(rows))]
        estimates[rep] = metric_value(sample, metric)
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def hierarchical_bootstrap(
    rows: list[dict[str, Any]], metric: str, reps: int, rng: np.random.Generator
) -> tuple[float, float]:
    # Targets are shared across the two key seeds of a geometry/cohort.  Treat
    # geometry/cohort as the outer sampling unit, sample one key run inside it,
    # and only then resample target records.  This avoids pretending duplicated
    # raw/prefix/oracle targets under a second key are independent documents.
    by_unit: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        unit = f"{row['geometry_id']}|{row.get('target_cohort', 'largest')}"
        by_unit.setdefault(unit, {}).setdefault(row["run_id"], []).append(row)
    units = sorted(by_unit)
    if len(units) < 2:
        return bootstrap_rows(rows, metric, reps, rng)
    estimates = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        sampled_rows: list[dict[str, Any]] = []
        for unit_pos in rng.integers(0, len(units), size=len(units)):
            key_runs = by_unit[units[int(unit_pos)]]
            run_ids = sorted(key_runs)
            run_id = run_ids[int(rng.integers(0, len(run_ids)))]
            run_rows = key_runs[run_id]
            sampled_rows.extend(
                run_rows[int(i)]
                for i in rng.integers(0, len(run_rows), size=len(run_rows))
            )
        estimates[rep] = metric_value(sampled_rows, metric)
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = dict(row)
    flat["pii_item_hits"] = json.dumps(flat.get("pii_item_hits", {}), sort_keys=True)
    return flat


def row_scalar_metric(row: dict[str, Any], metric: str) -> float:
    if metric == "pii_identifier_recall":
        values = [
            float(hit)
            for item_type, hit in row.get("pii_item_hits", {}).items()
            if item_type in {"email", "phone", "card"}
        ]
        return float(np.mean(values)) if values else float("nan")
    return float(row[metric])


def paired_delta_summaries(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Paired target-level deltas; intervals remain descriptive/conditional."""

    lookup = {
        (row["run_id"], row["method"], int(row["target_position"])): row
        for row in rows
    }
    methods = sorted({row["method"] for row in rows})
    metrics = (
        "token_f1",
        "bleu",
        "pii_identifier_recall",
        "geometry_cosine",
        "recovered_text_embedding_cosine",
    )
    summaries: list[dict[str, Any]] = []
    for baseline in ("raw", "prefix_only"):
        for method in methods:
            if method == baseline:
                continue
            for metric in metrics:
                delta_rows: list[dict[str, Any]] = []
                for row in rows:
                    if row["method"] != method:
                        continue
                    base = lookup.get(
                        (row["run_id"], baseline, int(row["target_position"]))
                    )
                    if base is None:
                        continue
                    value = row_scalar_metric(row, metric)
                    base_value = row_scalar_metric(base, metric)
                    if not (np.isfinite(value) and np.isfinite(base_value)):
                        continue
                    delta_rows.append(
                        {
                            "run_id": row["run_id"],
                            "geometry_id": row["geometry_id"],
                            "target_cohort": row.get("target_cohort", "largest"),
                            "paired_delta": value - base_value,
                        }
                    )
                for cohort in ("largest", "largest_pii"):
                    selected = [
                        row for row in delta_rows if row["target_cohort"] == cohort
                    ]
                    if not selected:
                        continue
                    low, high = hierarchical_bootstrap(
                        selected, "paired_delta", args.bootstrap_reps, rng
                    )
                    summaries.append(
                        {
                            "target_cohort": cohort,
                            "method": method,
                            "baseline": baseline,
                            "metric": metric,
                            "n_pairs": len(selected),
                            "mean_delta": float(
                                np.mean([row["paired_delta"] for row in selected])
                            ),
                            "ci_low": low,
                            "ci_high": high,
                            "interpretation": (
                                "descriptive conditional paired delta; interval is not "
                                "a significance test with only two geometry designs"
                            ),
                        }
                    )
    return summaries


def read_case_rows(out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((out / "cases").glob("case_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            rows.extend(payload["rows"])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_final_provenance(
    args: argparse.Namespace, out: Path, rows: list[dict[str, Any]]
) -> None:
    """Attach post-run annotations and hashes without touching case outputs."""

    source_path = Path(__file__).resolve()
    source_hash = file_sha256(source_path)
    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "attacker_side_information": (
                "Strengthened observer knows the exact full PCA basis V and document "
                "mean mu; only target cell key H is unknown in unknown-key/OLS/Procrustes."
            ),
            "normalization_scope": (
                "Native GTR vectors are intentionally unnormalized because the official "
                "corrector is checkpoint-compatible only with that representation; this "
                "is not the paper's literal unit-normalized e5 retrieval configuration."
            ),
            "pii_scope": (
                "Headline exact PII-ID recall excludes repeated names and includes visible "
                "email/phone/card identifiers. Names use a finite dictionary; cards have no "
                "visible denominator after 32-token truncation in this corpus."
            ),
            "bootstrap_scope": (
                "Descriptive conditional 95% hierarchical bootstrap; only two independent "
                "geometry designs, with key seed sampled within geometry/cohort."
            ),
            "finalizer_source_sha256": source_hash,
        }
    )
    critical_keys = (
        "profile", "n_ag", "n_pii", "n_test", "geometries",
        "known_pairs", "procrustes_pairs", "num_steps", "beam_width",
        "max_embed_tokens", "sample_seed", "pii_seed", "selection_seed",
        "kmeans_iters", "rank_rtol", "encoder",
        "vec2text_inversion_checkpoint", "vec2text_corrector_checkpoint",
        "embedding_path", "pooling", "cell_clustering",
    )
    protocol = {key: config.get(key) for key in critical_keys}
    config["protocol_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    dump_json(config_path, config)

    case_manifest = []
    for path in sorted((out / "cases").glob("case_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_manifest.append(
            {
                "file": path.name,
                "sha256": file_sha256(path),
                "status": payload.get("status"),
                "case_id": payload.get("case_id"),
                "n_rows": len(payload.get("rows", [])),
            }
        )
    dump_json(out / "case_manifest.json", case_manifest)

    visible_by_type: Counter[str] = Counter()
    unique_by_type: dict[str, set[str]] = {}
    corpus_path = out / "corpus.jsonl"
    if corpus_path.exists():
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            for item in record.get("pii_items", []):
                visible_by_type[item["type"]] += 1
                unique_by_type.setdefault(item["type"], set()).add(item["value"])
    pii_audit = {
        "visible_item_denominator_by_type": dict(visible_by_type),
        "unique_visible_values_by_type": {
            key: len(values) for key, values in unique_by_type.items()
        },
        "headline_metric": "pii_identifier_recall",
        "headline_excludes": "name (finite repeated dictionary)",
        "card_note": "No full card value survives the 32-token embedding truncation.",
        "n_prediction_rows": len(rows),
    }
    dump_json(out / "pii_audit.json", pii_audit)

    provenance = {
        "finalized_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "finalizer_source": str(source_path),
        "finalizer_source_sha256": source_hash,
        "protocol_fingerprint_sha256": config["protocol_fingerprint_sha256"],
        "case_files": len(case_manifest),
        "case_files_completed": sum(item["status"] == "completed" for item in case_manifest),
        "note": (
            "The long-running process loaded the core inference protocol at launch. "
            "During execution only summary/bootstrap logic and explanatory annotations "
            "were edited; immutable per-case JSON was then finalized with the recorded source."
        ),
    }
    dump_json(out / "provenance.json", provenance)

    runtime_path = out / "runtime.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime["finalizer_source_sha256"] = source_hash
        runtime["finalized_local"] = provenance["finalized_local"]
        dump_json(runtime_path, runtime)


def finalize_outputs(args: argparse.Namespace, out: Path, logger: logging.Logger) -> None:
    rows = read_case_rows(out)
    if not rows:
        raise RuntimeError(f"No completed case files under {out / 'cases'}")
    dump_json(out / "per_case.json", rows)
    write_csv(out / "per_case.csv", [flatten_row(row) for row in rows])

    rng = np.random.default_rng(args.bootstrap_seed)
    by_run_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_run_method.setdefault((row["run_id"], row["method"]), []).append(row)
    run_summary: list[dict[str, Any]] = []
    for (run_id, method), group in sorted(by_run_method.items()):
        result: dict[str, Any] = {
            "run_id": run_id,
            "geometry_id": group[0]["geometry_id"],
            "method": method,
            "m": group[0].get("m"),
            "n": len(group),
            "n_pii": sum(row["source"] == "synthetic_pii" for row in group),
        }
        for metric in METRICS:
            result[f"{metric}_mean"] = metric_value(group, metric)
            low, high = bootstrap_rows(group, metric, args.bootstrap_reps, rng)
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        run_summary.append(result)
    dump_json(out / "summary_by_run.json", run_summary)
    write_csv(out / "summary_by_run.csv", run_summary)

    by_cohort_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("target_cohort", "largest"), row["method"])
        by_cohort_method.setdefault(key, []).append(row)
    cohort_summary: list[dict[str, Any]] = []
    for (cohort, method), group in sorted(by_cohort_method.items()):
        result = {
            "target_cohort": cohort,
            "method": method,
            "m": group[0].get("m"),
            "n_rows": len(group),
            "n_runs": len({row["run_id"] for row in group}),
            "n_pii_rows": sum(row["source"] == "synthetic_pii" for row in group),
        }
        for metric in METRICS:
            result[f"{metric}_mean"] = metric_value(group, metric)
            low, high = hierarchical_bootstrap(
                group, metric, args.bootstrap_reps, rng
            )
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        cohort_summary.append(result)
    dump_json(out / "summary_by_cohort.json", cohort_summary)
    write_csv(out / "summary_by_cohort.csv", cohort_summary)

    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    pooled: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        result = {
            "method": method,
            "m": group[0].get("m"),
            "n_rows": len(group),
            "n_runs": len({row["run_id"] for row in group}),
            "n_pii_rows": sum(row["source"] == "synthetic_pii" for row in group),
        }
        for metric in METRICS:
            result[f"{metric}_mean"] = metric_value(group, metric)
            low, high = hierarchical_bootstrap(
                group, metric, args.bootstrap_reps, rng
            )
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        pooled.append(result)
    dump_json(out / "summary_pooled.json", pooled)
    write_csv(out / "summary_pooled.csv", pooled)
    paired = paired_delta_summaries(args, rows, rng)
    dump_json(out / "paired_deltas_by_cohort.json", paired)
    write_csv(out / "paired_deltas_by_cohort.csv", paired)
    write_readme(args, out, cohort_summary, pooled)
    update_final_provenance(args, out, rows)
    logger.info("Finalized %d prediction rows from %d cases", len(rows), len(by_run_method))


def fmt_ci(row: dict[str, Any], metric: str) -> str:
    value = row[f"{metric}_mean"]
    low = row[f"{metric}_ci_low"]
    high = row[f"{metric}_ci_high"]
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def write_readme(
    args: argparse.Namespace,
    out: Path,
    cohort_summary: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
) -> None:
    lines = [
        "# Experiment 29: SHARD outcome-level Vec2Text audit",
        "",
        "This directory contains a real text-reconstruction attack on SHARD's exposed public prefix and cell-keyed residual. It uses the exact mask-aware mean-pooled GTR embedding path bundled with the official `jxm/gtr__nq__32` / `jxm/gtr__nq__32__correct` Vec2Text corrector. The old CLS extraction is not used.",
        "",
        "The corpus combines locally cached AG News with controlled synthetic PII. PCA is full-dimensional; SHARD only splits the coordinates. The primary cohort targets the globally largest public-prefix cell and uses nested in-cell known pairs. When that cell has no PII, a separately labelled secondary diagnostic cohort targets the largest PII-containing cell solely to make the controlled PII outcome measurable. The intervals are descriptive, conditional 95% hierarchical bootstrap intervals; only two independent geometry designs are present, so they are not significance claims.",
        "",
    ]
    method_order = {
        "raw": 0,
        "prefix_only": 1,
        "unknown_key": 2,
        "ols_m8": 3,
        "ols_m16": 4,
        "ols_m32": 5,
        "ols_m64": 6,
        "procrustes_m32": 7,
        "procrustes_m64": 8,
        "oracle_residual": 9,
    }
    for cohort, title in (
        ("largest", "Globally largest-cell cohort"),
        ("largest_pii", "Secondary largest-PII-cell cohort"),
    ):
        selected = [row for row in cohort_summary if row["target_cohort"] == cohort]
        if not selected:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                "| Method | m | token-F1 | BLEU | exact PII-ID recall | input cosine | decoded-text cosine |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(selected, key=lambda item: method_order.get(item["method"], 100)):
            m = "--" if row.get("m") is None else str(row["m"])
            lines.append(
                f"| {row['method']} | {m} | {fmt_ci(row, 'token_f1')} | "
                f"{fmt_ci(row, 'bleu')} | {fmt_ci(row, 'pii_identifier_recall')} | "
                f"{fmt_ci(row, 'geometry_cosine')} | "
                f"{fmt_ci(row, 'recovered_text_embedding_cosine')} |"
            )
        lines.append("")
        if cohort == "largest_pii":
            lines.extend(
                [
                    "Exact controlled-PII recall by visible item type:",
                    "",
                    "| Method | all visible items | repeated-name | unique email | phone |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in sorted(selected, key=lambda item: method_order.get(item["method"], 100)):
                lines.append(
                    f"| {row['method']} | {fmt_ci(row, 'pii_recall')} | "
                    f"{fmt_ci(row, 'pii_name_recall')} | "
                    f"{fmt_ci(row, 'pii_email_recall')} | "
                    f"{fmt_ci(row, 'pii_phone_recall')} |"
                )
            lines.append("")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `raw_control.json`: mandatory positive-control sanity check for the official pooling path.",
            "- `config.json`, `runtime.json`, `corpus_metadata.json`, `embeddings_metadata.json`: exact protocol and environment.",
            "- `cases/`: checkpointed JSON for every geometry/key/method.",
            "- `per_case.csv` / `.json`: every decoded sample and outcome metric.",
            "- `summary_by_run.csv` / `.json`: within-run bootstrap summaries.",
            "- `summary_by_cohort.csv` / `.json`: primary and PII cohorts kept separate.",
            "- `paired_deltas_by_cohort.csv` / `.json`: paired descriptive deltas against raw and prefix-only controls.",
            "- `summary_pooled.csv` / `.json`: exploratory pooled summaries; do not use them in place of the cohort panels.",
            "- `run.log`: execution, OOM, and fallback log.",
            "",
            "## Scope and interpretation",
            "",
            "The reconstruction deliberately grants a strengthened observer the exact full PCA basis V and document mean mu; the only unknown in unknown-key/OLS/Procrustes is the target cell key H. The official corrector requires native unnormalized mean-pooled GTR vectors, so this is a checkpoint-compatible SHARD split stress test rather than the paper's literal unit-normalized e5 retrieval instance. Results are encoder- and attacker-specific. Prefix-only and unknown-key are baselines, not formal privacy guarantees. Raw and oracle-residual are exact-geometry reference controls: oracle must reconstruct the raw embedding numerically, but neither is an upper bound on token-F1 or BLEU because iterative neural decoding is non-monotone under embedding perturbations. Under an exact orthogonal key, minimum-norm OLS prediction depends on the anchor span and is algebraically invariant to the key seed; the clustered bootstrap therefore does not count the duplicated seed result as an independent target sample. Synthetic PII measures exact recovery of controlled visible items. Email usernames are unique; names repeat and are reported separately, some target names can also occur among anchors, and the card field has no visible denominator after 32-token truncation in this corpus. Headline PII-ID recall excludes names and includes only visible email/phone/card identifiers.",
            "",
            f"Profile recorded by the finalizer: `{args.profile}`; requested correction steps: `{args.num_steps}`.",
        ]
    )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    logger = setup_logging(out)
    if args.finalize_only:
        finalize_outputs(args, out, logger)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 29 requires CUDA for official Vec2Text correction")
    geometries = parse_geometries(args.geometries)
    known_pairs = parse_ints(args.known_pairs)
    procrustes_pairs = parse_ints(args.procrustes_pairs)
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    events_path = out / "runtime_events.json"
    events: list[dict[str, Any]] = (
        json.loads(events_path.read_text(encoding="utf-8"))
        if args.resume and events_path.exists()
        else []
    )
    started = time.time()
    config = {
        **vars(args),
        "out": str(out),
        "geometries_parsed": [asdict(g) for g in geometries],
        "known_pairs_parsed": known_pairs,
        "procrustes_pairs_parsed": procrustes_pairs,
        "encoder": "sentence-transformers/gtr-t5-base",
        "vec2text_inversion_checkpoint": "jxm/gtr__nq__32",
        "vec2text_corrector_checkpoint": "jxm/gtr__nq__32__correct",
        "embedding_path": "corrector.inversion_trainer.call_embedding_model",
        "pooling": "attention-mask mean pooling; no L2 normalization",
        "normalization_scope": (
            "Native GTR vectors are intentionally unnormalized because the official "
            "corrector is checkpoint-compatible only with that representation; this is "
            "not the paper's literal unit-normalized e5 retrieval configuration."
        ),
        "cell_clustering": "sklearn KMeans(k-means++, n_init=10, algorithm=lloyd)",
        "attacker_side_information": (
            "Strengthened observer knows the exact full PCA basis V and document "
            "mean mu; only target cell key H is unknown in unknown-key/OLS/Procrustes."
        ),
    }
    dump_json(out / "config.json", config)
    runtime = {
        "started_unix": started,
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": git_commit(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch", "transformers", "vec2text", "datasets", "numpy",
                "scikit-learn", "sacrebleu",
            )
        },
        "torch_cuda": torch.version.cuda,
        "gpu_start": gpu_snapshot(),
        "status": "running",
    }
    dump_json(out / "runtime.json", runtime)
    logger.info("Starting exp29 profile=%s; GPU=%s", args.profile, gpu_snapshot())
    torch.manual_seed(args.sample_seed)
    np.random.seed(args.sample_seed)

    corrector = load_official_corrector(logger)
    raw_control_sanity(args, corrector, out, logger, events)
    records, digest, ag_path = build_or_load_corpus(
        args, corrector.embedder_tokenizer, out, logger
    )
    x = load_or_encode_embeddings(
        args, corrector, records, digest, out, logger, events
    )
    pca_path = out / "full_pca.npz"
    if args.resume and pca_path.exists():
        stored = np.load(pca_path)
        mu, basis, eigenvalues = stored["mu"], stored["basis"], stored["eigenvalues"]
        if basis.shape != (x.shape[1], x.shape[1]):
            raise RuntimeError("Stored full PCA has incompatible shape")
        logger.info("Reusing full PCA from %s", pca_path)
    else:
        pca_started = time.time()
        mu, basis, eigenvalues = pca_full(x)
        np.savez_compressed(pca_path, mu=mu, basis=basis, eigenvalues=eigenvalues)
        logger.info("Fitted full %dx%d PCA in %.1fs", x.shape[1], x.shape[1], time.time() - pca_started)
    coordinates = ((x - mu) @ basis).astype(np.float32)
    max_reconstruction_error = float(
        np.max(np.abs(coordinates @ basis.T + mu - x))
    )
    logger.info("Full-PCA max absolute reconstruction error %.3e", max_reconstruction_error)

    geometry_cache: dict[str, dict[str, Any]] = {}
    run_manifest: list[dict[str, Any]] = []
    for geometry in geometries:
        seen_cells: set[int] = set()
        for target_cohort in ("largest", "largest_pii"):
            attack_cases, targets, run_info = build_attack_cases(
                x,
                records,
                coordinates,
                basis,
                mu,
                geometry,
                known_pairs,
                procrustes_pairs,
                args.n_test,
                args.selection_seed,
                args.kmeans_iters,
                args.rank_rtol,
                geometry_cache,
                target_cohort,
            )
            # If the globally largest cell is itself the largest PII cell, the
            # secondary cohort would be an exact duplicate.
            if run_info["target_cell"] in seen_cells:
                continue
            seen_cells.add(run_info["target_cell"])
            run_manifest.append(run_info)
            dump_json(out / "run_manifest.json", run_manifest)
            true_target = x[targets]
            logger.info(
                "%s cohort=%s cell=%d size=%d; targets AG=%d PII=%d",
                geometry.run_id,
                target_cohort,
                run_info["target_cell"],
                run_info["target_cell_size"],
                run_info["n_test_ag_news"],
                run_info["n_test_synthetic_pii"],
            )
            for method, (attack_embedding, m, fitted_rank) in attack_cases.items():
                case_id = f"{run_info['run_id']}__{method}"
                case_path = cases_dir / f"case_{safe_name(case_id)}.json"
                if args.resume and case_path.exists():
                    existing = json.loads(case_path.read_text(encoding="utf-8"))
                    if existing.get("status") == "completed":
                        logger.info("Skipping completed %s", case_id)
                        continue
                logger.info(
                    "Running %s: m=%s rank=%s geometry-cos=%.4f",
                    case_id,
                    m,
                    fitted_rank,
                    float(cosine_rows(attack_embedding, true_target).mean()),
                )
                hypotheses, inversion_meta = invert_embeddings(
                    corrector,
                    attack_embedding,
                    case_id,
                    args.invert_batch_size,
                    args.num_steps,
                    args.beam_width,
                    logger,
                    events,
                )
                recovered_embeddings = encode_texts(
                    corrector,
                    hypotheses,
                    args.max_embed_tokens,
                    min(args.encode_batch_size, max(1, len(hypotheses))),
                    logger,
                    events,
                )
                row_context = {
                    key: value
                    for key, value in run_info.items()
                    if key
                    not in {
                        "target_indices",
                        "anchor_indices_first_max_m",
                        "cell_sizes",
                        "cell_pii_sizes",
                    }
                }
                rows = evaluate_predictions(
                    records,
                    targets,
                    hypotheses,
                    attack_embedding,
                    true_target,
                    recovered_embeddings,
                    row_context,
                    method,
                    m,
                )
                payload = {
                    "status": "completed",
                    "case_id": case_id,
                    "method": method,
                    "m": m,
                    "fitted_rank": fitted_rank,
                    "run_info": run_info,
                    "inversion": inversion_meta,
                    "rows": rows,
                }
                dump_json(case_path, payload)
                dump_json(events_path, events)
                logger.info(
                    "Completed %s in %.1fs: token-F1=%.4f BLEU=%.4f PII=%s",
                    case_id,
                    inversion_meta["seconds"],
                    metric_value(rows, "token_f1"),
                    metric_value(rows, "bleu"),
                    metric_value(rows, "pii_recall"),
                )

    finalize_outputs(args, out, logger)
    runtime.update(
        {
            "status": "completed",
            "finished_unix": time.time(),
            "finished_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "elapsed_seconds": time.time() - started,
            "gpu_end": gpu_snapshot(),
            "max_full_pca_reconstruction_abs_error": max_reconstruction_error,
            "corpus_sha256": digest,
            "ag_news_cache": ag_path,
            "n_runtime_events": len(events),
        }
    )
    dump_json(events_path, events)
    dump_json(out / "runtime.json", runtime)
    logger.info("Experiment 29 complete in %.1fs", runtime["elapsed_seconds"])


if __name__ == "__main__":
    main()
