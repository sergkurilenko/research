"""Reference-corpus nearest-neighbour attack after known-plaintext alignment.

This experiment strengthens the manuscript beyond the pure embedding-space
Procrustes test. It simulates an attacker who has:

* a protected target vector y = SVD(x) R;
* m known plaintext/protected pairs for estimating the hidden rotation;
* a large reference corpus of candidate texts and their native embeddings.

If the reference corpus overlaps the protected collection, successful
alignment becomes a text-level lookup attack: the attacker can recover the
exact source paragraph by nearest-neighbour search. We also report a disjoint
reference-corpus lexical proxy, where the exact target paragraph is removed
and the top retrieved decoy is compared to the target text.

Outputs:
  results/exp9_outputs/exp9_reference_attack_by_seed.csv
  results/exp9_outputs/exp9_reference_attack_summary.csv
  results/exp9_outputs/exp9_reference_attack_summary.json
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
CACHE = NB / "_corpus_cache"
INDEX_CACHE = CACHE / "index_cache"
OUT = NB / "exp9_outputs"


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--rotation-seeds", default="11,23,31,47,53")
    ap.add_argument("--n-anchors", type=int, default=1200)
    ap.add_argument("--n-targets", type=int, default=500)
    ap.add_argument("--n-ref-decoys", type=int, default=100000)
    ap.add_argument("--known-pairs", default="0,10,25,50,100,192,500,1000")
    ap.add_argument("--chunk", type=int, default=128)
    return ap.parse_args()


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def make_random_orthogonal(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    c = source.T @ target
    u, _, vt = np.linalg.svd(c, full_matrices=False)
    return (u @ vt).astype(np.float32)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def search_topk(
    queries: np.ndarray,
    ref: np.ndarray,
    k: int = 10,
    chunk: int = 128,
) -> np.ndarray:
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], chunk):
        q = queries[start : start + chunk]
        scores = q @ ref.T
        out[start : start + len(q)] = topk_indices(scores, k)
    return out


def tokenize(text: str) -> set[str]:
    toks = {t.lower() for t in TOKEN_RE.findall(text)}
    return {t for t in toks if len(t) >= 3}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def mean_top1_jaccard(
    top1_positions: Iterable[int],
    target_tokens: list[set[str]],
    ref_tokens: list[set[str]],
) -> float:
    vals = [jaccard(target_tokens[i], ref_tokens[int(pos)]) for i, pos in enumerate(top1_positions)]
    return float(np.mean(vals))


def load_projected(ids: np.ndarray, k: int) -> np.ndarray:
    docs = np.load(CACHE / "E_docs_e5small_1000000.npy", mmap_mode="r")
    svd = np.load(INDEX_CACHE / "svd_e5small_proj336.npz")
    mu = svd["mu"].astype(np.float32)
    vk = svd["Vk"][:, :k].astype(np.float32)
    x = np.asarray(docs[ids], dtype=np.float32)
    return ((x - mu) @ vk).astype(np.float32)


def load_texts(ids: np.ndarray) -> list[str]:
    with open(CACHE / "corpus_wiki_ru_1000000.pkl", "rb") as f:
        corpus = pickle.load(f)
    docs = corpus["docs"]
    return [docs[int(i)] for i in ids]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "overlap_recall_at_1",
        "overlap_recall_at_10",
        "overlap_top1_token_jaccard",
        "disjoint_top1_token_jaccard",
        "mean_cosine_to_projected",
        "relative_l2_error",
    ]
    for m, sub in df.groupby("known_pairs"):
        row = {"known_pairs": int(m), "n_seeds": int(len(sub))}
        for col in metrics:
            vals = sub[col].astype(float).to_numpy()
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("known_pairs")


def main() -> None:
    args = parse_args()
    OUT.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    total = args.n_anchors + args.n_targets + args.n_ref_decoys
    ids = rng.choice(1_000_000, size=total, replace=False).astype(np.int64)
    anchor_ids = ids[: args.n_anchors]
    target_ids = ids[args.n_anchors : args.n_anchors + args.n_targets]
    decoy_ids = ids[args.n_anchors + args.n_targets :]

    print("Loading/projecting embeddings ...")
    z_anchors = load_projected(anchor_ids, args.k)
    z_targets = load_projected(target_ids, args.k)
    z_decoys = load_projected(decoy_ids, args.k)
    z_overlap_ref = np.vstack([z_targets, z_decoys]).astype(np.float32)
    z_disjoint_ref = z_decoys.astype(np.float32)

    z_targets_n = row_normalize(z_targets)
    z_overlap_ref_n = row_normalize(z_overlap_ref)
    z_disjoint_ref_n = row_normalize(z_disjoint_ref)

    print("Loading texts ...")
    target_texts = load_texts(target_ids)
    decoy_texts = load_texts(decoy_ids)
    overlap_texts = target_texts + decoy_texts
    target_tokens = [tokenize(t) for t in target_texts]
    overlap_tokens = [tokenize(t) for t in overlap_texts]
    decoy_tokens = [tokenize(t) for t in decoy_texts]

    known_pairs = [int(x) for x in args.known_pairs.split(",") if x.strip()]
    rotation_seeds = [int(x) for x in args.rotation_seeds.split(",") if x.strip()]
    rows = []

    for seed in rotation_seeds:
        print(f"rotation seed {seed}")
        r = make_random_orthogonal(seed, args.k)
        y_anchors = z_anchors @ r
        y_targets = z_targets @ r

        for m in known_pairs:
            if m == 0:
                w = np.eye(args.k, dtype=np.float32)
            else:
                w = procrustes(y_anchors[:m], z_anchors[:m])
            z_hat = (y_targets @ w).astype(np.float32)
            z_hat_n = row_normalize(z_hat)

            overlap_top10 = search_topk(z_hat_n, z_overlap_ref_n, k=10, chunk=args.chunk)
            disjoint_top10 = search_topk(z_hat_n, z_disjoint_ref_n, k=10, chunk=args.chunk)

            target_positions = np.arange(args.n_targets)[:, None]
            overlap_r1 = float(np.mean(overlap_top10[:, :1] == target_positions))
            overlap_r10 = float(np.mean(np.any(overlap_top10 == target_positions, axis=1)))
            overlap_j = mean_top1_jaccard(overlap_top10[:, 0], target_tokens, overlap_tokens)
            disjoint_j = mean_top1_jaccard(disjoint_top10[:, 0], target_tokens, decoy_tokens)

            rel = float(np.linalg.norm(z_hat - z_targets) / np.linalg.norm(z_targets))
            cos = float(np.mean(np.sum(z_hat_n * z_targets_n, axis=1)))
            row = {
                "rotation_seed": seed,
                "known_pairs": m,
                "reference_size_with_overlap": int(z_overlap_ref_n.shape[0]),
                "reference_size_disjoint": int(z_disjoint_ref_n.shape[0]),
                "overlap_recall_at_1": overlap_r1,
                "overlap_recall_at_10": overlap_r10,
                "overlap_top1_token_jaccard": overlap_j,
                "disjoint_top1_token_jaccard": disjoint_j,
                "mean_cosine_to_projected": cos,
                "relative_l2_error": rel,
            }
            rows.append(row)
            print(
                f"  m={m:4d}: exact R@1={overlap_r1:.3f}, "
                f"R@10={overlap_r10:.3f}, overlap J={overlap_j:.3f}, "
                f"disjoint J={disjoint_j:.3f}, cos={cos:.3f}"
            )

    by_seed = pd.DataFrame(rows)
    summary = summarize(by_seed)
    by_seed_path = OUT / "exp9_reference_attack_by_seed.csv"
    summary_path = OUT / "exp9_reference_attack_summary.csv"
    json_path = OUT / "exp9_reference_attack_summary.json"
    by_seed.to_csv(by_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "parameters": vars(args),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {by_seed_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
