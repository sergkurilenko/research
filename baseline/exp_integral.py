"""Multi-encoder integral experiment with PQ + CKKS architecture (v3).

CHANGE FROM v2: PQ index is now trained in the ROTATED space (E_rot), not in
the V_k-projected space. This closes the Procrustes-attack vulnerability:
in v2, server had E_rot (exact) plus access to PQ codes that decoded to
approximate E_proj vectors, allowing it to solve the orthogonal Procrustes
problem (E_proj, E_rot) = X X R^T -> recover R. In v3, both server's E_rot
and the PQ artifact live in the same rotated space, giving server no
additional information about R.

Per-seed PQ index is required (since R differs between seeds).

Architecture:
  1. SVD V_k (k = d/2 in the canonical operating point of §4.4;
     ENCODERS list also supports the auxiliary k = 7d/8 sweep via the
     `_half` / non-half encoder tag pairs) computed offline, client-secret with mu.
  2. Per-deployment rotation R, CLIENT-secret.
  3. Public artifact: PQ codebook + PQ codes per doc, IN ROTATED SPACE
     (faiss IndexPQ trained on E_rot = E_proj @ R^T).
  4. Per-query: client computes q_proj, q_tilde = q_proj @ R^T,
     runs PQ search using q_tilde (in rotated space) to pick top-K_CANDS,
     encrypts q_tilde, sends ct + cand_ids to server.
  5. Server: CKKS ct-pt rerank against E_rot[cand_ids], returns scores.
  6. Client: decrypts, ranks, returns top-10.

Output: exp5_outputs/v3_multi_results.json
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json
import os
import time
from pathlib import Path

import numpy as np
import faiss
import tenseal as ts
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

CACHE = DATA
INDEX_CACHE = CACHE / "index_cache"
INDEX_CACHE.mkdir(exist_ok=True)
OUT = (RESULTS / "exp5_outputs")
OUT.mkdir(exist_ok=True)

SEED_BASE = 42
SEEDS = [11, 23, 47, 31, 53]
T_CRIT_95 = {3: 4.303, 4: 3.182, 5: 2.776}

CKKS_N = 8192
CKKS_COEFF = [60, 40, 60]
CKKS_SCALE_BITS = 40

TOP_K = 12
K_CANDS = int(os.environ.get("V3_K_CANDS", 40))
PQ_NBITS = 8


def acc_at_k(pred, rel, k):
    return float(bool(set(pred[:k]) & rel))


def mrr_score(pred, rel):
    for r, p in enumerate(pred, 1):
        if p in rel:
            return 1.0 / r
    return 0.0


def ndcg_at_k(pred, rel, k=10):
    return sum(1.0 / np.log2(i + 2) for i, p in enumerate(pred[:k]) if p in rel)


def aggregate(preds, qrels, fn):
    return float(np.mean([fn(p, r) for p, r in zip(preds, qrels)]))


def metrics(preds, qrels):
    return {
        "acc1":   aggregate(preds, qrels, lambda p, r: acc_at_k(p, r, 1)),
        "acc10":  aggregate(preds, qrels, lambda p, r: acc_at_k(p, r, 10)),
        "mrr":    aggregate(preds, qrels, mrr_score),
        "ndcg10": aggregate(preds, qrels, ndcg_at_k),
    }


def t_ci(values):
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return float(arr.mean()), 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    t = T_CRIT_95.get(n, 1.96)
    return mean, std, float(t * std / np.sqrt(n))


def make_random_orthogonal(seed, dim):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def compute_svd(E, k):
    mu = E.mean(axis=0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0)
    n_sample = min(200_000, len(E))
    idx = rng.choice(len(E), size=n_sample, replace=False)
    _, _, Vt = randomized_svd(E[idx] - mu, n_components=k, random_state=42)
    return mu, Vt.T.astype(np.float32)


def build_or_load_pq_rotated(E_rot, m, nbits, encoder_tag, proj_k, seed):
    """Build (or load cached) faiss IndexPQ over ROTATED docs E_rot.

    v3: PQ artifact now lives in the rotated space.  Per-seed (depends on R).
    """
    path = INDEX_CACHE / (
        f"v3_indexpq_{encoder_tag}_proj{proj_k}_M{m}_b{nbits}_seed{seed}.faiss"
    )
    if path.exists():
        return faiss.read_index(str(path))
    print(f"  Training PQ M={m} (rotated, seed={seed}) for {encoder_tag} ...")
    t0 = time.time()
    index_pq = faiss.IndexPQ(proj_k, m, nbits, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(E_rot), size=min(200_000, len(E_rot)), replace=False)
    index_pq.train(E_rot[train_idx].astype(np.float32))
    index_pq.add(E_rot.astype(np.float32))
    faiss.write_index(index_pq, str(path))
    print(f"  PQ-rot built in {time.time()-t0:.0f}s")
    return index_pq


def make_ctx():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=CKKS_N,
                     coeff_mod_bit_sizes=CKKS_COEFF)
    ctx.global_scale = 2 ** CKKS_SCALE_BITS
    ctx.generate_galois_keys()
    return ctx


def run_seed(seed, E_proj, qrels, E_q_proj, encoder_tag, proj_k, pq_m):
    """Run seed with PQ trained in the rotated space (per seed)."""
    R = make_random_orthogonal(seed, proj_k)
    E_rot = (E_proj @ R.T).astype(np.float32)
    # Build (or load) per-seed PQ index over E_rot.
    index_pq = build_or_load_pq_rotated(
        E_rot, pq_m, PQ_NBITS, encoder_tag, proj_k, seed)
    N_q = int(os.environ.get("V3_N_QUERIES", len(E_q_proj)))
    N_q = min(N_q, len(E_q_proj))
    preds = []
    timings = {"encrypt_ms": [], "rerank_ms": [], "decrypt_ms": [],
               "stage1_ms": [], "total_server_ms": []}
    ctx = make_ctx()
    for i in tqdm(range(N_q), desc=f"  seed={seed} k={proj_k}", leave=False):
        q_proj = E_q_proj[i]
        q_tilde = (q_proj @ R.T).astype(np.float32)
        # Stage-1: PQ search in ROTATED space using q_tilde
        t0 = time.time()
        _, cand_ids_arr = index_pq.search(
            q_tilde.reshape(1, -1).astype(np.float32), K_CANDS)
        cand_ids = cand_ids_arr[0]
        timings["stage1_ms"].append(1000 * (time.time() - t0))
        t0 = time.time()
        ct_q = ts.ckks_vector(ctx, q_tilde.tolist())
        timings["encrypt_ms"].append(1000 * (time.time() - t0))
        t_server_start = time.time()
        t0 = time.time()
        scores = []
        for j in cand_ids:
            d_rot = E_rot[int(j)]
            score = float((ct_q * d_rot.tolist()).sum().decrypt()[0])
            scores.append(score)
        timings["rerank_ms"].append(1000 * (time.time() - t0))
        scores = np.array(scores)
        order = np.argsort(-scores)[:TOP_K]
        top_cand_ids = cand_ids[order]
        top_scores = scores[order]
        timings["total_server_ms"].append(1000 * (time.time() - t_server_start))
        t0 = time.time()
        final_order = np.argsort(-top_scores)[:10]
        timings["decrypt_ms"].append(1000 * (time.time() - t0))
        preds.append([int(top_cand_ids[o]) for o in final_order])
    m = metrics(preds, qrels)
    m["timing_p50_ms"] = {k: float(np.percentile(v, 50)) for k, v in timings.items()}
    m["timing_p95_ms"] = {k: float(np.percentile(v, 95)) for k, v in timings.items()}
    m["timing_p99_ms"] = {k: float(np.percentile(v, 99)) for k, v in timings.items()}
    m["k_cands"] = K_CANDS
    return m


def baseline_dense(E_docs, E_queries, qrels):
    ix = faiss.IndexFlatIP(E_docs.shape[1])
    ix.add(E_docs)
    _, ids = ix.search(E_queries, 10)
    preds = [list(int(j) for j in row) for row in ids]
    return metrics(preds, qrels)


def baseline_proj(E_proj, E_q_proj, qrels):
    ix = faiss.IndexFlatIP(E_proj.shape[1])
    ix.add(E_proj)
    _, ids = ix.search(E_q_proj, 10)
    preds = [list(int(j) for j in row) for row in ids]
    return metrics(preds, qrels)


ENCODERS = [
    # Standard operating point: k = 7d/8 (bandwidth-optimised, sigma_rec << 0.10)
    {"tag": "e5small",  "dim": 384,  "k": 336, "pq_m": 84,
     "docs": "E_docs_e5small_1000000.npy",
     "queries": "E_queries_e5small_self_q500.npy"},
    {"tag": "e5base",   "dim": 768,  "k": 672, "pq_m": 96,
     "docs": "E_docs_e5base_1000000.npy",
     "queries": "E_queries_e5base_self_q500.npy"},
    {"tag": "mpnet",    "dim": 768,  "k": 672, "pq_m": 96,
     "docs": "E_docs_mpnet_1000000.npy",
     "queries": "E_queries_mpnet_self_q500.npy"},
    # Extra high-Acc@1 encoders (d=1024). k = 7d/8 = 896.
    # PQ M = 128 (subquantizer dim = 7, matches e5-base granularity).
    {"tag": "e5large",  "dim": 1024, "k": 896, "pq_m": 128,
     "docs": "E_docs_e5large_1000000.npy",
     "queries": "E_queries_e5large_self_q500.npy"},
    {"tag": "bgem3",    "dim": 1024, "k": 896, "pq_m": 128,
     "docs": "E_docs_bgem3_1000000.npy",
     "queries": "E_queries_bgem3_self_q500.npy"},
    # High-protection operating point: k = d/2 (sigma_rec >= 0.10, proxy-R3 met).
    # PQ sub-quantiser dim = 4 (M = k/4) -- finer granularity at smaller k
    # to keep PQ-stage1 recall close to baseline_proj.
    {"tag": "e5small_half", "dim": 384,  "k": 192, "pq_m": 48,
     "docs": "E_docs_e5small_1000000.npy",
     "queries": "E_queries_e5small_self_q500.npy"},
    {"tag": "e5base_half",  "dim": 768,  "k": 384, "pq_m": 96,
     "docs": "E_docs_e5base_1000000.npy",
     "queries": "E_queries_e5base_self_q500.npy"},
    {"tag": "mpnet_half",   "dim": 768,  "k": 384, "pq_m": 96,
     "docs": "E_docs_mpnet_1000000.npy",
     "queries": "E_queries_mpnet_self_q500.npy"},
    {"tag": "e5large_half", "dim": 1024, "k": 512, "pq_m": 128,
     "docs": "E_docs_e5large_1000000.npy",
     "queries": "E_queries_e5large_self_q500.npy"},
    {"tag": "bgem3_half",   "dim": 1024, "k": 512, "pq_m": 128,
     "docs": "E_docs_bgem3_1000000.npy",
     "queries": "E_queries_bgem3_self_q500.npy"},
]


# Allow restricting to a subset via env var (e.g. V3_ENCODER_TAGS=e5large,bgem3)
_only = os.environ.get("V3_ENCODER_TAGS", "").strip()
if _only:
    _wanted = set(t.strip() for t in _only.split(",") if t.strip())
    ENCODERS = [e for e in ENCODERS if e["tag"] in _wanted]


def run_encoder(cfg):
    print(f"\n=== ENCODER: {cfg['tag']} d={cfg['dim']} k={cfg['k']} M={cfg['pq_m']} ===")
    E_docs    = np.load(CACHE / cfg["docs"]).astype(np.float32)
    E_queries = np.load(CACHE / cfg["queries"]).astype(np.float32)
    print(f"  E_docs={E_docs.shape}, E_queries={E_queries.shape}")
    rng_q = np.random.default_rng(SEED_BASE)
    qrels = [{int(i)} for i in rng_q.choice(len(E_docs), size=500, replace=False)]
    print(f"  SVD k={cfg['k']} ...")
    mu, Vk = compute_svd(E_docs, cfg["k"])
    sample = E_docs[:5000]
    e_proj_s = (sample - mu) @ Vk
    e_recon = e_proj_s @ Vk.T + mu
    sigma_rec = float(np.mean(np.linalg.norm(sample - e_recon, axis=1) /
                              (np.linalg.norm(sample, axis=1) + 1e-9)))
    print(f"  sigma_rec = {sigma_rec:.4f}")
    print("  baseline_dense (raw d) ...")
    bd = baseline_dense(E_docs, E_queries, qrels)
    print(f"    Acc@1={bd['acc1']:.3f} Acc@10={bd['acc10']:.3f} MRR={bd['mrr']:.3f} NDCG={bd['ndcg10']:.3f}")
    print("  Projecting ...")
    E_proj = ((E_docs - mu) @ Vk).astype(np.float32)
    E_q_proj = ((E_queries - mu) @ Vk).astype(np.float32)
    del E_docs
    print("  baseline_proj (V_k space) ...")
    bp = baseline_proj(E_proj, E_q_proj, qrels)
    print(f"    Acc@1={bp['acc1']:.3f} Acc@10={bp['acc10']:.3f} MRR={bp['mrr']:.3f} NDCG={bp['ndcg10']:.3f}")
    seeds_results = []
    for s in SEEDS:
        m = run_seed(s, E_proj, qrels, E_q_proj, cfg["tag"], cfg["k"], cfg["pq_m"])
        print(f"  seed={s}: Acc@1={m['acc1']:.3f} Acc@10={m['acc10']:.3f} "
              f"MRR={m['mrr']:.3f} NDCG={m['ndcg10']:.3f} "
              f"server p95={m['timing_p95_ms']['total_server_ms']:.0f}ms")
        seeds_results.append(m)
    a1  = [m["acc1"]   for m in seeds_results]
    a10 = [m["acc10"]  for m in seeds_results]
    mrr = [m["mrr"]    for m in seeds_results]
    nd  = [m["ndcg10"] for m in seeds_results]
    server_p50 = [m["timing_p50_ms"]["total_server_ms"] for m in seeds_results]
    server_p95 = [m["timing_p95_ms"]["total_server_ms"] for m in seeds_results]
    return {
        "encoder": cfg["tag"], "dim": cfg["dim"], "k": cfg["k"],
        "pq_m": cfg["pq_m"], "sigma_rec": sigma_rec,
        "baseline_dense": bd, "baseline_proj":  bp,
        "per_seed": seeds_results,
        "aggregate": {
            "acc1":   {"mean": t_ci(a1)[0],  "std": t_ci(a1)[1],  "ci_half": t_ci(a1)[2]},
            "acc10":  {"mean": t_ci(a10)[0], "std": t_ci(a10)[1], "ci_half": t_ci(a10)[2]},
            "mrr":    {"mean": t_ci(mrr)[0], "std": t_ci(mrr)[1], "ci_half": t_ci(mrr)[2]},
            "ndcg10": {"mean": t_ci(nd)[0],  "std": t_ci(nd)[1],  "ci_half": t_ci(nd)[2]},
            "server_p50_ms": {"mean": t_ci(server_p50)[0]},
            "server_p95_ms": {"mean": t_ci(server_p95)[0]},
        },
    }


def main():
    print(f"=== v3 PQ-in-rotated-space + CKKS rerank, multi-encoder ===")
    print(f"  CKKS: N={CKKS_N}, coeff={CKKS_COEFF}, scale=2^{CKKS_SCALE_BITS}")
    print(f"  Seeds: {SEEDS}, K_CANDS: {K_CANDS}")
    print(f"  Encoders: {[e['tag'] for e in ENCODERS]}")
    new_results = [run_encoder(cfg) for cfg in ENCODERS]

    # Output path: default v3_multi_results.json. If V3_OUT_TAG is set, write
    # to v3_multi_results_<tag>.json so that incremental runs do not overwrite
    # the canonical 5-encoder file.
    out_tag = os.environ.get("V3_OUT_TAG", "").strip()
    out_name = f"v3_multi_results_{out_tag}.json" if out_tag else "v3_multi_results.json"
    out_path = OUT / out_name

    # Merge mode: if V3_MERGE_INTO is set and the target file exists, merge
    # new encoders into it (preserving existing entries unless tag matches).
    merge_target = os.environ.get("V3_MERGE_INTO", "").strip()
    if merge_target:
        merge_path = OUT / merge_target
        if merge_path.exists():
            with open(merge_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            keep = [e for e in existing.get("encoders", [])
                    if e.get("encoder") not in {r.get("encoder") for r in new_results}]
            all_results = keep + new_results
            print(f"\n  merging into {merge_path}: kept {len(keep)} existing + "
                  f"{len(new_results)} new = {len(all_results)} total")
            out_path = merge_path
        else:
            print(f"\n  V3_MERGE_INTO target {merge_path} not found, writing fresh")
            all_results = new_results
    else:
        all_results = new_results

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "ckks_params": {"N": CKKS_N, "coeff": CKKS_COEFF, "scale_bits": CKKS_SCALE_BITS},
            "common_params": {"k_cands": K_CANDS, "pq_nbits": PQ_NBITS, "rerank_top": TOP_K},
            "architecture": "v3 (PQ in rotated space)",
            "encoders": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n=== SUMMARY ===")
    for r in all_results:
        a = r["aggregate"]
        bp = r["baseline_proj"]
        bd = r["baseline_dense"]
        print(f"\n{r['encoder']} (d={r['dim']}, k={r['k']}, sigma_rec={r['sigma_rec']:.3f}):")
        print(f"  baseline_dense: Acc@1={bd['acc1']:.3f}, Acc@10={bd['acc10']:.3f}")
        print(f"  baseline_proj:  Acc@1={bp['acc1']:.3f}, Acc@10={bp['acc10']:.3f}")
        print(f"  proposed (v3 PQ-rot+CKKS):")
        print(f"    Acc@1={a['acc1']['mean']:.3f} +/- {a['acc1']['ci_half']:.3f}")
        print(f"    Acc@10={a['acc10']['mean']:.3f} +/- {a['acc10']['ci_half']:.3f}")
        print(f"    MRR={a['mrr']['mean']:.3f} +/- {a['mrr']['ci_half']:.3f}")
        print(f"    NDCG@10={a['ndcg10']['mean']:.3f} +/- {a['ndcg10']['ci_half']:.3f}")
        print(f"    server p50={a['server_p50_ms']['mean']:.0f}, p95={a['server_p95_ms']['mean']:.0f} ms")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
