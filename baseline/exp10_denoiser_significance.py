"""Experiment 10: paired significance of the SVD-truncation effect on retrieval.

This addresses the reviewer concern that the +Acc@1 "linear denoiser" effect
reported in the integral experiment (Table 8) is within per-query sampling
noise and was only reported with an across-seed CI (which measures rotation
stability, not query-sampling uncertainty).

For each encoder at the canonical operating point k = d/2 we compare:
    - baseline_dense : exact inner-product retrieval in the raw R^d space
    - baseline_proj  : exact inner-product retrieval in span(V_k)
on the SAME 500 self-retrieval queries (paired). We then report:
    - McNemar exact test for Acc@1 and Acc@10 (paired binary outcomes)
    - paired bootstrap 95% CI for Delta Acc@1, Delta Acc@10, Delta MRR

Neither rotation nor CKKS nor PQ is involved here: the denoiser effect is a
property of the V_k projection alone, so the experiment needs only numpy +
scikit-learn on the cached embeddings (no faiss / tenseal).

Ground truth and SVD construction are byte-for-byte identical to
_rerun_exp5_v3_multi.py so the raw/SVD Acc numbers reproduce Table 8.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json
import time
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.stats import binomtest

CACHE = DATA
OUT = (RESULTS / "exp10_outputs")
OUT.mkdir(exist_ok=True)

SEED_BASE = 42
N_QUERIES = 500
DOC_CHUNK = 100_000
BOOT = 10_000
BOOT_SEED = 2026

# Canonical operating point k = d/2 (matches Table 8 "_half" configs).
ENCODERS = [
    {"tag": "e5-small", "dim": 384,  "k": 192,
     "docs": "E_docs_e5small_1000000.npy", "queries": "E_queries_e5small_self_q500.npy"},
    {"tag": "e5-base",  "dim": 768,  "k": 384,
     "docs": "E_docs_e5base_1000000.npy",  "queries": "E_queries_e5base_self_q500.npy"},
    {"tag": "mpnet",    "dim": 768,  "k": 384,
     "docs": "E_docs_mpnet_1000000.npy",   "queries": "E_queries_mpnet_self_q500.npy"},
    {"tag": "e5-large", "dim": 1024, "k": 512,
     "docs": "E_docs_e5large_1000000.npy", "queries": "E_queries_e5large_self_q500.npy"},
    {"tag": "bge-m3",   "dim": 1024, "k": 512,
     "docs": "E_docs_bgem3_1000000.npy",   "queries": "E_queries_bgem3_self_q500.npy"},
]


def compute_svd(E, k):
    """Identical to _rerun_exp5_v3_multi.compute_svd."""
    mu = E.mean(axis=0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0)
    n_sample = min(200_000, len(E))
    idx = rng.choice(len(E), size=n_sample, replace=False)
    _, _, Vt = randomized_svd(E[idx] - mu, n_components=k, random_state=42)
    return mu, Vt.T.astype(np.float32)


def topk_search(E_docs, Q, k=10):
    """Chunked exact top-k inner-product search (matches faiss IndexFlatIP).

    Returns the (n_q, k) array of doc ids sorted by descending score. Using the
    actual top-k avoids any cross-arithmetic self-comparison of the ground-truth
    document (which corrupts Acc@1 if gt-score is computed in a separate pass).
    """
    n, nq = len(E_docs), len(Q)
    best_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    best_ids = np.full((nq, k), -1, dtype=np.int64)
    for start in range(0, n, DOC_CHUNK):
        chunk = E_docs[start:start + DOC_CHUNK]
        s = Q @ chunk.T                       # (nq, csz), float32 (== faiss)
        kk = min(k, s.shape[1])
        part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        part_scores = np.take_along_axis(s, part, axis=1)
        cand_ids = part + start
        all_scores = np.concatenate([best_scores, part_scores], axis=1)
        all_ids = np.concatenate([best_ids, cand_ids], axis=1)
        sel = np.argsort(-all_scores, axis=1)[:, :k]
        best_scores = np.take_along_axis(all_scores, sel, axis=1)
        best_ids = np.take_along_axis(all_ids, sel, axis=1)
    return best_ids


def metrics_from_top(top_ids, gt_ids):
    hit1 = (top_ids[:, 0] == gt_ids).astype(np.float64)
    hit10 = np.any(top_ids == gt_ids[:, None], axis=1).astype(np.float64)
    rr = np.zeros(len(gt_ids), dtype=np.float64)
    for i in range(len(gt_ids)):
        pos = np.where(top_ids[i] == gt_ids[i])[0]
        rr[i] = 1.0 / (pos[0] + 1) if pos.size else 0.0
    return hit1, hit10, rr


def mcnemar(raw_hit, proj_hit):
    raw_hit = raw_hit.astype(bool)
    proj_hit = proj_hit.astype(bool)
    b = int(np.sum(raw_hit & ~proj_hit))   # raw correct, proj wrong
    c = int(np.sum(~raw_hit & proj_hit))   # raw wrong, proj correct
    nd = b + c
    if nd == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    p = binomtest(min(b, c), nd, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "p_value": float(p)}


def boot_ci(raw_vals, proj_vals, seed=BOOT_SEED, B=BOOT):
    rng = np.random.default_rng(seed)
    n = len(raw_vals)
    diff = proj_vals - raw_vals
    idx = rng.integers(0, n, size=(B, n))
    boot = diff[idx].mean(axis=1)
    return {
        "delta_mean": float(diff.mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
    }


def run_encoder(cfg):
    print(f"\n=== {cfg['tag']} (d={cfg['dim']}, k={cfg['k']}) ===", flush=True)
    t0 = time.time()
    E_docs = np.load(CACHE / cfg["docs"]).astype(np.float32)
    E_q = np.load(CACHE / cfg["queries"]).astype(np.float32)[:N_QUERIES]
    rng_q = np.random.default_rng(SEED_BASE)
    gt_ids = rng_q.choice(len(E_docs), size=N_QUERIES, replace=False)

    mu, Vk = compute_svd(E_docs, cfg["k"])
    sample = E_docs[:5000]
    recon = ((sample - mu) @ Vk) @ Vk.T + mu
    sigma_rec = float(np.mean(np.linalg.norm(sample - recon, axis=1) /
                              (np.linalg.norm(sample, axis=1) + 1e-9)))

    # Raw space
    raw_top = topk_search(E_docs, E_q, 10)
    # Projected space
    E_proj = ((E_docs - mu) @ Vk).astype(np.float32)
    Q_proj = ((E_q - mu) @ Vk).astype(np.float32)
    proj_top = topk_search(E_proj, Q_proj, 10)
    del E_docs, E_proj

    raw_h1, raw_h10, raw_rr = metrics_from_top(raw_top, gt_ids)
    pr_h1, pr_h10, pr_rr = metrics_from_top(proj_top, gt_ids)

    res = {
        "encoder": cfg["tag"], "dim": cfg["dim"], "k": cfg["k"],
        "sigma_rec": sigma_rec,
        "raw":  {"acc1": float(raw_h1.mean()), "acc10": float(raw_h10.mean()),
                 "mrr": float(raw_rr.mean())},
        "svd":  {"acc1": float(pr_h1.mean()), "acc10": float(pr_h10.mean()),
                 "mrr": float(pr_rr.mean())},
        "delta_acc1": float(pr_h1.mean() - raw_h1.mean()),
        "delta_acc10": float(pr_h10.mean() - raw_h10.mean()),
        "mcnemar_acc1": mcnemar(raw_h1, pr_h1),
        "mcnemar_acc10": mcnemar(raw_h10, pr_h10),
        "boot_acc1": boot_ci(raw_h1, pr_h1),
        "boot_acc10": boot_ci(raw_h10, pr_h10),
        "boot_mrr": boot_ci(raw_rr, pr_rr),
    }
    np.savez(OUT / f"hits_{cfg['tag']}.npz",
             raw_h1=raw_h1, raw_h10=raw_h10, raw_rr=raw_rr,
             proj_h1=pr_h1, proj_h10=pr_h10, proj_rr=pr_rr, gt_ids=gt_ids)
    m1, m10 = res["mcnemar_acc1"], res["mcnemar_acc10"]
    b1 = res["boot_acc1"]
    print(f"  sigma_rec={sigma_rec:.3f}", flush=True)
    print(f"  raw  Acc@1={res['raw']['acc1']:.3f} Acc@10={res['raw']['acc10']:.3f} MRR={res['raw']['mrr']:.3f}", flush=True)
    print(f"  svd  Acc@1={res['svd']['acc1']:.3f} Acc@10={res['svd']['acc10']:.3f} MRR={res['svd']['mrr']:.3f}", flush=True)
    print(f"  dAcc@1={res['delta_acc1']:+.3f}  boot95%=[{b1['ci_lo']:+.3f},{b1['ci_hi']:+.3f}]"
          f"  McNemar b={m1['b']} c={m1['c']} p={m1['p_value']:.3g}", flush=True)
    print(f"  dAcc@10={res['delta_acc10']:+.3f} McNemar b={m10['b']} c={m10['c']} p={m10['p_value']:.3g}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    results = [run_encoder(c) for c in ENCODERS]
    with open(OUT / "exp10_significance.json", "w", encoding="utf-8") as f:
        json.dump({"n_queries": N_QUERIES, "bootstrap": BOOT,
                   "operating_point": "k=d/2", "encoders": results},
                  f, indent=2, ensure_ascii=False)
    print("\n=== SUMMARY (paired, n=500 queries) ===")
    print(f"{'enc':10s} {'raw@1':>6s} {'svd@1':>6s} {'dAcc1':>7s} {'boot95':>18s} {'McNemar p':>10s}")
    for r in results:
        b = r["boot_acc1"]
        print(f"{r['encoder']:10s} {r['raw']['acc1']:6.3f} {r['svd']['acc1']:6.3f} "
              f"{r['delta_acc1']:+7.3f} [{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}]   "
              f"{r['mcnemar_acc1']['p_value']:10.3g}")
    print(f"\nSaved to {OUT/'exp10_significance.json'}")


if __name__ == "__main__":
    main()
