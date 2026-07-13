"""Experiment 12: utility of SHARD vs raw and the SVD-k/2 baseline.

Two-stage SHARD retrieval:
  stage-1: shortlist top-K_cands by the short PUBLIC prefix u (top d_pub PCA);
  stage-2: rerank the shortlist by the corrected FULL split score
           <q, x_i-mu>.  This equals <q,x_i>-<q,mu>, so it preserves the raw
           document ranking while remaining a prefix+residual CKKS score.

We compare, on the 10^6-doc self-retrieval probe, for each encoder:
  - raw            : top-k by <x_q, x_i>            (original metric, ceiling)
  - svd (k=d/2)    : top-k by <(x_q-mu)V_k,(x_i-mu)V_k>  (draft baseline)
  - shard(d_pub,Kc): prefix shortlist -> full-dim rerank
The hypothesis: SHARD recovers raw quality (full-dim rerank) while the
baseline does not (it reranks in the truncated half), even when the public
prefix is much shorter than k.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, time
from pathlib import Path
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp12_outputs"); OUT.mkdir(exist_ok=True)
N_Q = 500
KCANDS = [40, 100, 200]
PUB_FRACS = [1/8, 1/4, 1/2]


def full_pca(X, mu):
    """Exact full PCA basis (d x d) via eigendecomposition of the covariance
    on a 200k sample (fast and exact even for d=1024)."""
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    return V


def m_top(top, gt):
    h1, h10, rr = S.metrics_from_top(top, gt)
    return {"acc1": float(h1.mean()), "acc10": float(h10.mean()), "mrr": float(rr.mean())}


def shard_eval(Xrot, Qroute, Qscore, gt, d_pub, kc):
    U = np.ascontiguousarray(Xrot[:, :d_pub])
    Uq = np.ascontiguousarray(Qroute[:, :d_pub])
    short = S.topk_search(U, Uq, kc)               # (nq, kc) candidate ids
    # stage-2 full-dim rerank within the shortlist
    G = Xrot[short]                                # (nq, kc, d)
    sc = np.einsum("qkd,qd->qk", G, Qscore)
    order = np.argsort(-sc, axis=1)[:, :10]
    top = np.take_along_axis(short, order, axis=1)
    return m_top(top, gt), float((short == gt[:, None]).any(1).mean())  # +shortlist recall


def run(enc):
    print(f"\n=== {enc} ===", flush=True); t0 = time.time()
    X, Q, d = S.load(enc)
    X = np.asarray(X, dtype=np.float32)
    gt = S.qrels(len(X), N_Q)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    k = d // 2
    V = full_pca(X, mu)                             # full PCA basis (d x d)
    raw = m_top(S.topk_search(X, Q[:N_Q], 10), gt)
    Xrot = S.document_pca_coordinates(X, mu, V)
    Qrot = S.query_pca_coordinates(Q[:N_Q], V)
    Qrot_legacy = ((Q[:N_Q] - mu) @ V).astype(np.float32)
    del X
    svd = m_top(S.topk_search(np.ascontiguousarray(Xrot[:, :k]),
                              np.ascontiguousarray(Qrot[:, :k]), 10), gt)
    legacy_svd = m_top(S.topk_search(np.ascontiguousarray(Xrot[:, :k]),
                                     np.ascontiguousarray(Qrot_legacy[:, :k]), 10), gt)
    cfull = m_top(S.topk_search(Xrot, Qrot, 10), gt)
    res = {"encoder": enc, "d": d, "k": k, "raw": raw, "svd": svd,
           "legacy_centered_svd": legacy_svd, "corrected_full": cfull, "shard": {}}
    for pf in PUB_FRACS:
        d_pub = max(8, int(round(d * pf)))
        for kc in KCANDS:
            m, rec = shard_eval(Xrot, Qrot_legacy, Qrot, gt, d_pub, kc)
            res["shard"][f"dpub{d_pub}_kc{kc}"] = {**m, "shortlist_recall": rec,
                                                   "d_pub": d_pub, "kc": kc}
            print(f"  shard d_pub={d_pub:4d} Kc={kc:3d}: "
                  f"Acc@1={m['acc1']:.3f} Acc@10={m['acc10']:.3f} "
                  f"rec@Kc={rec:.3f}", flush=True)
    print(f"  raw Acc@1={raw['acc1']:.3f} | svd(k/2) Acc@1={svd['acc1']:.3f} "
          f"Acc@10={svd['acc10']:.3f} | corrected-full Acc@1={cfull['acc1']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    encs = ["e5-small", "e5-base", "mpnet", "e5-large", "bge-m3"]
    out = [run(e) for e in encs]
    json.dump({"n_queries": N_Q, "results": out},
              open(OUT / "exp12_shard_utility.json", "w"), indent=2)
    print("\nsaved", OUT / "exp12_shard_utility.json")


if __name__ == "__main__":
    main()
