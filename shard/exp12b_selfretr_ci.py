"""Experiment 12b: paired significance for SHARD self-retrieval Acc@1.

Re-runs the self-retrieval utility probe but keeps the per-query top-1 hit
outcomes, so the SHARD-vs-baseline and SHARD-vs-raw comparisons carry a
paired bootstrap CI on the Acc@1 delta and an exact McNemar test, rather than
point estimates alone (reviewer request). SHARD uses the canonical operating
point d_pub=d/4, K_cands=200.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import RESULTS
import json, os
import numpy as np
import scipy.stats as st
import shard_lib as S

OUT = (RESULTS / "exp12_outputs"); OUT.mkdir(exist_ok=True)
ENCODERS = os.environ.get("E12B_ENCS", "e5-small,e5-base").split(",")
N_Q = 500
KC = 200
NBOOT = 10_000
BOOT_SEED = 2026


def h1_of(top, gt):
    return (top[:, 0] == gt)


def shard_top1(Xrot, Qrot, d_pub, kc):
    U = np.ascontiguousarray(Xrot[:, :d_pub]); Uq = np.ascontiguousarray(Qrot[:, :d_pub])
    short = S.topk_search(U, Uq, kc)
    G = Xrot[short]
    sc = np.einsum("qkd,qd->qk", G, Qrot)
    return short[np.arange(len(Qrot)), sc.argmax(1)]          # per-query reranked top-1 doc id


def mcnemar(a, b):
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b)); n = n10 + n01
    p = float(st.binomtest(min(n10, n01), n, 0.5).pvalue) if n > 0 else 1.0
    return {"n_shard_only": n10, "n_other_only": n01, "p_mcnemar_exact": p}


def boot_delta(a, b):
    rng = np.random.default_rng(BOOT_SEED)
    d = a.astype(np.float64) - b.astype(np.float64); n = len(d)
    bs = d[rng.integers(0, n, (NBOOT, n))].mean(1)
    return {"delta": float(d.mean()), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5))}


def run(enc):
    print(f"=== exp12b self-retrieval CI: {enc} ===", flush=True)
    X, Q, d = S.load(enc)
    X = np.asarray(X, dtype=np.float32)
    gt = S.qrels(len(X), N_Q)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    k = d // 2; d_pub = d // 4
    rng = np.random.RandomState(0); idx = rng.choice(len(X), min(200_000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    h1_raw = h1_of(S.topk_search(X, Q[:N_Q], 10), gt)
    Xrot = ((X - mu) @ V).astype(np.float32); Qrot = ((Q[:N_Q] - mu) @ V).astype(np.float32); del X
    h1_svd = h1_of(S.topk_search(np.ascontiguousarray(Xrot[:, :k]),
                                 np.ascontiguousarray(Qrot[:, :k]), 10), gt)
    top1_shard = shard_top1(Xrot, Qrot, d_pub, KC)
    h1_shard = (top1_shard == gt)
    res = {"encoder": enc, "d": int(d), "d_pub": d_pub, "kc": KC, "n_q": N_Q,
           "acc1": {"raw": float(h1_raw.mean()), "svd": float(h1_svd.mean()),
                    "shard": float(h1_shard.mean())},
           "shard_vs_svd": {**boot_delta(h1_shard, h1_svd), **mcnemar(h1_shard, h1_svd)},
           "shard_vs_raw": {**boot_delta(h1_shard, h1_raw), **mcnemar(h1_shard, h1_raw)}}
    print(f"  Acc@1 raw={res['acc1']['raw']:.3f} svd={res['acc1']['svd']:.3f} "
          f"shard={res['acc1']['shard']:.3f}", flush=True)
    sv = res["shard_vs_svd"]; rv = res["shard_vs_raw"]
    print(f"  shard-svd: +{sv['delta']:.3f} [{sv['lo']:.3f},{sv['hi']:.3f}] "
          f"McNemar p={sv['p_mcnemar_exact']:.2e}", flush=True)
    print(f"  shard-raw: {rv['delta']:+.3f} [{rv['lo']:.3f},{rv['hi']:.3f}] "
          f"McNemar p={rv['p_mcnemar_exact']:.2e}", flush=True)
    json.dump(res, open(OUT / f"exp12b_selfretr_ci_{enc}.json", "w"), indent=2)
    print("saved", OUT / f"exp12b_selfretr_ci_{enc}.json", flush=True)
    return res


def main():
    out = [run(e) for e in ENCODERS]
    json.dump({"results": out}, open(OUT / "exp12b_selfretr_ci.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
