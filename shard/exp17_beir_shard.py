"""Experiment 17: SHARD utility on real IR (BEIR), the utility centerpiece.

exp11 showed the SVD-k/2 baseline significantly *reduces* nDCG@10 on BEIR
(it reranks in the truncated half). SHARD reranks in the FULL space (the
cell-keyed residual is included via orthogonal keys), so it should recover
raw nDCG while exposing only the short public prefix for stage-1.

For each (encoder, dataset) we compare, with paired bootstrap CIs:
  - raw           : full-dim inner product (ceiling)
  - svd (k=d/2)   : rerank in the truncated half (exp11 baseline)
  - shard(d_pub,Kc): prefix shortlist -> full-dim rerank
Encodes once and caches embeddings to npz.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from scipy.stats import binomtest

torch.set_num_threads(os.cpu_count() or 4)
OUT = (RESULTS / "exp17_outputs"); OUT.mkdir(exist_ok=True)
EMB = OUT / "emb"; EMB.mkdir(exist_ok=True)
MAXLEN, BATCH, BOOT = 128, 64, 10_000
ENCS = os.environ.get("E17_ENCODERS", "intfloat/multilingual-e5-base,intfloat/multilingual-e5-small").split(",")
DSETS = os.environ.get("E17_DATASETS", "scifact,nfcorpus").split(",")
KCANDS = [100, 200]
PUB_FRACS = [1/8, 1/4]


def mean_pool(h, m):
    m = m.unsqueeze(-1).float(); return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode(model, tok, texts, prefix):
    out = []
    for i in range(0, len(texts), BATCH):
        b = [f"{prefix}{t}" for t in texts[i:i+BATCH]]
        e = tok(b, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        h = model(**e)
        v = F.normalize(mean_pool(h.last_hidden_state, e["attention_mask"]), 2, 1)
        out.append(v.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def load_beir(name):
    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qr = load_dataset(f"BeIR/{name}-qrels"); split = "test" if "test" in qr else list(qr.keys())[0]
    cid2idx, titles, texts, cids = {}, [], [], []
    for r in corpus:
        cid2idx[str(r["_id"])] = len(cids); cids.append(str(r["_id"]))
        titles.append(r.get("title") or ""); texts.append(r.get("text") or "")
    doc_texts = [f"{t} {x}".strip() for t, x in zip(titles, texts)]
    qid2text = {str(r["_id"]): (r.get("text") or "") for r in queries}
    qrels = {}
    for r in qr[split]:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qids = [q for q in qrels if q in qid2text and any(c in cid2idx for c in qrels[q])]
    return doc_texts, cids, cid2idx, qids, [qid2text[q] for q in qids], qrels


def get_emb(enc, ds):
    tag = enc.split("/")[-1]
    f = EMB / f"{tag}_{ds}.npz"
    meta = EMB / f"{tag}_{ds}_meta.json"
    if f.exists():
        z = np.load(f); m = json.load(open(meta))
        return z["D"], z["Q"], m["cids"], m["qids"], {k: v for k, v in m["qrels"].items()}
    doc_texts, cids, cid2idx, qids, q_texts, qrels = load_beir(ds)
    tok = AutoTokenizer.from_pretrained(enc); model = AutoModel.from_pretrained(enc).eval()
    D = encode(model, tok, doc_texts, "passage: "); Q = encode(model, tok, q_texts, "query: ")
    np.savez(f, D=D, Q=Q)
    json.dump({"cids": cids, "qids": qids, "qrels": qrels}, open(meta, "w"))
    return D, Q, cids, qids, qrels


def dcg(rels, k=10):
    rels = np.asarray(rels[:k], float)
    return float((rels / np.log2(np.arange(2, rels.size + 2))).sum()) if rels.size else 0.0


def ndcg(ranked_cids, qrel, k=10):
    g = [qrel.get(c, 0) for c in ranked_cids[:k]]
    idcg = dcg(sorted(qrel.values(), reverse=True), k)
    return dcg(g, k) / idcg if idcg else 0.0


def per_query_ndcg(order_ids, cids, qids, qrels, k=10):
    out = []
    for qi, q in enumerate(qids):
        out.append(ndcg([cids[j] for j in order_ids[qi]], qrels[q], k))
    return np.array(out)


def boot(a, b, seed=2026):
    rng = np.random.default_rng(seed); diff = b - a; n = len(diff)
    bt = diff[rng.integers(0, n, (BOOT, n))].mean(1)
    return {"delta": float(diff.mean()), "lo": float(np.percentile(bt, 2.5)), "hi": float(np.percentile(bt, 97.5))}


def rank_all(D, Q):
    return np.argsort(-(Q @ D.T), axis=1)


def shard_rank(D, Q, mu, V, d_pub, kc):
    Drot = (D - mu) @ V; Qrot = (Q - mu) @ V
    U = np.ascontiguousarray(Drot[:, :d_pub]); Uq = np.ascontiguousarray(Qrot[:, :d_pub])
    short = np.argsort(-(Uq @ U.T), axis=1)[:, :kc]            # stage-1 prefix shortlist
    order = np.empty((len(Q), 10), np.int64)
    for qi in range(len(Q)):
        cand = short[qi]
        sc = D[cand] @ Q[qi]                                    # stage-2 full-dim rerank (raw)
        order[qi] = cand[np.argsort(-sc)[:10]]
    return order


def run(enc, ds):
    print(f"\n=== {enc.split('/')[-1]} on {ds} ===", flush=True); t0 = time.time()
    D, Q, cids, qids, qrels = get_emb(enc, ds)
    d = D.shape[1]; k = d // 2
    mu = D.mean(0, keepdims=True).astype(np.float32)
    import shard_lib as S
    rng = np.random.RandomState(0)
    idx = rng.choice(len(D), size=min(len(D), 200000), replace=False)
    V, _ = S.pca_basis((D[idx] - mu).astype(np.float32))
    raw = per_query_ndcg(rank_all(D, Q), cids, qids, qrels)
    Drot = (D - mu) @ V; Qrot = (Q - mu) @ V
    Pk = np.ascontiguousarray(Drot[:, :k]); Qk = np.ascontiguousarray(Qrot[:, :k])
    svd = per_query_ndcg(rank_all(Pk, Qk), cids, qids, qrels)
    res = {"encoder": enc, "dataset": ds, "d": d, "n_q": len(qids),
           "raw_ndcg10": float(raw.mean()), "svd_ndcg10": float(svd.mean()),
           "svd_vs_raw": boot(raw, svd), "shard": {}}
    for pf in PUB_FRACS:
        dp = max(8, int(round(d * pf)))
        for kc in KCANDS:
            sh = per_query_ndcg(shard_rank(D, Q, mu, V, dp, kc), cids, qids, qrels)
            res["shard"][f"dpub{dp}_kc{kc}"] = {"ndcg10": float(sh.mean()),
                "vs_raw": boot(raw, sh), "d_pub": dp, "kc": kc}
            print(f"  shard d_pub={dp} Kc={kc}: nDCG@10={sh.mean():.4f} "
                  f"(raw {raw.mean():.4f}, svd {svd.mean():.4f})", flush=True)
    print(f"  raw={raw.mean():.4f} svd={svd.mean():.4f} "
          f"(svd-raw {res['svd_vs_raw']['delta']:+.4f}) ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    out = []
    for e in ENCS:
        for ds in DSETS:
            try:
                out.append(run(e.strip(), ds.strip()))
                json.dump({"results": out}, open(OUT / "exp17_beir_shard.json", "w"), indent=2)
            except Exception as ex:
                print(f"  FAIL {e} {ds}: {type(ex).__name__}: {ex}", flush=True)
    print("\nsaved", OUT / "exp17_beir_shard.json")


if __name__ == "__main__":
    main()
