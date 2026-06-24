"""Experiment 11: does the SVD-truncation denoiser effect hold on real IR?

The integral experiment uses a self-retrieval geometry probe on Russian
Wikipedia. A reviewer concern (and a requirement for an IR/measurement venue)
is that the "linear denoiser" effect (SVD truncation to k=d/2 not hurting, and
sometimes improving, ranking on retrieval-trained encoders) might be an
artefact of self-retrieval. Here we test it on standard BEIR datasets with
real graded qrels.

For each (encoder, dataset):
  - encode the corpus and queries (e5-style: 'query:'/'passage:' prefixes,
    mean pooling, L2 normalisation);
  - compute a data-dependent SVD basis V_k (k = d/2) on the corpus embeddings;
  - evaluate raw R^d vs span(V_k) with exact inner-product retrieval;
  - report nDCG@10, Recall@10, Acc@1, plus a paired bootstrap CI for the
    nDCG@10 / Recall@10 deltas and a McNemar exact test for Acc@1.

CPU-only (no GPU dependency). Datasets are small (scifact, nfcorpus).
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
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from scipy.stats import binomtest

torch.set_num_threads(os.cpu_count() or 4)
OUT = (RESULTS / "exp11_outputs")
OUT.mkdir(exist_ok=True)

BOOT = 10_000
BOOT_SEED = 2026
MAXLEN = int(os.environ.get("E11_MAXLEN", 128))
BATCH = int(os.environ.get("E11_BATCH", 64))

ENCODERS = os.environ.get("E11_ENCODERS", "intfloat/multilingual-e5-base").split(",")
DATASETS = os.environ.get("E11_DATASETS", "scifact,nfcorpus").split(",")


def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode(model, tok, texts, prefix):
    embs = []
    for i in range(0, len(texts), BATCH):
        batch = [f"{prefix}{t}" for t in texts[i:i + BATCH]]
        enc = tok(batch, padding=True, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")
        out = model(**enc)
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])
        e = F.normalize(e, p=2, dim=1)
        embs.append(e.cpu().numpy().astype(np.float32))
    return np.concatenate(embs, 0)


def load_beir(name):
    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")
    split = "test" if "test" in qrels_ds else list(qrels_ds.keys())[0]
    qrels_rows = qrels_ds[split]

    cid2text, cid2idx = {}, {}
    titles, texts, cids = [], [], []
    for r in corpus:
        cid = str(r["_id"])
        title = r.get("title") or ""
        text = r.get("text") or ""
        cid2idx[cid] = len(cids)
        cids.append(cid)
        titles.append(title)
        texts.append(text)
    doc_texts = [f"{t} {x}".strip() for t, x in zip(titles, texts)]

    qid2text = {str(r["_id"]): (r.get("text") or "") for r in queries}

    qrels = {}
    for r in qrels_rows:
        qid = str(r["query-id"]); cid = str(r["corpus-id"])
        score = int(r["score"])
        if score <= 0:
            continue
        qrels.setdefault(qid, {})[cid] = score
    # keep only queries that (a) have text and (b) have >=1 relevant doc in corpus
    qids = [q for q in qrels if q in qid2text and
            any(c in cid2idx for c in qrels[q])]
    q_texts = [qid2text[q] for q in qids]
    return doc_texts, cids, cid2idx, qids, q_texts, qrels


def dcg_at_k(rels, k=10):
    rels = np.asarray(rels[:k], dtype=np.float64)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(ranked_rels, ideal_rels, k=10):
    idcg = dcg_at_k(sorted(ideal_rels, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_rels, k) / idcg


def eval_space(D, Q, qids, cids, qrels, k=10):
    """Return per-query ndcg@10, recall@10, hit@1 arrays under IP search."""
    scores = Q @ D.T                       # (n_q, n_docs)
    topk = np.argsort(-scores, axis=1)[:, :max(k, 1)]
    ndcgs, recalls, hits = [], [], []
    for qi, qid in enumerate(qids):
        rel = qrels[qid]
        ranked_cids = [cids[j] for j in topk[qi]]
        ranked_rels = [rel.get(c, 0) for c in ranked_cids]
        ideal = list(rel.values())
        ndcgs.append(ndcg_at_k(ranked_rels, ideal, k))
        n_rel = len([1 for c in rel if c in set(cids)])
        inter = len([1 for c in ranked_cids[:k] if rel.get(c, 0) > 0])
        recalls.append(inter / n_rel if n_rel else 0.0)
        hits.append(1.0 if ranked_rels[0] > 0 else 0.0)
    return (np.array(ndcgs), np.array(recalls), np.array(hits))


def compute_svd(E, k):
    from sklearn.utils.extmath import randomized_svd
    mu = E.mean(axis=0, keepdims=True).astype(np.float32)
    _, _, Vt = randomized_svd(E - mu, n_components=k, random_state=42)
    return mu, Vt.T.astype(np.float32)


def boot_ci(raw, proj, seed=BOOT_SEED, B=BOOT):
    rng = np.random.default_rng(seed)
    diff = proj - raw
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    bt = diff[idx].mean(axis=1)
    return {"delta": float(diff.mean()),
            "ci_lo": float(np.percentile(bt, 2.5)),
            "ci_hi": float(np.percentile(bt, 97.5))}


def mcnemar(raw_hit, proj_hit):
    raw_hit = raw_hit.astype(bool); proj_hit = proj_hit.astype(bool)
    b = int(np.sum(raw_hit & ~proj_hit)); c = int(np.sum(~raw_hit & proj_hit))
    nd = b + c
    p = 1.0 if nd == 0 else binomtest(min(b, c), nd, 0.5).pvalue
    return {"b": b, "c": c, "p_value": float(p)}


def run(enc_name, ds_name):
    print(f"\n=== {enc_name} on {ds_name} ===", flush=True)
    t0 = time.time()
    doc_texts, cids, cid2idx, qids, q_texts, qrels = load_beir(ds_name)
    print(f"  corpus={len(doc_texts)} queries={len(qids)}", flush=True)
    tok = AutoTokenizer.from_pretrained(enc_name)
    model = AutoModel.from_pretrained(enc_name).eval()
    D = encode(model, tok, doc_texts, "passage: ")
    Q = encode(model, tok, q_texts, "query: ")
    d = D.shape[1]; k = d // 2
    print(f"  encoded d={d}, SVD k={k} ({time.time()-t0:.0f}s)", flush=True)
    mu, Vk = compute_svd(D, k)
    sample = D[:min(5000, len(D))]
    recon = ((sample - mu) @ Vk) @ Vk.T + mu
    sigma_rec = float(np.mean(np.linalg.norm(sample - recon, axis=1) /
                              (np.linalg.norm(sample, axis=1) + 1e-9)))
    Dp = ((D - mu) @ Vk).astype(np.float32)
    Qp = ((Q - mu) @ Vk).astype(np.float32)

    rn, rr, rh = eval_space(D, Q, qids, cids, qrels)
    pn, pr, ph = eval_space(Dp, Qp, qids, cids, qrels)
    res = {
        "encoder": enc_name, "dataset": ds_name, "d": d, "k": k,
        "n_queries": len(qids), "n_docs": len(doc_texts), "sigma_rec": sigma_rec,
        "raw":  {"ndcg10": float(rn.mean()), "recall10": float(rr.mean()), "acc1": float(rh.mean())},
        "svd":  {"ndcg10": float(pn.mean()), "recall10": float(pr.mean()), "acc1": float(ph.mean())},
        "boot_ndcg10": boot_ci(rn, pn),
        "boot_recall10": boot_ci(rr, pr),
        "mcnemar_acc1": mcnemar(rh, ph),
    }
    bn = res["boot_ndcg10"]
    print(f"  sigma_rec={sigma_rec:.3f}", flush=True)
    print(f"  raw  nDCG@10={res['raw']['ndcg10']:.4f} R@10={res['raw']['recall10']:.4f} Acc@1={res['raw']['acc1']:.4f}", flush=True)
    print(f"  svd  nDCG@10={res['svd']['ndcg10']:.4f} R@10={res['svd']['recall10']:.4f} Acc@1={res['svd']['acc1']:.4f}", flush=True)
    print(f"  d.nDCG@10={bn['delta']:+.4f} boot95%=[{bn['ci_lo']:+.4f},{bn['ci_hi']:+.4f}]"
          f"  McNemar(Acc@1) p={res['mcnemar_acc1']['p_value']:.3g}", flush=True)
    return res


def main():
    results = []
    for enc in ENCODERS:
        for ds in DATASETS:
            try:
                results.append(run(enc.strip(), ds.strip()))
                with open(OUT / "exp11_beir.json", "w", encoding="utf-8") as f:
                    json.dump({"encoders": ENCODERS, "datasets": DATASETS,
                               "results": results}, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  FAILED {enc} {ds}: {type(e).__name__}: {e}", flush=True)
    print(f"\nSaved to {OUT/'exp11_beir.json'}")


if __name__ == "__main__":
    main()
