"""Experiment 17b: SHARD utility on a large multilingual benchmark (MIRACL).

Reviewer point 5 (rev1): "at least one large multilingual benchmark with
graded relevance, if computationally feasible." MIRACL is the canonical
multilingual IR benchmark; its relevance judgments are binary (0/1) and the
standard metric is nDCG@10 (a graded ranking metric). We run the same
full-corpus retrieval protocol as exp17 (BEIR) on two typologically distinct
MIRACL languages with manageable corpora:
  - Swahili (sw): ~132k passages  (Bantu, Latin script)
  - Bengali (bn): ~297k passages  (Indo-Aryan, Bengali script)
both covered by multilingual-e5. For each (encoder, language) we compare, with
paired bootstrap CIs over queries:
  - raw           : full-dim inner product (plaintext ceiling)
  - svd (k=d/2)   : rerank in the truncated half (truncation-defense baseline)
  - shard(d_pub,Kc): public-prefix shortlist -> full-dim (keyed, exact) rerank

Robustness: document encoding is CHECKPOINTED in 16k-document shards on disk,
so a crash loses at most one shard and a restart resumes; results are saved
incrementally per (encoder, language); thread count is capped to avoid MKL
oversubscription. Data are read straight from HF raw files (script loaders
were dropped in datasets>=4).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import RESULTS
import os
_NT = str(max(1, min(8, (os.cpu_count() or 4))))      # cap threads: avoid MKL oversubscription
os.environ.setdefault("OMP_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)
import gzip, json, shutil, time
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import list_repo_files, hf_hub_download
import shard_lib as S

torch.set_num_threads(int(_NT))
OUT = (RESULTS / "exp17b_outputs"); OUT.mkdir(exist_ok=True)
EMB = OUT / "emb"; EMB.mkdir(exist_ok=True)
LOGF = OUT / "progress.log"
MAXLEN, BATCH, BOOT = 128, 64, 10_000
ENCS = os.environ.get("E17B_ENCODERS",
                      "intfloat/multilingual-e5-small,intfloat/multilingual-e5-base").split(",")
LANGS = os.environ.get("E17B_LANGS", "sw,bn").split(",")
MAXDOCS = int(os.environ.get("E17B_MAXDOCS", 0))      # 0 = full corpus (smoke-test cap otherwise)
KCANDS = [100, 200]
PUB_FRACS = [1 / 8, 1 / 4]


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def mean_pool(h, m):
    m = m.unsqueeze(-1).float(); return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.no_grad()
def _encode_range(model, tok, texts, prefix):
    out = []
    for i in range(0, len(texts), BATCH):
        b = [f"{prefix}{t}" for t in texts[i:i + BATCH]]
        e = tok(b, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        h = model(**e)
        v = F.normalize(mean_pool(h.last_hidden_state, e["attention_mask"]), 2, 1)
        out.append(v.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def encode_sharded(model, tok, texts, prefix, shard_dir, chunk=16384):
    """Resumable encode: each chunk is saved to disk; a restart skips completed
    shards, so a crash costs at most one chunk instead of the whole pass."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts); nsh = max(1, (n + chunk - 1) // chunk)
    for si in range(nsh):
        fp = shard_dir / f"part_{si:04d}.npy"
        if fp.exists():
            continue
        lo, hi = si * chunk, min((si + 1) * chunk, n)
        np.save(fp, _encode_range(model, tok, texts[lo:hi], prefix))
        log(f"      shard {si + 1}/{nsh} ({hi}/{n}) saved")
    return np.concatenate([np.load(shard_dir / f"part_{si:04d}.npy") for si in range(nsh)], 0)


def load_miracl(lang):
    files = list_repo_files("miracl/miracl-corpus", repo_type="dataset")
    shards = sorted(f for f in files
                    if f.startswith(f"miracl-corpus-v1.0-{lang}/") and f.endswith(".jsonl.gz"))
    cid2idx, doc_texts, cids = {}, [], []
    for sh in shards:
        p = hf_hub_download("miracl/miracl-corpus", sh, repo_type="dataset")
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for ln in fh:
                r = json.loads(ln)
                cid = str(r["docid"])
                if cid in cid2idx:
                    continue
                cid2idx[cid] = len(cids); cids.append(cid)
                t = (r.get("title") or "").strip(); x = (r.get("text") or "").strip()
                doc_texts.append((t + " " + x).strip() if t else x)
                if MAXDOCS and len(cids) >= MAXDOCS:
                    break
        if MAXDOCS and len(cids) >= MAXDOCS:
            break
    tp = hf_hub_download("miracl/miracl",
                         f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv",
                         repo_type="dataset")
    qid2text = {}
    with open(tp, "rt", encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 2:
                qid2text[parts[0]] = parts[1]
    qp = hf_hub_download("miracl/miracl",
                         f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv",
                         repo_type="dataset")
    qrels = {}
    with open(qp, "rt", encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels.setdefault(parts[0], {})[str(parts[2])] = int(parts[3])
    qids = [q for q in qrels if q in qid2text and any(c in cid2idx for c in qrels[q])]
    return doc_texts, cids, cid2idx, qids, [qid2text[q] for q in qids], qrels


def get_emb(enc, lang):
    tag = enc.split("/")[-1]
    cap = f"_cap{MAXDOCS}" if MAXDOCS else ""
    f = EMB / f"{tag}_{lang}{cap}.npz"
    meta = EMB / f"{tag}_{lang}{cap}_meta.json"
    if f.exists() and meta.exists():
        z = np.load(f); m = json.load(open(meta, encoding="utf-8"))
        return z["D"], z["Q"], m["cids"], m["qids"], m["qrels"]
    doc_texts, cids, cid2idx, qids, q_texts, qrels = load_miracl(lang)
    log(f"    corpus={len(cids)} queries={len(qids)} (judged)")
    tok = AutoTokenizer.from_pretrained(enc); model = AutoModel.from_pretrained(enc).eval()
    ddir = EMB / f"{tag}_{lang}{cap}_docsh"; qdir = EMB / f"{tag}_{lang}{cap}_qsh"
    D = encode_sharded(model, tok, doc_texts, "passage: ", ddir)
    Q = encode_sharded(model, tok, q_texts, "query: ", qdir, chunk=4096)
    np.savez(f, D=D, Q=Q)
    json.dump({"cids": cids, "qids": qids, "qrels": qrels}, open(meta, "w", encoding="utf-8"))
    shutil.rmtree(ddir, ignore_errors=True); shutil.rmtree(qdir, ignore_errors=True)
    return D, Q, cids, qids, qrels


def dcg(rels, k=10):
    rels = np.asarray(rels[:k], float)
    return float((rels / np.log2(np.arange(2, rels.size + 2))).sum()) if rels.size else 0.0


def ndcg(ranked_cids, qrel, k=10):
    g = [qrel.get(c, 0) for c in ranked_cids[:k]]
    idcg = dcg(sorted(qrel.values(), reverse=True), k)
    return dcg(g, k) / idcg if idcg else 0.0


def per_query_ndcg(order_ids, cids, qids, qrels, k=10):
    return np.array([ndcg([cids[j] for j in order_ids[qi]], qrels[q], k)
                     for qi, q in enumerate(qids)])


def boot(a, b, seed=2026):
    rng = np.random.default_rng(seed); diff = b - a; n = len(diff)
    bt = diff[rng.integers(0, n, (BOOT, n))].mean(1)
    return {"delta": float(diff.mean()), "lo": float(np.percentile(bt, 2.5)),
            "hi": float(np.percentile(bt, 97.5))}


def top_order(scores, k=10):
    return np.argsort(-scores, axis=1)[:, :k]


def run(enc, lang):
    log(f"\n=== {enc.split('/')[-1]} on MIRACL/{lang} ===" ); t0 = time.time()
    D, Q, cids, qids, qrels = get_emb(enc, lang)
    d = D.shape[1]; k = d // 2
    mu = D.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(D), size=min(len(D), 200000), replace=False)
    V, _ = S.pca_basis((D[idx] - mu).astype(np.float32))
    Drot = ((D - mu) @ V).astype(np.float32); Qrot = ((Q - mu) @ V).astype(np.float32)

    raw = per_query_ndcg(top_order(Q @ D.T), cids, qids, qrels)
    Pk = np.ascontiguousarray(Drot[:, :k]); Qk = np.ascontiguousarray(Qrot[:, :k])
    svd = per_query_ndcg(top_order(Qk @ Pk.T), cids, qids, qrels)
    res = {"encoder": enc, "lang": lang, "d": int(d), "n_corpus": len(cids), "n_q": len(qids),
           "raw_ndcg10": float(raw.mean()), "svd_ndcg10": float(svd.mean()),
           "svd_vs_raw": boot(raw, svd), "shard": {}}
    for pf in PUB_FRACS:
        dp = max(8, int(round(d * pf)))
        U = np.ascontiguousarray(Drot[:, :dp]); Uq = np.ascontiguousarray(Qrot[:, :dp])
        short = np.argsort(-(Uq @ U.T), axis=1)
        for kc in KCANDS:
            order = np.empty((len(Q), 10), np.int64)
            for qi in range(len(Q)):
                cand = short[qi, :kc]
                sc = D[cand] @ Q[qi]
                order[qi] = cand[np.argsort(-sc)[:10]]
            sh = per_query_ndcg(order, cids, qids, qrels)
            res["shard"][f"dpub{dp}_kc{kc}"] = {"ndcg10": float(sh.mean()),
                "vs_raw": boot(raw, sh), "d_pub": dp, "kc": kc}
            log(f"  shard d_pub={dp} Kc={kc}: nDCG@10={sh.mean():.4f} "
                f"(raw {raw.mean():.4f}, svd {svd.mean():.4f})")
    log(f"  raw={raw.mean():.4f} svd={svd.mean():.4f} "
        f"(svd-raw {res['svd_vs_raw']['delta']:+.4f}) corpus={len(cids)} "
        f"n_q={len(qids)} ({time.time()-t0:.0f}s)")
    return res


def main():
    out_path = OUT / ("exp17b_miracl_smoke.json" if MAXDOCS else "exp17b_miracl.json")
    out, done = [], set()
    if out_path.exists():
        out = json.load(open(out_path, encoding="utf-8")).get("results", [])
        done = {(r["encoder"], r["lang"]) for r in out}
    log(f"START enc={ENCS} langs={LANGS} threads={_NT} cap={MAXDOCS} (done={len(done)})")
    for lang in LANGS:
        for e in ENCS:
            e = e.strip(); lang = lang.strip()
            if (e, lang) in done:
                log(f"skip {e} {lang} (cached result)"); continue
            try:
                out.append(run(e, lang))
                json.dump({"results": out}, open(out_path, "w", encoding="utf-8"), indent=2)
                log(f"SAVED cell {e} {lang} -> {out_path.name}")
            except Exception as ex:
                import traceback; traceback.print_exc()
                log(f"  FAIL {e} {lang}: {type(ex).__name__}: {ex}")
    log(f"DONE saved {out_path}")


if __name__ == "__main__":
    main()
