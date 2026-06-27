# -*- coding: utf-8 -*-
"""Stage-3 (UPD1 exp 2): evasion robustness of semantic-DLP detection under
obfuscation of the leaking text. Positives = 250 secret leaks under
V0 verbatim / V1 word-shuffle / V2 40% word-dropout / V3 second-sentence;
negatives = benign non-member candidates. Protected-space (uncentered V_k,
k=d/2) detection AUC per variant. Resumable per encoder (CPU)."""
import torch, numpy as np, pickle, re, json, time
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
torch.set_num_threads(8)

CACHE = Path("D:/PHD/research/RES1/notebooks/_corpus_cache")
OUT   = Path("C:/Users/zerg/AppData/Local/Temp/claude/D--PHD/3a3b6cb6-9a46-4118-968b-8f5396c7b1d3/scratchpad/paraphrase_results.json")
ENC = {
 "e5-small": dict(hf="intfloat/multilingual-e5-small", docs="E_docs_e5small_1000000.npy", q="E_queries_e5small_self_q500.npy", dim=384, prefix="query: ", pool="mean"),
 "e5-base":  dict(hf="intfloat/multilingual-e5-base",  docs="E_docs_e5base_1000000.npy",  q="E_queries_e5base_self_q500.npy",  dim=768, prefix="query: ", pool="mean"),
 "mpnet":    dict(hf="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", docs="E_docs_mpnet_1000000.npy", q="E_queries_mpnet_self_q500.npy", dim=768, prefix="", pool="mean"),
 "e5-large": dict(hf="intfloat/multilingual-e5-large", docs="E_docs_e5large_1000000.npy", q="E_queries_e5large_self_q500.npy", dim=1024, prefix="query: ", pool="mean"),
 "bge-m3":   dict(hf="BAAI/bge-m3", docs="E_docs_bgem3_1000000.npy", q="E_queries_bgem3_self_q500.npy", dim=1024, prefix="", pool="cls"),
}
N_DOCS, N_Q, N_POS, N_BG, N_FIT, SEED = 1_000_000, 500, 250, 50_000, 50_000, 42
SENT = re.compile(r"(?<=[.!?])\s+")
def first_sentence(t):
    p = SENT.split(t, maxsplit=1); s = p[0].strip() if p else t.strip()
    return s[:200] if len(s) >= 30 else t[:200]
def second_sentence(t):
    for s in SENT.split(t)[1:]:
        s = s.strip()
        if len(s) >= 30: return s[:200]
    return None
def shuffle_words(s, seed):
    w = s.split(); np.random.default_rng(seed).shuffle(w); return " ".join(w)
def dropout_words(s, seed, p=0.4):
    w = s.split()
    if len(w) <= 3: return s
    rng = np.random.default_rng(seed); keep = [x for x in w if rng.random() > p]
    return " ".join(keep) if len(keep) >= 3 else " ".join(w[:3])
def l2n(X):
    X = np.asarray(X, np.float32); return X/np.clip(np.linalg.norm(X,axis=1,keepdims=True),1e-12,None)
def auc(y, s):
    y = np.asarray(y).astype(bool); o = np.argsort(s,kind="mergesort"); r = np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1,n0 = int(y.sum()), int((~y).sum()); return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))

results = json.loads(OUT.read_text()) if OUT.exists() else {}
todo = [n for n in ENC if n not in results]
if not todo:
    print("all done:", list(results)); raise SystemExit
print("loading corpus text ...", flush=True)
docs = pickle.load(open(CACHE/"corpus_wiki_ru_1000000.pkl","rb"))["docs"]
sample_idx = np.random.default_rng(SEED).choice(N_DOCS, size=N_Q, replace=False)
secret = sample_idx[:N_POS]
fs_sec  = [first_sentence(docs[int(i)]) for i in secret]
V1_txt = [shuffle_words(s, 1000+j) for j,s in enumerate(fs_sec)]
V2_txt = [dropout_words(s, 2000+j) for j,s in enumerate(fs_sec)]
V3_txt = [second_sentence(docs[int(i)]) for i in secret]
v3_ok  = [j for j,s in enumerate(V3_txt) if s]
print(f"V3 has second sentence: {len(v3_ok)}/{N_POS}", flush=True)

for name in todo:
    c = ENC[name]; t0 = time.time(); k = c["dim"]//2
    X = np.load(CACHE/c["docs"], mmap_mode="r"); Eq = l2n(np.load(CACHE/c["q"]))
    # uncentered V_k (paper convention) + reference corpus R (250 secret + 50k bg)
    F = l2n(np.asarray(X[np.random.default_rng(7).choice(N_DOCS, N_FIT, replace=False)]))
    evals, evecs = np.linalg.eigh(((F.T@F)/len(F)).astype(np.float64))
    Vk = evecs[:, np.argsort(evals)[::-1][:k]].astype(np.float32)
    allsrc = set(int(i) for i in sample_idx)
    cand = np.random.default_rng(123).choice(N_DOCS, N_BG+4*N_Q, replace=False)
    bg = np.array([x for x in cand if int(x) not in allsrc][:N_BG], dtype=np.int64)
    R = l2n(np.asarray(X[np.concatenate([secret, bg])])); Rp = l2n(R@Vk)
    neg = l2n(Eq[N_POS:]); m_neg = (l2n(neg@Vk)@Rp.T).max(1)         # benign non-member candidates

    tok = AutoTokenizer.from_pretrained(c["hf"]); mdl = AutoModel.from_pretrained(c["hf"]).eval()
    @torch.no_grad()
    def enc(texts):
        out=[]
        for i in range(0, len(texts), 32):
            x = tok([c["prefix"]+t for t in texts[i:i+32]], return_tensors="pt", truncation=True, padding=True, max_length=256)
            o = mdl(**x)
            if c["pool"]=="mean":
                mk = x["attention_mask"].unsqueeze(-1).float(); v=(o.last_hidden_state*mk).sum(1)/mk.sum(1)
            else: v = o.last_hidden_state[:,0]
            out.append(torch.nn.functional.normalize(v,dim=-1).numpy())
        return np.concatenate(out).astype(np.float32)

    sanity = float((l2n(enc(fs_sec[:20]))*Eq[:20]).sum(1).mean())   # re-embed vs cached V0
    def aucv(emb, idx=None):
        e = l2n(emb); mp = (l2n(e@Vk)@Rp.T).max(1)
        if idx is None: pos = mp
        else: pos = mp[idx]
        return auc([1]*len(pos)+[0]*len(m_neg), np.concatenate([pos, m_neg]))
    A_V0 = auc([1]*N_POS+[0]*len(m_neg), np.concatenate([(l2n(Eq[:N_POS]@Vk)@Rp.T).max(1), m_neg]))
    A_V1 = aucv(enc(V1_txt)); A_V2 = aucv(enc(V2_txt))
    e3 = enc([V3_txt[j] for j in v3_ok]); A_V3 = aucv(e3, np.arange(len(v3_ok)))

    results[name] = dict(dim=c["dim"], sanity_cos=round(sanity,4),
        auc_V0_verbatim=round(A_V0,4), auc_V1_shuffle=round(A_V1,4),
        auc_V2_dropout=round(A_V2,4), auc_V3_secondsent=round(A_V3,4), n_V3=len(v3_ok),
        secs=round(time.time()-t0,1))
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[done] {name:9s} {time.time()-t0:5.0f}s sanity={sanity:.3f} | "
          f"AUC V0/V1/V2/V3 = {A_V0:.3f}/{A_V1:.3f}/{A_V2:.3f}/{A_V3:.3f}", flush=True)
    del mdl, tok

print("\n=== DONE ===", json.dumps(results, indent=2))
