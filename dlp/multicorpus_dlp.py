# -*- coding: utf-8 -*-
"""Reviewer pt.2: validate Corollary 1 + DLP detection across MORE datasets
(English BEIR: scifact/nfcorpus/arguana; multilingual MIRACL: bn/sw) with
bootstrap confidence intervals. Canary = query vs its relevant passage (qrels):
positives are queries whose relevant doc is in the protected set R, negatives
queries whose relevant docs are removed from R. Uncentered V_k (k=d/2), the
paper's convention. Pure numpy on cached embeddings."""
import numpy as np, json, time
from pathlib import Path
EMB = Path("D:/PHD/research/RES1/results")
OUT = Path("C:/Users/zerg/AppData/Local/Temp/claude/D--PHD/3a3b6cb6-9a46-4118-968b-8f5396c7b1d3/scratchpad/multicorpus_results.json")
DATA = [  # (label, lang/domain, npz, meta)
 ("scifact","en/science", "exp17_outputs/emb/{e}_scifact.npz","exp17_outputs/emb/{e}_scifact_meta.json"),
 ("nfcorpus","en/medical","exp17_outputs/emb/{e}_nfcorpus.npz","exp17_outputs/emb/{e}_nfcorpus_meta.json"),
 ("arguana","en/argument","exp17_outputs/emb/{e}_arguana.npz","exp17_outputs/emb/{e}_arguana_meta.json"),
 ("miracl-bn","bn/wiki","exp17b_outputs/emb/{e}_bn.npz","exp17b_outputs/emb/{e}_bn_meta.json"),
 ("miracl-sw","sw/wiki","exp17b_outputs/emb/{e}_sw.npz","exp17b_outputs/emb/{e}_sw_meta.json"),
]
ENC = {"multilingual-e5-small":384, "multilingual-e5-base":768}
N_BG_CAP = 60_000

def l2n(X):
    X=np.asarray(X,np.float32); return X/np.clip(np.linalg.norm(X,axis=1,keepdims=True),1e-12,None)
def auc(y,s):
    y=np.asarray(y).astype(bool); o=np.argsort(s,kind="mergesort"); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1,n0=int(y.sum()),int((~y).sum()); return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else float("nan")
def auc_ci(y,s,B=1000,seed=0):
    y=np.asarray(y); s=np.asarray(s); n=len(y); rng=np.random.default_rng(seed); a=[]
    for _ in range(B):
        i=rng.integers(0,n,n); yy=y[i]
        if 0<yy.sum()<len(yy): a.append(auc(yy,s[i]))
    return (round(float(np.percentile(a,2.5)),3), round(float(np.percentile(a,97.5)),3)) if a else (float("nan"),)*2

def parse_qrels(meta):
    cid2i={c:i for i,c in enumerate(meta["cids"])}; qrels=meta["qrels"]; qids=meta["qids"]
    qmap={}
    for qi,qid in enumerate(qids):
        rels=qrels.get(str(qid), qrels.get(qid, {}))
        cids=[c for c,r in rels.items() if float(r)>0] if isinstance(rels,dict) else list(rels)
        idx=[cid2i[c] for c in cids if c in cid2i]
        if idx: qmap[qi]=idx
    return qmap

results={}
for ename,dim in ENC.items():
    k=dim//2
    for label,tag,npzt,metat in DATA:
        npz=EMB/npzt.format(e=ename); meta=EMB/metat.format(e=ename)
        if not npz.exists(): continue
        t0=time.time(); key=f"{label}|{ename}"
        z=np.load(npz); D=l2n(z["D"]); Q=l2n(z["Q"]); m=json.load(open(meta))
        qmap=parse_qrels(m)
        qs=sorted(qmap);
        if len(qs)<20: continue
        rng=np.random.default_rng(0); rng.shuffle(qs)
        half=len(qs)//2; pos_q=qs[:half]; neg_q=qs[half:]
        neg_rel=set(i for q in neg_q for i in qmap[q])
        keep=np.array([i for i in range(len(D)) if i not in neg_rel])
        if len(keep)>N_BG_CAP:   # cap R but keep all positives' relevant docs
            posrel=set(i for q in pos_q for i in qmap[q])
            forced=np.array(sorted(posrel));
            others=np.array([i for i in keep if i not in posrel])
            others=np.random.default_rng(1).choice(others,N_BG_CAP-len(forced),replace=False)
            keep=np.concatenate([forced,others])
        R=D[keep]
        # uncentered V_k on a doc sample (paper convention) + sigma_rec(Frobenius)
        fit=D if len(D)<=50000 else D[np.random.default_rng(7).choice(len(D),50000,replace=False)]
        ev,evec=np.linalg.eigh(((fit.T@fit)/len(fit)).astype(np.float64))
        order=np.argsort(ev)[::-1]; Vk=evec[:,order[:k]].astype(np.float32)
        sigma=float(np.sqrt(max(0.0, ev[order][k:].sum()/ev[order].sum())))
        Rp=l2n(R@Vk)
        def sc(qset):
            qe=l2n(Q[np.array(qset)]); return (qe@R.T).max(1), (l2n(qe@Vk)@Rp.T).max(1)
        pp,pr=sc(pos_q); npl,npr=sc(neg_q)
        y=[1]*len(pos_q)+[0]*len(neg_q)
        A_pl=auc(y,np.concatenate([pp,npl])); A_pr=auc(y,np.concatenate([pr,npr]))
        ci_pl=auc_ci(y,np.concatenate([pp,npl])); ci_pr=auc_ci(y,np.concatenate([pr,npr]))
        # Corollary on random (query,R-doc) pairs
        sq=np.sqrt(np.clip(1-((l2n(Q)@Vk)**2).sum(1),0,None)); sr=np.sqrt(np.clip(1-((R@Vk)**2).sum(1),0,None))
        rg=np.random.default_rng(9); npair=min(200_000, len(Q)*len(R))
        qi=rg.integers(0,len(Q),npair); rj=rg.integers(0,len(R),npair)
        Qp=l2n(Q@Vk)
        s_p=(l2n(Q)[qi]*R[rj]).sum(1); sh=(Qp[qi]*Rp[rj]).sum(1)
        sc_,sr_=sq[qi],sr[rj]
        eps=(1-np.sqrt(np.clip((1-sc_**2)*(1-sr_**2),0,None)))+sc_*sr_
        dev=np.abs(sh-s_p); viol=int((dev>eps+1e-5).sum())
        results[key]=dict(dataset=label,lang=tag,encoder=ename,dim=dim,k=k,
            n_docs=int(len(D)),n_R=int(len(R)),n_pos=len(pos_q),n_neg=len(neg_q),
            sigma_rec=round(sigma,3),auc_plain=round(A_pl,3),auc_plain_ci=ci_pl,
            auc_protected=round(A_pr,3),auc_protected_ci=ci_pr,auc_gap=round(A_pr-A_pl,3),
            cor_max_dev=round(float(dev.max()),4),cor_violations=viol,n_pairs=int(npair),secs=round(time.time()-t0,1))
        OUT.write_text(json.dumps(results,indent=2))
        r=results[key]
        print(f"[{key:28s}] sigma={sigma:.3f} | AUC pl/pr={A_pl:.3f}{ci_pl}/{A_pr:.3f}{ci_pr} "
              f"gap={A_pr-A_pl:+.3f} | |dhat-s|max={dev.max():.4f} viol={viol} | n_pos/neg={len(pos_q)}/{len(neg_q)}")

tot_pairs=sum(r["n_pairs"] for r in results.values()); tot_viol=sum(r["cor_violations"] for r in results.values())
print(f"\n=== {len(results)} (corpus x encoder) runs | Corollary violations: {tot_viol} / {tot_pairs:,} pairs ===")
