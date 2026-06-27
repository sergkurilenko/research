# -*- coding: utf-8 -*-
"""Semantic-DLP detection: protected (SVD-truncated, span V_k) vs plaintext space.
Uncentered V_k = top-k right singular vectors of E (matches the paper's reported
sigma_rec(E;V_k) = ||E-pi_k(E)||_F/||E||_F exactly). Validates Corollary cor:detect
|s_hat - s| <= eps(sig_c,sig_r) <= 2 sigma_max^2 on real cached embeddings."""
import numpy as np, json, time
from pathlib import Path

CACHE = Path("D:/PHD/research/RES1/notebooks/_corpus_cache")
OUT   = Path("C:/Users/zerg/AppData/Local/Temp/claude/D--PHD/3a3b6cb6-9a46-4118-968b-8f5396c7b1d3/scratchpad/dlp_results.json")
ENC = {  # name: (docs_file, queries_file, dim, paper_sigma_rec @ k=d/2)
  "e5-small": ("E_docs_e5small_1000000.npy", "E_queries_e5small_self_q500.npy", 384, 0.239),
  "e5-base":  ("E_docs_e5base_1000000.npy",  "E_queries_e5base_self_q500.npy",  768, 0.129),
  "mpnet":    ("E_docs_mpnet_1000000.npy",   "E_queries_mpnet_self_q500.npy",   768, 0.107),
  "e5-large": ("E_docs_e5large_1000000.npy", "E_queries_e5large_self_q500.npy", 1024, 0.101),
  "bge-m3":   ("E_docs_bgem3_1000000.npy",   "E_queries_bgem3_self_q500.npy",   1024, 0.128),
}
N_DOCS, N_Q, N_POS, N_BG, N_FIT, SEED = 1_000_000, 500, 250, 50_000, 50_000, 42

def l2n(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

def auc(y, s):
    y = np.asarray(y).astype(bool)
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1*(n1+1)/2) / (n1*n0))

def eer_threshold(y, s):
    y = np.asarray(y).astype(bool); best, bt = 1e9, 0.0
    for t in np.unique(s):
        fpr = ((s>=t)&~y).sum()/max(1,(~y).sum()); fnr = ((s<t)&y).sum()/max(1,y.sum())
        if abs(fpr-fnr) < best: best, bt = abs(fpr-fnr), float(t)
    return bt

results = {}
for name, (df, qf, dim, paper_sig) in ENC.items():
    t0 = time.time(); k = dim // 2
    X = np.load(CACHE/df, mmap_mode="r")
    Q = l2n(np.load(CACHE/qf))
    sample_idx = np.random.default_rng(SEED).choice(N_DOCS, size=N_Q, replace=False)
    diag = float((Q*l2n(np.asarray(X[sample_idx]))).sum(1).mean())   # self-cos: seed/pairing sanity

    # --- uncentered V_k (top-k of E^T E) == paper's sigma_rec convention ---
    F = l2n(np.asarray(X[np.random.default_rng(7).choice(N_DOCS, size=N_FIT, replace=False)]))
    evals, evecs = np.linalg.eigh(((F.T@F)/len(F)).astype(np.float64))
    order = np.argsort(evals)[::-1]; ev = evals[order]
    Vk = evecs[:, order[:k]].astype(np.float32)
    sigma_frob = float(np.sqrt(max(0.0, ev[k:].sum()/ev.sum())))     # encoder-level sigma_rec(E;V_k)
    sig_vec = np.sqrt(np.clip(1-((F@Vk)**2).sum(1), 0, None))        # per-vector sigma_rec
    sig_mean, sig_p95, sig_max = float(sig_vec.mean()), float(np.quantile(sig_vec,0.95)), float(sig_vec.max())

    # --- reference corpus R = 250 secret sources + 50k background (no source leakage) ---
    allsrc = set(int(i) for i in sample_idx)
    cand = np.random.default_rng(123).choice(N_DOCS, size=N_BG+4*N_Q, replace=False)
    bg = np.array([c for c in cand if int(c) not in allsrc][:N_BG], dtype=np.int64)
    R_idx = np.concatenate([sample_idx[:N_POS], bg]); R = l2n(np.asarray(X[R_idx]))
    labels = np.array([1]*N_POS + [0]*(N_Q-N_POS))

    # --- plaintext vs protected (cosine of projections onto span V_k) ---
    Qp, Rp = l2n(Q@Vk), l2n(R@Vk)
    m_raw = (Q@R.T).max(1); m_pro = (Qp@Rp.T).max(1)
    A_raw, A_pro = auc(labels, m_raw), auc(labels, m_pro)

    # --- Corollary validation over 300k random (query, R-doc) pairs ---
    sig_q = np.sqrt(np.clip(1-((Q@Vk)**2).sum(1), 0, None))
    sig_r = np.sqrt(np.clip(1-((R@Vk)**2).sum(1), 0, None))
    rg = np.random.default_rng(9); npair = 300_000
    qi = rg.integers(0, N_Q, npair); rj = rg.integers(0, len(R), npair)
    s_pair = (Q[qi]*R[rj]).sum(1); shat_pair = (Qp[qi]*Rp[rj]).sum(1)
    sc, sr = sig_q[qi], sig_r[rj]
    eps_pair = (1 - np.sqrt(np.clip((1-sc**2)*(1-sr**2),0,None))) + sc*sr
    dev = np.abs(shat_pair - s_pair)
    viol_perpair = int((dev > eps_pair + 1e-5).sum())
    viol_uniform = int((dev > 2*sig_max**2 + 1e-5).sum())
    tau = eer_threshold(labels, m_raw)
    disagree = int(((m_raw>=tau) != (m_pro>=tau)).sum())

    results[name] = dict(dim=dim, k=k, diag_selfcos=round(diag,4),
        sigma_frob=round(sigma_frob,4), paper_sigma=paper_sig,
        sig_mean=round(sig_mean,4), sig_p95=round(sig_p95,4), sig_max=round(sig_max,4),
        auc_plain=round(A_raw,4), auc_protected=round(A_pro,4), auc_gap=round(A_pro-A_raw,4),
        cor_max_dev=round(float(dev.max()),4), cor_p999_dev=round(float(np.quantile(dev,0.999)),4),
        cor_eps_max=round(float(eps_pair.max()),4), two_sigmax2=round(2*sig_max**2,4),
        cor_viol_perpair=viol_perpair, cor_viol_uniform=viol_uniform,
        margin_shift_max=round(float(np.abs(m_pro-m_raw).max()),4),
        disagree_at_eer=disagree, secs=round(time.time()-t0,1))
    print(f"[done] {name:9s} {time.time()-t0:5.1f}s | diag={diag:.3f} | sigma_frob={sigma_frob:.3f}"
          f" (paper {paper_sig}) | AUC plain/prot={A_raw:.3f}/{A_pro:.3f} ({A_pro-A_raw:+.3f}) |"
          f" |dhat-s|max={float(dev.max()):.4f}<=eps={float(eps_pair.max()):.3f} | viol={viol_perpair}")

OUT.write_text(json.dumps(results, indent=2))
print("\nsaved", OUT)
print(json.dumps(results, indent=2))
