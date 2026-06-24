"""Experiment 20: the micro-key (C=N) variant -- residual leak and unlinkability.

Two cancellable-template properties as a function of key granularity:
  (a) residual neighbour-graph leak: the fraction of a document's true
      residual top-10 neighbours the server can still place using the keyed
      shards z (within a cell <z_i,z_j>=<r_i,r_j>; across cells it is noise).
      Per-document keys (cell size 1) leave no same-key pair, so this -> 0.
  (b) unlinkability of the residual channel: re-key the store twice and ask
      whether the same document is matchable across keys (mated vs non-mated
      cosine, AUC). Independent per-cell/per-doc keys make this ~0.5.
We also note (honestly) that the public prefix is NOT a cancellable
channel: it is identical across re-keyings unless the prefix global key is
also refreshed (Section on the reference-lookup limitation).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json, os
from pathlib import Path
import numpy as np
import shard_lib as S

OUT = (RESULTS / "exp20_outputs"); OUT.mkdir(exist_ok=True)
ENC = os.environ.get("E20_ENC", "e5-small")
N_POOL = int(os.environ.get("E20_POOL", 50_000))
D_PUB = int(os.environ.get("E20_DPUB", 96))
N_PROBE = 1000


def per_doc_keys_apply(R, seed):
    """Micro-key: an independent orthogonal key per document. We realise the
    pairwise effect <H_i r_i, H_j r_j> directly: for i!=j it is an inner
    product of independently random-rotated vectors (=> ~0 in expectation),
    so the recoverable same-key graph is empty. We model leak/unlinkability
    analytically below; here we return R rotated per-row by random orthogonals
    is O(N d^2) and unnecessary -- micro-key same-key pairs do not exist."""
    return None


def within_cell_recoverable(Rr, labels, probes, k=10):
    """Server's recoverable residual top-k using z: within-cell exact, else
    excluded. Returns mean overlap with the true residual top-k."""
    true_nn = np.argpartition(-(Rr[probes] @ Rr.T), k + 1, axis=1)
    # exclude self then take top-k
    out = []
    for pi, p in enumerate(probes):
        sims = Rr[p] @ Rr.T
        sims[p] = -np.inf
        tnn = np.argpartition(-sims, k)[:k]
        # server can only use same-cell candidates (within-cell scores are exact)
        same = np.where(labels == labels[p])[0]
        same = same[same != p]
        if len(same) == 0:
            out.append(0.0); continue
        ssc = Rr[p] @ Rr[same].T
        srec = same[np.argpartition(-ssc, min(k, len(same) - 1))[:k]]
        out.append(len(set(tnn) & set(srec)) / k)
    return float(np.mean(out))


def unlinkability_auc(Rr, labels, sample, seedA, seedB):
    """AUC of distinguishing mated (same doc, two keys) from non-mated."""
    ZA = S.apply_keys(Rr, labels, master_seed=seedA)
    ZB = S.apply_keys(Rr, labels, master_seed=seedB)
    a = ZA[sample]; b = ZB[sample]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    mated = np.sum(a * b, axis=1)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(sample))
    perm[perm == np.arange(len(sample))] = (perm[perm == np.arange(len(sample))] + 1) % len(sample)
    nonmated = np.sum(a * b[perm], axis=1)
    # AUC = P(mated > nonmated)
    return float((mated[:, None] > nonmated[None, :]).mean())


def main():
    print(f"=== exp20 micro-key: enc={ENC} d_pub={D_PUB} pool={N_POOL} ===", flush=True)
    X, _, d = S.load(ENC, n=N_POOL)
    mu = X.mean(0, keepdims=True).astype(np.float32)
    rng = np.random.RandomState(0); idx = rng.choice(len(X), min(50000, len(X)), replace=False)
    V, _ = S.pca_basis((X[idx] - mu).astype(np.float32))
    Xrot = ((X - mu) @ V).astype(np.float32); del X
    Rr = np.ascontiguousarray(Xrot[:, D_PUB:])
    U = np.ascontiguousarray(Xrot[:, :D_PUB])
    grng = np.random.default_rng(7)
    probes = grng.choice(len(Rr), N_PROBE, replace=False)
    res = {"encoder": ENC, "d": int(d), "d_pub": D_PUB, "schemes": {}}
    for C in [64, 256]:
        labels, _ = S.kmeans_cells(U, C, seed=0)
        leak = within_cell_recoverable(Rr, labels, probes)
        auc = unlinkability_auc(Rr, labels, probes, 3000, 4000)
        res["schemes"][f"cell_C{C}"] = {"residual_graph_recoverable": leak,
                                        "unlinkability_auc": auc}
        print(f"  cell C={C}: residual-graph recoverable={leak:.3f}  "
              f"unlinkability AUC={auc:.3f}", flush=True)
    # micro-key (per-document): cells of size 1 -> no same-key pairs.
    micro_labels = np.arange(len(Rr), dtype=np.int64)   # each doc its own cell
    leak_micro = within_cell_recoverable(Rr, micro_labels, probes)
    # unlinkability with a fresh per-document orthogonal key on each side,
    # computed directly on the probe sample (apply_keys over N docs would be
    # O(N d_priv^2)).
    d_priv = Rr.shape[1]
    P = Rr[probes]
    nrm = np.linalg.norm(P, axis=1) + 1e-9
    za = np.empty_like(P); zb = np.empty_like(P)
    for j in range(len(probes)):
        za[j] = P[j] @ S.random_orthogonal(d_priv, 7_000_000 + j).T
        zb[j] = P[j] @ S.random_orthogonal(d_priv, 8_000_000 + j).T
    cos_mated = np.sum(za * zb, axis=1) / (nrm * nrm)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(probes))
    cos_non = np.sum(za * zb[perm], axis=1) / (nrm * nrm[perm])
    auc_micro = float((cos_mated[:, None] > cos_non[None, :]).mean())
    res["schemes"]["microkey_perdoc"] = {"residual_graph_recoverable": leak_micro,
        "unlinkability_auc": auc_micro, "note": "cell size 1: no same-key pairs -> no graph leak"}
    print(f"  micro-key (per-doc): residual-graph recoverable={leak_micro:.3f}  "
          f"unlinkability AUC={auc_micro:.3f}", flush=True)
    json.dump(res, open(OUT / f"exp20_microkey_{ENC}.json", "w"), indent=2)
    print("saved", OUT / f"exp20_microkey_{ENC}.json", flush=True)


if __name__ == "__main__":
    main()
