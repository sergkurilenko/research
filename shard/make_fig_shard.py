"""Figures for the SHARD experiments: anchor-complexity, public leakage."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = FIGS
NB = RESULTS


def fig_alignment():
    d = json.load(open(NB / "exp13_outputs/exp13_alignment_e5-small.json"))
    sch = d["schemes"]
    order = [("global_key", "global key (baseline)", "#b42318", "o"),
             ("shard_C1", "SHARD C=1", "#7a5195", "s"),
             ("shard_C64", "SHARD C=64", "#1f77b4", "^"),
             ("shard_C256", "SHARD C=256", "#1a7f37", "D")]
    plt.figure(figsize=(6.4, 4.0))
    for key, lab, col, mk in order:
        if key not in sch:
            continue
        cur = sch[key]["curve"]
        ms = sorted(int(m) for m in cur)
        ys = [cur[str(m)] for m in ms]
        plt.plot(ms, ys, marker=mk, color=col, label=f"{lab} ($m_{{50}}$={sch[key]['m50']})", lw=1.8, ms=5)
    plt.xscale("log")
    plt.xlabel("known plaintext anchors $m$")
    plt.ylabel("residual re-identification R@1")
    plt.title("Alignment anchor-complexity: cell keys multiply the cost by $\\sim C$")
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=8, loc="center left")
    plt.tight_layout()
    plt.savefig(FIG / "fig_shard_alignment.png", dpi=170, bbox_inches="tight")
    print("wrote fig_shard_alignment.png")


def fig_leakage():
    d = json.load(open(NB / "exp14_outputs/exp14_leakage_e5-small.json"))
    pl = d["prefix_leakage"]
    dps = sorted(int(k) for k in pl)
    ys = [pl[str(x)] for x in dps]
    base = d["baseline_svd_k2"]
    dd = d["d"]
    plt.figure(figsize=(6.0, 3.8))
    plt.plot(dps, ys, "o-", color="#1f77b4", lw=1.8, ms=6, label="SHARD public prefix $u$")
    plt.axhline(base, color="#b42318", ls="--", lw=1.6,
                label=f"baseline geometry (SVD $k{{=}}d/2$): {base:.2f}")
    for x, y in zip(dps, ys):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    plt.xlabel("public prefix dimension $d_{pub}$")
    plt.ylabel("NN-overlap@10 with full space")
    plt.title("Public-index leakage: a short prefix reveals far less neighbour structure")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_shard_leakage.png", dpi=170, bbox_inches="tight")
    print("wrote fig_shard_leakage.png")


def fig_beir():
    d = json.load(open(NB / "exp17_outputs/exp17_beir_shard.json"))["results"]
    labels = [f"{r['encoder'].split('/')[-1].replace('multilingual-','')}\n{r['dataset']}" for r in d]
    raw = [r["raw_ndcg10"] for r in d]
    svd = [r["svd_ndcg10"] for r in d]
    # best SHARD (full prefix d/2, Kc=200) per cell
    sh = []
    for r in d:
        best = max(r["shard"].values(), key=lambda v: v["ndcg10"])
        sh.append(best["ndcg10"])
    x = np.arange(len(d)); w = 0.26
    plt.figure(figsize=(7.2, 3.8))
    plt.bar(x - w, raw, w, label="raw (full, ceiling)", color="#444444")
    plt.bar(x, svd, w, label="SVD $k{=}d/2$ baseline", color="#b42318")
    plt.bar(x + w, sh, w, label="SHARD (full-dim rerank)", color="#1a7f37")
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("nDCG@10")
    plt.title("BEIR utility: SHARD recovers the nDCG that SVD truncation loses")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig_shard_beir.png", dpi=170, bbox_inches="tight")
    print("wrote fig_shard_beir.png")


def fig_learned():
    d = json.load(open(NB / "exp22_outputs/exp22_learned_e5-small.json"))
    dp = d["d_priv"]
    plt.figure(figsize=(6.2, 3.9))
    styles = {"procrustes": ("Procrustes (orthogonal)", "#1a7f37", "o"),
              "ridge": ("Ridge (ALGEN core)", "#1f77b4", "s"),
              "mlp": ("MLP (nonlinear)", "#7a5195", "^")}
    for key, (lab, col, mk) in styles.items():
        cur = d["supervised"][key]
        ms = sorted(int(m) for m in cur)
        plt.plot(ms, [cur[str(m)] for m in ms], marker=mk, color=col, label=lab, lw=1.8, ms=5)
    u = d["unsupervised_covmatch_cos"]
    plt.axhline(u, color="#b42318", ls=":", lw=1.8,
                label=f"unsupervised cov-match (vec2vec core): {u:.2f}")
    plt.axvline(dp, color="grey", ls="--", lw=1.2)
    plt.text(dp + 5, 0.05, f"$d_{{priv}}={dp}$", color="grey", fontsize=9)
    plt.xlabel("in-cell known-plaintext anchors $m$")
    plt.ylabel("recovered-residual cosine to native")
    plt.title("No attacker beats the $d_{priv}$ barrier on SHARD's keyed residual")
    plt.ylim(-0.05, 1.05); plt.grid(alpha=0.3); plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "fig_shard_learned.png", dpi=170, bbox_inches="tight")
    print("wrote fig_shard_learned.png")


def fig_vs_dp():
    import os
    p = NB / "exp21_outputs/exp21_vs_dp_e5-small.json"
    if not os.path.exists(p):
        print("exp21 json not ready, skip fig_vs_dp"); return
    d = json.load(open(p)); m = d["m_budget"][0]
    plt.figure(figsize=(6.4, 4.2))
    # manual label offsets (dx,dy in points) to avoid collisions
    offs = {"DP-noise sigma=0.0": (6, -12), "DP-noise sigma=0.25": (6, 8),
            "DP-noise sigma=0.5": (-10, 12), "DP-noise sigma=1.0": (6, -12),
            "DP-noise sigma=2.0": (6, 6), "global key (C=1)": (6, 14),
            "SHARD C=64": (8, 2), "SHARD C=256": (8, -12)}
    jit = {"SHARD C=64": 0.012, "SHARD C=256": -0.012, "global key (C=1)": 0.012,
           "DP-noise sigma=0.0": -0.012}
    for de in d["defenses"]:
        name = de["name"]
        x = de["acc1"]; y = de["deanon_r1"][str(m)] if str(m) in de["deanon_r1"] else de["deanon_r1"][m]
        yj = y + jit.get(name, 0.0)
        if "SHARD" in name:
            col, mk, sz = "#1a7f37", "*", 200
        elif "DP" in name:
            col, mk, sz = "#b42318", "o", 60
        else:
            col, mk, sz = "#7a5195", "s", 60
        plt.scatter(x, yj, c=col, marker=mk, s=sz, zorder=3, edgecolors="k", linewidths=0.4)
        lab = name.replace("DP-noise sigma=", "DP $\\sigma$=").replace("global key (C=1)", "global key")
        plt.annotate(lab, (x, yj), textcoords="offset points",
                     xytext=offs.get(name, (6, 4)), fontsize=7.5)
    plt.xlabel("utility (self-retrieval Acc@1)")
    plt.ylabel(f"de-anonymisation R@1 at $m{{=}}{m}$ anchors")
    plt.title("Attack-aware vs. distortion-aware: SHARD reaches high utility + low de-anon",
              fontsize=10.5)
    plt.ylim(-0.08, 1.12); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG / "fig_shard_vs_dp.png", dpi=170, bbox_inches="tight")
    print("wrote fig_shard_vs_dp.png")


if __name__ == "__main__":
    fig_alignment()
    fig_leakage()
    fig_beir()
    fig_learned()
    fig_vs_dp()
