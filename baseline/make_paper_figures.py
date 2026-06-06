"""
Generate all figures referenced by paper_en.tex from cached experiment outputs.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGS / name)
    plt.close()


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    boxes = {
        "Client\nencoder": (0.05, 0.58),
        "Secret transform\nmu, Vk, R": (0.25, 0.58),
        "Local PQ\nshortlist": (0.45, 0.58),
        "CKKS encrypt\nrotated query": (0.65, 0.58),
        "Server ct-pt\nrerank": (0.65, 0.18),
        "Decrypt\nscores": (0.45, 0.18),
        "Top-k\nresults": (0.25, 0.18),
    }
    for text, (x, y) in boxes.items():
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.16,
                0.20,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                fc="#f5f7fb",
                ec="#4b5563",
                lw=1.0,
            )
        )
        ax.text(x + 0.08, y + 0.10, text, ha="center", va="center")

    arrows = [
        ("Client\nencoder", "Secret transform\nmu, Vk, R"),
        ("Secret transform\nmu, Vk, R", "Local PQ\nshortlist"),
        ("Local PQ\nshortlist", "CKKS encrypt\nrotated query"),
        ("CKKS encrypt\nrotated query", "Server ct-pt\nrerank"),
        ("Server ct-pt\nrerank", "Decrypt\nscores"),
        ("Decrypt\nscores", "Top-k\nresults"),
    ]
    for a, b in arrows:
        x1, y1 = boxes[a]
        x2, y2 = boxes[b]
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.16, y1 + 0.10),
                (x2, y2 + 0.10),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.0,
                color="#374151",
            )
        )
    ax.text(0.06, 0.90, "Offline: protected database and public PQ artifact", weight="bold")
    ax.text(0.06, 0.04, "Online: value privacy by CKKS; access pattern remains visible", color="#6b7280")
    plt.savefig(FIGS / "architecture_en.pdf", bbox_inches="tight")
    plt.close()


def exp1() -> None:
    imp = pd.read_csv(NB / "exp1_outputs" / "feature_importance_v2.csv").sort_values("importance")
    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    ax.barh(imp["feature"], imp["importance"], color="#2f6f9f")
    ax.set_xlabel("Importance")
    ax.set_title("CKKS latency surrogate")
    savefig("fig_exp1_feature_importance.png")

    metrics = pd.read_csv(NB / "exp1_outputs" / "regression_metrics_v2.csv")
    metrics = metrics.sort_values("r2_nested_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    ax.barh(metrics["model"], metrics["r2_nested_mean"], color="#5b8e7d")
    ax.errorbar(
        metrics["r2_nested_mean"],
        metrics["model"],
        xerr=metrics["r2_nested_std"],
        fmt="none",
        color="#111827",
        lw=0.8,
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Nested CV R2")
    ax.set_title("Regression quality")
    savefig("fig_exp1_regression_r2.png")


def exp2() -> None:
    df = pd.read_csv(NB / "exp2_outputs" / "exp2_ckks_timing.csv")
    df["time_s"] = df["time_ms"] / 1000.0
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    data = [df.loc[df["mode"] == m, "time_s"].values for m in ["ct-ct", "ct-pt"]]
    ax.boxplot(data, labels=["ct-ct", "ct-pt"], patch_artist=True)
    ax.set_ylabel("Latency, s")
    ax.set_title("Dot-product latency")
    savefig("fig_exp2_ctpt_vs_ctct.png")


def exp3() -> None:
    js = json.loads((NB / "exp3_outputs" / "exp3_summary.json").read_text(encoding="utf-8"))
    rows = pd.DataFrame(js["by_kfraction_rotation_rknown"])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    styles = {
        ("off", "n/a"): ("No rotation", "#1f77b4", "o"),
        ("on", "known"): ("Known R", "#ff7f0e", "s"),
        ("on", "unknown"): ("Unknown R", "#2ca02c", "^"),
    }
    for key, (label, color, marker) in styles.items():
        sub = rows[(rows["rotation"] == key[0]) & (rows["R_knowledge"] == key[1])].sort_values("k_fraction")
        if not len(sub):
            continue
        ax.plot(sub["k_fraction"], sub["bleu_mean"], marker=marker, color=color, label=label)
        ax.fill_between(sub["k_fraction"], sub["bleu_ci95_lo"], sub["bleu_ci95_hi"], color=color, alpha=0.15)
    ax.set_xlabel("k / d")
    ax.set_ylabel("BLEU")
    ax.legend(frameon=False)
    savefig("fig_exp3_bleu_vs_k.png")

    kd1 = rows[rows["k_fraction"] == 1.0]
    labels = ["No rotation", "Known R", "Unknown R"]
    vals = [
        kd1[(kd1["rotation"] == "off")]["bleu_mean"].iloc[0],
        kd1[(kd1["rotation"] == "on") & (kd1["R_knowledge"] == "known")]["bleu_mean"].iloc[0],
        kd1[(kd1["rotation"] == "on") & (kd1["R_knowledge"] == "unknown")]["bleu_mean"].iloc[0],
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("BLEU")
    ax.set_title("k / d = 1.0")
    ax.tick_params(axis="x", rotation=20)
    savefig("fig_exp3_bleu_kd1.png")


def exp5() -> None:
    js = json.loads((NB / "exp5_outputs" / "v3_multi_results_highprot.json").read_text(encoding="utf-8"))
    rows = []
    for enc in js["encoders"]:
        base = enc["baseline_proj"]["acc10"]
        for i, s in enumerate(enc["per_seed"]):
            rows.append({"encoder": enc["encoder"].replace("_half", ""), "seed": i + 1, "delta": s["acc10"] - base})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for enc, sub in df.groupby("encoder"):
        ax.plot(sub["seed"], sub["delta"], marker="o", label=enc)
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xlabel("Rotation seed index")
    ax.set_ylabel("Acc@10 delta vs SVD")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    savefig("fig_exp5_per_seed.png")


def exp6() -> None:
    js = json.loads((NB / "exp_cs_v2_outputs" / "cs_v2_results.json").read_text(encoding="utf-8"))
    pct = js["latency_pct_pooled"]
    stages = [
        ("Projection", pct["project_ms"]["p95"]),
        ("Local PQ", pct["client_pq_ms"]["p95"]),
        ("Encrypt", pct["encrypt_ms"]["p95"]),
        ("Server+HTTP", pct["network_ms"]["p95"]),
        ("Decrypt", pct["decrypt_ms"]["p95"]),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.bar([s[0] for s in stages], [s[1] for s in stages], color="#567a9f")
    ax.set_ylabel("p95 latency, ms")
    ax.tick_params(axis="x", rotation=25)
    savefig("fig_exp6_latency_decomposition.png")

    files = sorted((NB / "exp6_outputs").glob("exp6_per_query_latency_seed_*.csv"))
    vals = []
    for p in files:
        vals.extend(pd.read_csv(p)["total_ms"].tolist())
    vals = np.sort(np.array(vals))
    y = np.linspace(0, 1, len(vals), endpoint=True)
    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    ax.plot(vals, y, color="#5b8e7d")
    ax.axvline(np.percentile(vals, 95), color="#b91c1c", ls="--", lw=1, label="p95")
    ax.set_xlabel("Latency, ms")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False)
    savefig("fig_exp6_latency_cdf.png")


def exp7() -> None:
    pro = pd.read_csv(NB / "exp7_outputs" / "exp7_procrustes_summary.csv")
    pro = pro[pro["known_pairs"] >= 0]
    fig, ax1 = plt.subplots(figsize=(4.7, 3.0))
    ax1.plot(pro["known_pairs"], pro["target_recall_at_1_mean"], marker="o", color="#1f77b4", label="Recall@1")
    ax1.plot(pro["known_pairs"], pro["target_recall_at_10_mean"], marker="s", color="#2ca02c", label="Recall@10")
    ax1.set_xscale("symlog", linthresh=10)
    ax1.set_xlabel("Known plaintext pairs")
    ax1.set_ylabel("Target recovery")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(frameon=False, loc="lower right")
    savefig("fig_exp7_procrustes.png")

    pq = pd.read_csv(NB / "exp7_outputs" / "exp7_pq_leakage.csv")
    labels = [f"M={int(r.pq_m)}, {int(r.nbits)}b" for r in pq.itertuples()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    ax.bar(x - 0.2, pq["reconstruction_mean_cosine"], width=0.4, label="Cosine")
    ax.bar(x + 0.2, pq["neighbor_overlap_at_10"], width=0.4, label="NN overlap@10")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    savefig("fig_exp7_pq_leakage.png")


def exp8() -> None:
    df = pd.read_csv(NB / "exp8_outputs" / "exp8_tradeoff_summary.csv")
    colors = {"e5small": "#1f77b4", "e5base": "#2ca02c"}
    labels = {"e5small": "e5-small", "e5base": "e5-base"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
    for enc, sub in df.groupby("encoder"):
        sub = sub.sort_values("k_over_d")
        color = colors.get(enc, "#4b5563")
        label = labels.get(enc, enc)
        axes[0].plot(
            sub["k_over_d"],
            sub["svd_acc1"],
            marker="o",
            color=color,
            label=f"{label} SVD",
        )
        axes[0].plot(
            sub["k_over_d"],
            sub["noise_acc1"],
            marker="s",
            ls="--",
            color=color,
            label=f"{label} noise",
        )
        axes[1].plot(
            sub["k_over_d"],
            sub["svd_nn_overlap_at_10"],
            marker="o",
            color=color,
            label=f"{label} SVD",
        )
        axes[1].plot(
            sub["k_over_d"],
            sub["noise_nn_overlap_at_10"],
            marker="s",
            ls="--",
            color=color,
            label=f"{label} noise",
        )

    axes[0].set_ylabel("Acc@1")
    axes[0].set_xlabel("k / d")
    axes[0].set_title("Self-retrieval utility")
    axes[1].set_ylabel("NN overlap@10")
    axes[1].set_xlabel("k / d")
    axes[1].set_title("Raw-geometry leakage")
    axes[0].set_ylim(0.25, 0.95)
    axes[1].set_ylim(0.15, 1.02)
    axes[0].legend(frameon=False, fontsize=7)
    savefig("fig_exp8_tradeoff.png")


def exp9() -> None:
    df = pd.read_csv(NB / "exp9_outputs" / "exp9_reference_attack_summary.csv")
    df = df.sort_values("known_pairs")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)

    axes[0].plot(
        df["known_pairs"],
        df["overlap_recall_at_1_mean"],
        marker="o",
        color="#1f77b4",
        label="Exact R@1",
    )
    axes[0].plot(
        df["known_pairs"],
        df["overlap_recall_at_10_mean"],
        marker="s",
        color="#2ca02c",
        label="Exact R@10",
    )
    axes[0].set_ylabel("Exact paragraph recovery")
    axes[0].set_xlabel("Known plaintext pairs")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(
        df["known_pairs"],
        df["overlap_top1_token_jaccard_mean"],
        marker="o",
        color="#8c564b",
        label="Reference overlaps",
    )
    axes[1].plot(
        df["known_pairs"],
        df["disjoint_top1_token_jaccard_mean"],
        marker="s",
        color="#9467bd",
        label="Reference disjoint",
    )
    axes[1].set_ylabel("Top-1 token Jaccard")
    axes[1].set_xlabel("Known plaintext pairs")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)

    for ax in axes:
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(-0.5, float(df["known_pairs"].max()) * 1.15)
    savefig("fig_exp9_reference_attack.png")


def main() -> None:
    architecture()
    exp1()
    exp2()
    exp3()
    exp5()
    exp6()
    exp7()
    exp8()
    exp9()


if __name__ == "__main__":
    main()
