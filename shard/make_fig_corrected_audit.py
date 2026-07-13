"""Generate the corrected utility and partial-alignment manuscript figures.

Reads only the immutable exp23/exp24 CSV summaries.  Both vector PDF and a
high-resolution PNG preview are emitted so the paper can use vector artwork.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def utility_figure(rows: list[dict[str, str]], out: Path) -> None:
    beir = [r for r in rows if r["suite"] == "beir"]
    order = [
        ("multilingual-e5-base", "scifact"),
        ("multilingual-e5-base", "nfcorpus"),
        ("multilingual-e5-base", "arguana"),
        ("multilingual-e5-small", "scifact"),
        ("multilingual-e5-small", "nfcorpus"),
        ("multilingual-e5-small", "arguana"),
    ]
    by_case: dict[tuple[str, str], dict[str, float]] = {}
    dimensions: dict[tuple[str, str], int] = {}
    for row in beir:
        key = (row["encoder"], row["dataset"])
        by_case.setdefault(key, {})[row["method"]] = float(row["ndcg10"])
        dimensions[key] = int(row["dimension"])

    methods = ["raw", "old_centered_full", "corrected_half_pca", "shard"]
    labels = ["Raw", "Legacy centred\nfull", "Corrected\nhalf-PCA", "SHARD d/4"]
    colors = ["#333333", "#c96b59", "#5d8cc9", "#2c9a73"]
    values = {m: [] for m in methods}
    for key in order:
        d4 = dimensions[key] // 4
        shard = f"corrected_shard_centered_router_dpub{d4}_kc200"
        values["raw"].append(by_case[key]["raw"])
        values["old_centered_full"].append(by_case[key]["old_centered_full"])
        values["corrected_half_pca"].append(by_case[key]["corrected_half_pca"])
        values["shard"].append(by_case[key][shard])

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(order)); width = 0.19
    for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
        ax.bar(x + (j - 1.5) * width, values[method], width, label=label,
               color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e.replace('multilingual-e5-', 'e5-')}\n{d}" for e, d in order])
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0.24, 0.68)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    fig.tight_layout()
    save(fig, out, "fig_shard_beir")


def alignment_figure(rows: list[dict[str, str]], out: Path) -> None:
    def series(cells: int, method: str):
        picked = [r for r in rows if int(r["cells"]) == cells and r["method"] == method]
        picked.sort(key=lambda r: int(r["m"]))
        return picked

    colors = {1: "#333333", 16: "#8267b2", 64: "#2878b5", 256: "#d66b37"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    for cells in (1, 16, 64, 256):
        s = series(cells, "ols_pinv")
        x = np.array([int(r["m"]) for r in s])
        y = np.array([float(r["r1_mean"]) for r in s])
        lo = np.array([float(r["r1_ci_low"]) for r in s])
        hi = np.array([float(r["r1_ci_high"]) for r in s])
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.7,
                color=colors[cells], label=f"C={cells}")
        ax.fill_between(x, lo, hi, color=colors[cells], alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Global known pairs m")
    ax.set_ylabel("Residual-gallery R@1")
    ax.set_ylim(-0.02, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("Diffuse disclosure: OLS")

    ax = axes[1]
    for method, label, style in (
        ("ols_pinv", "Minimum-norm OLS", "-"),
        ("procrustes_partial", "Rank-deficient Procrustes", "--"),
    ):
        s = series(256, method)
        x = np.array([float(r["mean_anchors_in_target_cell"]) for r in s])
        y = np.array([float(r["r1_mean"]) for r in s])
        lo = np.array([float(r["r1_ci_low"]) for r in s])
        hi = np.array([float(r["r1_ci_high"]) for r in s])
        ax.plot(x, y, style, marker="o", markersize=3.5, linewidth=1.8, label=label)
        ax.fill_between(x, lo, hi, alpha=0.10)
    ax.axvline(288, color="#777777", linestyle=":", linewidth=1.5,
               label="full dimension (288)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Mean known pairs in target cell (C=256)")
    ax.set_ylim(-0.02, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Useful recovery precedes full rank")

    fig.tight_layout()
    save(fig, out, "fig_shard_alignment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "figs")
    parser.add_argument("--exp23", type=Path,
                        default=ROOT / "results" / "exp23_corrected_score" / "summary.csv")
    parser.add_argument("--exp24", type=Path,
                        default=ROOT / "results" / "exp24_partial_alignment_main_v2" / "summary.csv")
    parser.add_argument("--exp24-low", type=Path,
                        default=ROOT / "results" / "exp24_partial_alignment_lowm" / "summary.csv")
    args = parser.parse_args()
    utility_figure(read_csv(args.exp23), args.out)
    merged = read_csv(args.exp24)
    if args.exp24_low.exists():
        by_key = {(r["cells"], r["m"], r["method"]): r for r in merged}
        for row in read_csv(args.exp24_low):
            by_key[(row["cells"], row["m"], row["method"])] = row
        merged = list(by_key.values())
    alignment_figure(merged, args.out)
    print(f"saved corrected figures to {args.out}")


if __name__ == "__main__":
    main()
