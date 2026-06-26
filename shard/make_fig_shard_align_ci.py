"""Redraw fig_shard_alignment.png with bootstrap CI bands (exp13_ci data).

Two panels (e5-small, e5-base): residual re-identification R@1 vs.\ the number
of known-plaintext anchors m, for the global key and SHARD C=64 / C=256
(k-means cells), with shaded paired-bootstrap 95% CI bands.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results" / "exp13_outputs"
OUT = Path(r"D:/PHD/phd/article5/paper/figs/fig_shard_alignment.png")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160, "savefig.bbox": "tight"})

SCHEMES = [("global_key", "global key", "#b91c1c", "o"),
           ("shard_C64", "SHARD $C{=}64$", "#1f77b4", "s"),
           ("shard_C256", "SHARD $C{=}256$", "#2ca02c", "^")]


def panel(ax, enc, title):
    d = json.load(open(RES / f"exp13ci_alignment_{enc}.json"))
    for key, lab, col, mk in SCHEMES:
        cur = d["schemes"][key]["curve"]
        ms = sorted(int(m) for m in cur)
        rate = np.array([cur[str(m)]["rate"] for m in ms])
        lo = np.array([cur[str(m)]["lo"] for m in ms])
        hi = np.array([cur[str(m)]["hi"] for m in ms])
        ax.plot(ms, rate, marker=mk, color=col, label=lab, lw=1.4, ms=4)
        ax.fill_between(ms, lo, hi, color=col, alpha=0.18, linewidth=0)
    ax.axhline(0.5, color="#9ca3af", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("known-plaintext anchors $m$")
    ax.set_title(title)
    ax.set_ylim(-0.03, 1.05)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    panel(axes[0], "e5-small", "e5-small ($d_{\\mathrm{priv}}{=}288$)")
    panel(axes[1], "e5-base", "e5-base ($d_{\\mathrm{priv}}{=}576$)")
    axes[0].set_ylabel("residual re-identification R@1")
    axes[0].legend(frameon=False, fontsize=8, loc="center left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(OUT); plt.close()
    print("saved", OUT)


if __name__ == "__main__":
    main()
