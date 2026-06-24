"""Forest plot of the SVD-truncation effect (Delta Acc) with paired bootstrap
95% CIs, per encoder, for Acc@1 and Acc@10. Reads exp10_significance.json."""
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

OUT = FIGS
d = json.load(open(RESULTS / "exp10_outputs" / "exp10_significance.json", encoding="utf-8"))
encs = d["encoders"]
names = [e["encoder"] for e in encs]
y = np.arange(len(encs))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
for ax, key, title in [(axes[0], "boot_acc1", "$\\Delta$ Acc@1"),
                       (axes[1], "boot_acc10", "$\\Delta$ Acc@10")]:
    delt = [e[key]["delta_mean"] if "delta_mean" in e[key] else e[key]["delta"] for e in encs]
    lo = [e[key]["ci_lo"] for e in encs]
    hi = [e[key]["ci_hi"] for e in encs]
    pkey = "mcnemar_acc1" if key == "boot_acc1" else "mcnemar_acc10"
    sig = [e[pkey]["p_value"] < 0.05 for e in encs]
    for yi, dv, l, h, sg in zip(y, delt, lo, hi, sig):
        col = "#1a7f37" if (sg and dv > 0) else ("#b42318" if (sg and dv < 0) else "#667085")
        ax.plot([l, h], [yi, yi], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(dv, yi, "o", color=col, ms=6)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_title(title)
    ax.set_xlabel("SVD $-$ raw (paired, 95% CI)")
    ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "fig_exp10_significance.png", dpi=170, bbox_inches="tight")
print("wrote", OUT / "fig_exp10_significance.png")
