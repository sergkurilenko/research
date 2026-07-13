"""Generate the maximal-program figures from exp26--exp28 summaries.

The script deliberately reads aggregate, machine-readable summaries only.  It
does not inspect raw measurements, per-query outputs, or manuscript sources.
Every default representative slice is fixed in code and recorded in the
generated README so that figure construction is auditable and reproducible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]

# Colour-blind-safe Okabe--Ito-derived palette.  Method colours are kept
# identical between panels and encoder colours are kept identical in exp26/27.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#5B6573"
LIGHT_GREY = "#A7ADB4"
ENCODER_COLOURS = {
    "multilingual-e5-base": BLUE,
    "multilingual-e5-small": ORANGE,
}
ENCODER_LABELS = {
    "multilingual-e5-base": "e5-base",
    "multilingual-e5-small": "e5-small",
}

# Fixed, non-performance-conditioned default slices.
CKKS_CANDIDATE_COUNT = 128
CHURN_SCHEME = "cell_C64"
CHURN_N = 10_000
CHURN_ENCODER = "e5-base"
CHURN_CONDITIONS = ("clean", "int8")
CHURN_METHODS = (
    "public_prefix_nn",
    "residual_norm_nn",
    "gram_quantile_signature_nn",
    "combined_residual_invariants_nn",
)


def configure_matplotlib() -> None:
    """Set a compact, journal-safe style with embedded TrueType text."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"summary not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"summary is empty: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def as_float(row: dict[str, str], column: str) -> float:
    value = row[column].strip().lower()
    if value in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    return float(value)


def unique_row(rows: list[dict[str, str]], description: str) -> dict[str, str]:
    if len(rows) != 1:
        raise ValueError(f"expected one {description} row, found {len(rows)}")
    return rows[0]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def polish(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9DDE2", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={"Creator": "make_fig_maximal_program.py", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(out / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def ckks_figure(rows: list[dict[str, str]], source: Path, out: Path) -> None:
    require_columns(
        rows,
        (
            "encoder",
            "candidate_count",
            "packing_width",
            "query_ciphertext_count_median",
            "query_ciphertext_bytes_p50",
            "response_ciphertext_bytes_p50",
            "end_to_end_ms_p50",
            "server_ct_pt_eval_ms_p50",
            "upload_reduction_vs_width1_measured",
        ),
        source,
    )
    selected = [r for r in rows if int(r["candidate_count"]) == CKKS_CANDIDATE_COUNT]
    encoders = ("multilingual-e5-base", "multilingual-e5-small")
    for encoder in encoders:
        if not any(r["encoder"] == encoder for r in selected):
            raise ValueError(f"missing K={CKKS_CANDIDATE_COUNT} rows for {encoder}")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    ax = axes[0]
    all_widths: set[int] = set()
    response_by_width: dict[int, list[float]] = defaultdict(list)
    for encoder in encoders:
        series = sorted(
            (r for r in selected if r["encoder"] == encoder),
            key=lambda r: int(r["packing_width"]),
        )
        x = np.array([int(r["packing_width"]) for r in series])
        upload = np.array([as_float(r, "query_ciphertext_bytes_p50") / 2**20 for r in series])
        counts = np.array([int(round(as_float(r, "query_ciphertext_count_median"))) for r in series])
        all_widths.update(int(v) for v in x)
        for r in series:
            response_by_width[int(r["packing_width"])].append(
                as_float(r, "response_ciphertext_bytes_p50") / 2**20
            )
        ax.plot(
            x,
            upload,
            marker="o",
            color=ENCODER_COLOURS[encoder],
            label=f"{ENCODER_LABELS[encoder]} query",
            zorder=3,
        )
        # Counts are offset in opposite directions so the two encoders remain legible.
        for index, (width, value, count) in enumerate(zip(x, upload, counts)):
            reduction = as_float(series[index], "upload_reduction_vs_width1_measured")
            annotation = f"{count} ct"
            if index == len(series) - 1 and reduction > 0:
                annotation += f"\n−{100 * reduction:.0f}%"
            place_above = encoder.endswith("base") or index == len(series) - 1
            offset = (0, 6) if place_above else (0, -12)
            valign = "bottom" if place_above else "top"
            ax.annotate(
                annotation,
                (width, value),
                xytext=offset,
                textcoords="offset points",
                color=ENCODER_COLOURS[encoder],
                fontsize=6.7,
                ha="center",
                va=valign,
            )

    widths = np.array(sorted(all_widths))
    response = np.array([np.median(response_by_width[w]) for w in widths])
    ax.plot(
        widths,
        response,
        color=GREY,
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        label="response (both)",
        zorder=2,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Packing width")
    ax.set_ylabel("p50 ciphertext traffic (MiB)")
    ax.set_title("Traffic: upload shrinks, response does not")
    ax.legend(frameon=False, loc="lower left", handlelength=2.4)
    polish(ax)
    panel_label(ax, "a")

    ax = axes[1]
    server_fractions: list[float] = []
    for encoder in encoders:
        series = sorted(
            (r for r in selected if r["encoder"] == encoder),
            key=lambda r: int(r["packing_width"]),
        )
        x = np.array([int(r["packing_width"]) for r in series])
        e2e = np.array([as_float(r, "end_to_end_ms_p50") / 1000 for r in series])
        server = np.array([as_float(r, "server_ct_pt_eval_ms_p50") / 1000 for r in series])
        server_fractions.extend((server / e2e).tolist())
        ax.plot(
            x,
            e2e,
            marker="o",
            color=ENCODER_COLOURS[encoder],
            label=f"{ENCODER_LABELS[encoder]} E2E",
            zorder=3,
        )
        ax.plot(x, server, linestyle="--", color=ENCODER_COLOURS[encoder], alpha=0.72)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_xlabel("Packing width")
    ax.set_ylabel("p50 latency (s)")
    ax.set_ylim(1.75, 3.25)
    ax.set_title("Latency: server evaluation dominates")
    polish(ax)
    panel_label(ax, "b")
    ax.text(
        0.04,
        0.96,
        f"server eval = {100 * min(server_fractions):.0f}–{100 * max(server_fractions):.0f}% of E2E",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=GREY,
    )
    colour_legend = [
        Line2D([0], [0], color=ENCODER_COLOURS[e], marker="o", label=ENCODER_LABELS[e])
        for e in encoders
    ]
    style_legend = [
        Line2D([0], [0], color="#333333", linestyle="-", label="end to end"),
        Line2D([0], [0], color="#333333", linestyle="--", label="server CT–PT eval"),
    ]
    first = ax.legend(handles=colour_legend, frameon=False, loc="lower left")
    ax.add_artist(first)
    ax.legend(handles=style_legend, frameon=False, loc="lower right", handlelength=2.5)

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.18, top=0.86, wspace=0.34)
    save_figure(fig, out, "fig_maximal_ckks_blocksimd")


def raw_ndcg_by_case(utility_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    raw: dict[tuple[str, str, str], float] = {}
    for row in utility_rows:
        if np.isinf(as_float(row, "epsilon")):
            key = (row["suite"], row["encoder"], row["dataset"])
            if key in raw:
                raise ValueError(f"duplicate non-private utility row for {key}")
            raw[key] = as_float(row, "ndcg10")
    return raw


def choose_dp_cases(utility_rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    """Choose the lower raw-nDCG median in every observed suite/encoder stratum.

    This rule is fixed independently of all finite-epsilon utility and attack
    values.  The lower median makes the even-sized MIRACL stratum unambiguous.
    """
    raw = raw_ndcg_by_case(utility_rows)
    strata: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for (suite, encoder, dataset), ndcg in raw.items():
        strata[(suite, encoder)].append((ndcg, dataset))
    chosen: list[tuple[str, str, str]] = []
    for suite, encoder in sorted(strata):
        values = sorted(strata[(suite, encoder)], key=lambda item: (item[0], item[1]))
        _, dataset = values[(len(values) - 1) // 2]
        chosen.append((suite, encoder, dataset))
    return chosen


def case_label(key: tuple[str, str, str]) -> str:
    suite, encoder, dataset = key
    return f"{suite.upper()} / {ENCODER_LABELS[encoder]} / {dataset}"


def dp_figure(
    utility_rows: list[dict[str, str]],
    attack_rows: list[dict[str, str]],
    matched_rows: list[dict[str, str]],
    utility_source: Path,
    attack_source: Path,
    matched_source: Path,
    out: Path,
) -> list[tuple[str, str, str]]:
    require_columns(
        utility_rows,
        ("suite", "encoder", "dataset", "epsilon", "ndcg10", "ndcg10_ci95_low", "ndcg10_ci95_high"),
        utility_source,
    )
    require_columns(
        attack_rows,
        (
            "suite",
            "encoder",
            "dataset",
            "epsilon",
            "linkage_r1",
            "linkage_r1_ci95_low",
            "linkage_r1_ci95_high",
        ),
        attack_source,
    )
    require_columns(
        matched_rows,
        ("suite", "encoder", "dataset", "shard_ndcg10"),
        matched_source,
    )

    chosen = choose_dp_cases(utility_rows)
    if len(chosen) < 2:
        raise ValueError("DP figure requires at least two suite/encoder strata")
    raw_targets = raw_ndcg_by_case(utility_rows)
    matched_by_case = {
        (r["suite"], r["encoder"], r["dataset"]): r for r in matched_rows
    }
    palette = (BLUE, ORANGE, GREEN, PURPLE)
    colours = {key: palette[i % len(palette)] for i, key in enumerate(chosen)}

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), sharex=True)
    all_x: set[float] = set()
    ax = axes[0]
    for key in chosen:
        suite, encoder, dataset = key
        series = [
            r
            for r in utility_rows
            if (r["suite"], r["encoder"], r["dataset"]) == key
            and np.isfinite(as_float(r, "epsilon"))
        ]
        series.sort(key=lambda r: as_float(r, "epsilon"))
        if not series:
            raise ValueError(f"no finite-epsilon utility rows for {key}")
        x = np.log2([as_float(r, "epsilon") for r in series])
        y = np.array([as_float(r, "ndcg10") for r in series])
        lo = np.array([as_float(r, "ndcg10_ci95_low") for r in series])
        hi = np.array([as_float(r, "ndcg10_ci95_high") for r in series])
        all_x.update(float(v) for v in x)
        ax.plot(x, y, marker="o", markevery=2, color=colours[key], label=case_label(key))
        ax.fill_between(x, lo, hi, color=colours[key], alpha=0.09, linewidth=0)

        raw_target = raw_targets[key]
        matched = matched_by_case.get(key)
        if matched is None or not matched["shard_ndcg10"].strip():
            raise ValueError(f"missing corrected-SHARD target for {key}")
        shard_target = as_float(matched, "shard_ndcg10")
        # Short right-edge target segments keep six references from crossing the
        # entire panel while still presenting them as explicit target lines.
        ax.hlines(raw_target, 12.8, 15.15, color=colours[key], linestyle="--", linewidth=1.25, alpha=0.75)
        ax.hlines(shard_target, 12.8, 15.15, color=colours[key], linestyle=":", linewidth=1.55, alpha=0.9)

    ax.set_ylabel("nDCG@10")
    ax.set_ylim(-0.02, 0.73)
    ax.set_title("Retrieval utility")
    polish(ax)
    panel_label(ax, "a")
    ax.text(0.98, 0.04, "targets shown at high ε", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=GREY)

    ax = axes[1]
    for key in chosen:
        series = [
            r for r in attack_rows if (r["suite"], r["encoder"], r["dataset"]) == key
        ]
        series.sort(key=lambda r: as_float(r, "epsilon"))
        if not series:
            raise ValueError(f"no linkage rows for {key}")
        x = np.log2([as_float(r, "epsilon") for r in series])
        y = np.array([as_float(r, "linkage_r1") for r in series])
        lo = np.array([as_float(r, "linkage_r1_ci95_low") for r in series])
        hi = np.array([as_float(r, "linkage_r1_ci95_high") for r in series])
        ax.plot(x, y, marker="o", markevery=2, color=colours[key], label=case_label(key))
        ax.fill_between(x, lo, hi, color=colours[key], alpha=0.09, linewidth=0)
    ax.set_ylabel("Native-gallery linkage R@1")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Linkability of the released vectors")
    polish(ax)
    panel_label(ax, "b")

    ticks = sorted(all_x)
    labelled_ticks = [v for v in ticks if abs(v - round(v)) < 1e-9 and int(round(v)) % 2 == 1]
    if ticks and ticks[0] not in labelled_ticks:
        labelled_ticks.insert(0, ticks[0])
    for axis in axes:
        axis.set_xticks(labelled_ticks)
        axis.set_xlabel("log2 ε   (δ = 10⁻⁶)")

    case_handles = [Line2D([0], [0], color=colours[k], marker="o", label=case_label(k)) for k in chosen]
    target_handles = [
        Line2D([0], [0], color=GREY, linestyle="--", label="raw target"),
        Line2D([0], [0], color=GREY, linestyle=":", label="corrected SHARD target"),
    ]
    fig.legend(
        handles=case_handles + target_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        columnspacing=1.25,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.18, top=0.77, wspace=0.34)
    save_figure(fig, out, "fig_maximal_formal_dp")
    return chosen


def churn_figure(rows: list[dict[str, str]], source: Path, out: Path) -> None:
    require_columns(
        rows,
        (
            "encoder",
            "scheme",
            "n",
            "overlap",
            "condition",
            "linkage_method",
            "top1_recall_mean",
            "top1_recall_std",
        ),
        source,
    )
    selected = [
        r
        for r in rows
        if r["encoder"] == CHURN_ENCODER
        and r["scheme"] == CHURN_SCHEME
        and int(r["n"]) == CHURN_N
        and r["condition"] in CHURN_CONDITIONS
        and r["linkage_method"] in CHURN_METHODS
    ]
    labels = {
        "public_prefix_nn": "Public prefix",
        "residual_norm_nn": "Residual norm",
        "gram_quantile_signature_nn": "Cell-key Gram",
        "combined_residual_invariants_nn": "Cell-key combined",
    }
    colours = {
        "public_prefix_nn": BLUE,
        "residual_norm_nn": ORANGE,
        "gram_quantile_signature_nn": GREEN,
        "combined_residual_invariants_nn": PURPLE,
    }
    markers = {
        "public_prefix_nn": "o",
        "residual_norm_nn": "s",
        "gram_quantile_signature_nn": "^",
        "combined_residual_invariants_nn": "D",
    }
    linestyles = {
        "public_prefix_nn": "-",
        "residual_norm_nn": "--",
        "gram_quantile_signature_nn": "-.",
        "combined_residual_invariants_nn": "-",
    }
    plot_order = (
        "residual_norm_nn",
        "public_prefix_nn",
        "gram_quantile_signature_nn",
        "combined_residual_invariants_nn",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), sharex=True, sharey=True)
    for panel, condition in enumerate(CHURN_CONDITIONS):
        ax = axes[panel]
        for method in plot_order:
            series = sorted(
                (r for r in selected if r["condition"] == condition and r["linkage_method"] == method),
                key=lambda r: as_float(r, "overlap"),
            )
            if not series:
                raise ValueError(
                    f"missing churn rows for {CHURN_ENCODER}/{CHURN_SCHEME}/N={CHURN_N}/{condition}/{method}"
                )
            x = 100 * np.array([as_float(r, "overlap") for r in series])
            y = np.array([as_float(r, "top1_recall_mean") for r in series])
            sd = np.array([as_float(r, "top1_recall_std") for r in series])
            ax.plot(
                x,
                y,
                marker=markers[method],
                color=colours[method],
                linestyle=linestyles[method],
                markerfacecolor="white" if method == "public_prefix_nn" else colours[method],
                markeredgewidth=1.35 if method == "public_prefix_nn" else 0.8,
                label=labels[method],
                zorder=4 if method == "public_prefix_nn" else 3,
            )
            ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=colours[method], alpha=0.10, linewidth=0)
        ax.set_title("Clean releases" if condition == "clean" else "INT8-quantised releases")
        ax.set_xlabel("Common documents (%)")
        ax.set_xticks([25, 50, 75, 90, 100])
        ax.set_ylim(-0.03, 1.03)
        polish(ax)
        panel_label(ax, "a" if panel == 0 else "b")
        if condition == "clean":
            ax.text(
                0.04,
                0.94,
                "prefix and norm overlap at R@1 ≈ 1",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                color=GREY,
            )
    axes[0].set_ylabel("Common-item linkage R@1")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=colours[m],
                linestyle=linestyles[m],
                marker=markers[m],
                markerfacecolor="white" if m == "public_prefix_nn" else colours[m],
                label=labels[m],
            )
            for m in CHURN_METHODS
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        columnspacing=1.25,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.18, top=0.79, wspace=0.12)
    save_figure(fig, out, "fig_maximal_cross_release_churn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(
    out: Path,
    sources: dict[str, Path],
    dp_cases: list[tuple[str, str, str]],
) -> None:
    source_lines = "\n".join(
        f"- `{name}`: `{path.as_posix()}` (SHA-256 `{sha256(path)}`)" for name, path in sources.items()
    )
    case_lines = "\n".join(f"  - `{suite}/{encoder}/{dataset}`" for suite, encoder, dataset in dp_cases)
    text = f"""# Maximal-program figures (experiments 26--28)

These figures are generated by `shard/make_fig_maximal_program.py`.  The
generator reads only aggregate machine-readable summaries; raw measurements,
per-query/per-target files, experiment logs, exp29, and manuscript sources are
not read.

## Source summaries

{source_lines}

## Exact figure filters

### `fig_maximal_ckks_blocksimd`

- Source: exp26 `summary.csv`.
- Fixed candidate count: `K={CKKS_CANDIDATE_COUNT}`.
- Encoders: `multilingual-e5-base` and `multilingual-e5-small`.
- All available packing widths are plotted.  The missing base/width-8 point is
  not imputed (that layout is unavailable in the experiment summary).
- Traffic columns: p50 serialized query and response ciphertext bytes, converted
  to MiB.  Marker annotations are median query-ciphertext counts.
- Latency columns: p50 end-to-end and p50 server ciphertext--plaintext evaluation.
  The response line is the per-width median across available encoders; its near
  constancy exposes the response-traffic bottleneck rather than hiding it behind
  the query-upload reduction.

### `fig_maximal_formal_dp`

- Sources: exp27 `utility_summary.csv`, `attack_summary.csv`, and
  `matched_utility.csv`.
- All evaluated finite epsilon values are plotted on the x-axis as `log2(epsilon)`;
  no interpolation or point removal is used.
- Representative cases are selected without looking at any finite-epsilon
  utility or linkage result.  Within every observed `(suite, encoder)` stratum,
  the generator sorts datasets by the non-private (`epsilon=inf`) nDCG@10 and
  takes the lower median (dataset name breaks exact ties).  This produces:
{case_lines}
- Utility is nDCG@10 with its reported 95% CI.  Raw targets come from the
  non-private summary rows.  Corrected-SHARD target lines come from
  `matched_utility.csv`; they are shown only as short right-edge segments to
  avoid six full-width reference lines.
- Linkage is native-gallery R@1 with its reported 95% CI.  `delta=1e-6` is fixed
  by the experiment calibration.

### `fig_maximal_cross_release_churn`

- Source: exp28 `summary.csv`.
- Fixed slice: `encoder={CHURN_ENCODER}`, `scheme={CHURN_SCHEME}`, `N={CHURN_N}`.
- Conditions: clean and INT8.  These are the unperturbed baseline and the
  lowest-precision quantisation endpoint; fp16 is intentionally omitted from
  this compact endpoint comparison.
- All measured overlap levels are plotted: 25%, 50%, 75%, 90%, and 100%.
- Methods: `public_prefix_nn`, `residual_norm_nn`,
  `gram_quantile_signature_nn`, and `combined_residual_invariants_nn`.
- Metric: common-item top-1 recall (`top1_recall_mean`), labelled R@1, with a
  band of plus/minus one across-run standard deviation.  No values are imputed.

## Reproduction

From the repository root, using the experiment environment:

```powershell
..\\RES\\experiments\\.venv\\Scripts\\python.exe shard\\make_fig_maximal_program.py
```

Use `--out PATH` to write the same six PDF/PNG artifacts and this README to a
different directory.  PDFs are vector output; PNG previews are rendered at
320 dpi.
"""
    (out / "README.md").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from exp26--exp28 aggregate summaries only."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "maximal_program_figures")
    parser.add_argument(
        "--exp26",
        type=Path,
        default=ROOT / "results" / "exp26_ckks_blocksimd" / "summary.csv",
    )
    parser.add_argument(
        "--exp27-utility",
        type=Path,
        default=ROOT / "results" / "exp27_formal_dp_baseline" / "utility_summary.csv",
    )
    parser.add_argument(
        "--exp27-attack",
        type=Path,
        default=ROOT / "results" / "exp27_formal_dp_baseline" / "attack_summary.csv",
    )
    parser.add_argument(
        "--exp27-matched",
        type=Path,
        default=ROOT / "results" / "exp27_formal_dp_baseline" / "matched_utility.csv",
    )
    parser.add_argument(
        "--exp28",
        type=Path,
        default=ROOT / "results" / "exp28_cross_release_churn" / "summary.csv",
    )
    args = parser.parse_args()

    configure_matplotlib()
    exp26_rows = read_csv(args.exp26)
    utility_rows = read_csv(args.exp27_utility)
    attack_rows = read_csv(args.exp27_attack)
    matched_rows = read_csv(args.exp27_matched)
    exp28_rows = read_csv(args.exp28)

    ckks_figure(exp26_rows, args.exp26, args.out)
    dp_cases = dp_figure(
        utility_rows,
        attack_rows,
        matched_rows,
        args.exp27_utility,
        args.exp27_attack,
        args.exp27_matched,
        args.out,
    )
    churn_figure(exp28_rows, args.exp28, args.out)
    sources = {
        "exp26 summary": args.exp26,
        "exp27 utility summary": args.exp27_utility,
        "exp27 attack summary": args.exp27_attack,
        "exp27 matched-utility summary": args.exp27_matched,
        "exp28 summary": args.exp28,
    }
    write_readme(args.out, sources, dp_cases)
    print(f"saved maximal-program figures to {args.out}")


if __name__ == "__main__":
    main()
