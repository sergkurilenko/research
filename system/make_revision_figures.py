"""Create the vector figures for the revised system paper.

The script deliberately reads the completed experiment JSON files rather than
copying numbers into plotting code.  It creates four grayscale-safe, vector
PDFs in ``paper_revised``:

* ``fig_kernel_scaling.pdf`` -- packed CKKS kernel scaling;
* ``fig_beir_utility.pdf`` -- official BEIR nDCG@10 comparison;
* ``fig_tradeoff.pdf`` -- held-out million-document payload/utility/latency;
* ``fig_leakage.pdf`` -- candidate-ID, public-index, and score-oracle leakage.

No experimental values are generated, rounded into source data, or inferred.
All values plotted here are loaded from the JSON artefacts produced by the
benchmark harnesses.  The PDF backend keeps text and marks as vector objects
for inclusion in LaTeX.

Usage
-----
``python system/make_revision_figures.py``

Optional paths make the script usable from a copied release bundle as well:
``python system/make_revision_figures.py --results-dir ... --output-dir ...``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


_WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = _WORKSPACE / "results" / "system_revision"
DEFAULT_OUTPUT_DIR = _WORKSPACE / "paper_revised"

_PDF_METADATA = {
    "Creator": "system/make_revision_figures.py",
    "Producer": "Matplotlib",
}

# Each series is distinguishable by both marker and line style.  The gray
# levels merely add a secondary cue, so figures stay interpretable in
# monochrome printing.
METHOD_STYLES: dict[str, dict[str, Any]] = {
    "naive_per_candidate": {
        "label": "Naive per-candidate",
        "color": "0.08",
        "marker": "s",
        "linestyle": "-",
    },
    "block_packed": {
        "label": "Block packed",
        "color": "0.36",
        "marker": "o",
        "linestyle": "-",
    },
    "segmented_score_packed": {
        "label": "Single response",
        "color": "0.64",
        "marker": "^",
        "linestyle": "-",
    },
    "raw_exact": {
        "label": "Raw exact",
        "color": "0.08",
        "marker": "s",
        "linestyle": "-",
    },
    "projected_exact": {
        "label": "Projected exact",
        "color": "0.31",
        "marker": "D",
        "linestyle": "-",
    },
    "pq_only": {
        "label": "PQ only",
        "color": "0.62",
        "marker": "o",
        "linestyle": "-",
    },
    "pq_projected_rerank": {
        "label": "Plaintext rerank",
        "color": "0.48",
        "marker": "^",
        "linestyle": "-",
    },
    "ckks_segmented_score_packed": {
        "label": "CKKS rerank",
        "color": "0.15",
        "marker": "X",
        "linestyle": "-",
    },
    "packed_ckks": {
        "label": "CKKS rerank",
        "color": "0.15",
        "marker": "X",
        "linestyle": "-",
    },
    "projected_float32_return": {
        "label": "Projected FP32",
        "color": "0.31",
        "marker": "D",
        "linestyle": "-",
    },
    "projected_float16_return": {
        "label": "Projected FP16",
        "color": "0.48",
        "marker": "^",
        "linestyle": "-",
    },
    "projected_int8_symmetric_return": {
        "label": "Projected int8",
        "color": "0.68",
        "marker": "v",
        "linestyle": "-",
    },
    "raw_float32_return": {
        "label": "Raw FP32",
        "color": "0.08",
        "marker": "s",
        "linestyle": "-",
    },
    "raw_float16_return": {
        "label": "Raw FP16",
        "color": "0.58",
        "marker": "P",
        "linestyle": "-",
    },
}

BEIR_METHODS = (
    "raw_exact",
    "projected_exact",
    "pq_only",
    "pq_projected_rerank",
    "ckks_segmented_score_packed",
)
BEIR_DATASET_ORDER = (
    ("arguana", "ArguAna"),
    ("fiqa", "FiQA"),
    ("nfcorpus", "NFCorpus"),
    ("scidocs", "SciDocs"),
    ("scifact", "SciFact"),
    ("trec-covid", "TREC-COVID"),
)
TRADEOFF_METHODS = (
    "pq_only",
    "projected_int8_symmetric_return",
    "projected_float16_return",
    "raw_float16_return",
    "projected_float32_return",
    "raw_float32_return",
    "packed_ckks",
)


def _configure_matplotlib() -> None:
    """Apply compact journal-oriented defaults before any figure is created."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "lines.markersize": 5.1,
            "grid.color": "0.86",
            "grid.linewidth": 0.55,
            "grid.linestyle": "-",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required result file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON result file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object at {location}, got {type(value).__name__}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric value at {location}, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected finite numeric value at {location}, got {value!r}")
    return number


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    location = "root"
    for key in keys:
        current = _mapping(current, location)
        if key not in current:
            raise ValueError(f"Missing key {key!r} at {location}")
        current = current[key]
        location = f"{location}.{key}"
    return current


def _metric_summary(
    record: Mapping[str, Any], method: str, metric: str
) -> tuple[float, float, float]:
    summary = _mapping(_nested(record, "metrics", method, metric), f"{method}.{metric}")
    mean = _number(summary.get("mean"), f"{method}.{metric}.mean")
    low = _number(summary.get("ci_low"), f"{method}.{metric}.ci_low")
    high = _number(summary.get("ci_high"), f"{method}.{metric}.ci_high")
    if low > mean or high < mean:
        raise ValueError(f"Invalid confidence interval for {method}.{metric}")
    return mean, low, high


def _style(method: str) -> Mapping[str, Any]:
    try:
        return METHOD_STYLES[method]
    except KeyError as exc:
        raise ValueError(f"No plotting style is registered for {method!r}") from exc


def _save_vector_pdf(figure: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=_PDF_METADATA,
    )
    plt.close(figure)
    return output


def _common_axis_style(axis: plt.Axes) -> None:
    axis.grid(axis="y", zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.65)


def _kernel_series(
    sweep: Mapping[str, Any], *, varying: str, fixed: int
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract x, median, and p95 server times for one completed sweep."""

    points = sweep.get("points")
    if not isinstance(points, list):
        raise ValueError("ckks_sweep.json has no point list")
    selected: list[Mapping[str, Any]] = []
    for point in points:
        point_map = _mapping(point, "ckks_sweep.points[]")
        dimension = int(_number(point_map.get("dimension"), "point.dimension"))
        candidates = int(_number(point_map.get("candidate_count"), "point.candidate_count"))
        if varying == "dimension" and candidates == fixed:
            selected.append(point_map)
        elif varying == "candidate_count" and dimension == fixed:
            selected.append(point_map)
    if not selected:
        raise ValueError(f"No CKKS sweep data for {varying} with fixed value {fixed}")
    selected.sort(key=lambda point: int(point[varying]))
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method in ("naive_per_candidate", "block_packed", "segmented_score_packed"):
        xs: list[float] = []
        medians: list[float] = []
        p95s: list[float] = []
        for point in selected:
            timing = _mapping(
                _nested(point, "methods", method, "server_ms"),
                f"{point.get('id')}.{method}.server_ms",
            )
            xs.append(_number(point[varying], f"{point.get('id')}.{varying}"))
            median = _number(timing.get("median"), f"{method}.server_ms.median")
            p95 = _number(timing.get("p95"), f"{method}.server_ms.p95")
            if p95 < median:
                raise ValueError(f"P95 is smaller than median in {method}")
            medians.append(median)
            p95s.append(p95)
        result[method] = (
            np.asarray(xs, dtype=float),
            np.asarray(medians, dtype=float),
            np.asarray(p95s, dtype=float),
        )
    return result


def make_kernel_scaling(results_dir: Path, output_dir: Path) -> Path:
    """Render median and P95 public-server CKKS kernel scaling curves."""

    sweep = _load_json(results_dir / "ckks_sweep.json")
    dimension_data = _kernel_series(sweep, varying="dimension", fixed=100)
    candidate_data = _kernel_series(sweep, varying="candidate_count", fixed=672)

    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.58), sharey=True)
    panels = (
        (axes[0], dimension_data, "Candidate dimension $d$"),
        (axes[1], candidate_data, "Shortlist size $K$"),
    )
    for axis, series, xlabel in panels:
        for method in ("naive_per_candidate", "block_packed", "segmented_score_packed"):
            xs, medians, p95s = series[method]
            style = _style(method)
            axis.plot(
                xs,
                medians,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                zorder=3,
            )
            axis.plot(
                xs,
                p95s,
                color=style["color"],
                marker=style["marker"],
                linestyle="--",
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=0.85,
                zorder=2,
            )
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        _common_axis_style(axis)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Public-server time (ms; log scale)")
    axes[0].set_ylim(35, 5000)
    axes[0].set_xticks(dimension_data["naive_per_candidate"][0])
    axes[1].set_xticks(candidate_data["naive_per_candidate"][0])

    method_handles = [
        Line2D(
            [],
            [],
            color=_style(method)["color"],
            marker=_style(method)["marker"],
            linestyle="-",
            label=_style(method)["label"],
        )
        for method in ("naive_per_candidate", "block_packed", "segmented_score_packed")
    ]
    summary_handles = [
        Line2D([], [], color="0.2", linestyle="-", label="Median"),
        Line2D(
            [], [], color="0.2", marker="o", markerfacecolor="white", linestyle="--", label="P95"
        ),
    ]
    figure.legend(
        handles=method_handles + summary_handles,
        loc="upper center",
        ncol=5,
        columnspacing=1.15,
        handletextpad=0.42,
        bbox_to_anchor=(0.5, 1.035),
        frameon=False,
    )
    figure.subplots_adjust(left=0.095, right=0.995, bottom=0.24, top=0.76, wspace=0.17)
    return _save_vector_pdf(figure, output_dir / "fig_kernel_scaling.pdf")


def _load_beir_records(results_dir: Path) -> dict[str, Mapping[str, Any]]:
    """Join official IR results with actual encrypted replay results."""

    official = _load_json(results_dir / "graded_ir_official_test.json")
    datasets = _mapping(official.get("datasets"), "graded_ir_official_test.datasets")
    result: dict[str, Mapping[str, Any]] = {}
    for dataset_id, _display_name in BEIR_DATASET_ORDER:
        if dataset_id not in datasets:
            raise ValueError(f"Official BEIR results omit required dataset {dataset_id!r}")
        official_record = _mapping(datasets[dataset_id], f"datasets.{dataset_id}")
        replay = _load_json(results_dir / f"graded_ckks_replay_{dataset_id}_full.json")
        replay_dataset = replay.get("dataset")
        if replay_dataset != dataset_id:
            raise ValueError(
                f"CKKS replay {dataset_id!r} carries incompatible dataset {replay_dataset!r}"
            )
        official_metrics = _mapping(official_record.get("metrics"), f"{dataset_id}.metrics")
        replay_metrics = _mapping(replay.get("metrics"), f"{dataset_id}.replay.metrics")
        merged_metrics: dict[str, Any] = {}
        for method in ("raw_exact", "projected_exact", "pq_only", "pq_projected_rerank"):
            if method not in official_metrics:
                raise ValueError(f"{dataset_id}: missing official method {method}")
            merged_metrics[method] = official_metrics[method]
        if "ckks_segmented_score_packed" not in replay_metrics:
            raise ValueError(f"{dataset_id}: missing encrypted replay method")
        merged_metrics["ckks_segmented_score_packed"] = replay_metrics[
            "ckks_segmented_score_packed"
        ]
        result[dataset_id] = {"metrics": merged_metrics}
    return result


def make_beir_utility(results_dir: Path, output_dir: Path) -> Path:
    """Render the frozen post-exploratory nDCG@10 and candidate Recall@100."""

    split_artifact = _load_json(
        results_dir / "ir_expansion" / "confirmatory_splits.json"
    )
    records = _mapping(split_artifact.get("datasets"), "confirmatory_splits.datasets")
    x_positions = np.arange(len(BEIR_DATASET_ORDER), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(BEIR_METHODS))

    figure, axes = plt.subplots(
        1, 2, figsize=(7.15, 2.72), gridspec_kw={"width_ratios": [1.55, 0.9]}
    )
    axis, recall_axis = axes
    for offset, method in zip(offsets, BEIR_METHODS, strict=True):
        means: list[float] = []
        for dataset_id, _display_name in BEIR_DATASET_ORDER:
            dataset_record = _mapping(records.get(dataset_id), f"splits.{dataset_id}")
            confirmatory = _mapping(
                dataset_record.get("confirmatory"), f"splits.{dataset_id}.confirmatory"
            )
            if method == "ckks_segmented_score_packed":
                actual = _mapping(
                    confirmatory.get("actual_ckks_evaluable_subset"),
                    f"{dataset_id}.actual_ckks_evaluable_subset",
                )
                metrics = _mapping(
                    _mapping(
                        actual.get("metrics_canonical_primary"),
                        f"{dataset_id}.actual.metrics_canonical_primary",
                    ).get(method),
                    f"{dataset_id}.actual.{method}",
                )
            else:
                metrics = _mapping(
                    _mapping(confirmatory.get("ir_metrics"), f"{dataset_id}.ir_metrics").get(
                        method
                    ),
                    f"{dataset_id}.ir_metrics.{method}",
                )
            means.append(
                _number(
                    _mapping(metrics.get("ndcg_at_10"), f"{dataset_id}.{method}.ndcg").get(
                        "mean"
                    ),
                    f"{dataset_id}.{method}.ndcg.mean",
                )
            )
        style = _style(method)
        face = "white" if method == "pq_projected_rerank" else style["color"]
        axis.plot(
            x_positions + offset,
            means,
            marker=style["marker"],
            color=style["color"],
            markerfacecolor=face,
            markeredgecolor=style["color"],
            markersize=5.1,
            markeredgewidth=0.8,
            linestyle="none",
            label=style["label"],
            zorder=3,
        )

    axis.set_ylim(0.10, 0.74)
    axis.set_xlim(-0.55, len(x_positions) - 0.45)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([display for _dataset, display in BEIR_DATASET_ORDER])
    axis.tick_params(axis="x", labelrotation=22)
    axis.set_ylabel("Frozen revision nDCG@10")
    _common_axis_style(axis)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.29),
        ncol=3,
        columnspacing=1.15,
        handletextpad=0.42,
        frameon=False,
    )

    recall_projected: list[float] = []
    recall_pq: list[float] = []
    for dataset_id, _display_name in BEIR_DATASET_ORDER:
        confirmatory = _mapping(
            _mapping(records.get(dataset_id), f"splits.{dataset_id}").get("confirmatory"),
            f"splits.{dataset_id}.confirmatory",
        )
        metrics = _mapping(confirmatory.get("ir_metrics"), f"{dataset_id}.ir_metrics")
        for method, destination in (
            ("projected_exact", recall_projected),
            ("pq_only", recall_pq),
        ):
            method_metrics = _mapping(metrics.get(method), f"{dataset_id}.{method}")
            destination.append(
                _number(
                    _mapping(
                        method_metrics.get("recall_at_100"),
                        f"{dataset_id}.{method}.recall_at_100",
                    ).get("mean"),
                    f"{dataset_id}.{method}.recall_at_100.mean",
                )
            )
    recall_axis.plot(
        x_positions,
        recall_projected,
        "D-",
        color="0.18",
        label="Projected exact",
    )
    recall_axis.plot(
        x_positions,
        recall_pq,
        "o--",
        color="0.58",
        markerfacecolor="white",
        label="PQ candidate set",
    )
    recall_axis.set_xticks(x_positions)
    recall_axis.set_xticklabels([display for _dataset, display in BEIR_DATASET_ORDER])
    recall_axis.tick_params(axis="x", labelrotation=35)
    recall_axis.set_ylabel("Recall@100")
    recall_axis.set_ylim(0.05, 1.0)
    _common_axis_style(recall_axis)
    recall_axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.21), frameon=False)
    figure.subplots_adjust(left=0.08, right=0.995, bottom=0.29, top=0.76, wspace=0.38)
    return _save_vector_pdf(figure, output_dir / "fig_beir_utility.pdf")


def _find_network_scenario(
    network: Mapping[str, Any], *, link_mbps: float, rtt_ms: float
) -> Mapping[str, Any]:
    scenarios = network.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("network_tradeoff.json has no scenario list")
    for scenario in scenarios:
        candidate = _mapping(scenario, "network_tradeoff.scenarios[]")
        if (
            math.isclose(_number(candidate.get("link_mbps"), "scenario.link_mbps"), link_mbps)
            and math.isclose(_number(candidate.get("rtt_ms"), "scenario.rtt_ms"), rtt_ms)
        ):
            return candidate
    raise ValueError(f"No network scenario for {link_mbps:g} Mb/s and RTT {rtt_ms:g} ms")


def _tradeoff_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Load held-out Hit@10, application payload, and modeled latency together."""

    network = _load_json(results_dir / "network_tradeoff.json")
    scenario = _find_network_scenario(network, link_mbps=100.0, rtt_ms=20.0)
    network_methods = _mapping(scenario.get("methods"), "network scenario methods")
    unified = _load_json(results_dir / "unified_ckks_e5base_k672_K100_test.json")
    unified_methods = _mapping(unified.get("methods"), "unified CKKS methods")
    vector = _load_json(results_dir / "vector_return_baselines_e5base_K100_test.json")
    vector_methods = _mapping(vector.get("methods"), "vector-return methods")

    rows: list[dict[str, Any]] = []
    for method in TRADEOFF_METHODS:
        if method not in network_methods:
            raise ValueError(f"Network model omits {method}")
        network_record = _mapping(network_methods[method], f"network.{method}")
        if method == "pq_only":
            source = _mapping(unified_methods.get(method), f"unified.{method}")
        elif method == "packed_ckks":
            source = _mapping(
                unified_methods.get("ckks_projected_shortlist"),
                "unified.ckks_projected_shortlist",
            )
        else:
            source = _mapping(vector_methods.get(method), f"vector.{method}")
        metrics = _mapping(source.get("metrics"), f"{method}.metrics")
        hit = _number(
            _mapping(metrics.get("hit_at_10"), f"{method}.hit_at_10").get("mean"),
            f"{method}.hit_at_10.mean",
        )
        rows.append(
            {
                "method": method,
                "label": _style(method)["label"],
                "hit_at_10": hit,
                "payload_bytes": _number(
                    network_record.get("payload_bytes"), f"network.{method}.payload_bytes"
                ),
                "modeled_p50_ms": _number(
                    network_record.get("modeled_p50_ms"), f"network.{method}.modeled_p50_ms"
                ),
            }
        )
    return rows


def make_tradeoff(results_dir: Path, output_dir: Path) -> Path:
    """Render held-out payload--utility points and modeled 100 Mb/s latency."""

    rows = _tradeoff_rows(results_dir)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.85),
        gridspec_kw={"width_ratios": [1.2, 0.95]},
    )
    payload_axis, latency_axis = axes

    for row in rows:
        method = str(row["method"])
        style = _style(method)
        face = "white" if method in {"pq_only", "projected_float16_return"} else style["color"]
        payload_axis.plot(
            row["payload_bytes"] / 1024.0,
            row["hit_at_10"],
            marker=style["marker"],
            color=style["color"],
            markerfacecolor=face,
            markeredgecolor=style["color"],
            markeredgewidth=0.85,
            linestyle="none",
            zorder=3,
        )
    payload_axis.set_xscale("symlog", linthresh=1.0, linscale=0.65, base=10)
    payload_axis.set_xlim(-0.22, 620)
    payload_axis.set_xticks([0, 1, 10, 100, 1000])
    payload_axis.set_xticklabels(["0", "1", "10", "100", "1000"])
    payload_axis.set_ylim(0.86, 0.928)
    payload_axis.set_xlabel("Online application payload (KiB)\nheld-out 1M corpus")
    payload_axis.set_ylabel("Hit@10")
    _common_axis_style(payload_axis)

    ascending = sorted(rows, key=lambda row: float(row["modeled_p50_ms"]))
    y_positions = np.arange(len(ascending), dtype=float)
    for y, row in zip(y_positions, ascending, strict=True):
        method = str(row["method"])
        style = _style(method)
        face = "white" if method in {"pq_only", "projected_float16_return"} else style["color"]
        latency_axis.plot(
            row["modeled_p50_ms"],
            y,
            marker=style["marker"],
            color=style["color"],
            markerfacecolor=face,
            markeredgecolor=style["color"],
            markeredgewidth=0.85,
            linestyle="none",
            zorder=3,
        )
        latency_axis.annotate(
            f"{row['modeled_p50_ms']:.0f}",
            xy=(row["modeled_p50_ms"], y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=6.6,
            color="0.15",
        )
    latency_axis.set_xscale("log")
    latency_axis.set_xlim(36, 520)
    latency_axis.set_xticks([40, 100, 200, 400])
    latency_axis.set_xticklabels(["40", "100", "200", "400"])
    latency_axis.set_yticks(y_positions)
    latency_axis.set_yticklabels([str(row["label"]) for row in ascending])
    latency_axis.set_xlabel("Modeled p50 latency (ms)\n100 Mb/s, 20-ms RTT")
    _common_axis_style(latency_axis)
    latency_axis.grid(axis="y", visible=False)
    latency_axis.set_ylim(-0.65, len(ascending) - 0.35)

    legend_handles = [
        Line2D(
            [],
            [],
            color=_style(str(row["method"]))["color"],
            marker=_style(str(row["method"]))["marker"],
            markerfacecolor=(
                "white"
                if row["method"] in {"pq_only", "projected_float16_return"}
                else _style(str(row["method"]))["color"]
            ),
            markeredgecolor=_style(str(row["method"]))["color"],
            linestyle="none",
            label=str(row["label"]),
        )
        for row in rows
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        columnspacing=1.1,
        handletextpad=0.38,
        frameon=False,
    )
    figure.subplots_adjust(left=0.095, right=0.995, bottom=0.25, top=0.72, wspace=0.48)
    return _save_vector_pdf(figure, output_dir / "fig_tradeoff.pdf")


def _direct_candidate_labels(axis: plt.Axes, docs: Iterable[Mapping[str, Any]]) -> None:
    for document in docs:
        candidate_id = int(_number(document.get("candidate_id"), "oracle.document.candidate_id"))
        metrics = _mapping(document.get("metrics"), "oracle.document.metrics")
        relative = _number(metrics.get("relative_l2_error"), "oracle.relative_l2_error")
        residual_cosine = (1.0 - _number(metrics.get("cosine_similarity"), "oracle.cosine")) * 1e9
        axis.annotate(
            f"id {candidate_id}",
            xy=(relative * 1e4, residual_cosine),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.6,
            color="0.15",
        )


def make_leakage(results_dir: Path, output_dir: Path) -> Path:
    """Render candidate-ID, public-PQ, and adaptive score-oracle leakage."""

    pq = _load_json(results_dir / "pq_leakage_e5base_k672_sample100k.json")
    reconstruction = _mapping(pq.get("reconstruction"), "PQ leakage reconstruction")
    candidate = _load_json(
        results_dir / "security_expansion" / "candidate_id_million.json"
    )
    candidate_collection = _mapping(
        _mapping(candidate.get("collections"), "candidate collections").get(
            "million_vector_heldout"
        ),
        "candidate collections.million_vector_heldout",
    )
    candidate_methods = _mapping(
        candidate_collection.get("methods"), "candidate collection methods"
    )
    oracle = _load_json(results_dir / "score_oracle_extraction_d672_3docs.json")
    documents = oracle.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("score_oracle_extraction has no documents")
    document_maps = [_mapping(doc, "oracle.documents[]") for doc in documents]

    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.25))
    candidate_axis, pq_axis, oracle_axis, cost_axis = axes.ravel()

    # The provider needs only the disclosed IDs and its own exact candidate
    # rows.  Linkability is stated directly because its AUC is too close to one
    # for a useful shared-scale plot.
    candidate_ks = [20, 50, 100, 200]
    for method, label, marker, color, face in (
        ("centroid", "Set centroid", "s", "0.58", "white"),
        ("log_rank", "Ordered log-rank", "o", "0.10", "0.10"),
        ("ridge_rank_ls", "Ordered ridge", "^", "0.36", "white"),
    ):
        overlaps: list[float] = []
        for k in candidate_ks:
            record = _mapping(candidate_methods.get(f"K{k}:{method}"), f"K{k}:{method}")
            overlap = _mapping(
                record.get("exact_top10_overlap_fraction"),
                f"K{k}:{method}.exact_top10_overlap_fraction",
            )
            overlaps.append(_number(overlap.get("mean"), f"K{k}:{method}.overlap.mean"))
        candidate_axis.plot(
            candidate_ks,
            overlaps,
            marker=marker,
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markeredgewidth=0.8,
            label=label,
        )
    set_aucs = [
        _number(
            _mapping(
                _mapping(candidate_methods[f"K{k}:centroid"], f"K{k}:centroid").get(
                    "linkability"
                ),
                f"K{k}:centroid.linkability",
            ).get("roc_auc"),
            f"K{k}:centroid.linkability.roc_auc",
        )
        for k in candidate_ks
    ]
    candidate_axis.text(
        0.98,
        0.97,
        f"set-only link AUC\n{min(set_aucs):.5f}--{max(set_aucs):.5f}",
        transform=candidate_axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="0.20",
    )
    candidate_axis.set_xlabel("Disclosed shortlist size K")
    candidate_axis.set_ylabel("Exact top-10 overlap")
    candidate_axis.set_xticks(candidate_ks)
    candidate_axis.set_ylim(0.15, 0.43)
    _common_axis_style(candidate_axis)
    candidate_axis.legend(loc="lower left", frameon=False, handletextpad=0.4)

    # Public PQ reconstruction: exact sample means and P05--P95 ranges from
    # the 100,000-document audit.  The lifted point is explicitly a lifted
    # reconstruction, not an unmeasured original-space estimate.
    pq_spaces = (
        ("projected_space", "Projected space", "o", "0.55"),
        ("lifted_original_space", "Lifted original", "D", "0.14"),
    )
    for space_key, label, marker, color in pq_spaces:
        space = _mapping(reconstruction.get(space_key), f"reconstruction.{space_key}")
        cosine_map = _mapping(space.get("cosine"), f"{space_key}.cosine")
        l2_map = _mapping(space.get("relative_l2_error"), f"{space_key}.relative_l2_error")
        cosine_mean = _number(cosine_map.get("mean"), f"{space_key}.cosine.mean")
        cosine_low = _number(cosine_map.get("p05"), f"{space_key}.cosine.p05")
        cosine_high = _number(cosine_map.get("p95"), f"{space_key}.cosine.p95")
        l2_mean = _number(l2_map.get("mean"), f"{space_key}.l2.mean")
        l2_low = _number(l2_map.get("p05"), f"{space_key}.l2.p05")
        l2_high = _number(l2_map.get("p95"), f"{space_key}.l2.p95")
        pq_axis.errorbar(
            l2_mean,
            cosine_mean,
            xerr=np.asarray([[l2_mean - l2_low], [l2_high - l2_mean]]),
            yerr=np.asarray([[cosine_mean - cosine_low], [cosine_high - cosine_mean]]),
            fmt=marker,
            color=color,
            markerfacecolor="white" if marker == "o" else color,
            markeredgecolor=color,
            markeredgewidth=0.85,
            capsize=2.0,
            capthick=0.7,
            elinewidth=0.7,
            label=label,
            zorder=3,
        )
    pq_axis.set_xlabel("Relative L2 reconstruction error\n(mean; P05--P95)")
    pq_axis.set_ylabel("Cosine similarity\n(mean; P05--P95)")
    pq_axis.set_xlim(0.20, 0.59)
    pq_axis.set_ylim(0.79, 0.995)
    _common_axis_style(pq_axis)
    pq_axis.legend(loc="lower left", frameon=False, handletextpad=0.4)

    # Adaptive chosen-query reconstruction.  The transformed ordinate avoids
    # hiding the measured differences behind 0.999999998... tick labels.
    for index, document in enumerate(document_maps):
        metrics = _mapping(document.get("metrics"), "oracle.document.metrics")
        relative = _number(metrics.get("relative_l2_error"), "oracle.relative_l2_error")
        residual_cosine = (1.0 - _number(metrics.get("cosine_similarity"), "oracle.cosine")) * 1e9
        marker = ("s", "^", "D")[index % 3]
        oracle_axis.plot(
            relative * 1e4,
            residual_cosine,
            marker=marker,
            color="0.14",
            markerfacecolor=("0.14" if index != 1 else "white"),
            markeredgewidth=0.85,
            linestyle="none",
            zorder=3,
        )
    _direct_candidate_labels(oracle_axis, document_maps)
    oracle_axis.set_xlabel("Relative L2 error\n(x$10^{-4}$)")
    oracle_axis.set_ylabel("1 - cosine\n(x$10^{-9}$)")
    relative_values = [
        _number(_mapping(doc["metrics"], "metrics").get("relative_l2_error"), "relative")
        * 1e4
        for doc in document_maps
    ]
    residual_values = [
        (1.0 - _number(_mapping(doc["metrics"], "metrics").get("cosine_similarity"), "cosine"))
        * 1e9
        for doc in document_maps
    ]
    _set_padded_limits(oracle_axis, relative_values, residual_values, fraction=0.16)
    _common_axis_style(oracle_axis)

    total_bytes = _number(
        _mapping(oracle.get("application_payload_bytes"), "oracle.application_payload_bytes").get(
            "online_bidirectional_total"
        ),
        "oracle.online_bidirectional_total",
    )
    total_queries = _number(
        _mapping(oracle.get("experiment"), "oracle.experiment").get("actual_online_query_count"),
        "oracle.actual_online_query_count",
    )
    cost_x: list[float] = []
    cost_y: list[float] = []
    for index, document in enumerate(document_maps):
        wall_ms = _number(document.get("wall_ms"), "oracle.document.wall_ms")
        traffic_mib = total_bytes / len(document_maps) / (1024.0**2)
        cost_x.append(traffic_mib)
        cost_y.append(wall_ms / 1000.0)
        marker = ("s", "^", "D")[index % 3]
        cost_axis.plot(
            traffic_mib,
            wall_ms / 1000.0,
            marker=marker,
            color="0.14",
            markerfacecolor=("0.14" if index != 1 else "white"),
            markeredgewidth=0.85,
            linestyle="none",
            zorder=3,
        )
        cost_axis.annotate(
            f"id {int(_number(document.get('candidate_id'), 'candidate_id'))}",
            xy=(traffic_mib, wall_ms / 1000.0),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.6,
            color="0.15",
        )
    queries_per_document = total_queries / len(document_maps)
    cost_axis.set_xlabel("Traffic / document (MiB)")
    cost_axis.set_ylabel("Online wall time / document (s)")
    cost_axis.text(
        0.03,
        0.04,
        f"{queries_per_document:.0f} encrypted queries/document",
        transform=cost_axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=6.35,
        color="0.22",
    )
    _set_padded_limits(cost_axis, cost_x, cost_y, fraction=0.18)
    _common_axis_style(cost_axis)

    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.13,
        top=0.98,
        hspace=0.42,
        wspace=0.34,
    )
    return _save_vector_pdf(figure, output_dir / "fig_leakage.pdf")


def _set_padded_limits(
    axis: plt.Axes, xs: Sequence[float], ys: Sequence[float], *, fraction: float
) -> None:
    """Set non-degenerate data limits with a small, deterministic margin."""

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, max(abs(x_min), 1.0) * 0.02)
    y_span = max(y_max - y_min, max(abs(y_min), 1.0) * 0.02)
    axis.set_xlim(x_min - fraction * x_span, x_max + fraction * x_span)
    axis.set_ylim(y_min - fraction * y_span, y_max + fraction * y_span)


def make_all_figures(results_dir: Path, output_dir: Path) -> list[Path]:
    """Create all four figures and return their stable output paths."""

    _configure_matplotlib()
    return [
        make_kernel_scaling(results_dir, output_dir),
        make_beir_utility(results_dir, output_dir),
        make_tradeoff(results_dir, output_dir),
        make_leakage(results_dir, output_dir),
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = make_all_figures(args.results_dir.resolve(), args.output_dir.resolve())
    summary = [
        {"path": str(path), "bytes": path.stat().st_size, "format": "vector-pdf"}
        for path in outputs
    ]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
