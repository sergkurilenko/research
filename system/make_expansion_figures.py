"""Create figures for the IR and provider-scaling expansion experiments.

The script reads completed JSON artifacts and writes only vector PDF figures.
It does not calculate or modify experimental results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = WORKSPACE / "results" / "system_revision"
DEFAULT_OUTPUT = WORKSPACE / "paper_revised"
DATASETS = (
    ("arguana", "ArguAna"),
    ("fiqa", "FiQA-2018"),
    ("nfcorpus", "NFCorpus"),
    ("scidocs", "SciDocs"),
    ("scifact", "SciFact"),
    ("trec-covid", "TREC-COVID"),
)


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.7,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.7,
            "grid.color": "0.86",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected an object at {name}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a number at {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value at {name}")
    return result


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Creator": "system/make_expansion_figures.py",
            "Producer": "Matplotlib",
        },
    )
    plt.close(figure)
    return path


def _axis_style(axis: plt.Axes) -> None:
    axis.grid(True, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.6, length=2.8)


def make_system_scaling(results: Path, output: Path) -> Path:
    data = _load(results / "systems_expansion" / "windows_full.json")
    if data.get("schema") != "systems_expansion.v1":
        raise ValueError("Unexpected systems-expansion schema")
    aggregate = data.get("scaling", {}).get("aggregate")
    if not isinstance(aggregate, list):
        raise ValueError("systems scaling aggregate is missing")
    rows = [_mapping(row, "scaling.aggregate[]") for row in aggregate]
    packed = sorted(
        (row for row in rows if row.get("method") == "segmented_score_packed"),
        key=lambda row: int(row["concurrency"]),
    )
    naive = sorted(
        (row for row in rows if row.get("method") == "naive_per_candidate"),
        key=lambda row: int(row["concurrency"]),
    )
    if len(packed) != 5 or len(naive) != 2:
        raise ValueError("Expected five packed and two naive scaling points")

    def series(source: Sequence[Mapping[str, Any]], field: str, statistic: str) -> list[float]:
        return [
            _number(_mapping(row[field], field).get(statistic), f"{field}.{statistic}")
            for row in source
        ]

    concurrency = [int(row["concurrency"]) for row in packed]
    naive_concurrency = [int(row["concurrency"]) for row in naive]
    packed_qps = series(packed, "throughput_qps_across_restarts", "median")
    naive_qps = series(naive, "throughput_qps_across_restarts", "median")
    packed_p50 = series(packed, "pooled_pipe_roundtrip_ms", "p50")
    packed_p95 = series(packed, "pooled_pipe_roundtrip_ms", "p95")
    packed_rss = [
        _number(row.get("peak_sum_worker_rss_bytes"), "peak_sum_worker_rss_bytes")
        / (1024.0**2)
        for row in packed
    ]

    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
    qps_axis, latency_axis, memory_axis = axes

    qps_axis.plot(concurrency, packed_qps, "o-", color="0.12", label="Block packed")
    qps_axis.plot(
        naive_concurrency,
        naive_qps,
        "s--",
        color="0.58",
        markerfacecolor="white",
        label="Per candidate",
    )
    qps_axis.set_ylabel("Throughput (requests/s)")
    qps_axis.set_xlabel("Concurrent clients")
    qps_axis.legend(frameon=False, loc="upper left")
    _axis_style(qps_axis)

    latency_axis.plot(concurrency, packed_p50, "o-", color="0.18", label="p50")
    latency_axis.plot(
        concurrency,
        packed_p95,
        "^--",
        color="0.55",
        markerfacecolor="white",
        label="p95",
    )
    latency_axis.fill_between(concurrency, packed_p50, packed_p95, color="0.90", zorder=0)
    latency_axis.set_ylabel("Pipe round trip (ms)")
    latency_axis.set_xlabel("Concurrent clients")
    latency_axis.legend(frameon=False, loc="upper left")
    _axis_style(latency_axis)

    memory_axis.plot(concurrency, packed_rss, "D-", color="0.26")
    memory_axis.set_ylabel("Peak summed RSS (MiB)")
    memory_axis.set_xlabel("Concurrent clients")
    _axis_style(memory_axis)

    for axis in axes:
        axis.set_xticks(concurrency)
        axis.set_xlim(0.5, 16.5)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.98, wspace=0.43)
    return _save(figure, output / "fig_system_scaling.pdf")


def _strict_ndcg(point: Mapping[str, Any]) -> tuple[float, float]:
    strict = _mapping(
        _mapping(point.get("splits"), "point.splits").get("strict_confirmatory"),
        "point.splits.strict_confirmatory",
    )
    exact = _mapping(
        strict.get("projected_exact_metrics_canonical_primary"),
        "strict.projected_exact_metrics_canonical_primary",
    )
    pq = _mapping(strict.get("pq_projected_rerank"), "strict.pq_projected_rerank")
    pq_metrics = _mapping(
        pq.get("metrics_canonical_primary"), "pq.metrics_canonical_primary"
    )
    return (
        _number(exact.get("ndcg_at_10"), "projected.ndcg_at_10"),
        _number(pq_metrics.get("ndcg_at_10"), "pq.ndcg_at_10"),
    )


def make_svd_pareto(results: Path, output: Path) -> Path:
    pareto = _load(results / "ir_expansion" / "svd_pareto.json")
    controls = _load(results / "ir_expansion" / "projection_controls.json")
    if pareto.get("schema") != "ir_expansion.svd_pareto.v2":
        raise ValueError("Unexpected SVD-Pareto schema")
    pareto_datasets = _mapping(pareto.get("datasets"), "pareto.datasets")
    control_datasets = _mapping(controls.get("datasets"), "controls.datasets")

    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.15), sharex=True)
    dimensions_expected = [192, 256, 384, 512, 672, 768]
    for axis, (dataset, display) in zip(axes.ravel(), DATASETS, strict=True):
        dataset_record = _mapping(pareto_datasets.get(dataset), f"pareto.{dataset}")
        points = _mapping(dataset_record.get("dimensions"), f"pareto.{dataset}.dimensions")
        dimensions = sorted(int(value) for value in points)
        if dimensions != dimensions_expected:
            raise ValueError(f"Incomplete Pareto dimensions for {dataset}: {dimensions}")
        exact_values: list[float] = []
        pq_values: list[float] = []
        for dimension in dimensions:
            exact, pq_value = _strict_ndcg(
                _mapping(points[str(dimension)], f"{dataset}.{dimension}")
            )
            exact_values.append(exact)
            pq_values.append(pq_value)
        axis.plot(dimensions, exact_values, "o-", color="0.10", label="SVD exact")
        axis.plot(
            dimensions,
            pq_values,
            "^--",
            color="0.52",
            markerfacecolor="white",
            label="SVD + PQ rerank",
        )

        dataset_controls = _mapping(control_datasets.get(dataset), f"controls.{dataset}")
        for method, marker, color, label in (
            ("coordinate", "s", "0.70", "Coordinate"),
            ("random", "x", "0.38", "Random orthoproject"),
        ):
            xs: list[int] = []
            ys: list[float] = []
            for dimension in (384, 672):
                dimension_record = _mapping(
                    dataset_controls.get(str(dimension)), f"controls.{dataset}.{dimension}"
                )
                method_record = _mapping(
                    dimension_record.get(method), f"controls.{dataset}.{dimension}.{method}"
                )
                strict = _mapping(
                    _mapping(method_record.get("splits"), "control.splits").get(
                        "strict_confirmatory"
                    ),
                    "control.strict_confirmatory",
                )
                metrics = _mapping(
                    strict.get("metrics_canonical_primary"), "control.metrics"
                )
                xs.append(dimension)
                ys.append(_number(metrics.get("ndcg_at_10"), "control.ndcg_at_10"))
            axis.plot(
                xs,
                ys,
                linestyle="none",
                marker=marker,
                color=color,
                markerfacecolor="white" if marker == "s" else color,
                markeredgewidth=0.9,
                label=label,
                zorder=3,
            )
        axis.axvline(672, color="0.78", linewidth=0.7, linestyle=":", zorder=0)
        axis.set_title(display, pad=2.5)
        axis.set_xticks(dimensions_expected)
        axis.tick_params(axis="x", labelrotation=30)
        _axis_style(axis)

    for axis in axes[:, 0]:
        axis.set_ylabel("Revision nDCG@10")
    for axis in axes[1, :]:
        axis.set_xlabel("Projected dimension")
    legend = [
        Line2D([], [], color="0.10", marker="o", label="SVD exact"),
        Line2D(
            [], [], color="0.52", marker="^", markerfacecolor="white", linestyle="--",
            label="SVD + PQ rerank"
        ),
        Line2D(
            [], [], color="0.70", marker="s", markerfacecolor="white", linestyle="none",
            label="Coordinate"
        ),
        Line2D([], [], color="0.38", marker="x", linestyle="none", label="Random orthoproject"),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.4,
    )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.91, hspace=0.30, wspace=0.28)
    return _save(figure, output / "fig_svd_pareto.pdf")


def make_all(results: Path, output: Path) -> list[Path]:
    _configure()
    return [make_system_scaling(results, output), make_svd_pareto(results, output)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = make_all(args.results_dir.resolve(), args.output_dir.resolve())
    print(json.dumps([{"path": str(path), "bytes": path.stat().st_size} for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
