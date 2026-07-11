"""Validate and tabulate the completed IR-expansion artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "system_revision" / "ir_expansion"
DATASETS = ("arguana", "fiqa", "nfcorpus", "scidocs", "scifact", "trec-covid")
DIMENSIONS = (192, 256, 384, 512, 672, 768)
SPLITS = ("validation", "strict_confirmatory")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard/non-finite JSON constant: {value}")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def all_numbers(value: Any, prefix: str = "$") -> Iterable[tuple[str, float]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from all_numbers(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from all_numbers(child, f"{prefix}[{index}]")
    elif isinstance(value, float):
        yield prefix, value


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def validate(results_dir: Path = RESULTS) -> dict[str, Any]:
    paths = {
        "splits": results_dir / "confirmatory_splits.json",
        "controls": results_dir / "projection_controls.json",
        "pareto": results_dir / "svd_pareto.json",
        "pq": results_dir / "pq_sensitivity.json",
    }
    data = {name: load(path) for name, path in paths.items()}
    splits, controls, pareto, pq = (
        data["splits"], data["controls"], data["pareto"], data["pq"]
    )
    checks: list[dict[str, Any]] = []

    def check(condition: bool, name: str, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"validation failed: {name}: {detail}")

    check(
        set(pareto["datasets"]) == set(DATASETS),
        "Pareto has exactly six datasets",
    )
    check(set(pq["datasets"]) == set(DATASETS), "PQ has exactly six datasets")
    check(bool(pareto.get("completed_utc")), "Pareto completion timestamp exists")
    check(bool(pq.get("completed_utc")), "PQ completion timestamp exists")
    check(bool(controls.get("completed_utc")), "control completion timestamp exists")

    nonfinite = [
        f"{name}:{path}"
        for name, artifact in data.items()
        for path, value in all_numbers(artifact)
        if not math.isfinite(value)
    ]
    check(not nonfinite, "all JSON numbers are finite", nonfinite[:10])

    for dataset in DATASETS:
        validation_ids = set(
            (results_dir / "splits" / f"{dataset}_validation_qids.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        confirmatory_ids = set(
            (results_dir / "splits" / f"{dataset}_confirmatory_qids.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        check(
            validation_ids.isdisjoint(confirmatory_ids),
            f"{dataset}: validation and confirmatory IDs are disjoint",
        )
        expected_validation = splits["datasets"][dataset]["validation"]["count"]
        expected_confirmatory = splits["datasets"][dataset]["confirmatory"][
            "canonical_count"
        ]
        check(
            len(validation_ids) == expected_validation,
            f"{dataset}: validation ID count matches manifest",
        )
        check(
            len(confirmatory_ids) == expected_confirmatory,
            f"{dataset}: confirmatory ID count matches manifest",
        )

        dimensions = pareto["datasets"][dataset]["dimensions"]
        check(
            set(map(int, dimensions)) == set(DIMENSIONS),
            f"{dataset}: complete dimension grid",
        )
        for dimension in DIMENSIONS:
            point = dimensions[str(dimension)]
            check(
                set(point["splits"]) == set(SPLITS),
                f"{dataset} d={dimension}: both split branches",
            )
        for dimension in (384, 672):
            for split in SPLITS:
                from_control = controls["datasets"][dataset][str(dimension)]["svd"][
                    "splits"
                ][split]["metrics_canonical_primary"]
                from_pareto = dimensions[str(dimension)]["splits"][split][
                    "projected_exact_metrics_canonical_primary"
                ]
                check(
                    all(close(from_control[key], from_pareto[key]) for key in from_control),
                    f"{dataset} d={dimension} {split}: control SVD equals Pareto SVD",
                )

        required_points = {f"M{m}_seed42" for m in (32, 48, 84, 96)}
        if dataset in {"fiqa", "trec-covid"}:
            required_points |= {
                f"M{m}_seed{seed}" for m in (84, 96) for seed in (17, 2026)
            }
        check(
            required_points.issubset(pq["datasets"][dataset]),
            f"{dataset}: complete M/seed grid",
        )
        for point_name in required_points:
            point = pq["datasets"][dataset][point_name]
            for split in SPLITS:
                k_points = point["splits"][split]["k"]
                check(
                    set(k_points) == {"20", "50", "100", "200"},
                    f"{dataset} {point_name} {split}: complete K grid",
                )
                candidate_recall = [
                    k_points[str(k)]["metrics_canonical_primary"][
                        "candidate_recall_at_k"
                    ]
                    for k in (20, 50, 100, 200)
                ]
                check(
                    all(a <= b + 1e-12 for a, b in zip(candidate_recall, candidate_recall[1:])),
                    f"{dataset} {point_name} {split}: candidate recall monotone in K",
                    candidate_recall,
                )
                for k in (20, 50, 100):
                    metrics = k_points[str(k)]["metrics_canonical_primary"]
                    check(
                        close(
                            metrics["candidate_recall_at_k"],
                            metrics["reranked_recall_at_100"],
                        ),
                        f"{dataset} {point_name} {split} K={k}: full shortlist Recall@100 invariant",
                    )

        # The d=672, M=84 point is shared byte-for-byte between both studies.
        for split in SPLITS:
            pareto_metrics = dimensions["672"]["splits"][split][
                "pq_projected_rerank"
            ]["metrics_canonical_primary"]
            sensitivity_metrics = pq["datasets"][dataset]["M84_seed42"]["splits"][
                split
            ]["k"]["100"]["metrics_canonical_primary"]
            check(
                all(close(pareto_metrics[key], sensitivity_metrics[key]) for key in pareto_metrics),
                f"{dataset} {split}: shared d672/M84 metrics agree",
            )

    # Validate each unique serialized index once.
    index_records: dict[str, Mapping[str, Any]] = {}
    for dataset in DATASETS:
        for point in pareto["datasets"][dataset]["dimensions"].values():
            index_records[point["pq_index"]["path"]] = point["pq_index"]
        for name, point in pq["datasets"][dataset].items():
            if name.startswith("M"):
                index_records[point["index"]["path"]] = point["index"]
    total_index_bytes = 0
    for raw_path, metadata in index_records.items():
        path = Path(raw_path)
        check(path.exists(), f"index exists: {path.name}", str(path))
        check(
            path.stat().st_size == metadata["serialized_bytes"],
            f"index byte count matches: {path.name}",
        )
        check(sha256(path) == metadata["sha256"], f"index hash matches: {path.name}")
        total_index_bytes += path.stat().st_size

    report = {
        "schema": "ir_expansion.validation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "check_count": len(checks),
        "unique_index_count": len(index_records),
        "unique_index_bytes": total_index_bytes,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "checks": checks,
    }
    atomic_json(results_dir / "validation_report.json", report)
    write_tables(data, results_dir)
    return report


def write_tables(data: Mapping[str, Any], results_dir: Path) -> None:
    splits, controls, pareto, pq = (
        data["splits"], data["controls"], data["pareto"], data["pq"]
    )
    ckks_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        confirmatory = splits["datasets"][dataset]["confirmatory"]
        delta = confirmatory["actual_ckks_evaluable_subset"][
            "ckks_minus_plaintext_ndcg_at_10_canonical_primary"
        ]
        ckks_rows.append(
            {
                "dataset": dataset,
                "canonical_queries": confirmatory["canonical_count"],
                "evaluable_queries": confirmatory["evaluable_count"],
                "mean_delta_ndcg10": delta["mean_delta"],
                "ci95_low": delta["paired_two_sided_ci"][0],
                "ci95_high": delta["paired_two_sided_ci"][1],
                "non_inferior_margin_0.002": delta["non_inferior"],
                "equivalent_margin_0.002": delta[
                    "equivalent_within_symmetric_margin"
                ],
            }
        )
    write_csv(results_dir / "confirmatory_ckks.csv", ckks_rows)

    control_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for dimension in (384, 672):
            for method in ("svd", "random", "coordinate"):
                for split in SPLITS:
                    point = controls["datasets"][dataset][str(dimension)][method][
                        "splits"
                    ][split]
                    metrics = point["metrics_canonical_primary"]
                    control_rows.append(
                        {
                            "dataset": dataset,
                            "dimension": dimension,
                            "method": method,
                            "split": split,
                            "ndcg_at_10": metrics["ndcg_at_10"],
                            "recall_at_100": metrics["recall_at_100"],
                            "mrr_at_10": metrics["mrr_at_10"],
                            "exact_total_ms": point["timing"]["total_ms"],
                        }
                    )
    write_csv(results_dir / "projection_controls.csv", control_rows)

    pareto_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for dimension in DIMENSIONS:
            point = pareto["datasets"][dataset]["dimensions"][str(dimension)]
            for split in SPLITS:
                branch = point["splits"][split]
                exact = branch["projected_exact_metrics_canonical_primary"]
                candidate = branch["pq_projected_rerank"]["metrics_canonical_primary"]
                pareto_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "dimension": dimension,
                        "explained_variance": point[
                            "explained_variance_ratio_on_frozen_fit_sample"
                        ],
                        "projected_exact_ndcg10": exact["ndcg_at_10"],
                        "projected_exact_recall100": exact["recall_at_100"],
                        "pq_rerank_ndcg10": candidate["ndcg_at_10"],
                        "candidate_recall_at_100": candidate["candidate_recall_at_k"],
                        "pq_m": point["pq_index"]["m"],
                        "pq_code_bytes_per_doc": point["pq_index"][
                            "code_bytes_per_document"
                        ],
                        "pq_serialized_bytes": point["pq_index"]["serialized_bytes"],
                    }
                )
    write_csv(results_dir / "svd_pareto.csv", pareto_rows)

    pq_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for point_name, point in pq["datasets"][dataset].items():
            if not point_name.startswith("M"):
                continue
            m = int(point_name.split("_")[0][1:])
            seed = int(point_name.split("seed")[1])
            for split in SPLITS:
                for k in (20, 50, 100, 200):
                    branch = point["splits"][split]["k"][str(k)]
                    metrics = branch["metrics_canonical_primary"]
                    delta = branch["rerank_minus_projected_exact_ndcg_at_10"]
                    pq_rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "m": m,
                            "seed": seed,
                            "k": k,
                            "ndcg_at_10": metrics["ndcg_at_10"],
                            "mrr_at_10": metrics["mrr_at_10"],
                            "candidate_recall_at_k": metrics[
                                "candidate_recall_at_k"
                            ],
                            "reranked_recall_at_100": metrics[
                                "reranked_recall_at_100"
                            ],
                            "delta_ndcg10_vs_projected_exact": delta["mean_delta"],
                            "delta_ci95_low": delta["paired_two_sided_ci"][0],
                            "delta_ci95_high": delta["paired_two_sided_ci"][1],
                            "pq_search_mean_ms": branch[
                                "pq_search_mean_per_query_ms"
                            ],
                            "rerank_p50_ms": branch["rerank_latency_ms"]["p50"],
                            "code_bytes_per_doc": point["index"][
                                "code_bytes_per_document"
                            ],
                            "serialized_index_bytes": point["index"][
                                "serialized_bytes"
                            ],
                        }
                    )
    write_csv(results_dir / "pq_sensitivity.csv", pq_rows)


def main() -> int:
    report = validate()
    print(
        f"IR validation passed: {report['check_count']} checks, "
        f"{report['unique_index_count']} unique indices"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
