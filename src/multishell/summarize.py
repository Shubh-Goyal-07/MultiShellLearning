"""Aggregate evaluation artifacts without silently dropping failed seeds.

Results are reported in three separate controlled tables (representation
quality, common affine probe, method-specific native classification) plus a
reference table for unchanged external reproductions, which never enter paired
statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .artifacts import save_json

REPORT_TABLES: dict[str, tuple[str, ...]] = {
    "representation": (
        "raw_euclidean.accuracy",
        "raw_euclidean.macro_f1",
        "raw_euclidean.balanced_accuracy",
        "raw_euclidean.worst_class_accuracy",
        "raw_euclidean.retrieval.recall_at_1",
        "raw_euclidean.retrieval.recall_at_5",
        "raw_euclidean.retrieval.map_at_r",
        "raw_euclidean_selected_k.accuracy",
        "raw_euclidean.cosine_1nn_accuracy",
    ),
    "common_probe": (
        "linear_probe.accuracy",
        "linear_probe.macro_f1",
        "linear_probe.balanced_accuracy",
        "linear_probe.worst_class_accuracy",
    ),
    "native": (
        "shellmetric_native.accuracy",
        "shellmetric_native.shell_selection_accuracy",
        "shellmetric_native.conditional_cosine_knn_accuracy",
        "shellmetric_native.oracle_shell_cosine_knn_accuracy",
        "shellmetric_native_selected_k.accuracy",
        "native_head.accuracy",
    ),
}

DEFAULT_METRICS = (
    *REPORT_TABLES["representation"],
    *REPORT_TABLES["common_probe"],
    *REPORT_TABLES["native"],
    "geometry.global_covariance.effective_rank",
    "geometry.global_covariance.effective_rank_fraction",
    "geometry.within_class_covariance.effective_rank",
    "geometry.spoke.macro_mean_spoke_fraction",
    "geometry.shells.nearest_shell_adherence_rate",
    "trainable_parameter_count",
    "total_parameter_count",
    "training_runtime_seconds",
    "efficiency.embedding_seconds_per_sample",
    "efficiency.raw_knn_seconds_per_query",
    "efficiency.gallery_memory_bytes",
    "evaluation_seconds",
    "evaluation_samples_per_second",
)
DEFAULT_GROUP_BY = ("dataset", "partition", "method", "embedding_dim")


# A small alias surface maps matched-baseline artifacts onto the common raw
# representation columns. ShellMetric artifacts already use canonical names.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "raw_euclidean.accuracy": (
        "raw_euclidean.accuracy",
        "raw_euclidean_1nn_accuracy",
        "accuracy",
        "top1_accuracy",
    ),
    "raw_euclidean.macro_f1": ("raw_euclidean.macro_f1", "macro_f1"),
    "raw_euclidean.balanced_accuracy": (
        "raw_euclidean.balanced_accuracy",
        "balanced_accuracy",
    ),
    "raw_euclidean.worst_class_accuracy": (
        "raw_euclidean.worst_class_accuracy",
        "worst_class_accuracy",
    ),
    "raw_euclidean.retrieval.recall_at_1": (
        "raw_euclidean.retrieval.recall_at_1",
        "retrieval.recall_at_1",
    ),
    "raw_euclidean.retrieval.recall_at_5": (
        "raw_euclidean.retrieval.recall_at_5",
        "retrieval.recall_at_5",
    ),
    "raw_euclidean.retrieval.map_at_r": (
        "raw_euclidean.retrieval.map_at_r",
        "retrieval.map_at_r",
    ),
    "shellmetric_native.accuracy": (
        "shellmetric_native.accuracy",
        "native.accuracy",
        "native_accuracy",
    ),
    "trainable_parameter_count": (
        "trainable_parameter_count",
        "parameter_counts.trainable",
        "parameter_count",
    ),
    "total_parameter_count": (
        "total_parameter_count",
        "parameter_counts.total",
        "parameter_count",
    ),
    "training_runtime_seconds": (
        "training_runtime_seconds",
        "runtime_seconds",
        "training.runtime_seconds",
    ),
    "evaluation_seconds": (
        "evaluation_seconds",
        "timing.evaluation_seconds",
    ),
    "evaluation_samples_per_second": (
        "evaluation_samples_per_second",
        "samples_per_second",
        "timing.samples_per_second",
    ),
}


METRIC_LABELS = {
    "raw_euclidean.accuracy": "Raw Euclidean accuracy",
    "raw_euclidean.macro_f1": "Raw Euclidean macro-F1",
    "raw_euclidean.balanced_accuracy": "Raw Euclidean balanced accuracy",
    "raw_euclidean.worst_class_accuracy": "Raw Euclidean worst-class accuracy",
    "raw_euclidean.retrieval.recall_at_1": "Retrieval Recall@1",
    "raw_euclidean.retrieval.recall_at_5": "Retrieval Recall@5",
    "raw_euclidean.retrieval.map_at_r": "Retrieval mAP@R",
    "raw_euclidean_selected_k.accuracy": "Raw Euclidean selected-k accuracy",
    "raw_euclidean.cosine_1nn_accuracy": "Cosine 1-NN (diagnostic)",
    "shellmetric_native.accuracy": "ShellMetric-native 1-NN accuracy",
    "shellmetric_native.shell_selection_accuracy": "Shell selection accuracy",
    "shellmetric_native.conditional_cosine_knn_accuracy": "Conditional cosine kNN",
    "shellmetric_native.oracle_shell_cosine_knn_accuracy": "Oracle-shell cosine kNN",
    "shellmetric_native_selected_k.accuracy": "ShellMetric-native selected-k accuracy",
    "native_head.accuracy": "Native-head accuracy",
    "linear_probe.accuracy": "Linear-probe accuracy",
    "linear_probe.macro_f1": "Linear-probe macro-F1",
    "linear_probe.balanced_accuracy": "Linear-probe balanced accuracy",
    "linear_probe.worst_class_accuracy": "Linear-probe worst-class accuracy",
    "geometry.global_covariance.effective_rank": "Embedding effective rank",
    "geometry.global_covariance.effective_rank_fraction": "Embedding rank fraction",
    "geometry.within_class_covariance.effective_rank": "Within-class effective rank",
    "geometry.spoke.macro_mean_spoke_fraction": "Mean spoke fraction",
    "trainable_parameter_count": "Trainable parameters",
    "total_parameter_count": "Total parameters",
    "training_runtime_seconds": "Training time (s)",
    "evaluation_seconds": "Evaluation time (s)",
    "evaluation_samples_per_second": "Evaluation samples/s",
}


@dataclass(frozen=True)
class RunRecord:
    path: str
    status: str
    metadata: dict[str, Any]
    metrics: dict[str, float]
    error: str | None = None


def _flatten_numeric(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_numeric(item, name))
        elif isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(item, bool):
            numeric = float(item)
            if math.isfinite(numeric):
                result[name] = numeric
    return result


def _training_metadata_numbers(evaluation_root: Path) -> dict[str, float]:
    """Load efficiency fields stored beside an evaluation partition."""

    candidates = (
        evaluation_root.parent / "training_result.json",
        evaluation_root.parent / "job_result.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"training result must be a JSON object: {path}")
        return _flatten_numeric(raw)
    return {}


def discover_run_records(paths: Iterable[str | os.PathLike[str]]) -> list[RunRecord]:
    """Read evaluation and failure artifacts below one or more paths."""

    metric_files: set[Path] = set()
    failure_files: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            if path.name == "metrics.json":
                metric_files.add(path)
            elif path.name in ("failure.json", "failed.json"):
                failure_files.add(path)
            continue
        direct_metrics = path / "metrics.json"
        direct_failure = path / "failure.json"
        direct_failed = path / "failed.json"
        if direct_metrics.exists():
            metric_files.add(direct_metrics)
        if direct_failure.exists():
            failure_files.add(direct_failure)
        if direct_failed.exists():
            failure_files.add(direct_failed)
        metric_files.update(path.rglob("metrics.json"))
        failure_files.update(path.rglob("failure.json"))
        failure_files.update(path.rglob("failed.json"))

    records: list[RunRecord] = []
    successful_roots: set[Path] = set()
    for metrics_path in sorted(metric_files):
        root = metrics_path.parent
        successful_roots.add(root.resolve())
        metadata_path = root / "evaluation_metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
        if metadata.get("artifact_type") == "job_evaluation":
            continue  # job-store copies; study rows carry the reporting identity
        metrics_raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata.setdefault("run_id", metadata.get("row_id", root.name))
        metadata.setdefault("partition", metadata.get("partition", "unknown"))
        metrics = _flatten_numeric(metrics_raw)
        # Baseline artifacts keep training cost and parameter counts in their
        # evaluation metadata.  Promote only recognized reporting quantities;
        # numeric identity fields such as ``seed`` must never become metrics.
        metadata_numbers = _flatten_numeric(metadata)
        training_numbers = _training_metadata_numbers(root)
        for canonical, aliases in METRIC_ALIASES.items():
            if canonical in metrics:
                continue
            for candidate in aliases:
                if candidate in metrics:
                    metrics[canonical] = metrics[candidate]
                    break
                if candidate in metadata_numbers:
                    metrics[canonical] = metadata_numbers[candidate]
                    break
                if candidate in training_numbers:
                    metrics[canonical] = training_numbers[candidate]
                    break
        records.append(
            RunRecord(path=str(root), status="success", metadata=metadata, metrics=metrics)
        )
    for failure_path in sorted(failure_files):
        if failure_path.parent.resolve() in successful_roots:
            continue
        raw = json.loads(failure_path.read_text(encoding="utf-8"))
        metadata = dict(raw.get("metadata", {}))
        for key in ("dataset", "method", "seed", "embedding_dim", "partition", "run_id", "table"):
            if key in raw:
                metadata.setdefault(key, raw[key])
        metadata.setdefault("run_id", failure_path.parent.name)
        records.append(
            RunRecord(
                path=str(failure_path.parent),
                status="failure",
                metadata=metadata,
                metrics={},
                error=str(raw.get("exception") or raw.get("error") or "unknown failure"),
            )
        )
    return records


def _critical_value(confidence: float, degrees_of_freedom: int) -> float:
    probability = 0.5 + confidence / 2.0
    try:
        from scipy.stats import t

        return float(t.ppf(probability, degrees_of_freedom))
    except Exception:  # pragma: no cover - scipy is an optional reporting dependency
        return float(NormalDist().inv_cdf(probability))


def summarize_values(values: Sequence[float], *, confidence: float = 0.95) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    mean = float(array.mean())
    if count == 1:
        return {"n": 1, "mean": mean, "std": None, "ci_low": None, "ci_high": None}
    std = float(array.std(ddof=1))
    half_width = _critical_value(confidence, count - 1) * std / math.sqrt(count)
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def _metadata_key(record: RunRecord, names: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(record.metadata.get(name, "<missing>")) for name in names)


def _metric_value(record: RunRecord, metric: str) -> float | None:
    """Return a metric through its stable reporting aliases."""

    for candidate in METRIC_ALIASES.get(metric, (metric,)):
        if candidate in record.metrics:
            return record.metrics[candidate]
    return None


def aggregate_records(
    records: Sequence[RunRecord],
    *,
    group_by: Sequence[str] = ("dataset", "method"),
    metrics: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    metric_names = tuple(metrics or DEFAULT_METRICS)
    groups: dict[tuple[str, ...], list[RunRecord]] = {}
    for record in records:
        groups.setdefault(_metadata_key(record, group_by), []).append(record)
    rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        success = [item for item in members if item.status == "success"]
        failed = [item for item in members if item.status != "success"]
        identity = dict(zip(group_by, key, strict=True))
        for metric in metric_names:
            resolved = [_metric_value(item, metric) for item in success]
            values = [value for value in resolved if value is not None]
            row = {
                **identity,
                "metric": metric,
                **summarize_values(values, confidence=confidence),
                "successful_runs": len(success),
                "failed_runs": len(failed),
                "missing_metric_runs": len(success) - len(values),
                "run_paths": [item.path for item in members],
            }
            rows.append(row)
    return rows


def paired_comparisons(
    records: Sequence[RunRecord],
    *,
    reference_method: str,
    metric: str = "raw_euclidean.accuracy",
    method_field: str = "method",
    group_fields: Sequence[str] = ("dataset", "partition", "embedding_dim"),
    pair_field: str = "seed",
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """Method-minus-reference differences over seeds matched within each cell.

    Partitions, datasets, and dimensions are never pooled, and reference-table
    rows (unchanged external reproductions) never enter paired statistics.
    """

    cells: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}
    for record in records:
        value = _metric_value(record, metric)
        if (
            record.status != "success"
            or record.metadata.get("table") == "reference"
            or value is None
        ):
            continue
        method = str(record.metadata.get(method_field, "<missing>"))
        by_method = cells.setdefault(_metadata_key(record, group_fields), {})
        by_method.setdefault(method, {})[str(record.metadata.get(pair_field, "<missing>"))] = value
    result: list[dict[str, Any]] = []
    for cell in sorted(cells):
        reference = cells[cell].get(reference_method)
        if reference is None:
            continue
        for method in sorted(set(cells[cell]) - {reference_method}):
            values = cells[cell][method]
            matched = sorted(set(values) & set(reference))
            result.append(
                {
                    **dict(zip(group_fields, cell, strict=True)),
                    "method": method,
                    "reference_method": reference_method,
                    "metric": metric,
                    **summarize_values(
                        [values[key] - reference[key] for key in matched], confidence=confidence
                    ),
                    "matched_pairs": matched,
                }
            )
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in scalar_keys and not isinstance(value, (list, dict)):
                scalar_keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _format_estimate(mean: Any, std: Any, n: int) -> str:
    """Format an estimate without inventing a standard deviation for n=1."""

    if mean is None or n == 0:
        return ""
    mean_value = float(mean)
    if std is None:
        return f"{mean_value:.6g} (n={n})"
    return f"{mean_value:.6g} ± {float(std):.6g} (n={n})"


def build_results_table(
    aggregate: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Pivot the long aggregate into one publication-oriented row per group.

    Each metric contributes numeric mean/std/CI/count columns plus a formatted
    ``mean_std`` column.  Numeric columns keep the CSV analysis-friendly while
    the formatted column makes it immediately usable in a draft results table.
    """

    indexed = {
        tuple(str(row[name]) for name in group_by) + (str(row["metric"]),): row for row in aggregate
    }
    identities = sorted({tuple(str(row[name]) for name in group_by) for row in aggregate})
    table: list[dict[str, Any]] = []
    for identity in identities:
        row: dict[str, Any] = dict(zip(group_by, identity, strict=True))
        representative = next(
            indexed[identity + (metric,)] for metric in metrics if identity + (metric,) in indexed
        )
        successful = int(representative["successful_runs"])
        failed = int(representative["failed_runs"])
        row.update(
            {
                "total_runs": successful + failed,
                "successful_runs": successful,
                "failed_runs": failed,
            }
        )
        for metric in metrics:
            summary = indexed.get(identity + (metric,))
            if summary is None:  # pragma: no cover - aggregate_records is rectangular
                continue
            prefix = f"{metric}__"
            count = int(summary["n"])
            row[f"{prefix}mean"] = summary["mean"]
            row[f"{prefix}std"] = summary["std"]
            row[f"{prefix}ci_low"] = summary["ci_low"]
            row[f"{prefix}ci_high"] = summary["ci_high"]
            row[f"{prefix}n"] = count
            row[f"{prefix}missing_metric_runs"] = int(summary["missing_metric_runs"])
            row[f"{prefix}mean_std"] = _format_estimate(summary["mean"], summary["std"], count)
        table.append(row)
    return table


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_results_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
    metrics: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [*group_by, "Runs (successful/total; failed)"] + [
        METRIC_LABELS.get(metric, metric) for metric in metrics
    ]
    lines = [
        "| " + " | ".join(_markdown_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [row[name] for name in group_by]
        values.append(f"{row['successful_runs']}/{row['total_runs']}; {row['failed_runs']} failed")
        for metric in metrics:
            values.append(row.get(f"{metric}__mean_std") or "—")
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize_runs(
    paths: Iterable[str | os.PathLike[str]],
    *,
    group_by: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    confidence: float = 0.95,
    reference_method: str | None = None,
    paired_metric: str = "raw_euclidean.accuracy",
    output_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    records = discover_run_records(paths)
    if not records:
        raise ValueError("no evaluation or failure artifacts were found")
    if group_by is None:  # default grouping skips identity fields no record declares
        group_by = tuple(
            name for name in DEFAULT_GROUP_BY if any(name in item.metadata for item in records)
        )
    metric_names = tuple(metrics or DEFAULT_METRICS)
    controlled = [item for item in records if item.metadata.get("table") != "reference"]
    reference = [item for item in records if item.metadata.get("table") == "reference"]
    aggregate = aggregate_records(
        controlled, group_by=group_by, metrics=metric_names, confidence=confidence
    )
    results_table = build_results_table(aggregate, group_by=group_by, metrics=metric_names)
    table_metrics = {
        name: [metric for metric in names if metric in metric_names]
        for name, names in REPORT_TABLES.items()
    }
    tables = {
        name: build_results_table(aggregate, group_by=group_by, metrics=names)
        for name, names in table_metrics.items()
        if aggregate and names
    }
    if reference:
        table_metrics["reference"] = [*REPORT_TABLES["representation"], *REPORT_TABLES["native"]]
        tables["reference"] = build_results_table(
            aggregate_records(reference, group_by=group_by, metrics=table_metrics["reference"]),
            group_by=group_by,
            metrics=table_metrics["reference"],
        )
    comparisons = (
        []
        if reference_method is None
        else paired_comparisons(
            records,
            reference_method=reference_method,
            metric=paired_metric,
            confidence=confidence,
        )
    )
    failures = [
        {"path": item.path, "error": item.error, **item.metadata}
        for item in records
        if item.status != "success"
    ]
    result = {
        "group_by": list(group_by),
        "confidence": confidence,
        "run_count": len(records),
        "success_count": sum(item.status == "success" for item in records),
        "failure_count": sum(item.status != "success" for item in records),
        "aggregate": aggregate,
        "results_table": results_table,
        "tables": tables,
        "paired_comparisons": comparisons,
        "failures": failures,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "summary.json", result, overwrite=True)
        _write_csv(root / "summary.csv", aggregate)
        _write_csv(root / "results_table.csv", results_table)
        _write_results_markdown(
            root / "results_table.md",
            results_table,
            group_by=group_by,
            metrics=metric_names,
        )
        for name, table in tables.items():
            _write_results_markdown(
                root / f"table_{name}.md", table, group_by=group_by, metrics=table_metrics[name]
            )
        _write_csv(root / "paired_comparisons.csv", comparisons)
        _write_csv(root / "failures.csv", failures)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-by", nargs="+", help=f"default: {' '.join(DEFAULT_GROUP_BY)}")
    parser.add_argument("--metric", action="append", dest="metrics")
    parser.add_argument("--reference-method")
    parser.add_argument("--paired-metric", default="raw_euclidean.accuracy")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    result = summarize_runs(
        args.paths,
        group_by=args.group_by,
        metrics=args.metrics,
        confidence=args.confidence,
        reference_method=args.reference_method,
        paired_metric=args.paired_metric,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("run_count", "success_count", "failure_count")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
