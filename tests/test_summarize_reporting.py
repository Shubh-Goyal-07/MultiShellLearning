from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from multishell.summarize import (
    RunRecord,
    aggregate_records,
    paired_comparisons,
    summarize_runs,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _successful_run(
    root: Path,
    *,
    seed: int,
    accuracy_name: str,
    accuracy: float,
    macro_f1: float | None,
    parameters: int,
    runtime: float,
) -> None:
    metrics: dict[str, float | dict[str, float]] = {
        accuracy_name: accuracy,
        "balanced_accuracy": accuracy - 0.01,
        "evaluation_seconds": 2.0 + seed,
        "evaluation_samples_per_second": 500.0 - seed,
        "retrieval": {"recall_at_1": accuracy - 0.02},
    }
    if macro_f1 is not None:
        metrics["macro_f1"] = macro_f1
    _write_json(root / "metrics.json", metrics)
    _write_json(
        root / "evaluation_metadata.json",
        {
            "dataset": "mnist",
            "method": "arcface",
            "seed": seed,
            "partition": "validation",
            "parameter_counts": {"trainable": parameters, "total": parameters + 10},
            "runtime_seconds": runtime,
        },
    )


def test_summary_writes_long_wide_and_markdown_tables(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _successful_run(
        runs / "seed_0",
        seed=0,
        accuracy_name="accuracy",
        accuracy=0.7,
        macro_f1=0.65,
        parameters=100,
        runtime=10.0,
    )
    _successful_run(
        runs / "seed_1",
        seed=1,
        accuracy_name="top1_accuracy",
        accuracy=0.9,
        macro_f1=None,
        parameters=120,
        runtime=14.0,
    )
    # ``failed.json`` is accepted recursively as well as the historical
    # ``failure.json`` name, and contributes to the same dataset/method group.
    _write_json(
        runs / "seed_2" / "failed.json",
        {
            "dataset": "mnist",
            "method": "arcface",
            "seed": 2,
            "partition": "validation",
            "error": "simulated failure",
        },
    )
    output = tmp_path / "report"

    result = summarize_runs([runs], output_dir=output)

    assert result["run_count"] == 3
    assert result["success_count"] == 2
    assert result["failure_count"] == 1
    long_rows = {row["metric"]: row for row in result["aggregate"]}
    assert long_rows["raw_euclidean.accuracy"]["mean"] == pytest.approx(0.8)
    assert long_rows["raw_euclidean.accuracy"]["n"] == 2
    assert long_rows["raw_euclidean.accuracy"]["failed_runs"] == 1
    assert long_rows["raw_euclidean.macro_f1"]["n"] == 1
    assert long_rows["raw_euclidean.macro_f1"]["missing_metric_runs"] == 1
    assert long_rows["trainable_parameter_count"]["mean"] == pytest.approx(110.0)
    assert long_rows["training_runtime_seconds"]["mean"] == pytest.approx(12.0)

    wide = result["results_table"]
    assert len(wide) == 1
    assert wide[0]["dataset"] == "mnist"
    assert wide[0]["method"] == "arcface"
    assert wide[0]["total_runs"] == 3
    assert wide[0]["successful_runs"] == 2
    assert wide[0]["failed_runs"] == 1
    assert wide[0]["raw_euclidean.accuracy__mean"] == pytest.approx(0.8)
    assert wide[0]["raw_euclidean.accuracy__n"] == 2
    assert "±" in wide[0]["raw_euclidean.accuracy__mean_std"]
    assert wide[0]["raw_euclidean.macro_f1__missing_metric_runs"] == 1

    expected_files = {
        "summary.json",
        "summary.csv",
        "results_table.csv",
        "results_table.md",
        "paired_comparisons.csv",
        "failures.csv",
    }
    assert expected_files <= {path.name for path in output.iterdir()}
    with (output / "results_table.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["raw_euclidean.accuracy__n"] == "2"
    assert csv_rows[0]["training_runtime_seconds__mean"] == "12.0"
    markdown = (output / "results_table.md").read_text(encoding="utf-8")
    assert "Raw Euclidean accuracy" in markdown
    assert "2/3; 1 failed" in markdown
    assert "0.8 ±" in markdown


def test_accuracy_alias_is_used_by_aggregation_and_pairing() -> None:
    records = [
        RunRecord(
            path="reference",
            status="success",
            metadata={"dataset": "mnist", "method": "ce", "seed": 4},
            metrics={"top1_accuracy": 0.8},
        ),
        RunRecord(
            path="candidate",
            status="success",
            metadata={"dataset": "mnist", "method": "arcface", "seed": 4},
            metrics={"accuracy": 0.85},
        ),
    ]

    aggregate = aggregate_records(records, metrics=["raw_euclidean.accuracy"])
    assert {row["method"]: row["mean"] for row in aggregate} == {
        "arcface": pytest.approx(0.85),
        "ce": pytest.approx(0.8),
    }
    comparisons = paired_comparisons(
        records, reference_method="ce", metric="raw_euclidean.accuracy"
    )
    assert comparisons[0]["method"] == "arcface"
    assert comparisons[0]["n"] == 1
    assert comparisons[0]["mean"] == pytest.approx(0.05)


def test_shellmetric_job_cost_is_discovered_beside_validation(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "seed_000" / "validation"
    _write_json(validation / "metrics.json", {"raw_euclidean": {"accuracy": 0.75}})
    _write_json(
        validation / "evaluation_metadata.json",
        {
            "dataset": "mnist",
            "method": "ShellMetric-AutoK",
            "seed": 0,
            "partition": "validation",
        },
    )
    _write_json(
        tmp_path / "seed_000" / "job_result.json",
        {
            "training_runtime_seconds": 123.5,
            "trainable_parameter_count": 234_243,
            "total_parameter_count": 234_243,
        },
    )

    result = summarize_runs([tmp_path], output_dir=tmp_path / "report")
    row = result["results_table"][0]
    assert row["training_runtime_seconds__mean"] == pytest.approx(123.5)
    assert row["trainable_parameter_count__mean"] == pytest.approx(234_243)
    assert row["total_parameter_count__mean"] == pytest.approx(234_243)


def test_partitions_are_never_pooled_and_reference_rows_never_pair() -> None:
    def record(method: str, partition: str, value: float, **extra: str) -> RunRecord:
        metadata = {"dataset": "cifar100", "method": method, "seed": 0, "partition": partition}
        return RunRecord(
            path=f"{method}/{partition}",
            status="success",
            metadata={**metadata, **extra},
            metrics={"raw_euclidean.accuracy": value},
        )

    records = [
        record("ShellMetric-AutoK", "validation", 0.9),
        record("ShellMetric-AutoK", "test", 0.7),
        record("CE", "validation", 0.8),
        record("CE", "test", 0.6),
        record("HyperSpaceX-Official", "validation", 0.99, table="reference"),
    ]
    comparisons = paired_comparisons(records, reference_method="CE")
    assert {(row["partition"], row["method"]): row["n"] for row in comparisons} == {
        ("test", "ShellMetric-AutoK"): 1,
        ("validation", "ShellMetric-AutoK"): 1,
    }
    assert all(row["mean"] == pytest.approx(0.1) for row in comparisons)
    aggregate = aggregate_records(
        records[:4], group_by=("dataset", "partition", "method"), metrics=["raw_euclidean.accuracy"]
    )
    assert {row["n"] for row in aggregate} == {1}
