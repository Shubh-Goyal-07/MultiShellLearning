from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")
pytest.importorskip("matplotlib")

from multishell.artifacts import save_npz  # noqa: E402
from multishell.visualize import find_rows, project, visualize_study  # noqa: E402


def _row(study: Path, method: str, dim: int, *, plan: str | None = None) -> None:
    rng = np.random.default_rng(dim)
    labels = np.repeat(np.arange(3), 20)
    job = study / "jobs" / f"{method}-{dim}"
    save_npz(
        job / "validation" / "embeddings.npz",
        {
            "validation_embeddings": rng.normal(size=(labels.size, dim)),
            "validation_labels": labels,
            "validation_sample_ids": np.arange(labels.size),
        },
    )
    results = study / "results" / method / f"d{dim}" / "seed0" / "validation"
    results.mkdir(parents=True)
    meta = {
        "method": method,
        "embedding_dim": dim,
        "seed": 0,
        "partition": "validation",
        "job_dir": str(job),
        "training_job_id": job.name,
        "plan_provenance_hash": plan,
    }
    (results / "evaluation_metadata.json").write_text(json.dumps(meta))
    metrics = {"raw_euclidean": {"accuracy": 0.5, "retrieval": {"map_at_r": 0.4}}}
    if plan:
        metrics["geometry"] = {"shells": {"radii": [0.5, 1.2]}}
    (results / "metrics.json").write_text(json.dumps(metrics))


def test_every_method_gets_3d_and_2d_figures_with_shells(tmp_path: Path) -> None:
    study = tmp_path / "study"
    plan = study / "plans" / "p" / "plan_metadata.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"class_ids": [0, 1, 2], "assignment": [1, 2, 2]}))
    _row(study, "ShellMetric-FixedS(2)", 3, plan="p")
    _row(study, "SupCon", 3)
    _row(study, "CE", 16)

    rows = {row.method: row for row in find_rows(study)}
    assert rows["ShellMetric-FixedS(2)"].assignment == {0: 1, 1: 2, 2: 2}
    assert rows["SupCon"].radii is None

    index = visualize_study(study)
    figures = study / "visualizations" / "validation"
    for cell, methods in {"d3": ("ShellMetric-FixedS(2)", "SupCon"), "d16": ("CE",)}.items():
        for method in methods:
            assert (figures / cell / "seed0" / f"{method}_3d.html").stat().st_size > 0
            assert (figures / cell / "seed0" / f"{method}_2d.png").stat().st_size > 0
        assert (figures / cell / "seed0" / "overview.png").is_file()
    assert "shell 2" in (figures / "d3" / "seed0" / "ShellMetric-FixedS(2)_3d.html").read_text()
    assert index.read_text().count("3D</a>") == 3
    assert "ShellMetric-FixedS%282%29_3d.html" in index.read_text()

    # A filtered re-run refreshes only its own figures; the index still links everything.
    visualize_study(study, methods=["CE"])
    assert index.read_text().count("3D</a>") == 3


def test_projection_keeps_native_coordinates_and_origin() -> None:
    values = np.random.default_rng(0).normal(size=(50, 8))
    assert np.array_equal(project(values[:, :2], 3)[:, :2], values[:, :2])
    projected = project(values, 3)
    assert projected.shape == (50, 3)
    assert np.all(np.linalg.norm(projected, axis=1) <= np.linalg.norm(values, axis=1) + 1e-9)
