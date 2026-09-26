from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")
pytest.importorskip("matplotlib")

from multishell import visualize  # noqa: E402
from multishell.artifacts import save_npz  # noqa: E402
from multishell.visualize import find_rows, project, reduce, visualize_study  # noqa: E402


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

    index = visualize_study(study, reducers=("pca",))
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
    visualize_study(study, methods=["CE"], reducers=("pca",))
    assert index.read_text().count("3D</a>") == 3


@pytest.mark.parametrize(
    ("reducer", "module", "name"), [("umap", "umap", "UMAP"), ("tsne", "sklearn", "t-SNE")]
)
def test_nonlinear_views_are_added_only_above_three_dimensions(
    tmp_path: Path, reducer: str, module: str, name: str
) -> None:
    pytest.importorskip(module)
    study = tmp_path / "study"
    _row(study, "SupCon", 3)
    _row(study, "CE", 16)

    index = visualize_study(study, reducers=("pca", reducer))
    high, low = (study / "visualizations" / "validation" / cell / "seed0" for cell in ("d16", "d3"))
    for figure in (f"CE_3d_{reducer}.html", f"CE_2d_{reducer}.png", f"overview_{reducer}.png"):
        assert (high / figure).stat().st_size > 0
    assert (high / "CE_3d.html").is_file() and (high / "overview.png").is_file()
    assert not list(low.glob(f"*_{reducer}.*"))
    assert f"3D {name}</a>" in index.read_text()


def test_missing_reducer_package_warns_and_keeps_pca(tmp_path: Path, monkeypatch) -> None:
    study = tmp_path / "study"
    _row(study, "CE", 16)
    real = visualize.importlib.util.find_spec
    monkeypatch.setattr(
        visualize.importlib.util,
        "find_spec",
        lambda name, *args: None if name == "umap" else real(name, *args),
    )
    with pytest.warns(UserWarning, match="pip install umap-learn"):
        visualize_study(study, reducers=("pca", "umap"))
    cell = study / "visualizations" / "validation" / "d16" / "seed0"
    assert (cell / "CE_3d.html").is_file() and not (cell / "CE_3d_umap.html").exists()


def test_projection_keeps_native_coordinates_and_origin() -> None:
    values = np.random.default_rng(0).normal(size=(50, 8))
    assert np.array_equal(project(values[:, :2], 3)[:, :2], values[:, :2])
    projected = project(values, 3)
    assert projected.shape == (50, 3)
    assert np.all(np.linalg.norm(projected, axis=1) <= np.linalg.norm(values, axis=1) + 1e-9)
    for reducer in ("pca", "umap", "tsne"):  # low d is always drawn natively
        assert np.array_equal(reduce(values[:, :3], 3, reducer), values[:, :3])
    with pytest.raises(ValueError, match="unknown reducer"):
        reduce(values, 2, "isomap")
