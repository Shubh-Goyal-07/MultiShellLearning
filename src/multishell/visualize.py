"""Embedding-space figures for every evaluated study row.

For each (method, d, seed, partition) row this writes an interactive 3D HTML
scatter and a 2D PNG (projection + per-class radius strip), plus one overview
grid per (partition, d, seed) and an ``index.html`` linking everything.
Embeddings with d > 3 are projected onto their top principal directions about
the origin (uncentered SVD), so norms and shells stay meaningful. ShellMetric
rows also show their learned radii: spheres in native 3D, lines in the strip.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np


@dataclass(frozen=True)
class RowEmbeddings:
    method: str
    embedding_dim: int
    seed: int
    partition: str
    embeddings: np.ndarray
    labels: np.ndarray
    radii: np.ndarray | None = None
    assignment: dict[int, int] | None = None  # class id -> 1-based shell index
    metrics: dict[str, float] | None = None

    @property
    def slug(self) -> str:
        """File-safe method name; keeps the parentheses of names like FixedS(3)."""

        return re.sub(r"[^A-Za-z0-9_.+()-]+", "_", self.method).strip("_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _job_dir(study_dir: Path, meta: dict[str, Any]) -> Path:
    """The recorded job directory, or the study's job store if the run moved machines."""

    recorded = Path(str(meta.get("job_dir", "")))
    if (recorded / meta["partition"] / "embeddings.npz").is_file():
        return recorded
    provenance = study_dir / "provenance.json"
    if provenance.is_file():
        store = Path(_load_json(provenance)["job_store"])
        roots = [store]  # relative stores resolve from the working dir or the repo root
        if not store.is_absolute() and len(study_dir.resolve().parents) > 2:
            roots.append(study_dir.resolve().parents[2] / store)
        for root in roots:
            candidate = root / str(meta["training_job_id"])
            if (candidate / meta["partition"]).is_dir():
                return candidate
    return recorded


def _shells(study_dir: Path, meta: dict[str, Any], metrics: dict[str, Any]):
    radii = (metrics.get("geometry", {}).get("shells") or {}).get("radii")
    plan_path = study_dir / "plans" / str(meta.get("plan_provenance_hash")) / "plan_metadata.json"
    if radii is None or not plan_path.is_file():
        return None, None
    plan = _load_json(plan_path)
    assignment = dict(zip(map(int, plan["class_ids"]), map(int, plan["assignment"]), strict=True))
    return np.asarray(radii, dtype=np.float64), assignment


def _headline(metrics: dict[str, Any]) -> dict[str, float]:
    raw = metrics.get("raw_euclidean", {})
    values = {
        "1-NN acc": raw.get("accuracy"),
        "mAP@R": raw.get("retrieval", {}).get("map_at_r"),
        "probe acc": metrics.get("linear_probe", {}).get("accuracy"),
    }
    return {name: float(value) for name, value in values.items() if value is not None}


def find_rows(
    study_dir: str | Path,
    *,
    partitions: Sequence[str] = ("validation", "test"),
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> Iterable[RowEmbeddings]:
    """Yield every evaluated row of a study output directory that has embeddings."""

    study_dir = Path(study_dir)
    for meta_path in sorted(study_dir.glob("results/*/d*/seed*/*/evaluation_metadata.json")):
        meta = _load_json(meta_path)
        if meta.get("partition") not in partitions:
            continue
        if methods and meta["method"] not in methods:
            continue
        if seeds is not None and int(meta["seed"]) not in seeds:
            continue
        job_dir = _job_dir(study_dir, meta)
        arrays_path = job_dir / meta["partition"] / "embeddings.npz"
        if not arrays_path.is_file():
            continue
        with np.load(arrays_path, allow_pickle=False) as archive:  # skips the train gallery
            embeddings = archive[f"{meta['partition']}_embeddings"]
            labels = archive[f"{meta['partition']}_labels"]
        metrics_path = meta_path.with_name("metrics.json")
        metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
        radii, assignment = _shells(study_dir, meta, metrics)
        yield RowEmbeddings(
            method=str(meta["method"]),
            embedding_dim=int(meta["embedding_dim"]),
            seed=int(meta["seed"]),
            partition=str(meta["partition"]),
            embeddings=np.asarray(embeddings, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int64),
            radii=radii,
            assignment=assignment,
            metrics=_headline(metrics),
        )


def project(embeddings: np.ndarray, components: int) -> np.ndarray:
    """Native coordinates when d <= components, else top directions about the origin."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.shape[1] <= components:
        return np.pad(values, ((0, 0), (0, components - values.shape[1])))
    _, _, directions = np.linalg.svd(values, full_matrices=False)
    return values @ directions[:components].T


def stratified_subsample(labels: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Indices of at most ``max_points`` samples, spread evenly over classes."""

    if labels.size <= max_points:
        return np.arange(labels.size)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, max_points // classes.size)
    chosen = [rng.permutation(np.flatnonzero(labels == label))[:per_class] for label in classes]
    return np.sort(np.concatenate(chosen))


def _class_order(row: RowEmbeddings) -> list[int]:
    classes = sorted(int(label) for label in np.unique(row.labels))
    if row.assignment:
        classes.sort(key=lambda label: (row.assignment.get(label, 0), label))
    return classes


def _colors(labels: np.ndarray) -> dict[int, str]:
    """One fixed color per class id, identical across every plot of a dataset."""

    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    classes = sorted(int(label) for label in np.unique(labels))
    count = len(classes)
    cmap = colormaps["tab10" if count <= 10 else "tab20" if count <= 20 else "turbo"]
    return {
        label: to_hex(cmap(i / max(count - 1, 1) if count > 20 else i))
        for i, label in enumerate(classes)
    }


def _title(row: RowEmbeddings) -> str:
    stats = ", ".join(f"{name} {100 * value:.1f}%" for name, value in (row.metrics or {}).items())
    return f"{row.method} · d={row.embedding_dim} · seed {row.seed} · {row.partition}" + (
        f"<br><sub>{stats}</sub>" if stats else ""
    )


def plot_3d(
    row: RowEmbeddings, path: Path, *, max_points: int = 4000, offline: bool = False
) -> Path:
    """Interactive scatter; ``offline`` embeds plotly.js (~4 MB) instead of loading its CDN."""

    import plotly.graph_objects as go

    keep = stratified_subsample(row.labels, max_points, seed=row.seed)
    points, labels = project(row.embeddings[keep], 3), row.labels[keep]
    norms = np.linalg.norm(row.embeddings[keep], axis=1)
    classes, colors = _class_order(row), _colors(row.labels)
    figure = go.Figure()
    for label in classes:
        mask = labels == label
        shell = f" (shell {row.assignment[label]})" if row.assignment else ""
        figure.add_trace(
            go.Scatter3d(
                x=points[mask, 0],
                y=points[mask, 1],
                z=points[mask, 2],
                mode="markers",
                name=f"class {label}{shell}",
                marker={"size": 2, "color": colors[label], "opacity": 0.75},
                customdata=norms[mask],
                hovertemplate=f"class {label}{shell}<br>‖z‖=%{{customdata:.3f}}<extra></extra>",
                showlegend=len(classes) <= 20,
            )
        )
    if row.radii is not None and row.embedding_dim == 3:
        theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
        for index, radius in enumerate(row.radii, start=1):
            figure.add_trace(
                go.Surface(
                    x=radius * np.cos(theta) * np.sin(phi),
                    y=radius * np.sin(theta) * np.sin(phi),
                    z=radius * np.cos(phi),
                    opacity=0.08,
                    showscale=False,
                    colorscale=[[0, "gray"], [1, "gray"]],
                    name=f"shell {index}: ρ={radius:.3f}",
                    hoverinfo="name",
                )
            )
    axis = "PC" if row.embedding_dim > 3 else "z"
    figure.update_layout(
        title=_title(row),
        scene={
            "aspectmode": "data",
            **{f"{a}axis_title": f"{axis}{i}" for i, a in enumerate("xyz", start=1)},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True if offline else "cdn")
    return path


def _scatter_2d(axis, row: RowEmbeddings, keep: np.ndarray, colors: dict[int, str]) -> None:
    points, labels = project(row.embeddings[keep], 2), row.labels[keep]
    for label, color in colors.items():
        mask = labels == label
        axis.scatter(points[mask, 0], points[mask, 1], s=2, color=color, alpha=0.6, linewidths=0)
    if row.radii is not None and row.embedding_dim <= 2:
        for radius in row.radii:
            axis.add_patch(_circle(radius))
    axis.set_aspect("equal", adjustable="datalim")
    prefix = "PC" if row.embedding_dim > 2 else "z"
    axis.set_xlabel(f"{prefix}1")
    axis.set_ylabel(f"{prefix}2")


def _circle(radius: float):
    from matplotlib.patches import Circle

    return Circle((0, 0), radius, fill=False, linestyle="--", color="gray", linewidth=0.8)


def _figure(**kwargs):
    """A pyplot-free figure, so plotting never changes a notebook's backend."""

    from matplotlib.figure import Figure

    return Figure(**kwargs)


def plot_2d(row: RowEmbeddings, path: Path, *, max_points: int = 4000) -> Path:
    keep = stratified_subsample(row.labels, max_points, seed=row.seed)
    classes, colors = _class_order(row), _colors(row.labels)
    figure = _figure(figsize=(13, 5.5), layout="tight")
    left, right = figure.subplots(1, 2)
    _scatter_2d(left, row, keep, colors)
    left.set_title("2D projection" if row.embedding_dim > 2 else "embedding")

    labels, norms = row.labels[keep], np.linalg.norm(row.embeddings[keep], axis=1)
    rng = np.random.default_rng(0)
    for position, label in enumerate(classes):
        values = norms[labels == label]
        jitter = rng.uniform(-0.3, 0.3, values.size)
        right.scatter(position + jitter, values, s=2, color=colors[label], alpha=0.5)
    if row.radii is not None:
        for index, radius in enumerate(row.radii, start=1):
            right.axhline(radius, color="gray", linestyle="--", linewidth=0.8)
            right.annotate(f"ρ{index}", (len(classes) - 0.5, radius), fontsize=8, va="bottom")
    step = max(1, len(classes) // 25)
    right.set_xticks(range(0, len(classes), step), [str(c) for c in classes[::step]], fontsize=7)
    right.set_xlabel("class" + (" (ordered by assigned shell)" if row.assignment else ""))
    right.set_ylabel("‖z‖")
    right.set_title("radius per class")
    figure.suptitle(_title(row).replace("<br><sub>", "\n").replace("</sub>", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    return path


OVERVIEW_POINTS = 2000


def plot_overview(
    rows: Sequence[RowEmbeddings], path: Path, *, max_points: int = OVERVIEW_POINTS
) -> Path:
    """One 2D-projection panel per method, for a single (partition, d, seed)."""

    from matplotlib.lines import Line2D

    columns = min(4, len(rows))
    grid_rows = -(-len(rows) // columns)
    figure = _figure(figsize=(4.2 * columns + 1.0, 4.2 * grid_rows), layout="constrained")
    axes = figure.subplots(grid_rows, columns, squeeze=False)
    colors = _colors(rows[0].labels)
    for axis, row in zip(axes.flat, rows, strict=False):
        _scatter_2d(axis, row, stratified_subsample(row.labels, max_points, row.seed), colors)
        accuracy = (row.metrics or {}).get("1-NN acc")
        suffix = "" if accuracy is None else f"\n1-NN {100 * accuracy:.1f}%"
        axis.set_title(row.method + suffix, fontsize=9)
    for axis in axes.flat[len(rows) :]:
        axis.axis("off")
    first = rows[0]
    figure.suptitle(f"d={first.embedding_dim} · seed {first.seed} · {first.partition}")
    if len(colors) <= 20:
        handles = [
            Line2D([], [], marker="o", linestyle="", color=color, label=str(label))
            for label, color in colors.items()
        ]
        figure.legend(handles=handles, title="class", loc="outside right center")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    return path


def visualize_study(
    study_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    partitions: Sequence[str] = ("validation", "test"),
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    max_points: int = 4000,
    offline: bool = False,
) -> Path:
    """Write every row's figures, per-cell overviews, and an index; return the index path."""

    study_dir = Path(study_dir)
    output = Path(output_dir) if output_dir else study_dir / "visualizations"
    overviews: dict[Path, list[RowEmbeddings]] = {}
    for row in find_rows(study_dir, partitions=partitions, methods=methods, seeds=seeds):
        cell = output / row.partition / f"d{row.embedding_dim}" / f"seed{row.seed}"
        plot_3d(row, cell / f"{row.slug}_3d.html", max_points=max_points, offline=offline)
        plot_2d(row, cell / f"{row.slug}_2d.png", max_points=max_points)
        keep = stratified_subsample(row.labels, OVERVIEW_POINTS, seed=row.seed)
        thumbnail = replace(row, embeddings=row.embeddings[keep], labels=row.labels[keep])
        overviews.setdefault(cell, []).append(thumbnail)  # small, so memory stays flat
    if not overviews:
        raise FileNotFoundError(f"no evaluated rows with embeddings under {study_dir / 'results'}")
    for cell, thumbnails in overviews.items():
        plot_overview(thumbnails, cell / "overview.png")
    return write_index(output, title=study_dir.name)


def _cell_key(cell: Path) -> tuple[str, int, int]:
    partition, dim, seed = cell.parts[-3:]
    return partition, int(dim.removeprefix("d")), int(seed.removeprefix("seed"))


def _link(output: Path, path: Path, label: str) -> str:
    href = quote(path.relative_to(output).as_posix())
    return f'<a href="{href}">{label}</a>' if path.is_file() else ""


def write_index(output: str | Path, *, title: str) -> Path:
    """Link every figure present under ``output``, including those of earlier runs."""

    output = Path(output)
    cells = [
        cell
        for cell in output.glob("*/d*/seed*")
        if cell.is_dir() and re.fullmatch(r"d\d+/seed\d+", f"{cell.parent.name}/{cell.name}")
    ]
    sections = []
    for cell in sorted(cells, key=_cell_key):
        partition, dim, seed = _cell_key(cell)
        methods = sorted(
            {path.name.removesuffix("_3d.html") for path in cell.glob("*_3d.html")}
            | {path.name.removesuffix("_2d.png") for path in cell.glob("*_2d.png")}
        )
        items = "".join(
            f"<li>{html.escape(method)}: {_link(output, cell / f'{method}_3d.html', '3D')} · "
            f"{_link(output, cell / f'{method}_2d.png', '2D')}</li>"
            for method in methods
        )
        overview = cell / "overview.png"
        image = (
            f'<img src="{quote(overview.relative_to(output).as_posix())}" style="max-width:100%">'
            if overview.is_file()
            else ""
        )
        sections.append(f"<h2>{partition} · d={dim} · seed {seed}</h2>{image}<ul>{items}</ul>")
    index = output / "index.html"
    index.write_text(
        "<!doctype html><meta charset='utf-8'><title>Embedding spaces</title>"
        f"<h1>{html.escape(title)}</h1>" + "".join(sections),
        encoding="utf-8",
    )
    return index


__all__ = [
    "RowEmbeddings",
    "find_rows",
    "plot_2d",
    "plot_3d",
    "plot_overview",
    "project",
    "stratified_subsample",
    "visualize_study",
    "write_index",
]
