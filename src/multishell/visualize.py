"""Embedding-space figures for every evaluated study row.

For each (method, d, seed, partition) row this writes an interactive 3D HTML
scatter and a 2D PNG (2D view + per-class radius strip), plus one overview grid
per (partition, d, seed) and an ``index.html`` linking everything.

Reduction is for display only; saved embeddings and metrics are never touched.
d <= 3 is drawn in native coordinates. For d > 3 the default view projects onto
the top principal directions about the origin (uncentered PCA), so norms and
shells stay readable; UMAP and t-SNE views can be added to show neighbourhood
structure (they do not preserve radii). The radius strip always uses the true
d-dimensional norms. ShellMetric rows also show their learned radii.
"""

from __future__ import annotations

import html
import importlib.util
import json
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

REDUCERS = ("pca", "umap", "tsne")
_REDUCER_NAMES = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}
_REDUCER_PACKAGES = {"umap": ("umap", "umap-learn"), "tsne": ("sklearn", "scikit-learn")}
OVERVIEW_POINTS = 2000
_FIGURE_NAME = re.compile(
    r"^(?P<method>.+)_(?P<kind>3d|2d)(?:_(?P<reducer>umap|tsne))?\.(html|png)$"
)


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


@dataclass(frozen=True)
class Projection:
    """Plot coordinates of a row's sampled points under one reducer."""

    row: RowEmbeddings
    reducer: str
    points: np.ndarray  # [n, 2] or [n, 3]
    norms: np.ndarray  # true d-dimensional norms of the same points

    @property
    def native(self) -> bool:
        return self.row.embedding_dim <= self.points.shape[1]

    @property
    def axis_prefix(self) -> str:
        return "z" if self.native else _REDUCER_NAMES[self.reducer]

    @property
    def view_name(self) -> str:
        return "native coordinates" if self.native else f"{_REDUCER_NAMES[self.reducer]} view"


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


def reduce(embeddings: np.ndarray, components: int, reducer: str = "pca", *, seed: int = 0):
    """Display coordinates: native for d <= components, else PCA, UMAP, or t-SNE."""

    values = np.asarray(embeddings, dtype=np.float64)
    if reducer not in REDUCERS:
        raise ValueError(f"unknown reducer {reducer!r}; choose from {REDUCERS}")
    if reducer == "pca" or values.shape[1] <= components:
        return project(values, components)
    count = values.shape[0]
    if reducer == "umap":
        from umap import UMAP

        model = UMAP(
            n_components=components,
            n_neighbors=min(15, count - 1),
            min_dist=0.1,
            random_state=seed,
            n_jobs=1,
        )
    else:
        from sklearn.manifold import TSNE

        model = TSNE(
            n_components=components,
            perplexity=min(30.0, max(5.0, (count - 1) / 3)),
            init="pca",
            random_state=seed,
        )
    return np.asarray(model.fit_transform(values), dtype=np.float64)


def available_reducers(requested: Sequence[str]) -> tuple[str, ...]:
    """PCA first, then each requested reducer whose package is installed (others warn)."""

    unknown = sorted(set(requested) - set(REDUCERS))
    if unknown:
        raise ValueError(f"unknown reducers {unknown}; choose from {REDUCERS}")
    chosen = ["pca"]
    for reducer in dict.fromkeys(requested):
        if reducer == "pca":
            continue
        module, package = _REDUCER_PACKAGES[reducer]
        if importlib.util.find_spec(module) is None:
            warnings.warn(
                f"{_REDUCER_NAMES[reducer]} views skipped: pip install {package}", stacklevel=2
            )
            continue
        chosen.append(reducer)
    return tuple(chosen)


def stratified_subsample(labels: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Indices of at most ``max_points`` samples, spread evenly over classes."""

    if labels.size <= max_points:
        return np.arange(labels.size)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, max_points // classes.size)
    chosen = [rng.permutation(np.flatnonzero(labels == label))[:per_class] for label in classes]
    return np.sort(np.concatenate(chosen))


def sample_row(row: RowEmbeddings, max_points: int) -> RowEmbeddings:
    keep = stratified_subsample(row.labels, max_points, seed=row.seed)
    return replace(row, embeddings=row.embeddings[keep], labels=row.labels[keep])


def project_row(row: RowEmbeddings, components: int, reducer: str = "pca") -> Projection:
    return Projection(
        row,
        reducer,
        reduce(row.embeddings, components, reducer, seed=row.seed),
        np.linalg.norm(row.embeddings, axis=1),
    )


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


def _title(row: RowEmbeddings, view: str | None = None) -> str:
    stats = ", ".join(f"{name} {100 * value:.1f}%" for name, value in (row.metrics or {}).items())
    header = f"{row.method} · d={row.embedding_dim} · seed {row.seed} · {row.partition}"
    return header + (f" · {view}" if view else "") + (f"<br><sub>{stats}</sub>" if stats else "")


def plot_3d(projection: Projection, path: Path, *, offline: bool = False) -> Path:
    """Interactive scatter; ``offline`` embeds plotly.js (~4 MB) instead of loading its CDN."""

    import plotly.graph_objects as go

    row, points, labels = projection.row, projection.points, projection.row.labels
    classes, colors = _class_order(row), _colors(labels)
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
                customdata=projection.norms[mask],
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
    prefix = projection.axis_prefix
    figure.update_layout(
        title=_title(row, projection.view_name),
        scene={
            "aspectmode": "data",
            **{f"{a}axis_title": f"{prefix}{i}" for i, a in enumerate("xyz", start=1)},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True if offline else "cdn")
    return path


def _scatter_2d(axis, projection: Projection, colors: dict[int, str]) -> None:
    points, labels = projection.points, projection.row.labels
    for label, color in colors.items():
        mask = labels == label
        axis.scatter(points[mask, 0], points[mask, 1], s=2, color=color, alpha=0.6, linewidths=0)
    radii = projection.row.radii
    if radii is not None and projection.native:
        for radius in radii:
            axis.add_patch(_circle(radius))
    if projection.native or projection.reducer == "pca":
        axis.set_aspect("equal", adjustable="datalim")  # UMAP/t-SNE axes carry no scale
    axis.set_xlabel(f"{projection.axis_prefix}1")
    axis.set_ylabel(f"{projection.axis_prefix}2")


def _circle(radius: float):
    from matplotlib.patches import Circle

    return Circle((0, 0), radius, fill=False, linestyle="--", color="gray", linewidth=0.8)


def _figure(**kwargs):
    """A pyplot-free figure, so plotting never changes a notebook's backend."""

    from matplotlib.figure import Figure

    return Figure(**kwargs)


def plot_2d(projection: Projection, path: Path) -> Path:
    """2D view (left) and each class's true ‖z‖ against the learned radii (right)."""

    row = projection.row
    classes, colors = _class_order(row), _colors(row.labels)
    figure = _figure(figsize=(13, 5.5), layout="tight")
    left, right = figure.subplots(1, 2)
    _scatter_2d(left, projection, colors)
    left.set_title(projection.view_name)

    rng = np.random.default_rng(0)
    for position, label in enumerate(classes):
        values = projection.norms[row.labels == label]
        jitter = rng.uniform(-0.3, 0.3, values.size)
        right.scatter(position + jitter, values, s=2, color=colors[label], alpha=0.5)
    if row.radii is not None:
        for index, radius in enumerate(row.radii, start=1):
            right.axhline(radius, color="gray", linestyle="--", linewidth=0.8)
            right.annotate(f"ρ{index}", (len(classes) - 0.5, radius), fontsize=8, va="bottom")
    step = max(1, len(classes) // 25)
    right.set_xticks(range(0, len(classes), step), [str(c) for c in classes[::step]], fontsize=7)
    right.set_xlabel("class" + (" (ordered by assigned shell)" if row.assignment else ""))
    right.set_ylabel("‖z‖ (true d-dimensional norm)")
    right.set_title("radius per class")
    figure.suptitle(_title(row).replace("<br><sub>", "\n").replace("</sub>", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    return path


def plot_overview(projections: Sequence[Projection], path: Path) -> Path:
    """One 2D panel per method, for a single (partition, d, seed) and reducer."""

    from matplotlib.lines import Line2D

    columns = min(4, len(projections))
    grid_rows = -(-len(projections) // columns)
    figure = _figure(figsize=(4.2 * columns + 1.0, 4.2 * grid_rows), layout="constrained")
    axes = figure.subplots(grid_rows, columns, squeeze=False)
    colors = _colors(projections[0].row.labels)
    for axis, projection in zip(axes.flat, projections, strict=False):
        _scatter_2d(axis, projection, colors)
        accuracy = (projection.row.metrics or {}).get("1-NN acc")
        suffix = "" if accuracy is None else f"\n1-NN {100 * accuracy:.1f}%"
        axis.set_title(projection.row.method + suffix, fontsize=9)
    for axis in axes.flat[len(projections) :]:
        axis.axis("off")
    first = projections[0]
    figure.suptitle(
        f"d={first.row.embedding_dim} · seed {first.row.seed} · {first.row.partition}"
        f" · {first.view_name}"
    )
    if len(colors) <= 20:
        handles = [
            Line2D([], [], marker="o", linestyle="", color=color, label=str(label))
            for label, color in colors.items()
        ]
        figure.legend(handles=handles, title="class", loc="outside right center")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    return path


def _thumbnail(projection: Projection) -> Projection:
    """A small copy for the overview: plot coordinates only, no d-dimensional data."""

    row = projection.row
    keep = stratified_subsample(row.labels, OVERVIEW_POINTS, seed=row.seed)
    small = replace(row, embeddings=row.embeddings[keep, :0], labels=row.labels[keep])
    return Projection(small, projection.reducer, projection.points[keep], projection.norms[keep])


def _suffix(reducer: str) -> str:
    return "" if reducer == "pca" else f"_{reducer}"


def visualize_study(
    study_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    partitions: Sequence[str] = ("validation", "test"),
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    max_points: int = 4000,
    offline: bool = False,
    reducers: Sequence[str] = ("pca", "umap"),
) -> Path:
    """Write every row's figures, per-cell overviews, and an index; return the index path.

    PCA/native views are always written; UMAP and t-SNE views are added for d > 3.
    """

    study_dir = Path(study_dir)
    output = Path(output_dir) if output_dir else study_dir / "visualizations"
    chosen = available_reducers(reducers)
    overviews: dict[tuple[Path, str], list[Projection]] = {}
    for row in find_rows(study_dir, partitions=partitions, methods=methods, seeds=seeds):
        cell = output / row.partition / f"d{row.embedding_dim}" / f"seed{row.seed}"
        sample = sample_row(row, max_points)
        for reducer in chosen if row.embedding_dim > 3 else ("pca",):
            suffix = _suffix(reducer)
            plot_3d(
                project_row(sample, 3, reducer),
                cell / f"{row.slug}_3d{suffix}.html",
                offline=offline,
            )
            flat = project_row(sample, 2, reducer)
            plot_2d(flat, cell / f"{row.slug}_2d{suffix}.png")
            overviews.setdefault((cell, reducer), []).append(_thumbnail(flat))
    if not overviews:
        raise FileNotFoundError(f"no evaluated rows with embeddings under {study_dir / 'results'}")
    for (cell, reducer), thumbnails in overviews.items():
        plot_overview(thumbnails, cell / f"overview{_suffix(reducer)}.png")
    return write_index(output, title=study_dir.name)


def _cell_key(cell: Path) -> tuple[str, int, int]:
    partition, dim, seed = cell.parts[-3:]
    return partition, int(dim.removeprefix("d")), int(seed.removeprefix("seed"))


def _href(output: Path, path: Path) -> str:
    return quote(path.relative_to(output).as_posix())


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
        figures: dict[str, list[tuple[int, str, Path]]] = {}
        for path in sorted(cell.iterdir()):
            match = _FIGURE_NAME.match(path.name)
            if match:
                reducer = match["reducer"] or "pca"
                label = match["kind"].upper() + (
                    "" if reducer == "pca" else f" {_REDUCER_NAMES[reducer]}"
                )
                order = (REDUCERS.index(reducer), match["kind"] != "3d")
                figures.setdefault(match["method"], []).append((order, label, path))
        items = "".join(
            f"<li>{html.escape(method)}: "
            + " · ".join(
                f'<a href="{_href(output, path)}">{label}</a>' for _, label, path in sorted(links)
            )
            + "</li>"
            for method, links in sorted(figures.items())
        )
        images = "".join(
            f'<img src="{_href(output, cell / f"overview{_suffix(reducer)}.png")}" '
            'style="max-width:100%">'
            for reducer in REDUCERS
            if (cell / f"overview{_suffix(reducer)}.png").is_file()
        )
        sections.append(f"<h2>{partition} · d={dim} · seed {seed}</h2>{images}<ul>{items}</ul>")
    index = output / "index.html"
    index.write_text(
        "<!doctype html><meta charset='utf-8'><title>Embedding spaces</title>"
        f"<h1>{html.escape(title)}</h1>" + "".join(sections),
        encoding="utf-8",
    )
    return index


__all__ = [
    "REDUCERS",
    "Projection",
    "RowEmbeddings",
    "available_reducers",
    "find_rows",
    "plot_2d",
    "plot_3d",
    "plot_overview",
    "project",
    "project_row",
    "reduce",
    "sample_row",
    "stratified_subsample",
    "visualize_study",
    "write_index",
]
