"""Cached, leakage-safe planning stages: split, pilot, confusion, and AutoK.

Only the training partition reaches the pilot, so validation and test labels
cannot influence confusion, AutoK, or any class-to-shell plan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import Dataset

from ..artifacts import hash_array, load_json, save_json, stable_hash
from ..confusion import (
    ConfusionArtifact,
    build_confusion_artifact,
    load_confusion_artifact,
    save_confusion_artifact,
)
from ..data import IndexedDataset, dataset_labels, dataset_metadata, dataset_sample_ids
from ..splits import SplitManifest, load_split_manifest, make_split_manifest, save_split_manifest
from ..train_pilot import OOFPredictions, PilotTrainingConfig, train_pilot_crossfit
from .autok import AutoKResult, load_autok, save_autok, select_shell_count
from .plan import make_radius_settings

DatasetFactory = Callable[[str, bool], Dataset[Any]]
PilotFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class Planning:
    """Every planning artifact, with how each was obtained and where it lives."""

    split: SplitManifest
    oof: OOFPredictions
    confusion: ConfusionArtifact
    autok: AutoKResult
    statuses: Mapping[str, str]
    directories: Mapping[str, Path]

    @property
    def oof_hash(self) -> str:
        return oof_hash(self.oof)


def save_immutable_json(path: Path, payload: Any) -> str:
    """Save once, or prove an existing immutable JSON artifact is identical."""

    if path.exists():
        if stable_hash(load_json(path)) != stable_hash(payload):
            raise ValueError(f"immutable artifact differs from the requested study: {path}")
        return "reused"
    save_json(path, payload)
    return "created"


def partition(dataset: Dataset[Any], manifest: SplitManifest, mask: np.ndarray) -> IndexedDataset:
    """View of ``dataset`` holding exactly the manifest rows selected by ``mask``."""

    source = {sample_id: index for index, sample_id in enumerate(dataset_sample_ids(dataset))}
    requested = manifest.sample_ids[mask]
    try:
        view = IndexedDataset(dataset, [source[str(sample_id)] for sample_id in requested])
    except KeyError as error:
        raise ValueError(f"split sample {error.args[0]!r} is absent from its dataset") from error
    if not np.array_equal(view.sample_ids, requested) or not np.array_equal(
        view.targets, manifest.labels[mask]
    ):
        raise ValueError("dataset IDs or labels do not match the immutable split manifest")
    return view


def prepare_split(
    config: Mapping[str, Any],
    dataset_factory: DatasetFactory,
    official_train: Dataset[Any],
) -> tuple[SplitManifest, str]:
    """Load the immutable manifest, or create it once from official IDs/labels."""

    path = Path(config["data"]["split_manifest"])
    train_ids, train_labels = dataset_sample_ids(official_train), dataset_labels(official_train)
    if path.exists():
        manifest = load_split_manifest(path)
        rows = ~manifest.test_mask
        expected = dict(
            zip(manifest.sample_ids[rows].tolist(), manifest.labels[rows].tolist(), strict=True)
        )
        if dict(zip(train_ids.tolist(), train_labels.tolist(), strict=True)) != expected:
            raise ValueError("official training data do not match the immutable split manifest")
        return manifest, "reused"
    # Only official test IDs/labels are read; no test example enters planning.
    official_test = dataset_factory("test", False)
    train_meta, test_meta = dataset_metadata(official_train), dataset_metadata(official_test)
    if tuple(train_meta.class_ids) != tuple(test_meta.class_ids):
        raise ValueError("official train/test class orders differ")
    manifest = make_split_manifest(
        train_labels,
        sample_ids=train_ids,
        test_labels=dataset_labels(official_test),
        test_sample_ids=dataset_sample_ids(official_test),
        dataset_name=train_meta.name,
        dataset_version=str(config["data"].get("version", "unspecified")),
        train_checksum=train_meta.fingerprint,
        test_checksum=test_meta.fingerprint,
    )
    save_split_manifest(manifest, path)
    return manifest, "created"


def oof_hash(oof: OOFPredictions) -> str:
    return stable_hash(
        {
            name: hash_array(getattr(oof, name))
            for name in (
                "sample_ids",
                "labels",
                "fold_ids",
                "seeds",
                "logits",
                "probabilities",
                "class_ids",
            )
        }
    )


def pilot_cache_key(config: Mapping[str, Any], manifest: SplitManifest) -> str:
    """One OOF artifact per dataset/backbone/inner-activation/pilot configuration."""

    model, data = config["model"], config["data"]
    return stable_hash(
        {
            "split_planning_hash": manifest.planning_hash,
            "dataset": data["dataset"],
            "version": data.get("version"),
            "input_shape": data.get("input_shape"),
            "model": {
                key: model.get(key)
                for key in (
                    "backbone",
                    "backbone_activation",
                    "pretrained",
                    "pretrained_weights",
                    "feature_dim",
                )
            },
            "pilot": config["pilot"],
            "deterministic_oof_transform": True,
        }
    )


def _complete_pair(root: Path, basenames: Sequence[str]) -> bool:
    present = [(root / name).exists() for name in basenames]
    if any(present) and not all(present):
        raise ValueError(f"incomplete cached artifact: {root}")
    return all(present)


def _prepare_pilot(
    config: Mapping[str, Any],
    manifest: SplitManifest,
    train: Dataset[Any],
    train_eval: Dataset[Any],
    factory: PilotFactory,
    cache_root: Path,
) -> tuple[OOFPredictions, str, Path]:
    key = pilot_cache_key(config, manifest)
    root = cache_root / "pilot" / key
    if _complete_pair(root, ("oof_predictions.npz", "oof_metadata.json")):
        oof, status = OOFPredictions.load(root), "reused"
        if oof.metadata.get("cache_key") != key:
            raise ValueError("pilot cache key mismatch")
    else:
        oof = train_pilot_crossfit(
            train,
            manifest.fold_ids[manifest.train_mask],
            factory,
            evaluation_dataset=train_eval,
            labels=manifest.labels[manifest.train_mask],
            sample_ids=manifest.sample_ids[manifest.train_mask],
            class_ids=manifest.class_ids,
            config=PilotTrainingConfig.from_mapping(config["pilot"]),
            split_hash=manifest.planning_hash,
            output_dir=root,
            show_progress=None,
        )
        oof.metadata.update(cache_key=key, planning_hash=manifest.planning_hash)
        oof.save(root)
        status = "created"
    if (
        not np.array_equal(oof.sample_ids, manifest.sample_ids[manifest.train_mask])
        or not np.array_equal(oof.labels, manifest.labels[manifest.train_mask])
        or not np.array_equal(oof.fold_ids, manifest.fold_ids[manifest.train_mask])
        or tuple(oof.seeds.tolist()) != tuple(int(seed) for seed in config["pilot"]["seeds"])
    ):
        raise ValueError("cached OOF predictions do not match the resolved study")
    return oof, status, root


def _prepare_confusion(
    oof: OOFPredictions,
    manifest: SplitManifest,
    class_names: Sequence[str],
    cache_root: Path,
) -> tuple[ConfusionArtifact, str, Path]:
    content_hash = oof_hash(oof)
    key = stable_hash(
        {
            "oof_hash": content_hash,
            "split_planning_hash": manifest.planning_hash,
            "definition": "raw_symmetric_off_diagonal_v2",
        }
    )
    root = cache_root / "confusion" / key
    if _complete_pair(root, ("confusion.npz", "confusion_metadata.json")):
        confusion, status = load_confusion_artifact(root), "reused"
    else:
        confusion = build_confusion_artifact(
            oof.labels,
            oof.mean_probabilities,
            class_ids=oof.class_ids,
            class_names=class_names,
            sample_ids=oof.sample_ids,
            fold_ids=oof.fold_ids,
            pilot_seeds=oof.seeds,
            split_manifest=manifest,
            metadata={"oof_hash": content_hash, "uncalibrated_raw_softmax": True},
        )
        save_confusion_artifact(confusion, root)
        status = "created"
    if confusion.split_hash != manifest.planning_hash:
        raise ValueError("confusion cache references a different split")
    if confusion.metadata.get("oof_hash") != content_hash:
        raise ValueError("confusion cache references different OOF predictions")
    return confusion, status, root


def _prepare_autok(
    config: Mapping[str, Any],
    confusion: ConfusionArtifact,
    oof: OOFPredictions,
    cache_root: Path,
) -> tuple[AutoKResult, str, Path]:
    repeats = int(config["shells"]["auto_k"]["bootstrap_repeats"])
    radius = make_radius_settings(config["shells"]["radius_gap_floor_fraction"])
    key = stable_hash(
        {"confusion_hash": confusion.artifact_hash, "bootstrap_repeats": repeats, "radius": radius}
    )
    root = cache_root / "autok" / key
    names = ("plan.npz", "plan_metadata.json", "autok.npz", "autok_metadata.json")
    if _complete_pair(root, names):
        result = load_autok(root)
        if result.plan.confusion_hash != confusion.artifact_hash:
            raise ValueError("cached AutoK plan references another confusion artifact")
        return result, "reused", root
    result = select_shell_count(
        oof.labels,
        oof.mean_probabilities,
        confusion,
        bootstrap_repeats=repeats,
        radius_settings=radius,
    )
    save_autok(result, root)
    return result, "created", root


def prepare_planning(
    config: Mapping[str, Any],
    manifest: SplitManifest,
    train: Dataset[Any],
    train_eval: Dataset[Any],
    pilot_factory: PilotFactory,
    cache_root: Path,
    class_names: Sequence[str],
) -> Planning:
    """Pilot -> confusion -> AutoK, each cached by the hash of its inputs."""

    oof, pilot_status, pilot_dir = _prepare_pilot(
        config, manifest, train, train_eval, pilot_factory, cache_root
    )
    confusion, confusion_status, confusion_dir = _prepare_confusion(
        oof, manifest, class_names, cache_root
    )
    autok, autok_status, autok_dir = _prepare_autok(config, confusion, oof, cache_root)
    return Planning(
        split=manifest,
        oof=oof,
        confusion=confusion,
        autok=autok,
        statuses={"pilot": pilot_status, "confusion": confusion_status, "autok": autok_status},
        directories={"pilot": pilot_dir, "confusion": confusion_dir, "autok": autok_dir},
    )


__all__ = [
    "DatasetFactory",
    "PilotFactory",
    "Planning",
    "oof_hash",
    "partition",
    "pilot_cache_key",
    "prepare_planning",
    "prepare_split",
    "save_immutable_json",
]
