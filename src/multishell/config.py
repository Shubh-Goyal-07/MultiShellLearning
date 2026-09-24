"""Strict configuration resolution for the ShellMetric experiment path."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import save_json, stable_hash
from .baselines import EXTERNAL_BASELINES
from .shellmetric.architecture import validate_architecture_study
from .shellmetric.plan import expand_fixed_shell_counts
from .shellmetric.study import CONTROLS, expand_baselines, expand_embedding_dims


class ConfigError(ValueError):
    """Raised when a resolved configuration violates the final protocol."""


DEFAULT_CONFIG: dict[str, Any] = {
    "method": "shellmetric",
    "cache_dir": None,
    "experiment": {"name": "shellmetric", "deterministic": True},
    "data": {
        "dataset": "mnist",
        "root": "data",
        "split_manifest": "artifacts/splits/mnist.json",
        "split_seed": 12345,
        "validation_fraction": 0.10,
        "sealed_test": True,
    },
    "model": {
        "backbone": "small_cnn",
        "backbone_activation": "relu",
        "embedding_dim": 3,
        "embedding_head": "linear_no_bias",
        "output_bias": False,
        "pretrained": False,
    },
    "pilot": {
        "loss": "cross_entropy",
        "folds": 3,
        "seeds": [0, 1, 2],
        "batch_size": 128,
        "epochs": 30,
        "optimizer": "adamw",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "momentum": 0.9,
        "scheduler": "cosine",
        "warmup_epochs": 0,
        "label_smoothing": 0.0,
        "temperature_calibration": False,
    },
    "shells": {
        "smax_rule": "ceil_sqrt",
        "capacity_rule": "index_squared",
        "assignment_rule": "difficulty_sorted",
        "radii": "learned_simplex_gaps_ordered_rms",
        "radius_gap_floor_fraction": 0.10,
        "auto_k": {"bootstrap_repeats": 200, "rule": "paired_predictive_risk_one_se"},
    },
    "loss": {
        "name": "shellmetric_pair_margin",
        "positive_margin": 0.50,
        "negative_margin_base": 1.00,
        "confusion_margin_delta": 0.50,
        "negative_weight": 1.00,
        "shell_weight": 1.00,
        "shell_huber_beta": 0.10,
        "epsilon": 1.0e-8,
    },
    # null P/K resolve to P=min(C,32), K=max(4,floor(128/P)).
    "sampling": {"classes_per_batch": None, "samples_per_class": None, "target_batch_size": 128},
    "training": {
        "epochs": 30,
        "optimizer": "adamw",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "momentum": 0.9,
        "warmup_epochs": 0,
        "scheduler": "cosine",
        "mixed_precision": False,
        "workers": 0,
        "keep_epoch_checkpoints": True,
    },
    "baseline": {
        "label_smoothing": 0.1,
        "scale": 30.0,
        "arcface_margin": 0.50,
        "cosface_margin": 0.35,
        "pair_margin": 1.0,
        "supcon_temperature": 0.07,
        "triplet_margin": 0.20,
        "multi_similarity_alpha": 2.0,
        "multi_similarity_beta": 50.0,
        "multi_similarity_base": 0.5,
        "multi_similarity_miner_epsilon": 0.1,
    },
    "study": {
        "fixed_shell_counts": "compact",
        "ensure_one_shell_control": False,
        "run_auto_k": True,
        "controls": [],
        "seeds": [0, 1, 2, 3, 4],
        "embedding_dims": None,
        "baselines": [],
        "external_baselines": [],
        "architecture": None,
        # Rehearsal lifts the canonical-cell restriction on controls and the
        # Section 5.4 stages so the full workflow can be exercised on a cheap
        # dataset (e.g. MNIST). Rehearsal results are never publication results.
        "rehearsal": False,
    },
    "external_pins": {},
    "evaluation": {
        "euclidean_knn_k": [1, 3, 5, 11],
        "retrieval": ["recall_at_1", "recall_at_5", "map_at_r"],
        "shellmetric_native_classifier": "shell_then_cosine_knn",
        "shellmetric_native_k": [1, 3, 5, 11],
        "train_linear_probe_on_frozen_encoder": True,
        "probe_learning_rate": 1.0e-2,
        "probe_weight_decays": [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3],
        "probe_max_epochs": 200,
        "probe_patience": 20,
        "probe_batch_size": None,
        "locked": False,
    },
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_override_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered in {"none", "null"}:
            return None
        return value


def apply_overrides(
    config: Mapping[str, Any], overrides: Mapping[str, Any] | Sequence[str] | None
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    if overrides is None:
        return result
    if isinstance(overrides, Mapping):
        return deep_merge(result, overrides)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        if any(not part for part in parts):
            raise ConfigError(f"invalid override key: {dotted!r}")
        cursor: MutableMapping[str, Any] = result
        for part in parts[:-1]:
            current = cursor.setdefault(part, {})
            if not isinstance(current, MutableMapping):
                raise ConfigError(f"cannot set a child of non-mapping {part!r}")
            cursor = current
        cursor[parts[-1]] = _parse_override_value(raw)
    return result


_FORBIDDEN_KEYS = {
    "assignment",
    "manual_assignment",
    "class_directions",
    "directions",
    "prototypes",
    "prototype",
    "proxy",
    "codebook",
    "fixed_radii",
    "raw_radii",
    "radius_candidates",
    "representation_classifier",
    "primary_decoder",
    "test_derived_planning",
}
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PIN_SCHEMAS: dict[str, dict[str, re.Pattern[str] | None]] = {
    "hyperspacex": {
        "repository": None,
        "commit": _SHA1,
        "environment": None,
        "command": None,
        "checkpoint_rule": None,
    },
    "vit": {
        "provider": None,
        "model_id": None,
        "dependency_version": None,
        "weight_sha256": _SHA256,
        "preprocessing": None,
    },
}
# Matched additionally needs the official reproduction and its deviation manifest.
_OPTIONAL_PIN_FIELDS = {
    "hyperspacex": {"official_reproduction_hash": _SHA256, "matched_deviations": None},
    "vit": {},
}


def _walk_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in _FORBIDDEN_KEYS:
                found.append(path)
            found.extend(_walk_keys(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found.extend(_walk_keys(child, f"{prefix}[{index}]"))
    return found


def _positive_number(mapping: Mapping[str, Any], key: str) -> float:
    value = float(mapping[key])
    if value <= 0:
        raise ConfigError(f"{key} must be positive")
    return value


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value)


def _validate_optimizer(section: Mapping[str, Any], name: str, optimizers: set[str]) -> None:
    epochs = int(section.get("epochs", 0))
    if epochs < 1:
        raise ConfigError(f"{name}.epochs must be positive")
    _positive_number(section, "learning_rate")
    if float(section.get("weight_decay", 0.0)) < 0:
        raise ConfigError(f"{name}.weight_decay must be non-negative")
    if str(section.get("optimizer")) not in optimizers:
        raise ConfigError(f"{name}.optimizer must be one of {sorted(optimizers)}")
    if str(section.get("scheduler")) not in {"cosine", "none"}:
        raise ConfigError(f"{name}.scheduler must be cosine or none")
    if not 0 <= int(section.get("warmup_epochs", 0)) < epochs:
        raise ConfigError(f"{name}.warmup_epochs must lie in [0, epochs)")


def _validate_pins(pins: Mapping[str, Any]) -> None:
    unknown = sorted(set(pins) - set(_PIN_SCHEMAS))
    if unknown:
        raise ConfigError(f"unknown external pins {unknown}; expected {sorted(_PIN_SCHEMAS)}")
    for name, pin in pins.items():
        if not pin:
            continue
        schema = {**_PIN_SCHEMAS[name], **_OPTIONAL_PIN_FIELDS[name]}
        extra = sorted(set(pin) - set(schema))
        missing = sorted(set(_PIN_SCHEMAS[name]) - set(pin))
        if extra or missing:
            raise ConfigError(f"external_pins.{name}: missing {missing}, unexpected {extra}")
        for key, pattern in schema.items():
            value = pin.get(key)
            if value in (None, "") and key in _OPTIONAL_PIN_FIELDS[name]:
                continue
            if value in (None, "") or (pattern is not None and not pattern.match(str(value))):
                raise ConfigError(
                    f"external_pins.{name}.{key} must be an immutable, fully specified value"
                )


def validate_config(config: Mapping[str, Any], *, class_count: int | None = None) -> None:
    if str(config.get("method", "")) != "shellmetric":
        raise ConfigError("this runner accepts only method: shellmetric")
    required = {
        "data",
        "model",
        "pilot",
        "shells",
        "loss",
        "sampling",
        "training",
        "baseline",
        "study",
        "evaluation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ConfigError(f"missing required sections: {missing}")
    forbidden = _walk_keys(config)
    if forbidden:
        raise ConfigError(f"forbidden ShellMetric configuration fields: {forbidden}")

    data = config["data"]
    if int(data.get("split_seed", -1)) != 12345:
        raise ConfigError("data.split_seed must be 12345")
    fraction = float(data.get("validation_fraction", 0.0))
    if not 0.0 < fraction < 1.0 or not bool(data.get("sealed_test", False)):
        raise ConfigError("data must use a sealed test split and a valid validation fraction")

    pilot = config["pilot"]
    if pilot.get("loss") != "cross_entropy" or int(pilot.get("folds", 0)) != 3:
        raise ConfigError("pilot must use cross_entropy with exactly 3 folds")
    if bool(pilot.get("temperature_calibration", False)):
        raise ConfigError("OOF temperature calibration is forbidden")
    if float(pilot.get("label_smoothing", 0.0)) != 0.0:
        raise ConfigError("the preliminary pilot must use ordinary cross-entropy")
    if not _sequence(pilot.get("seeds")):
        raise ConfigError("pilot.seeds must be a non-empty sequence")
    _validate_optimizer(pilot, "pilot", {"adam", "adamw", "sgd"})

    model = config["model"]
    if int(model.get("embedding_dim", 0)) < 1:
        raise ConfigError("model.embedding_dim must be positive")
    if bool(model.get("output_bias", True)):
        raise ConfigError("ShellMetric output_bias must be false")
    if model.get("embedding_head") not in {"linear_no_bias", "radial_power_gate"}:
        raise ConfigError("unsupported ShellMetric embedding_head")
    if model.get("backbone_activation") not in {
        "relu",
        "silu",
        "gelu_exact",
        "channel_radial_silu",
        "native",
    }:
        raise ConfigError("unsupported backbone_activation")

    shells = config["shells"]
    locked = {
        "smax_rule": "ceil_sqrt",
        "capacity_rule": "index_squared",
        "assignment_rule": "difficulty_sorted",
        "radii": "learned_simplex_gaps_ordered_rms",
    }
    for key, expected in locked.items():
        if shells.get(key) != expected:
            raise ConfigError(f"shells.{key} must be {expected!r}")
    if not 0.0 < float(shells.get("radius_gap_floor_fraction", 0.0)) < 1.0:
        raise ConfigError("radius_gap_floor_fraction must lie in (0, 1)")
    auto_k = shells.get("auto_k", {})
    if auto_k.get("rule") != "paired_predictive_risk_one_se":
        raise ConfigError("unsupported AutoK rule")
    if int(auto_k.get("bootstrap_repeats", 0)) < 2:
        raise ConfigError("AutoK bootstrap_repeats must be at least 2")

    loss = config["loss"]
    if loss.get("name") != "shellmetric_pair_margin":
        raise ConfigError("only shellmetric_pair_margin is allowed")
    for key in (
        "positive_margin",
        "negative_margin_base",
        "confusion_margin_delta",
        "negative_weight",
        "shell_weight",
    ):
        if float(loss.get(key, -1.0)) < 0:
            raise ConfigError(f"loss.{key} must be non-negative")
    _positive_number(loss, "shell_huber_beta")
    _positive_number(loss, "epsilon")

    sampling = config["sampling"]
    for key in ("classes_per_batch", "samples_per_class"):
        if sampling.get(key) is not None and int(sampling[key]) < 2:
            raise ConfigError(f"sampling.{key} must be null (automatic) or at least 2")
    if class_count is not None and int(sampling.get("classes_per_batch") or 2) > class_count:
        raise ConfigError(f"sampling.classes_per_batch cannot exceed the {class_count} classes")
    if int(sampling.get("target_batch_size", 0)) < 2:
        raise ConfigError("sampling.target_batch_size must be at least 2")
    training = config["training"]
    _validate_optimizer(training, "training", {"adamw", "sgd"})
    if int(training.get("workers", 0)) < 0:
        raise ConfigError("training.workers must be non-negative")

    baseline = config["baseline"]
    unknown = sorted(set(baseline) - set(DEFAULT_CONFIG["baseline"]))
    if unknown:
        raise ConfigError(f"baseline options {unknown} are not declared in Section 8.3")
    if any(float(value) < 0 for value in baseline.values()):
        raise ConfigError("baseline options must be non-negative")

    study = config["study"]
    controls = list(study.get("controls") or [])
    if len(controls) != len(set(controls)) or not set(controls) <= set(CONTROLS):
        raise ConfigError("study.controls contains an unsupported or duplicate control")
    if not isinstance(study.get("rehearsal", False), bool):
        raise ConfigError("study.rehearsal must be true or false")
    seeds = study.get("seeds")
    if not _sequence(seeds) or len({int(seed) for seed in seeds}) != len(seeds):
        raise ConfigError("study.seeds must be a non-empty sequence of unique integers")
    external = list(study.get("external_baselines") or [])
    if not set(external) <= set(EXTERNAL_BASELINES):
        raise ConfigError(
            f"study.external_baselines must be a subset of {sorted(EXTERNAL_BASELINES)}"
        )
    try:
        expand_embedding_dims(study.get("embedding_dims"), model["embedding_dim"])
        expand_baselines(study.get("baselines"))
        validate_architecture_study(config)
        if class_count is not None:
            expand_fixed_shell_counts(
                study.get("fixed_shell_counts"),
                class_count,
                ensure_one_shell_control=bool(study.get("ensure_one_shell_control", False)),
            )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    _validate_pins(config.get("external_pins") or {})

    evaluation = config["evaluation"]
    expected_k = [1, 3, 5, 11]
    if list(evaluation.get("euclidean_knn_k", [])) != expected_k:
        raise ConfigError(f"evaluation.euclidean_knn_k must equal {expected_k}")
    if list(evaluation.get("shellmetric_native_k", [])) != expected_k:
        raise ConfigError(f"evaluation.shellmetric_native_k must equal {expected_k}")
    if not set(evaluation.get("retrieval", [])) <= {"recall_at_1", "recall_at_5", "map_at_r"}:
        raise ConfigError("evaluation.retrieval supports recall_at_1, recall_at_5, and map_at_r")
    if evaluation.get("shellmetric_native_classifier") != "shell_then_cosine_knn":
        raise ConfigError("ShellMetric native classifier must be shell_then_cosine_knn")
    if not bool(evaluation.get("train_linear_probe_on_frozen_encoder", False)):
        raise ConfigError("the common frozen-encoder affine probe is required")
    _positive_number(evaluation, "probe_learning_rate")
    if (
        int(evaluation.get("probe_max_epochs", 0)) < 1
        or int(evaluation.get("probe_patience", 0)) < 1
    ):
        raise ConfigError("probe_max_epochs and probe_patience must be positive")


def resolve_config(
    config: Mapping[str, Any] | None = None,
    *,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    resolved = apply_overrides(deep_merge(DEFAULT_CONFIG, config or {}), overrides)
    if validate:
        validate_config(resolved)
    return resolved


def _load_with_includes(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source in stack:
        raise ConfigError("configuration include cycle: " + " -> ".join(map(str, (*stack, source))))
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        loaded = json.loads(source.read_text(encoding="utf-8"))
    else:
        import yaml

        loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ConfigError("configuration root must be a mapping")
    local = dict(loaded)
    includes = local.pop("include", [])
    if isinstance(includes, (str, Path)):
        includes = [includes]
    if not isinstance(includes, Sequence):
        raise ConfigError("include must be a path or sequence")
    merged: dict[str, Any] = {}
    for include in includes:
        child = Path(include)
        if not child.is_absolute():
            child = source.parent / child
        merged = deep_merge(merged, _load_with_includes(child, (*stack, source)))
    return deep_merge(merged, local)


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    with_defaults: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    loaded = _load_with_includes(Path(path), ())
    result = (
        resolve_config(loaded, overrides=overrides, validate=False)
        if with_defaults
        else apply_overrides(loaded, overrides)
    )
    if validate:
        validate_config(result)
    return result


def config_hash(config: Mapping[str, Any]) -> str:
    return stable_hash(config)


def save_resolved_config(
    path: str | Path, config: Mapping[str, Any], *, overwrite: bool = False
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() == ".json":
        return save_json(destination, config, overwrite=overwrite)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite resolved config: {destination}")
    import yaml

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8", newline="\n"
    )
    return destination


__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG",
    "apply_overrides",
    "config_hash",
    "deep_merge",
    "load_config",
    "resolve_config",
    "save_resolved_config",
    "validate_config",
]
