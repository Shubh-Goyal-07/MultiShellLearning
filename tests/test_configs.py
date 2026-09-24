from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from multishell.config import ConfigError, load_config, resolve_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "shellmetric"
SUITES = sorted(path for path in CONFIGS.glob("*.yaml"))
TEMPLATES = sorted((CONFIGS / "architecture").glob("*.yaml"))


@pytest.mark.parametrize("path", SUITES, ids=lambda path: path.stem)
def test_every_shipped_suite_resolves_under_the_strict_schema(path: Path) -> None:
    config = load_config(path)
    assert config["model"]["output_bias"] is False
    assert config["sampling"]["classes_per_batch"] is None  # P = min(C, 32)
    if config["model"]["backbone"] == "resnet18" and not config["model"].get("pretrained"):
        assert (config["model"]["backbone_activation"], config["model"]["embedding_head"]) == (
            "relu",
            "linear_no_bias",
        )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.stem)
def test_architecture_templates_require_locked_references(path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = raw["study"]["architecture"]
    for key, value in section.items():
        if str(value).startswith("REPLACE_WITH_STAGE_A_WINNER"):
            section[key] = "silu"
        elif str(value).startswith("REPLACE_WITH_STAGE_B_WINNER"):
            section[key] = "radial_power_gate"
        elif str(value).startswith("REPLACE_WITH"):
            section[key] = "0" * 64
    base = load_config(CONFIGS / "base" / "cifar100.yaml", validate=False)
    raw.pop("include")
    filled = resolve_config(
        {**base, **{k: v for k, v in raw.items() if k != "study"}, "study": raw["study"]}
    )
    assert filled["study"]["architecture"]["stage"] == section["stage"]
