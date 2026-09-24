from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fill_architecture_config.py"
spec = importlib.util.spec_from_file_location("fill_architecture_config", SCRIPT)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)


def test_fill_replaces_primary_and_stage_placeholders(tmp_path: Path) -> None:
    primary, stage_a = tmp_path / "primary", tmp_path / "stage_a"
    primary.mkdir(), stage_a.mkdir()
    (primary / "provenance.json").write_text(json.dumps({"locked_loss_hash": "a" * 64}))
    decision = {"selected": "silu", "decision_hash": "b" * 64}
    (stage_a / "architecture_decision.json").write_text(json.dumps(decision))
    template = tmp_path / "stage_b.yaml"
    template.write_text(
        "include: ../base.yaml  # kept\n"
        "activation: REPLACE_WITH_STAGE_A_WINNER\n"
        "activation_decision_hash: REPLACE_WITH_STAGE_A_decision_hash\n"
        "locked_loss_hash: REPLACE_WITH_PRIMARY_locked_loss_hash\n"
    )

    assert (
        fill_module.main([str(template), "--primary", str(primary), "--stage-a", str(stage_a)]) == 0
    )
    filled = (tmp_path / "stage_b.filled.yaml").read_text()
    assert "REPLACE_WITH" not in filled and "# kept" in filled
    assert f"activation: silu\nactivation_decision_hash: {'b' * 64}" in filled
    assert f"locked_loss_hash: {'a' * 64}" in filled

    with pytest.raises(SystemExit, match="--stage-a"):
        fill_module.main([str(template), "--primary", str(primary)])
