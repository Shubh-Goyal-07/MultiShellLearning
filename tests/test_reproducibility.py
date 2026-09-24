from __future__ import annotations

from multishell import reproducibility
from multishell.train_pilot import seed_everything as seed_pilot


def test_deterministic_seeding_configures_cublas_workspace(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    metadata = reproducibility.seed_everything(17, deterministic=True)
    assert metadata["cublas_workspace_config"] == ":4096:8"

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seed_pilot(17, deterministic=True)
    assert reproducibility.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_seeding_respects_user_workspace(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    metadata = reproducibility.seed_everything(17, deterministic=True)
    assert metadata["cublas_workspace_config"] == ":16:8"

