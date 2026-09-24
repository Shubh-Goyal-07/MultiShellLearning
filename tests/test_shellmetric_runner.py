from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from torch import nn

from multishell.config import resolve_config
from multishell.data import TensorSampleDataset
from multishell.shellmetric.runner import run_shellmetric_study


def _datasets() -> dict[str, TensorSampleDataset]:
    rng = np.random.default_rng(4)
    centers = np.asarray(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float32,
    )
    result = {}
    for split, count in (("train", 20), ("test", 6)):
        labels = np.repeat(np.arange(3), count)
        inputs = centers[labels] + rng.normal(0.0, 0.25, size=(labels.size, 4))
        result[split] = TensorSampleDataset(
            inputs.astype(np.float32),
            labels,
            sample_ids=[f"fixture:{split}:{index:04d}" for index in range(labels.size)],
            dataset_name="fixture",
            split=split,
        )
    return result


def _config(root: Path) -> dict:
    return resolve_config(
        {
            "experiment": {"name": "integration"},
            "data": {
                "dataset": "mnist",
                "split_manifest": str(root / "split.json"),
            },
            "pilot": {"seeds": [0], "epochs": 1},
            "shells": {"auto_k": {"bootstrap_repeats": 2}},
            "sampling": {
                "classes_per_batch": 3,
                "samples_per_class": 4,
                "target_batch_size": 12,
            },
            "training": {"epochs": 1},
            "study": {
                "fixed_shell_counts": [1, 2],
                "ensure_one_shell_control": False,
                "controls": [],
                "seeds": [0],
            },
            "evaluation": {
                "probe_weight_decays": [0.0],
                "probe_max_epochs": 2,
                "probe_patience": 1,
                "probe_batch_size": 64,
            },
        }
    )


def test_restartable_synthetic_end_to_end_and_semantic_deduplication(tmp_path: Path) -> None:
    datasets = _datasets()

    def dataset_factory(split: str, training: bool):
        del training
        return datasets[split]

    def pilot_factory(class_count: int) -> nn.Module:
        return nn.Linear(4, class_count)

    def encoder_factory(spec) -> nn.Module:
        return nn.Linear(4, spec.embedding_dim, bias=False)

    config = _config(tmp_path)
    dry = run_shellmetric_study(
        config,
        tmp_path / "run",
        dry_run=True,
        dataset_factory=dataset_factory,
        pilot_factory=pilot_factory,
        encoder_factory=encoder_factory,
    )
    assert len(dry.manifest.reporting_rows) == 3
    assert len(dry.manifest.training_jobs) == 2
    assert any(row.aliases for row in dry.manifest.reporting_rows)

    completed = run_shellmetric_study(
        config,
        tmp_path / "run",
        resume=True,
        dataset_factory=dataset_factory,
        pilot_factory=pilot_factory,
        encoder_factory=encoder_factory,
    )
    assert {job.status for job in completed.jobs} == {"completed"}
    for job in completed.jobs:
        assert (job.output_dir / "checkpoints" / "best.pt").is_file()
        assert (job.output_dir / "validation" / "raw_metrics.json").is_file()
        assert (job.output_dir / "validation" / "native_metrics.json").is_file()
        assert (job.output_dir / "validation" / "probe_metrics.json").is_file()
        assert (job.output_dir / "validation" / "geometry.json").is_file()

    reused = run_shellmetric_study(
        config,
        tmp_path / "run",
        resume=True,
        dataset_factory=dataset_factory,
        pilot_factory=pilot_factory,
        encoder_factory=encoder_factory,
    )
    assert {job.status for job in reused.jobs} == {"reused"}


def test_test_access_requires_a_locked_configuration(tmp_path: Path) -> None:
    datasets = _datasets()
    config = _config(tmp_path)
    try:
        run_shellmetric_study(
            config,
            tmp_path / "guard",
            dry_run=True,
            allow_test=True,
            dataset_factory=lambda split, training: datasets[split],
            pilot_factory=lambda class_count: nn.Linear(4, class_count),
        )
    except ValueError as error:
        assert "evaluation.locked" in str(error)
    else:  # pragma: no cover - explicit failure gives a clearer diagnostic
        raise AssertionError("unlocked test access was accepted")


def _mlp_encoder(spec) -> nn.Module:
    """Tiny encoder honoring the job's inner activation and output map."""

    from multishell.shellmetric.activations import build_activation
    from multishell.shellmetric.heads import build_embedding_head

    return nn.Sequential(
        nn.Linear(4, 16),
        build_activation(spec.backbone_activation),
        build_embedding_head(spec.embedding_head, 16, spec.embedding_dim),
    )


def _canonical(root: Path, **study) -> dict:
    """The CIFAR-100/ResNet-18/3D cell by name; tiny factories stand in for real data."""

    config = _config(root)
    config["data"]["dataset"] = "cifar100"
    config["model"]["backbone"] = "resnet18"
    config["cache_dir"] = str(root / "cache")
    config["study"].update(study)
    return resolve_config(config)


def _run(config: dict, output: Path, **options):
    datasets = _datasets()
    return run_shellmetric_study(
        config,
        output,
        dataset_factory=lambda split, training: datasets[split],
        pilot_factory=lambda class_count: nn.Linear(4, class_count),
        encoder_factory=_mlp_encoder,
        device="cpu",
        **options,
    )


def test_canonical_suite_with_controls_baselines_and_locked_test(tmp_path: Path) -> None:
    import json

    from multishell.summarize import summarize_runs
    from multishell.training import load_checkpoint

    config = _canonical(
        tmp_path,
        fixed_shell_counts="compact",
        ensure_one_shell_control=True,
        controls=["shuffled_confusion", "no_shell_loss"],
        baselines=["cross_entropy", "supcon", "batch_hard_triplet"],
    )
    config["evaluation"]["locked"] = True
    run = _run(config, tmp_path / "suite", allow_test=True)
    assert run.jobs and {job.status for job in run.jobs} == {"completed"}
    rows = {row.method: row for row in run.manifest.reporting_rows}
    assert {"ShellMetric-AutoK-ShuffledConfusion", "ShellMetric-AutoK-NoShellLoss", "CE"} <= set(
        rows
    )
    assert [item["name"] for item in run.manifest.comparisons] == [
        "multi_shell_vs_one_shell",
        "shuffled_confusion",
        "no_shell_loss",
    ]

    def metrics(method: str, partition: str = "validation") -> dict:
        path = tmp_path / "suite" / "results" / method / "d3" / "seed0" / partition / "metrics.json"
        return json.loads(path.read_text(encoding="utf-8"))

    assert (
        metrics("ShellMetric-AutoK-NoShellLoss")["shellmetric_native"]["status"] == "not_applicable"
    )
    assert metrics("CE")["shellmetric_native"]["status"] == "not_applicable"
    assert metrics("CE")["native_head"]["selection_rule"] == "validation_native_top1"
    assert metrics("SupCon")["native_head"] == {"status": "not_applicable"}
    autok_test = metrics("ShellMetric-AutoK", "test")
    assert autok_test["raw_euclidean"]["sample_count"] == 18
    assert autok_test["shellmetric_native"]["k"] == 1
    assert autok_test["shellmetric_native"]["radius_hash"]

    store = tmp_path / "cache" / "jobs"
    no_shell = load_checkpoint(
        store / rows["ShellMetric-AutoK-NoShellLoss"].training_job_id / "checkpoints" / "best.pt"
    )
    assert no_shell["radii"] is None
    history = json.loads(
        (store / rows["ShellMetric-AutoK-NoShellLoss"].training_job_id / "history.json").read_text()
    )
    assert all(record["shell"] == 0.0 for record in history)
    autok = load_checkpoint(
        store / rows["ShellMetric-AutoK"].training_job_id / "checkpoints" / "best.pt"
    )
    assert set(autok) >= {"encoder", "radii"} and not {"classifier", "decoder", "prototypes"} & set(
        autok
    )
    shapes = [tuple(value.shape) for value in autok["encoder"].values()]
    assert (3, 3) not in shapes  # no [C, d] class-point tensor anywhere in the representation

    summary = summarize_runs([tmp_path / "suite" / "results"], output_dir=tmp_path / "report")
    assert {"representation", "common_probe", "native"} <= set(summary["tables"])
    assert summary["failure_count"] == 0


def test_architecture_protocol_reuses_primary_and_locks_decisions(tmp_path: Path) -> None:
    import json

    primary_config = _canonical(tmp_path, fixed_shell_counts="none")
    primary = _run(primary_config, tmp_path / "primary")
    provenance = json.loads((tmp_path / "primary" / "provenance.json").read_text(encoding="utf-8"))
    references = {
        "locked_loss_hash": provenance["locked_loss_hash"],
        "reference_plan_semantic_hash": provenance["autok_plan_semantic_hash"],
        "reference_plan_provenance_hash": provenance["autok_plan_provenance_hash"],
    }

    stage_a = _canonical(
        tmp_path, fixed_shell_counts="none", architecture={"stage": "activation", **references}
    )
    locked = copy.deepcopy(stage_a)
    locked["evaluation"]["locked"] = True
    with pytest.raises(ValueError, match="validation-only"):
        _run(locked, tmp_path / "stage_a_test", allow_test=True)
    result_a = _run(stage_a, tmp_path / "stage_a")
    primary_job = primary.jobs[0].job_id
    assert {job.job_id: job.status for job in result_a.jobs}[primary_job] == "reused"
    decision_a = result_a.decision
    assert decision_a is not None and decision_a["stage"] == "activation"
    winner = decision_a["selected"]

    stage_b = _canonical(
        tmp_path,
        fixed_shell_counts="none",
        architecture={
            "stage": "output_map",
            "activation": winner,
            "activation_decision_hash": decision_a["decision_hash"],
            **references,
        },
    )
    result_b = _run(stage_b, tmp_path / "stage_b")
    linear = next(
        row for row in result_b.manifest.reporting_rows if row.method.endswith("(linear_no_bias)")
    )
    winner_row = next(
        row
        for row in result_a.manifest.reporting_rows
        if row.method == f"ShellMetric-StageA({winner})"
    )
    assert linear.training_job_id == winner_row.training_job_id
    decision_b = result_b.decision
    assert decision_b is not None and decision_b["fixed_activation"] == winner

    wrong = copy.deepcopy(stage_b)
    wrong["study"]["architecture"]["activation"] = "silu" if winner != "silu" else "relu"
    with pytest.raises(ValueError, match="selected"):
        _run(wrong, tmp_path / "wrong_b", dry_run=True)

    auxiliary = _canonical(
        tmp_path,
        fixed_shell_counts="none",
        seeds=[0, 3],
        architecture={
            "stage": "auxiliary",
            "activation": winner,
            "embedding_head": decision_b["selected"],
            "activation_decision_hash": decision_a["decision_hash"],
            "output_map_decision_hash": decision_b["decision_hash"],
            **references,
        },
    )
    planned = _run(auxiliary, tmp_path / "auxiliary", dry_run=True)
    expected = (
        "ShellMetric-AutoK"
        if (winner, decision_b["selected"]) == ("relu", "linear_no_bias")
        else "ShellMetric-ActivationVariant"
    )
    assert {row.method for row in planned.manifest.reporting_rows} == {expected}
    assert {job.status for job in planned.jobs if job.job_id == linear.training_job_id} <= {
        "complete"
    }


def test_planning_ignores_validation_and_test_labels(tmp_path: Path) -> None:
    from dataclasses import replace

    from multishell.shellmetric.stages import partition, prepare_planning, prepare_split

    datasets = _datasets()
    config = _config(tmp_path)
    config["data"]["input_shape"] = [4]
    manifest, _ = prepare_split(config, lambda split, training: datasets[split], datasets["train"])
    labels = manifest.labels.copy()
    held_out = np.flatnonzero(~manifest.train_mask)
    labels[held_out] = (labels[held_out] + 1) % 3
    relabeled = replace(manifest, labels=labels)
    assert relabeled.planning_hash == manifest.planning_hash

    def plan(split, cache: str):
        train = partition(datasets["train"], manifest, split.train_mask)
        return prepare_planning(
            config,
            split,
            train,
            train,
            lambda count: nn.Linear(4, count),
            tmp_path / cache,
            ("0", "1", "2"),
        )

    first, second = plan(manifest, "a"), plan(relabeled, "b")
    assert first.statuses["pilot"] == second.statuses["pilot"] == "created"
    assert first.confusion.artifact_hash == second.confusion.artifact_hash
    assert first.autok.plan_provenance_hash == second.autok.plan_provenance_hash


def test_resume_reproduces_plan_radii_encoder_and_run_hash(tmp_path: Path, monkeypatch) -> None:
    import shutil

    import torch

    import multishell.shellmetric.train as train_module
    from multishell.confusion import build_confusion_artifact
    from multishell.shellmetric.plan import build_shell_plan
    from multishell.shellmetric.train import TrainingConfig, train_shellmetric
    from multishell.training import load_checkpoint

    datasets = _datasets()
    probabilities = np.full((3, 3), 0.1) + np.eye(3) * 0.7
    probabilities[0, 1], probabilities[0, 2] = 0.15, 0.05
    confusion = build_confusion_artifact(np.arange(3), probabilities)
    plan = build_shell_plan(confusion.h, confusion.W, 2)
    config = TrainingConfig(
        epochs=2, classes_per_batch=3, samples_per_class=4, target_batch_size=12
    )
    saved = train_module.atomic_torch_save

    def keep_first_epoch(path, payload):
        if Path(path).name == "last.pt" and payload["epoch"] == 1:
            saved(Path(path).with_name("last_epoch1.pt"), payload)
        return saved(path, payload)

    monkeypatch.setattr(train_module, "atomic_torch_save", keep_first_epoch)

    def train(directory: Path, *, resume: bool = False):
        torch.manual_seed(0)
        return train_shellmetric(
            nn.Linear(4, 3, bias=False),
            datasets["train"],
            datasets["test"],
            plan,
            config=config,
            seed=0,
            output_dir=directory,
            job_hash="job",
            resume=resume,
        )

    straight = train(tmp_path / "straight")
    resumed_dir = tmp_path / "resumed" / "checkpoints"
    resumed_dir.mkdir(parents=True)
    shutil.copy(tmp_path / "straight" / "checkpoints" / "last_epoch1.pt", resumed_dir / "last.pt")
    resumed = train(tmp_path / "resumed", resume=True)
    assert resumed.job_hash == straight.job_hash
    assert resumed.history == straight.history
    final = [
        load_checkpoint(path / "checkpoints" / "last.pt")
        for path in (tmp_path / "straight", tmp_path / "resumed")
    ]
    assert final[0]["radii_values"] == final[1]["radii_values"]
    for key, value in final[0]["encoder"].items():
        assert torch.equal(value, final[1]["encoder"][key])
    with pytest.raises(ValueError, match="resume checkpoint"):
        train_shellmetric(
            nn.Linear(4, 3, bias=False),
            datasets["train"],
            datasets["test"],
            plan,
            config=config,
            seed=0,
            output_dir=tmp_path / "resumed",
            job_hash="other",
            resume=True,
        )
