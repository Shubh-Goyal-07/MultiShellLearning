from __future__ import annotations

import copy

import numpy as np
import pytest

from multishell.baselines import METRIC_BASELINES
from multishell.config import ConfigError, resolve_config
from multishell.confusion import build_confusion_artifact
from multishell.shellmetric.architecture import (
    STAGE_A_CANDIDATES,
    decide_stage,
    select_architecture_candidate,
    verify_decision,
)
from multishell.shellmetric.autok import select_shell_count
from multishell.shellmetric.study import (
    AUTOK,
    BLOCKED_EXTERNAL_PIN,
    DIMENSION_PRESETS,
    REQUIRES_EXTERNAL_ADAPTER,
    blocked_study_manifest,
    build_study_manifest,
    shuffled_confusion_plan,
)

HASH = "a" * 64


def _planning(class_count: int = 9, *, flat: bool = False):
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(class_count), 12)
    probabilities = np.empty((labels.size, class_count))
    for row, label in enumerate(labels):
        correct = 0.7 if flat else 0.96 - 0.075 * label
        probabilities[row] = (1.0 - correct) / (class_count - 1)
        probabilities[row, label] = correct
        if not flat:
            probabilities[row] = np.maximum(
                probabilities[row] + rng.normal(0, 0.004, class_count), 0
            )
            probabilities[row] /= probabilities[row].sum()
    confusion = build_confusion_artifact(labels, probabilities, split_hash="train-only")
    return confusion, select_shell_count(labels, probabilities, confusion, bootstrap_repeats=8)


def _config(**study) -> dict:
    return resolve_config(
        {
            "data": {"dataset": "cifar100"},
            "model": {"backbone": "resnet18", "embedding_dim": 3},
            "study": {"fixed_shell_counts": "none", "seeds": [0, 1, 2], **study},
        }
    )


def _manifest(config: dict, planning=None):
    confusion, autok = planning or _planning()
    schedules = {int(seed): f"schedule-{seed}" for seed in config["study"]["seeds"]}
    return build_study_manifest(
        config, confusion, autok, split_hash="split", schedule_hashes=schedules
    )


def _architecture(stage: str, **fields) -> dict:
    return {
        "stage": stage,
        "locked_loss_hash": HASH,
        "reference_plan_semantic_hash": HASH,
        "reference_plan_provenance_hash": HASH,
        **fields,
    }


def _job(manifest, method: str, seed: int, dim: int = 3) -> str:
    return next(
        row.training_job_id
        for row in manifest.reporting_rows
        if row.method == method and row.seed == seed and row.embedding_dim == dim
    )


def test_dimension_suite_expands_exactly_and_reuses_core_jobs() -> None:
    assert DIMENSION_PRESETS["dimension_curve"] == (2, 3, 8, 16, 32, 128, 512, 1024)
    baselines = ["cross_entropy", "arcface", "supcon", "batch_hard_triplet"]
    core, _ = _manifest(_config(embedding_dims="core", baselines="all"))
    curve, _ = _manifest(_config(embedding_dims="dimension_curve", baselines=baselines))
    assert curve.embedding_dims == DIMENSION_PRESETS["dimension_curve"]
    assert len(curve.reporting_rows) == 5 * 8 * 3
    core_jobs = {job.job_id for job in core.training_jobs}
    reused = [job for job in curve.training_jobs if job.job_id in core_jobs]
    assert {job.encoder.embedding_dim for job in reused} == {3, 32, 128}
    assert len(reused) == 5 * 3 * 3  # five methods x three shared dims x three seeds


def test_metric_baselines_share_the_shellmetric_pk_schedule() -> None:
    manifest, _ = _manifest(_config(baselines="all"))
    jobs = {job.job_id: job for job in manifest.training_jobs}
    for row in manifest.reporting_rows:
        job = jobs[row.training_job_id]
        expected = "pk" if job.method in {*METRIC_BASELINES, "shellmetric"} else "shuffled"
        assert job.spec["sampler"]["kind"] == expected
        if expected == "pk":
            assert job.spec["sampler"]["schedule_hash"] == f"schedule-{row.seed}"


def test_stage_a_varies_only_activation_and_relu_aliases_primary() -> None:
    primary, _ = _manifest(_config())
    stage, _ = _manifest(_config(architecture=_architecture("activation")))
    assert stage.architecture_stage == "activation"
    assert {row.method for row in stage.reporting_rows} == {
        f"ShellMetric-StageA({name})" for name in STAGE_A_CANDIDATES
    }
    assert all(row.evaluation_scope == "validation_only" for row in stage.reporting_rows)
    for seed in (0, 1, 2):
        assert _job(stage, "ShellMetric-StageA(relu)", seed) == _job(primary, AUTOK, seed)
    encoders = {job.encoder for job in stage.training_jobs}
    assert {encoder.embedding_head for encoder in encoders} == {"linear_no_bias"}
    assert {encoder.backbone_activation for encoder in encoders} == set(STAGE_A_CANDIDATES)
    assert len({job.plan_semantic_hash for job in stage.training_jobs}) == 1


def test_stage_b_linear_entry_aliases_the_stage_a_winner() -> None:
    stage_a, _ = _manifest(_config(architecture=_architecture("activation")))
    stage_b, _ = _manifest(
        _config(
            architecture=_architecture(
                "output_map", activation="silu", activation_decision_hash=HASH
            )
        )
    )
    for seed in (0, 1, 2):
        assert _job(stage_b, "ShellMetric-StageB(linear_no_bias)", seed) == _job(
            stage_a, "ShellMetric-StageA(silu)", seed
        )
        assert _job(stage_b, "ShellMetric-StageB(radial_power_gate)", seed) != _job(
            stage_a, "ShellMetric-StageA(silu)", seed
        )


@pytest.mark.parametrize(
    "architecture",
    [
        _architecture("activation", embedding_head="radial_power_gate"),
        _architecture("activation", candidates=["relu", "radial_power_gate"]),
        _architecture(
            "output_map",
            activation="silu",
            activation_decision_hash=HASH,
            candidates=["linear_no_bias", "silu"],
        ),
        _architecture(
            "activation", candidates=["silu", "relu", "gelu_exact", "channel_radial_silu"]
        ),
        _architecture("activation", locked_loss_hash="latest"),
    ],
)
def test_architecture_stages_reject_cartesian_products_and_unlocked_references(
    architecture: dict,
) -> None:
    with pytest.raises(ConfigError):
        _config(architecture=architecture)


def test_architecture_stages_are_single_cell_and_selection_seed_only() -> None:
    with pytest.raises(ConfigError, match="seeds"):
        _config(architecture=_architecture("activation"), seeds=[0, 3])
    with pytest.raises(ConfigError, match="AutoK only"):
        _config(architecture=_architecture("activation"), fixed_shell_counts="compact")
    with pytest.raises(ConfigError, match="official ReLU"):
        config = copy.deepcopy(_config())
        config["model"]["backbone_activation"] = "silu"
        config["study"]["architecture"] = _architecture("activation")
        resolve_config(config)


def test_auxiliary_winner_equal_to_primary_is_an_alias_not_a_variant() -> None:
    primary, _ = _manifest(_config(seeds=[0, 1, 2, 3, 4]))
    fields = {"activation_decision_hash": HASH, "output_map_decision_hash": HASH}
    same, _ = _manifest(
        _config(
            seeds=[0, 1, 2, 3, 4],
            architecture=_architecture(
                "auxiliary", activation="relu", embedding_head="linear_no_bias", **fields
            ),
        )
    )
    assert {row.method for row in same.reporting_rows} == {AUTOK}
    assert all(row.note and "no distinct" in row.note for row in same.reporting_rows)
    assert {job.job_id for job in same.training_jobs} == {
        job.job_id for job in primary.training_jobs
    }
    variant, _ = _manifest(
        _config(
            seeds=[0, 1, 2, 3, 4],
            architecture=_architecture(
                "auxiliary", activation="silu", embedding_head="radial_power_gate", **fields
            ),
        )
    )
    assert {row.method for row in variant.reporting_rows} == {"ShellMetric-ActivationVariant"}
    assert all(row.evaluation_scope == "test_eligible" for row in variant.reporting_rows)


def test_decision_rule_uses_tolerance_map_spoke_then_preferred_name() -> None:
    order = ("relu", "silu", "gelu_exact", "channel_radial_silu")
    metrics = {
        "relu": {
            "euclidean_1nn": 0.700,
            "map_at_r": 0.5,
            "undefined_spoke_classes": 0,
            "mean_spoke_fraction": 0.3,
        },
        "silu": {
            "euclidean_1nn": 0.701,
            "map_at_r": 0.6,
            "undefined_spoke_classes": 0,
            "mean_spoke_fraction": 0.2,
        },
        "gelu_exact": {
            "euclidean_1nn": 0.702,
            "map_at_r": 0.6,
            "undefined_spoke_classes": 1,
            "mean_spoke_fraction": 0.1,
        },
        "channel_radial_silu": {
            "euclidean_1nn": 0.690,
            "map_at_r": 0.9,
            "undefined_spoke_classes": 0,
            "mean_spoke_fraction": 0.0,
        },
    }
    decision = select_architecture_candidate(
        metrics, declared_order=order, preferred_exact_tie="relu"
    )
    assert decision["selected"] == "silu"
    assert [step["retained"] for step in decision["trace"]] == [
        ["relu", "silu", "gelu_exact"],
        ["silu", "gelu_exact"],
        ["silu"],
    ]
    tied = {name: {**metrics["relu"], "euclidean_1nn": 0.7} for name in order}
    assert (
        select_architecture_candidate(tied, declared_order=order, preferred_exact_tie="relu")[
            "selected"
        ]
        == "relu"
    )
    assert (
        select_architecture_candidate(tied, declared_order=order, preferred_exact_tie="none")[
            "selected"
        ]
        == "relu"
    )  # first declared name


def test_stage_decision_artifact_is_content_hashed_and_verifiable() -> None:
    config = _config(architecture=_architecture("activation"))
    per_candidate = {
        name: [
            {
                "seed": seed,
                "euclidean_1nn": 0.5 + 0.01 * index,
                "map_at_r": 0.4,
                "undefined_spoke_classes": 0,
                "mean_spoke_fraction": None,
            }
            for seed in (0, 1, 2)
        ]
        for index, name in enumerate(STAGE_A_CANDIDATES)
    }
    decision = decide_stage(config, per_candidate)
    assert decision["selected"] == "channel_radial_silu"
    assert decision["selection_uses_test"] is False
    verify_decision(decision, stage="activation", selected="channel_radial_silu")
    with pytest.raises(ValueError):
        verify_decision({**decision, "selected": "relu"}, stage="activation", selected="relu")


def test_external_baselines_are_gated_and_official_is_reference_only() -> None:
    config = _config(external_baselines=["hyperspacex_matched", "hyperspacex_official"])
    manifest, _ = _manifest(config)
    rows = {row.method: row for row in manifest.reporting_rows if "HyperSpaceX" in row.method}
    assert rows["HyperSpaceX-Matched"].status == BLOCKED_EXTERNAL_PIN
    assert rows["HyperSpaceX-Official"].table == "reference"
    assert all(row.training_job_id is None for row in rows.values())

    pin = {
        "repository": "https://github.com/IAB-IITJ/HyperSpaceX",
        "commit": "046d0fbe9ee9cec841b8d78efa65019da0a6307c",
        "environment": "conda-env-export.yaml sha256:" + HASH,
        "command": "python train_objects.py",
        "checkpoint_rule": "final epoch",
    }
    pinned = resolve_config({**config, "external_pins": {"hyperspacex": pin}})
    statuses = {row.method: row.status for row in _manifest(pinned)[0].reporting_rows}
    assert statuses["HyperSpaceX-Official"] == REQUIRES_EXTERNAL_ADAPTER
    assert statuses["HyperSpaceX-Matched"] == BLOCKED_EXTERNAL_PIN  # official not reproduced
    with pytest.raises(ConfigError, match="immutable"):
        resolve_config({**config, "external_pins": {"hyperspacex": {**pin, "commit": "latest"}}})


def test_unpinned_backbone_blocks_every_row_without_planning() -> None:
    config = resolve_config(
        {
            "data": {"dataset": "cifar100"},
            "model": {"backbone": "vit_s16", "backbone_activation": "native", "embedding_dim": 128},
            "study": {
                "fixed_shell_counts": "none",
                "baselines": ["cross_entropy"],
                "seeds": [0, 1],
            },
        }
    )
    manifest = blocked_study_manifest(config, class_count=100, split_hash="split")
    assert {row.status for row in manifest.reporting_rows} == {BLOCKED_EXTERNAL_PIN}
    assert len(manifest.reporting_rows) == 4 and not manifest.training_jobs


def test_one_shell_comparison_is_not_applicable_when_autok_picks_one_shell() -> None:
    planning = _planning(flat=True)
    assert planning[1].chosen_shell_count == 1
    config = _config(
        fixed_shell_counts="compact",
        ensure_one_shell_control=True,
        controls=["shuffled_confusion", "no_shell_loss"],
        seeds=[0],
    )
    manifest, _ = _manifest(config, planning)
    comparison = next(
        item for item in manifest.comparisons if item["name"] == "multi_shell_vs_one_shell"
    )
    assert comparison["status"] == "not_applicable"
    assert _job(manifest, AUTOK, 0) == _job(manifest, "ShellMetric-FixedS(1)", 0)
    no_shell = next(row for row in manifest.reporting_rows if row.method.endswith("NoShellLoss"))
    assert no_shell.parents["autok_plan_provenance_hash"] == planning[1].plan_provenance_hash


def test_shuffled_confusion_is_a_structure_preserving_non_identity_derangement() -> None:
    confusion, autok = _planning()
    plan, permutation = shuffled_confusion_plan(confusion, autok.plan, seed=5)
    permutation = np.asarray(permutation)
    assert np.all(permutation != np.arange(permutation.size))
    shuffled = confusion.W[np.ix_(permutation, permutation)]
    assert np.array_equal(shuffled, shuffled.T) and np.all(np.diag(shuffled) == 0)
    assert np.array_equal(np.sort(shuffled, axis=None), np.sort(confusion.W, axis=None))
    assert not np.array_equal(shuffled, confusion.W)
    assert plan.shell_count == autok.plan.shell_count and plan.capacities == autok.plan.capacities
    assert plan.assignment != autok.plan.assignment or not np.allclose(plan.w_hat, autok.plan.w_hat)
    assert shuffled_confusion_plan(confusion, autok.plan, seed=5)[1] == tuple(permutation)


def test_rehearsal_lifts_only_the_canonical_cell_restriction() -> None:
    def mnist(**study) -> dict:
        return {
            "data": {"dataset": "mnist"},
            "model": {"backbone": "resnet18", "embedding_dim": 3},
            "study": {"fixed_shell_counts": "none", "seeds": [0, 1, 2], **study},
        }

    stage = {"architecture": _architecture("activation")}
    with pytest.raises(ConfigError, match="CIFAR-100/ResNet-18/3D"):
        resolve_config(mnist(**stage))
    rehearsal = resolve_config(mnist(rehearsal=True, **stage))
    assert rehearsal["study"]["rehearsal"] is True
    with pytest.raises(ConfigError, match="scratch ResNet-18"):
        resolve_config({**mnist(rehearsal=True, **stage), "model": {"backbone": "small_cnn"}})
    with pytest.raises(ConfigError, match="true or false"):
        resolve_config(mnist(rehearsal="yes"))

    controls = {"fixed_shell_counts": [1], "controls": ["no_shell_loss"]}
    with pytest.raises(ValueError, match="canonical cell"):
        _manifest(resolve_config(mnist(**controls)))
    manifest, _ = _manifest(resolve_config(mnist(rehearsal=True, **controls)))
    assert any(row.method.endswith("NoShellLoss") for row in manifest.reporting_rows)


def test_shuffled_control_plans_never_leak_into_other_seeds() -> None:
    config = _config(
        fixed_shell_counts=[1],
        ensure_one_shell_control=True,
        controls=["shuffled_confusion"],
    )
    manifest, plans = _manifest(config)
    shuffled = [row for row in manifest.reporting_rows if "Shuffled" in row.method]
    assert {row.method for row in shuffled} == {"ShellMetric-AutoK-ShuffledConfusion"}
    assert sorted(row.seed for row in shuffled) == [0, 1, 2]
    assert {f"ShellMetric-AutoK-ShuffledConfusion/seed={seed}" for seed in (0, 1, 2)} <= set(plans)
