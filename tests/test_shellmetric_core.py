from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch
from torch import nn

from multishell.artifacts import stable_hash
from multishell.config import ConfigError, resolve_config, validate_config
from multishell.confusion import build_confusion_artifact
from multishell.shellmetric.autok import select_shell_count
from multishell.shellmetric.loss import shellmetric_loss
from multishell.shellmetric.plan import (
    assign_classes_to_shells,
    build_shell_plan,
    shell_capacities,
    smax,
)
from multishell.shellmetric.probe import module_checksum, train_linear_probe
from multishell.shellmetric.radii import OrderedRadii
from multishell.shellmetric.sampler import (
    ManifestBatchSampler,
    PKSchedule,
    build_batch_manifest,
    default_pk,
)
from multishell.shellmetric.study import EncoderSpec, build_study_manifest, job_spec
from multishell.splits import make_split_manifest


def _fractional_capacities(class_count: int, shell_count: int) -> tuple[int, ...]:
    capacities = [1] * shell_count
    for _ in range(class_count - shell_count):
        best = max(
            range(1, shell_count + 1),
            key=lambda shell: (Fraction(shell * shell, capacities[shell - 1] + 1), shell),
        )
        capacities[best - 1] += 1
    return tuple(capacities)


def _confusion_fixture():
    class_probabilities = np.asarray(
        [
            [0.70, 0.20, 0.08, 0.02],
            [0.10, 0.65, 0.20, 0.05],
            [0.02, 0.13, 0.65, 0.20],
            [0.01, 0.02, 0.17, 0.80],
        ],
        dtype=np.float64,
    )
    labels = np.repeat(np.arange(4), 6)
    probabilities = class_probabilities[labels]
    artifact = build_confusion_artifact(
        labels,
        probabilities,
        pilot_seeds=[0, 1, 2],
        split_hash="train-only",
    )
    return labels, probabilities, artifact


def _study_config(*, controls: list[str] | None = None) -> dict:
    return resolve_config(
        {
            "data": {"dataset": "cifar100"},
            "model": {"backbone": "resnet18", "embedding_dim": 3},
            "study": {
                "fixed_shell_counts": "all",
                "ensure_one_shell_control": False,
                "controls": ["shuffled_confusion", "no_shell_loss"]
                if controls is None
                else controls,
                "seeds": [7],
            },
        }
    )


def test_shell_capacities_use_the_exact_integer_priority_rule() -> None:
    assert (smax(10), smax(100), smax(1000)) == (4, 10, 32)
    assert shell_capacities(10, 4) == (1, 1, 3, 5)

    for class_count in range(1, 65):
        for shell_count in range(1, smax(class_count) + 1):
            capacities = shell_capacities(class_count, shell_count)
            assert capacities == _fractional_capacities(class_count, shell_count)
            assert sum(capacities) == class_count
            assert all(value >= 1 for value in capacities)
            assert tuple(sorted(capacities)) == capacities


def test_shell_assignment_and_hashes_separate_semantics_from_provenance() -> None:
    difficulty = np.asarray([1.0, 0.5, 0.5, 2.0])
    class_ids = np.asarray([30, 10, 20, 40])
    assignment, ordered = assign_classes_to_shells(difficulty, 2, class_ids=class_ids)
    np.testing.assert_array_equal(ordered, [10, 20, 30, 40])
    np.testing.assert_array_equal(assignment, [2, 1, 2, 2])

    confusion = np.asarray(
        [
            [0.0, 0.4, 0.2, 0.1],
            [0.4, 0.0, 0.3, 0.2],
            [0.2, 0.3, 0.0, 0.5],
            [0.1, 0.2, 0.5, 0.0],
        ]
    )
    fixed = build_shell_plan(
        difficulty,
        confusion,
        2,
        class_ids=class_ids,
        family="fixed_s",
        selection={"requested_shell_count": 2},
    )
    autok = build_shell_plan(
        difficulty,
        confusion,
        2,
        class_ids=class_ids,
        family="autok",
        selection={"route": "bootstrap"},
    )
    assert fixed.plan_semantic_hash == autok.plan_semantic_hash
    assert fixed.plan_provenance_hash != autok.plan_provenance_hash

    changed = confusion.copy()
    changed[0, 1] = changed[1, 0] = 0.1
    changed_plan = build_shell_plan(difficulty, changed, 2, class_ids=class_ids)
    assert changed_plan.plan_semantic_hash != fixed.plan_semantic_hash


def test_ordered_radii_are_strict_positive_ordered_and_rms_normalized() -> None:
    one = OrderedRadii([10]).double()
    assert list(one.parameters()) == []
    torch.testing.assert_close(one(), torch.ones(1, dtype=torch.float64))

    module = OrderedRadii([1, 1, 3, 5]).double()
    assert sum(parameter.numel() for parameter in module.parameters()) == 3
    initial = module()
    assert torch.all(initial > 0)
    assert torch.all(torch.diff(initial) > 0)
    torch.testing.assert_close(
        torch.diff(initial),
        torch.full_like(torch.diff(initial), torch.diff(initial)[0].item()),
    )

    with torch.no_grad():
        assert module.gamma_tail is not None
        module.gamma_tail.copy_(torch.tensor([100.0, -100.0, 25.0]))
    radii = module()
    weights = module.capacities / module.capacities.sum()
    assert torch.all(torch.diff(radii) > 0)
    torch.testing.assert_close(
        torch.sum(weights * radii.square()), torch.tensor(1.0, dtype=torch.float64)
    )
    radii.sum().backward()
    assert module.gamma_tail is not None
    assert torch.isfinite(module.gamma_tail.grad).all()


def test_shellmetric_loss_is_rotation_invariant_and_shell_gradient_is_radial() -> None:
    embeddings = torch.tensor(
        [[0.8, 0.2, -0.1], [0.9, 0.1, 0.0], [-0.2, 1.1, 0.1], [0.0, 1.2, -0.1]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 1, 1])
    w_hat = torch.tensor([[0.0, 0.75], [0.75, 0.0]], dtype=torch.float64)
    assignment = torch.tensor([1, 2])
    radii = torch.tensor([0.8, 1.2], dtype=torch.float64)
    options = dict(class_to_shell=assignment, radii=radii, epsilon=1.0e-10)
    original = shellmetric_loss(embeddings, labels, w_hat, **options)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    rotated = shellmetric_loss(embeddings @ rotation, labels, w_hat, **options)
    for name in ("total", "positive", "negative", "shell"):
        torch.testing.assert_close(getattr(rotated, name), getattr(original, name))

    radial_inputs = torch.tensor([[1.4, 0.2], [1.3, 0.1]], dtype=torch.float64, requires_grad=True)
    radial = shellmetric_loss(
        radial_inputs,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.float64),
        class_to_shell=torch.ones(1, dtype=torch.long),
        radii=torch.ones(1, dtype=torch.float64),
        positive_margin=10.0,
    )
    gradient = torch.autograd.grad(radial.total, radial_inputs)[0]
    tangent = torch.stack((-radial_inputs[:, 1], radial_inputs[:, 0]), dim=1)
    torch.testing.assert_close((gradient * tangent).sum(dim=1), torch.zeros(2, dtype=torch.float64))


def test_shellmetric_pair_margins_and_no_shell_mode_are_exact() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    weighted = shellmetric_loss(
        embeddings,
        labels,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        class_to_shell=None,
        radii=None,
        negative_margin_base=1.0,
        confusion_margin_delta=0.5,
        shell_weight=0.0,
    )
    unweighted = shellmetric_loss(
        embeddings,
        labels,
        torch.zeros(2, 2, dtype=torch.float64),
        class_to_shell=None,
        radii=None,
        negative_margin_base=1.0,
        confusion_margin_delta=0.5,
        shell_weight=0.0,
    )
    torch.testing.assert_close(weighted.negative, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(unweighted.negative, torch.tensor(0.25, dtype=torch.float64))
    assert weighted.positive_pairs == 0
    assert weighted.negative_pairs == 1
    assert weighted.shell.item() == 0.0
    torch.testing.assert_close(weighted.total, weighted.negative)


def test_pk_sampler_is_deterministic_balanced_and_replayable() -> None:
    assert default_pk(10) == (10, 12)
    assert default_pk(100) == (32, 4)
    labels = np.repeat(np.arange(5), [2, 3, 4, 5, 6])
    first = build_batch_manifest(labels, seed=19, steps=4, classes_per_batch=3, samples_per_class=4)
    second = build_batch_manifest(
        labels, seed=19, steps=4, classes_per_batch=3, samples_per_class=4
    )
    different = build_batch_manifest(
        labels, seed=20, steps=4, classes_per_batch=3, samples_per_class=4
    )
    assert first == second
    assert first != different
    for batch in first:
        _, counts = np.unique(labels[np.asarray(batch)], return_counts=True)
        np.testing.assert_array_equal(counts, [4, 4, 4])

    replay = ManifestBatchSampler(first, dataset_size=labels.size)
    assert tuple(tuple(batch) for batch in replay) == first
    assert tuple(tuple(batch) for batch in replay) == first


def test_affine_probe_cannot_mutate_encoder_parameters_or_batchnorm_buffers() -> None:
    encoder = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4))
    encoder.train()
    before = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
    checksum = module_checksum(encoder)
    train_x = torch.tensor(
        [[-2.0, -1.0], [-1.5, -0.8], [-1.0, -1.4], [1.0, 1.4], [1.5, 0.8], [2.0, 1.0]]
    )
    train_y = torch.tensor([0, 0, 0, 1, 1, 1])
    validation_x = torch.tensor([[-1.2, -1.0], [1.2, 1.0]])
    validation_y = torch.tensor([0, 1])

    result = train_linear_probe(
        train_x,
        train_y,
        validation_x,
        validation_y,
        encoder=encoder,
        seed=11,
        weight_decays=(0.0,),
        max_epochs=8,
        patience=3,
        batch_size=3,
    )

    assert isinstance(result.model, nn.Linear)
    assert result.model.in_features == 2
    assert result.model.out_features == 2
    assert sum(parameter.numel() for parameter in result.model.parameters()) == 2 * (2 + 1)
    assert result.encoder_checksum == checksum == module_checksum(encoder)
    assert encoder.training and encoder[1].training
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "override",
    [
        {"shells": {"manual_assignment": [1, 2]}},
        {"shells": {"class_directions": [[1.0, 0.0]]}},
        {"loss": {"prototype": "learned"}},
        {"model": {"representation_classifier": "linear"}},
        {"study": {"test_derived_planning": True}},
        {"model": {"output_bias": True}},
        {"pilot": {"temperature_calibration": True}},
    ],
)
def test_shellmetric_config_strictly_rejects_forbidden_protocols(override: dict) -> None:
    with pytest.raises(ConfigError):
        resolve_config(override)

    invalid_count = resolve_config(validate=False)
    invalid_count["study"]["fixed_shell_counts"] = [1, 5]
    with pytest.raises(ConfigError, match="invalid shell counts"):
        validate_config(invalid_count, class_count=10)


def test_split_planning_hash_uses_train_only_and_tracks_training_identity() -> None:
    train_labels = np.repeat(np.arange(2), 20)
    test_labels = np.repeat(np.arange(2), 4)
    manifest = make_split_manifest(
        train_labels,
        sample_ids=[f"train-{index}" for index in range(train_labels.size)],
        test_labels=test_labels,
        test_sample_ids=[f"test-{index}" for index in range(test_labels.size)],
        dataset_name="fixture",
        dataset_version="v1",
        train_checksum="train-a",
        test_checksum="test-a",
    )

    held_out_labels = manifest.labels.copy()
    for mask in (manifest.validation_mask, manifest.test_mask):
        zero = np.flatnonzero(mask & (held_out_labels == 0))[0]
        one = np.flatnonzero(mask & (held_out_labels == 1))[0]
        held_out_labels[[zero, one]] = held_out_labels[[one, zero]]
    held_out_changed = replace(manifest, labels=held_out_labels, test_checksum="test-b")
    assert held_out_changed.planning_hash == manifest.planning_hash
    assert held_out_changed.manifest_hash != manifest.manifest_hash

    training_labels = manifest.labels.copy()
    zero = np.flatnonzero(manifest.train_mask & (training_labels == 0))[0]
    one = np.flatnonzero(manifest.train_mask & (training_labels == 1))[0]
    training_labels[[zero, one]] = training_labels[[one, zero]]
    assert replace(manifest, labels=training_labels).planning_hash != manifest.planning_hash
    assert replace(manifest, train_checksum="train-b").planning_hash != manifest.planning_hash


def test_study_deduplicates_autok_fixeds_and_emits_only_canonical_controls() -> None:
    labels, probabilities, artifact = _confusion_fixture()
    autok = select_shell_count(labels, probabilities, artifact, bootstrap_repeats=4)
    config = _study_config()
    manifest, plans = build_study_manifest(config, artifact, autok, split_hash="split")

    assert len(manifest.reporting_rows) == 5
    assert len({row.row_id for row in manifest.reporting_rows}) == 5
    assert len(manifest.training_jobs) == 4
    fixed_name = f"ShellMetric-FixedS({autok.chosen_shell_count})"
    fixed = next(row for row in manifest.reporting_rows if row.method == fixed_name)
    automatic = next(row for row in manifest.reporting_rows if row.method == "ShellMetric-AutoK")
    assert fixed.training_job_id == automatic.training_job_id
    assert fixed.plan_semantic_hash == automatic.plan_semantic_hash
    assert fixed.plan_provenance_hash != automatic.plan_provenance_hash
    assert automatic.row_id in fixed.aliases and fixed.row_id in automatic.aliases

    shuffled = plans["ShellMetric-AutoK-ShuffledConfusion/seed=7"]
    assert shuffled.shell_count == autok.chosen_shell_count
    assert shuffled.capacities == autok.plan.capacities
    assert shuffled.plan_semantic_hash != autok.plan.plan_semantic_hash
    no_shell = next(job for job in manifest.training_jobs if job.no_shell_loss)
    assert no_shell.control and no_shell.plan_semantic_hash is None

    noncanonical = copy.deepcopy(config)
    noncanonical["data"]["dataset"] = "mnist"
    with pytest.raises(ValueError, match="canonical cell"):
        build_study_manifest(noncanonical, artifact, autok, split_hash="split")


def _job_id(config: dict, plan, *, schedule: str = "batches", seed: int = 3) -> str:
    return stable_hash(
        job_spec(
            config,
            kind="shellmetric",
            method="shellmetric",
            seed=seed,
            encoder=EncoderSpec(3, "relu", "linear_no_bias"),
            split_hash="split",
            schedule_hash=schedule,
            plan=plan,
        )
    )


def test_encoder_job_hash_ignores_reporting_provenance_but_tracks_training_semantics() -> None:
    _, _, artifact = _confusion_fixture()
    fixed = build_shell_plan(
        artifact.h, artifact.W, 2, family="fixed_s", selection={"requested": 2}
    )
    automatic = build_shell_plan(
        artifact.h, artifact.W, 2, family="autok", selection={"bootstrap": "trace"}
    )
    config = _study_config(controls=[])
    fixed_hash = _job_id(config, fixed)
    assert fixed_hash == _job_id(config, automatic)

    reporting_only = copy.deepcopy(config)
    reporting_only["study"]["seeds"] = [99]
    reporting_only["experiment"]["name"] = "renamed"
    reporting_only["training"]["keep_epoch_checkpoints"] = False
    assert fixed_hash == _job_id(reporting_only, fixed)
    changed_loss = copy.deepcopy(config)
    changed_loss["loss"]["positive_margin"] += 0.1
    assert fixed_hash != _job_id(changed_loss, fixed)
    assert fixed_hash != _job_id(config, fixed, schedule="different-batches")
    assert fixed_hash != _job_id(config, fixed, seed=4)


def test_harder_classes_are_never_assigned_inward_of_easier_classes() -> None:
    rng = np.random.default_rng(11)
    for class_count in (10, 37, 100):
        difficulty = rng.random(class_count)
        for shell_count in range(1, smax(class_count) + 1):
            assignment, _ = assign_classes_to_shells(difficulty, shell_count)
            order = np.argsort(difficulty)
            assert np.all(np.diff(assignment[order]) >= 0)
            assert assignment[order[-1]] == shell_count


def test_fixed_shell_count_forms_are_exact_and_independent_of_autok() -> None:
    from multishell.shellmetric.plan import expand_fixed_shell_counts

    assert expand_fixed_shell_counts("compact", 10) == (1, 2, 3, 4)
    assert expand_fixed_shell_counts("compact", 100) == (1, 3, 5, 8, 10)
    assert expand_fixed_shell_counts("compact", 200) == (1, 4, 8, 12, 15)
    assert expand_fixed_shell_counts("compact", 1000) == (1, 8, 16, 24, 32)
    assert expand_fixed_shell_counts("all", 10) == (1, 2, 3, 4)
    assert expand_fixed_shell_counts("none", 10) == ()
    assert expand_fixed_shell_counts([], 10) == ()
    assert expand_fixed_shell_counts([4, 2, 4], 10) == (2, 4)
    assert expand_fixed_shell_counts([4, 2], 10, ensure_one_shell_control=True) == (1, 2, 4)
    with pytest.raises(ValueError):
        expand_fixed_shell_counts([5], 10)

    labels, probabilities, artifact = _confusion_fixture()
    autok = select_shell_count(labels, probabilities, artifact, bootstrap_repeats=4)
    assert autok.candidate_shell_counts == tuple(range(1, smax(4) + 1))
    for fixed in ("none", [2], "all"):
        config = _study_config(controls=[])
        config["study"]["fixed_shell_counts"] = fixed
        manifest, _ = build_study_manifest(config, artifact, autok, split_hash="split")
        assert manifest.autok_shell_count == autok.chosen_shell_count


def test_radius_gaps_sum_to_one_respect_the_floor_and_follow_the_plan() -> None:
    module = OrderedRadii([1, 1, 3, 5], gap_floor_fraction=0.2).double()
    with torch.no_grad():
        module.gamma_tail.copy_(torch.tensor([50.0, -50.0, 3.0]))
    gaps = module.gaps()
    torch.testing.assert_close(gaps.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.all(gaps >= 0.2 / 4 - 1e-12)

    _, _, artifact = _confusion_fixture()
    from multishell.shellmetric.plan import make_radius_settings

    plan = build_shell_plan(artifact.h, artifact.W, 2, radius_settings=make_radius_settings(0.25))
    assert OrderedRadii.from_plan(plan).gap_floor_fraction == 0.25
    default = build_shell_plan(artifact.h, artifact.W, 2)
    assert plan.plan_semantic_hash != default.plan_semantic_hash


def test_positive_hinge_is_zero_inside_margin_and_confusion_never_shrinks_margins() -> None:
    labels = torch.tensor([0, 0])
    inside = shellmetric_loss(
        torch.tensor([[0.0, 0.0], [0.4, 0.0]], dtype=torch.float64),
        labels,
        torch.zeros(1, 1, dtype=torch.float64),
        class_to_shell=None,
        radii=None,
        shell_weight=0.0,
        positive_margin=0.5,
    )
    assert inside.positive.item() == 0.0 and inside.positive_pairs == 1

    embeddings = torch.tensor([[0.0, 0.0], [0.6, 0.0], [0.0, 0.6]], dtype=torch.float64)
    w_hat = torch.tensor([[0.0, 1.0, 0.2], [1.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64)
    options = dict(class_to_shell=None, radii=None, shell_weight=0.0)
    confused = shellmetric_loss(embeddings[:2], torch.tensor([0, 1]), w_hat, **options)
    mild = shellmetric_loss(embeddings[[0, 2]], torch.tensor([0, 2]), w_hat, **options)
    assert confused.negative > mild.negative


def test_pk_schedule_varies_by_epoch_covers_the_split_and_always_pairs() -> None:
    labels = np.repeat(np.arange(10), 60)
    schedule = PKSchedule.create(labels, seed=3)
    assert (schedule.classes_per_batch, schedule.samples_per_class) == (10, 12)
    first, second = schedule.epoch(1), schedule.epoch(2)
    assert first != second and first == PKSchedule.create(labels, seed=3).epoch(1)
    assert len(first) == int(np.ceil(labels.size / 128))
    for batch in first:
        _, counts = np.unique(labels[np.asarray(batch)], return_counts=True)
        assert counts.size >= 2 and np.all(counts >= 2)  # positives and negatives exist
    covered = np.unique(np.concatenate([np.asarray(batch) for batch in first]))
    assert covered.size / labels.size > 0.95
    assert PKSchedule.create(labels, seed=4).schedule_hash != schedule.schedule_hash
