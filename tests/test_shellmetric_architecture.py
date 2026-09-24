from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from multishell.models import (
    CIFARResNet,
    PilotClassifier,
    SmallCNNBackbone,
    build_backbone,
    build_encoder,
    build_pilot_classifier,
    count_trainable_parameters,
)
from multishell.shellmetric.activations import (
    ACTIVATION_NAMES,
    ChannelRadialSiLU,
    build_activation,
)
from multishell.shellmetric.heads import LinearNoBias, RadialPowerGate, build_embedding_head


def _orthogonal(dimension: int) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(dimension, dimension, dtype=torch.float64))
    return matrix


def test_locked_backbone_widths_and_parameter_counts() -> None:
    expected = {
        "small_cnn": (128, 109_184, (1, 28, 28)),
        "resnet18": (512, 11_168_832, (3, 32, 32)),
        "resnet50": (2_048, 23_500_352, (3, 32, 32)),
    }
    for name, (width, parameter_count, shape) in expected.items():
        backbone = build_backbone(name, input_shape=shape)
        assert backbone.output_dim == width
        assert count_trainable_parameters(backbone) == parameter_count


def test_obsolete_mlp_and_projection_heads_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported backbone"):
        build_backbone("mlp")
    with pytest.raises(ValueError, match="unsupported embedding head"):
        build_encoder("small_cnn", head="cartesian_linear")
    with pytest.raises(ValueError, match="unsupported embedding head"):
        build_encoder("small_cnn", head="mlp")
    with pytest.raises(ValueError, match="translation bias"):
        build_encoder({"model": {"backbone": "small_cnn", "output_bias": True}})


def test_resnet18_has_exactly_seventeen_parameter_free_activation_sites() -> None:
    counts = set()
    for name in ACTIVATION_NAMES:
        backbone = build_backbone("resnet18", input_shape=(3, 32, 32), activation=name)
        assert isinstance(backbone, CIFARResNet)
        assert len(backbone.activation_sites) == 17
        assert len({id(site) for site in backbone.activation_sites}) == 17
        assert all(count_trainable_parameters(site) == 0 for site in backbone.activation_sites)
        assert not any(
            isinstance(module, tuple(type(build_activation(item)) for item in ACTIVATION_NAMES))
            for block in backbone.modules()
            if hasattr(block, "shortcut")
            for module in block.shortcut.modules()
        )
        counts.add(count_trainable_parameters(backbone))
    assert counts == {11_168_832}


def test_channel_radial_silu_geometry_and_gradients() -> None:
    activation = ChannelRadialSiLU().double()
    zeros = torch.zeros(2, 4, 3, 3, dtype=torch.float64, requires_grad=True)
    zero_output = activation(zeros)
    assert torch.equal(zero_output, zeros)
    zero_output.sum().backward()
    assert torch.isfinite(zeros.grad).all()

    values = torch.randn(2, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    output = activation(values)
    assert output.shape == values.shape
    assert output.dtype == values.dtype
    assert output.device == values.device
    assert torch.autograd.gradcheck(activation, (values,))

    rotation = _orthogonal(4)
    rotated = torch.einsum("ij,bjhw->bihw", rotation, values)
    expected = torch.einsum("ij,bjhw->bihw", rotation, output)
    assert torch.allclose(activation(rotated), expected, atol=1e-12, rtol=1e-12)

    radii = torch.linspace(0.0, 5.0, 100, dtype=torch.float64)
    radial_inputs = radii[:, None, None, None].expand(-1, 4, 1, 1)
    output_radii = activation(radial_inputs).square().mean(dim=1).sqrt().flatten()
    assert torch.all(output_radii[1:] > output_radii[:-1])


def test_radial_power_gate_identity_bounds_geometry_and_stability() -> None:
    gate = RadialPowerGate(5, 3).double()
    with torch.no_grad():
        gate.projection.weight.copy_(torch.randn_like(gate.projection.weight))
    inputs = torch.randn(7, 5, dtype=torch.float64)
    projected = gate.projection(inputs)
    assert torch.equal(gate(inputs), projected)
    assert count_trainable_parameters(gate) == 5 * 3 + 1
    assert gate.projection.bias is None

    for beta in (-100.0, -1.0, 0.0, 1.0, 100.0):
        with torch.no_grad():
            gate.beta.fill_(beta)
        assert 0.5 < gate.power.item() < 2.0

    with torch.no_grad():
        gate.beta.fill_(0.75)
    vectors = torch.tensor(
        [[0.0, 0.0, 0.0], [1e-30, 0.0, 0.0], [1e6, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    transformed = gate.transform(vectors)
    transformed.sum().backward()
    assert torch.isfinite(transformed).all()
    assert torch.isfinite(vectors.grad).all()
    assert torch.equal(transformed[0], vectors.detach()[0])
    assert torch.allclose(
        transformed[1:] / transformed[1:].norm(dim=1, keepdim=True),
        vectors.detach()[1:] / vectors.detach()[1:].norm(dim=1, keepdim=True),
    )

    rotation = _orthogonal(3)
    samples = torch.randn(12, 3, dtype=torch.float64)
    assert torch.allclose(
        gate.transform(samples @ rotation.T),
        gate.transform(samples) @ rotation.T,
        atol=1e-12,
        rtol=1e-12,
    )
    radii = torch.logspace(-6, 6, 100, dtype=torch.float64)
    mapped = gate.transform(
        torch.stack((radii, torch.zeros_like(radii), torch.zeros_like(radii)), dim=1)
    )
    assert torch.all(mapped.norm(dim=1)[1:] > mapped.norm(dim=1)[:-1])


def test_encoder_is_raw_and_pilot_is_direct_h_to_c() -> None:
    config = {
        "data": {"input_shape": [1, 28, 28]},
        "model": {
            "backbone": "small_cnn",
            "backbone_activation": "relu",
            "embedding_dim": 3,
            "embedding_head": "linear_no_bias",
            "output_bias": False,
        },
    }
    encoder = build_encoder(config)
    assert isinstance(encoder.backbone, SmallCNNBackbone)
    assert isinstance(encoder.head, LinearNoBias)
    assert encoder.head.bias is None
    assert not any(
        isinstance(module, (nn.BatchNorm1d, nn.Softmax)) for module in encoder.head.modules()
    )
    assert encoder(torch.randn(4, 1, 28, 28)).shape == (4, 3)

    pilot = build_pilot_classifier("small_cnn", input_shape=(1, 28, 28), num_classes=10)
    assert isinstance(pilot, PilotClassifier)
    assert pilot.classifier.in_features == pilot.backbone.output_dim
    assert pilot.classifier.out_features == 10
    assert pilot(torch.randn(4, 1, 28, 28)).shape == (4, 10)


def test_channel_radial_silu_has_unit_gain_at_unit_radius_and_keeps_direction() -> None:
    activation = ChannelRadialSiLU().double()
    unit = torch.full((1, 4, 1, 1), math.sqrt(1.0 - 1.0e-6), dtype=torch.float64)
    torch.testing.assert_close(activation(unit), unit)
    values = torch.randn(3, 4, 2, 2, dtype=torch.float64)
    output = activation(values)
    torch.testing.assert_close(
        output / output.norm(dim=1, keepdim=True), values / values.norm(dim=1, keepdim=True)
    )


def test_cartesian_projection_is_normal_variance_preserving_and_nonzero() -> None:
    torch.manual_seed(0)
    weights = LinearNoBias(512, 64).weight.detach().flatten()
    assert weights.std().item() == pytest.approx(1.0 / math.sqrt(512), rel=0.05)
    centered = weights - weights.mean()
    kurtosis = (centered**4).mean() / centered.var(unbiased=False) ** 2 - 3.0
    assert abs(kurtosis.item()) < 0.2  # normal, not uniform (-1.2)
    encoder = build_encoder(
        {"data": {"input_shape": [1, 28, 28]}, "model": {"backbone": "small_cnn"}}
    )
    variance = encoder(torch.randn(16, 1, 28, 28)).var()
    assert torch.isfinite(variance) and variance > 0


def test_output_maps_have_exactly_hd_and_hd_plus_one_parameters() -> None:
    for width, dimension in ((128, 3), (512, 1024)):
        linear = build_embedding_head("linear_no_bias", width, dimension)
        gate = build_embedding_head("radial_power_gate", width, dimension)
        assert count_trainable_parameters(linear) == width * dimension
        assert count_trainable_parameters(gate) == width * dimension + 1


def test_radial_gate_is_equivariant_while_a_final_coordinatewise_relu_is_not() -> None:
    rotation = _orthogonal(3)
    samples = torch.randn(32, 3, dtype=torch.float64)
    relu = nn.ReLU()
    assert not torch.allclose(relu(samples @ rotation.T), relu(samples) @ rotation.T)


def test_linear_and_gate_encoders_start_from_the_same_seeded_state() -> None:
    config = {"data": {"input_shape": [1, 28, 28]}, "model": {"backbone": "small_cnn"}}
    torch.manual_seed(7)
    linear = build_encoder(config, head="linear_no_bias")
    torch.manual_seed(7)
    gate = build_encoder(config, head="radial_power_gate")
    gate_backbone = gate.backbone.state_dict()
    for name, value in linear.backbone.state_dict().items():
        assert torch.equal(value, gate_backbone[name])
    assert torch.equal(linear.head.weight, gate.head.projection.weight)
    inputs = torch.randn(4, 1, 28, 28)
    torch.testing.assert_close(linear(inputs), gate(inputs))  # beta=0 is the identity


def test_imagenet_weights_are_pinned_inventories_hold_and_vit_is_gated() -> None:
    with pytest.raises(ValueError, match="immutable"):
        build_backbone(
            "resnet18", input_shape=(3, 224, 224), pretrained=True, pretrained_weights="DEFAULT"
        )
    for name, expected in (("resnet18", 11_176_512), ("resnet50", 23_508_032)):
        backbone = build_backbone(name, input_shape=(3, 224, 224))
        assert count_trainable_parameters(backbone) == expected
    with pytest.raises(ValueError, match="blocked_external_pin"):
        build_backbone("vit_s16", input_shape=(3, 224, 224))
