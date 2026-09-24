"""Locked ShellMetric backbones, encoders, and pilot classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from torch import Tensor, nn

from .shellmetric.activations import build_activation
from .shellmetric.heads import build_embedding_head

# Section 8.5 inventory: trainable backbone parameters (no embedding map/classifier).
BACKBONE_INVENTORY: dict[tuple[str, str], int] = {
    ("small_cnn", "mnist_stem"): 109_184,
    ("resnet18", "cifar_stem"): 11_168_832,
    ("resnet18", "imagenet_stem"): 11_176_512,
    ("resnet50", "cifar_stem"): 23_500_352,
    ("resnet50", "imagenet_stem"): 23_508_032,
}
PINNED_IMAGENET_WEIGHTS = "IMAGENET1K_V1"
EXTERNAL_BACKBONES = frozenset({"vit_s16", "vit_small_patch16_224"})


class SmallCNNBackbone(nn.Module):
    """The fixed 109,184-parameter MNIST/Fashion-MNIST smoke backbone."""

    output_dim = 128

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, self.output_dim)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs).flatten(1)
        return self.activation(self.projection(features))


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.activation1 = build_activation(activation)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = self._shortcut(in_channels, channels, stride)
        self.activation2 = build_activation(activation)

    @staticmethod
    def _shortcut(in_channels: int, out_channels: int, stride: int) -> nn.Module:
        if stride == 1 and in_channels == out_channels:
            return nn.Identity()
        return nn.Sequential(
            _conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
        )

    @property
    def activation_sites(self) -> tuple[nn.Module, nn.Module]:
        return self.activation1, self.activation2

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        output = self.activation1(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.activation2(output + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int,
        activation: str,
    ) -> None:
        super().__init__()
        width = channels
        out_channels = channels * self.expansion
        self.conv1 = _conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.activation1 = build_activation(activation)
        self.conv2 = _conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.activation2 = build_activation(activation)
        self.conv3 = _conv1x1(width, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = BasicBlock._shortcut(in_channels, out_channels, stride)
        self.activation3 = build_activation(activation)

    @property
    def activation_sites(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self.activation1, self.activation2, self.activation3

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        output = self.activation1(self.bn1(self.conv1(inputs)))
        output = self.activation2(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        return self.activation3(output + residual)


ResidualBlock: TypeAlias = type[BasicBlock] | type[Bottleneck]


class CIFARResNet(nn.Module):
    """Scratch ResNet with a 3x3 stride-one stem and no max-pool."""

    def __init__(
        self,
        block: ResidualBlock,
        layers: Sequence[int],
        *,
        in_channels: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if len(layers) != 4 or any(depth < 1 for depth in layers):
            raise ValueError("layers must contain four positive depths")
        if in_channels < 1:
            raise ValueError("in_channels must be positive")

        self._activation_name = activation
        self._in_channels = 64
        self.conv1 = _conv3x3(in_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.stem_activation = build_activation(activation)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 512 * block.expansion
        self._initialize_weights()

    def _make_layer(
        self,
        block: ResidualBlock,
        channels: int,
        depth: int,
        *,
        stride: int,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [
            block(self._in_channels, channels, stride, self._activation_name)
        ]
        self._in_channels = channels * block.expansion
        blocks.extend(
            block(self._in_channels, channels, 1, self._activation_name) for _ in range(1, depth)
        )
        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def activation_sites(self) -> tuple[nn.Module, ...]:
        sites = [self.stem_activation]
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                sites.extend(block.activation_sites)
        return tuple(sites)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.stem_activation(self.bn1(self.conv1(inputs)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.pool(output).flatten(1)


class CartesianEncoder(nn.Module):
    """Backbone plus a raw, origin-preserving Cartesian output map."""

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dimension: int,
        *,
        feature_dim: int | None = None,
        head: str = "linear_no_bias",
    ) -> None:
        super().__init__()
        backbone_width = int(getattr(backbone, "output_dim", 0))
        if feature_dim is not None and int(feature_dim) != backbone_width:
            raise ValueError("feature_dim must match backbone.output_dim")
        if backbone_width < 1 or embedding_dimension < 1:
            raise ValueError("backbone width and embedding_dimension must be positive")
        self.backbone = backbone
        self.embedding_dimension = int(embedding_dimension)
        self.head_name = head.strip().lower().replace("-", "_")
        self.head = build_embedding_head(self.head_name, backbone_width, self.embedding_dimension)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.backbone(inputs))


class PilotClassifier(nn.Module):
    """Ordinary affine H-to-C classifier used only for cross-fitted OOF logits."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        *,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        backbone_width = int(getattr(backbone, "output_dim", 0))
        if feature_dim is not None and int(feature_dim) != backbone_width:
            raise ValueError("feature_dim must match backbone.output_dim")
        if backbone_width < 1 or num_classes < 2:
            raise ValueError("backbone width must be positive and num_classes at least two")
        self.backbone = backbone
        self.classifier = nn.Linear(backbone_width, int(num_classes))
        self.num_classes = int(num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.backbone(inputs))


def _normalize_architecture(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _build_imagenet_resnet(architecture: str, *, pretrained: bool, weights: str) -> nn.Module:
    """Standard-stem ResNet with explicitly pinned weights (never a moving DEFAULT)."""

    try:
        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
    except Exception as exc:  # pragma: no cover - depends on the optional binary build
        raise RuntimeError("ImageNet ResNets require a compatible torchvision build") from exc
    constructor, enum = (
        (resnet18, ResNet18_Weights) if architecture == "resnet18" else (resnet50, ResNet50_Weights)
    )
    if pretrained and weights.upper() in {"DEFAULT", "LATEST"}:
        raise ValueError("pretrained weights must name an immutable enum, not DEFAULT")
    model = constructor(weights=enum[weights] if pretrained else None)
    model.output_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model


def _verify_inventory(backbone: nn.Module, key: tuple[str, str]) -> nn.Module:
    expected = BACKBONE_INVENTORY[key]
    actual = count_trainable_parameters(backbone)
    if actual != expected:
        raise RuntimeError(
            f"{key[0]} ({key[1]}) has {actual} trainable parameters; the locked inventory "
            f"requires {expected}. A mismatch is an architecture change."
        )
    return backbone


def build_backbone(
    architecture: str = "small_cnn",
    *,
    input_shape: Sequence[int] = (1, 28, 28),
    feature_dim: int = 128,
    pretrained: bool = False,
    pretrained_weights: str = PINNED_IMAGENET_WEIGHTS,
    activation: str = "relu",
    **obsolete: Any,
) -> nn.Module:
    """Build the fixed smoke CNN, scratch CIFAR ResNet, or pinned ImageNet form.

    Inventory variants are checked against Section 8.5's parameter counts.
    """

    if obsolete:
        names = ", ".join(sorted(obsolete))
        raise ValueError(f"obsolete backbone options are not supported: {names}")
    if len(input_shape) != 3:
        raise ValueError("input_shape must be [channels, height, width]")
    name = _normalize_architecture(architecture)
    activation = _normalize_architecture(activation)
    if activation == "native":
        activation = "relu"
    in_channels = int(input_shape[0])
    if name in EXTERNAL_BACKBONES:
        raise ValueError(
            f"{architecture!r} is gated (blocked_external_pin): pin its provider, model ID, "
            "dependency version, weight checksum, and preprocessing before use"
        )
    if name == "small_cnn":
        if pretrained:
            raise ValueError("the smoke CNN has no pretrained variant")
        if feature_dim != SmallCNNBackbone.output_dim:
            raise ValueError("the locked small CNN has feature_dim=128")
        if activation != "relu":
            raise ValueError("the smoke CNN uses its fixed native ReLU activation")
        backbone = SmallCNNBackbone(in_channels)
        return _verify_inventory(backbone, (name, "mnist_stem")) if in_channels == 1 else backbone
    if name not in {"resnet18", "resnet50"}:
        raise ValueError(
            f"unsupported backbone {architecture!r}; choose small_cnn, resnet18, or resnet50"
        )
    if pretrained or max(int(value) for value in input_shape[-2:]) > 64:
        if in_channels != 3 or activation != "relu":
            raise ValueError(f"ImageNet {name} retains its native 3-channel ReLU architecture")
        backbone = _build_imagenet_resnet(name, pretrained=pretrained, weights=pretrained_weights)
        return _verify_inventory(backbone, (name, "imagenet_stem"))
    block, layers = (BasicBlock, (2, 2, 2, 2)) if name == "resnet18" else (Bottleneck, (3, 4, 6, 3))
    backbone = CIFARResNet(block, layers, in_channels=in_channels, activation=activation)
    return _verify_inventory(backbone, (name, "cifar_stem")) if in_channels == 3 else backbone


def _config_values(
    config_or_architecture: Mapping[str, Any] | str | None,
) -> tuple[str, dict[str, Any], Sequence[int] | None]:
    if not isinstance(config_or_architecture, Mapping):
        return str(config_or_architecture or "small_cnn"), {}, None

    model = config_or_architecture.get("model", config_or_architecture)
    if not isinstance(model, Mapping):
        raise TypeError("model configuration must be a mapping")
    values = dict(model)
    architecture = str(values.pop("backbone", values.pop("architecture", "small_cnn")))

    configured_shape: Sequence[int] | None = None
    data = config_or_architecture.get("data")
    if isinstance(data, Mapping) and data.get("input_shape") is not None:
        configured_shape = tuple(int(value) for value in data["input_shape"])
    return architecture, values, configured_shape


def build_encoder(
    config_or_architecture: Mapping[str, Any] | str | None = None,
    *,
    embedding_dimension: int | None = None,
    input_shape: Sequence[int] = (1, 28, 28),
    feature_dim: int = 128,
    head: str | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> CartesianEncoder:
    architecture, values, configured_shape = _config_values(config_or_architecture)
    if configured_shape is not None:
        input_shape = configured_shape
    configured_dimension = values.pop(
        "embedding_dim", values.pop("embedding_dimension", values.pop("dimension", 3))
    )
    configured_head = values.pop("embedding_head", values.pop("head", "linear_no_bias"))
    configured_activation = values.pop("backbone_activation", values.pop("activation", "relu"))
    embedding_dimension = int(
        configured_dimension if embedding_dimension is None else embedding_dimension
    )
    head_name = str(configured_head if head is None else head)
    activation_name = str(configured_activation if activation is None else activation)
    output_bias = bool(values.pop("output_bias", False))
    if output_bias:
        raise ValueError("ShellMetric embedding heads cannot have a translation bias")
    values.update(kwargs)
    backbone = build_backbone(
        architecture,
        input_shape=input_shape,
        feature_dim=int(values.pop("feature_dim", feature_dim)),
        activation=activation_name,
        **values,
    )
    return CartesianEncoder(backbone, embedding_dimension, head=head_name)


def build_pilot_classifier(
    config_or_architecture: Mapping[str, Any] | str | None = None,
    *,
    num_classes: int,
    input_shape: Sequence[int] = (1, 28, 28),
    feature_dim: int = 128,
    activation: str | None = None,
    **kwargs: Any,
) -> PilotClassifier:
    architecture, values, configured_shape = _config_values(config_or_architecture)
    if configured_shape is not None:
        input_shape = configured_shape
    configured_activation = values.pop("backbone_activation", values.pop("activation", "relu"))
    activation_name = str(configured_activation if activation is None else activation)
    for key in (
        "embedding_dim",
        "embedding_dimension",
        "dimension",
        "embedding_head",
        "head",
        "output_bias",
    ):
        values.pop(key, None)
    values.update(kwargs)
    backbone = build_backbone(
        architecture,
        input_shape=input_shape,
        feature_dim=int(values.pop("feature_dim", feature_dim)),
        activation=activation_name,
        **values,
    )
    return PilotClassifier(backbone, num_classes)


def build_model(
    config_or_architecture: Mapping[str, Any] | str | None = None,
    *,
    task: str = "encoder",
    num_classes: int | None = None,
    **kwargs: Any,
) -> nn.Module:
    if task in {"encoder", "embedding"}:
        return build_encoder(config_or_architecture, **kwargs)
    if task in {"pilot", "classifier"}:
        if num_classes is None:
            raise ValueError("num_classes is required for a pilot classifier")
        return build_pilot_classifier(config_or_architecture, num_classes=num_classes, **kwargs)
    raise ValueError(f"unsupported model task: {task!r}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "BACKBONE_INVENTORY",
    "BasicBlock",
    "Bottleneck",
    "CIFARResNet",
    "CartesianEncoder",
    "PilotClassifier",
    "SmallCNNBackbone",
    "build_backbone",
    "build_encoder",
    "build_model",
    "build_pilot_classifier",
    "count_trainable_parameters",
]
