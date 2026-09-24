"""Inner activations used by the ShellMetric architecture study."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class ChannelRadialSiLU(nn.Module):
    """Scale each channel vector by a bounded function of its RMS radius."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 2:
            raise ValueError("channel-radial activation expects shape [batch, channels, ...]")
        if not inputs.is_floating_point():
            raise TypeError("channel-radial activation requires floating-point inputs")

        work_dtype = (
            torch.float32
            if inputs.dtype in {torch.float16, torch.bfloat16}
            else inputs.dtype
        )
        work = inputs.to(dtype=work_dtype)
        radius = torch.sqrt(work.square().mean(dim=1, keepdim=True) + self.epsilon)
        multiplier = 2.0 * torch.sigmoid(radius - 1.0)
        return inputs * multiplier.to(dtype=inputs.dtype)


ActivationFactory = Callable[[], nn.Module]

_ACTIVATION_FACTORIES: dict[str, ActivationFactory] = {
    "relu": lambda: nn.ReLU(inplace=False),
    "silu": lambda: nn.SiLU(inplace=False),
    "gelu_exact": lambda: nn.GELU(approximate="none"),
    "channel_radial_silu": ChannelRadialSiLU,
}

ACTIVATION_NAMES = tuple(_ACTIVATION_FACTORIES)


def build_activation(name: str) -> nn.Module:
    """Create one parameter-free activation from the locked Stage-A registry."""

    normalized = name.strip().lower().replace("-", "_")
    try:
        return _ACTIVATION_FACTORIES[normalized]()
    except KeyError as exc:
        choices = ", ".join(ACTIVATION_NAMES)
        raise ValueError(f"unsupported activation {name!r}; choose one of: {choices}") from exc


__all__ = ["ACTIVATION_NAMES", "ChannelRadialSiLU", "build_activation"]
