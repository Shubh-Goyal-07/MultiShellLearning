"""Bias-free Cartesian output maps for ShellMetric encoders."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LinearNoBias(nn.Linear):
    """A raw Cartesian projection whose origin cannot be translated."""

    def __init__(self, in_features: int, out_features: int) -> None:
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        super().__init__(int(in_features), int(out_features), bias=False)

    def reset_parameters(self) -> None:
        # Variance-preserving (LeCun) normal initialization; never zero.
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.in_features))


class RadialPowerGate(nn.Module):
    """Bias-free projection followed by a learned equivariant radial power map."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.projection = LinearNoBias(in_features, out_features)
        self.beta = nn.Parameter(torch.zeros(()))
        self.epsilon = float(epsilon)
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    @staticmethod
    def _constrained_power(beta: Tensor) -> Tensor:
        power = torch.exp(math.log(2.0) * torch.tanh(beta))
        lower = torch.nextafter(power.new_tensor(0.5), power.new_tensor(float("inf")))
        upper = torch.nextafter(power.new_tensor(2.0), power.new_tensor(float("-inf")))
        return power.clamp(min=lower, max=upper)

    @property
    def power(self) -> Tensor:
        """Return the constrained scalar exponent in the open interval (0.5, 2)."""

        return self._constrained_power(self.beta)

    def transform(self, projected: Tensor) -> Tensor:
        """Apply only the radial map to already projected vectors."""

        if projected.ndim < 1 or projected.shape[-1] != self.out_features:
            raise ValueError(f"projected vectors must end in dimension {self.out_features}")
        if not projected.is_floating_point():
            raise TypeError("radial power gate requires floating-point inputs")

        work_dtype = (
            torch.float32 if projected.dtype in {torch.float16, torch.bfloat16} else projected.dtype
        )
        work = projected.to(dtype=work_dtype)
        beta = self.beta.to(device=work.device, dtype=work_dtype)
        power = self._constrained_power(beta)
        radius = torch.linalg.vector_norm(work, dim=-1, keepdim=True)
        scale = torch.exp((power - 1.0) * torch.log(radius + self.epsilon))
        return (work * scale).to(dtype=projected.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.transform(self.projection(inputs))


def build_embedding_head(
    name: str,
    in_features: int,
    embedding_dimension: int,
) -> nn.Module:
    """Build one of the two output maps admitted by the final specification."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized == "linear_no_bias":
        return LinearNoBias(in_features, embedding_dimension)
    if normalized == "radial_power_gate":
        return RadialPowerGate(in_features, embedding_dimension)
    raise ValueError(
        f"unsupported embedding head {name!r}; choose linear_no_bias or radial_power_gate"
    )


__all__ = ["LinearNoBias", "RadialPowerGate", "build_embedding_head"]
