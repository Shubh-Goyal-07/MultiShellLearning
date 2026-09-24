"""Learned, strictly ordered, scale-constrained ShellMetric radii."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn


class OrderedRadii(nn.Module):
    """Represent ``S`` radii with ``S-1`` gap logits and weighted RMS one.

    The first logit is fixed at zero, removing the softmax translation degree
    of freedom.  A one-shell instance has no parameters and always returns 1.
    """

    def __init__(
        self,
        capacities: Sequence[int],
        *,
        gap_floor_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(tuple(int(value) for value in capacities), dtype=torch.float64)
        if values.ndim != 1 or values.numel() < 1 or torch.any(values <= 0):
            raise ValueError("capacities must be a non-empty sequence of positive integers")
        if not 0.0 < float(gap_floor_fraction) < 1.0:
            raise ValueError("gap_floor_fraction must lie in (0, 1)")
        self.shell_count = int(values.numel())
        self.gap_floor_fraction = float(gap_floor_fraction)
        self.register_buffer("capacities", values, persistent=True)
        if self.shell_count > 1:
            self.gamma_tail = nn.Parameter(torch.zeros(self.shell_count - 1))
        else:
            self.register_parameter("gamma_tail", None)

    @classmethod
    def from_plan(cls, plan: Any) -> OrderedRadii:
        """Build the radii exactly as recorded in a plan's semantic settings."""

        return cls(
            plan.capacities,
            gap_floor_fraction=float(plan.radius_settings["gap_floor_fraction"]),
        )

    def gaps(self) -> Tensor:
        if self.shell_count == 1:
            return self.capacities.new_ones(1)
        assert self.gamma_tail is not None
        gamma = torch.cat((self.gamma_tail.new_zeros(1), self.gamma_tail))
        q = torch.softmax(gamma, dim=0)
        return self.gap_floor_fraction / self.shell_count + (1.0 - self.gap_floor_fraction) * q

    def forward(self) -> Tensor:
        if self.shell_count == 1:
            # Follow the parameter/buffer dtype while exposing no trainable scalar.
            return self.capacities.new_ones(1)
        gaps = self.gaps()
        unscaled = torch.cumsum(gaps, dim=0)
        weights = self.capacities.to(device=unscaled.device, dtype=unscaled.dtype)
        weights = weights / weights.sum()
        rms = torch.sqrt(torch.sum(weights * unscaled.square()))
        return unscaled / rms

    @property
    def radii(self) -> Tensor:
        return self()

    def extra_repr(self) -> str:
        return f"shell_count={self.shell_count}, gap_floor_fraction={self.gap_floor_fraction}"


__all__ = ["OrderedRadii"]
