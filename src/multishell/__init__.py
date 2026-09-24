"""ShellMetric: prototype-free supervised multi-shell metric learning."""

from __future__ import annotations

from .shellmetric import (
    AutoKResult,
    OrderedRadii,
    ShellMetricLoss,
    ShellPlan,
    select_shell_count,
    shell_capacities,
    smax,
)

__version__ = "1.0.0"

__all__ = [
    "AutoKResult",
    "OrderedRadii",
    "ShellMetricLoss",
    "ShellPlan",
    "__version__",
    "select_shell_count",
    "shell_capacities",
    "smax",
]
