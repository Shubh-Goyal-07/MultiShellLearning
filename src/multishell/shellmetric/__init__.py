"""Prototype-free ShellMetric planning, training, and evaluation."""

from .autok import AutoKResult, select_shell_count
from .heads import LinearNoBias, RadialPowerGate, build_embedding_head
from .loss import ShellMetricLoss, shellmetric_loss
from .plan import ShellPlan, assign_classes_to_shells, shell_capacities, smax
from .radii import OrderedRadii

__all__ = [
    "AutoKResult",
    "LinearNoBias",
    "OrderedRadii",
    "RadialPowerGate",
    "ShellMetricLoss",
    "ShellPlan",
    "assign_classes_to_shells",
    "build_embedding_head",
    "select_shell_count",
    "shell_capacities",
    "shellmetric_loss",
    "smax",
]
