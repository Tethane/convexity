"""Owned problem/result records and structural extension protocols for control.

Ordinary workflows may import these names directly from convexity.math.control.
See docs/math/reference/control.md for public signatures and mathematical conventions.
"""
from .problems import (
    FiniteMDP,
    ControlResult,
    BeliefResult,
    LQRResult,
    ScenarioTree,
    TreeControlResult,
    ControlSolver,
)

__all__ = [
    "FiniteMDP",
    "ControlResult",
    "BeliefResult",
    "LQRResult",
    "ScenarioTree",
    "TreeControlResult",
    "ControlSolver",
]
