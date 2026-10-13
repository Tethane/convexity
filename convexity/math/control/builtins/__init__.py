"""Concrete specifications and convenience functions for control.

Ordinary workflows may import these names directly from convexity.math.control.
See docs/math/reference/control.md for public signatures and mathematical conventions.
"""
from .solvers import (
    finite_horizon,
    value_iteration,
    belief_update,
    belief_plan,
    lqr,
    scenario_mpc,
    generator_hjb,
    impulse_control,
)

__all__ = [
    "finite_horizon",
    "value_iteration",
    "belief_update",
    "belief_plan",
    "lqr",
    "scenario_mpc",
    "generator_hjb",
    "impulse_control",
]
