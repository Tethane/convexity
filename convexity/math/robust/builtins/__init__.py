"""Concrete specifications and convenience functions for robust.

Ordinary workflows may import these names directly from convexity.math.robust.
"""

from .formulations import (
    Box,
    Ellipsoid,
    minimax,
    mean_cvar,
    robust_allocation,
    gaussian_chance_constraint,
    kl_worst_case,
    chance_probability,
)

__all__ = [
    "Box",
    "Ellipsoid",
    "minimax",
    "mean_cvar",
    "robust_allocation",
    "gaussian_chance_constraint",
    "kl_worst_case",
    "chance_probability",
]
