"""Concrete specifications and convenience functions for utility.

Ordinary workflows may import these names directly from convexity.math.utility.
See docs/math/reference/utility.md for public signatures and mathematical conventions.
"""
from .functions import (
    LinearUtility,
    LogUtility,
    ExponentialUtility,
    PowerUtility,
    SmoothUtility,
    expected_utility,
    certainty_equivalent,
    mean_variance,
    mean_cvar,
    preference_penalty,
)

__all__ = [
    "LinearUtility",
    "LogUtility",
    "ExponentialUtility",
    "PowerUtility",
    "SmoothUtility",
    "expected_utility",
    "certainty_equivalent",
    "mean_variance",
    "mean_cvar",
    "preference_penalty",
]
