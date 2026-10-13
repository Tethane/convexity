"""Concrete specifications and convenience functions for sensitivities.

Ordinary workflows may import these names directly from convexity.math.sensitivities.
See docs/math/reference/sensitivities.md for public signatures and mathematical conventions.
"""
from .geometry import sensitivities, bump_sensitivities, hedge, curve_sensitivities

__all__ = ["sensitivities", "bump_sensitivities", "hedge", "curve_sensitivities"]
