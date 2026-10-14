"""Concrete specifications and convenience functions for sensitivities.

Ordinary workflows may import these names directly from convexity.math.sensitivities.
"""

from .geometry import sensitivities, bump_sensitivities, hedge, curve_sensitivities

__all__ = ["sensitivities", "bump_sensitivities", "hedge", "curve_sensitivities"]
