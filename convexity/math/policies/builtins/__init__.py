"""Concrete specifications and convenience functions for policies.

Ordinary workflows may import these names directly from convexity.math.policies.
"""

from .thresholds import Hysteresis, no_action

__all__ = ["Hysteresis", "no_action"]
