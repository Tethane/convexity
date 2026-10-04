"""Concrete specifications and convenience functions for implied_volatility.

Ordinary workflows may import these names directly from convexity.math.implied_volatility.
See docs/reference/implied_volatility.md for public signatures and mathematical conventions.
"""
from .surfaces import (
    implied_volatility,
    SVI,
    SSVI,
    SmileSurface,
    sabr_volatility,
    black_variance_surface,
    dupire_local_volatility,
    risk_neutral_density,
)

__all__ = [
    "implied_volatility",
    "SVI",
    "SSVI",
    "SmileSurface",
    "sabr_volatility",
    "black_variance_surface",
    "dupire_local_volatility",
    "risk_neutral_density",
]
