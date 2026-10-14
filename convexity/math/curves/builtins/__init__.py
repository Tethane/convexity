"""Concrete specifications and convenience functions for curves.

Ordinary workflows may import these names directly from convexity.math.curves.
"""

from .curves import (
    YieldCurve,
    flat_curve,
    discount_curve,
    zero_curve,
    bootstrap,
    forward_price,
    futures_basis,
    fit_discount_curve,
)
from .curves import deposit_quote, swap_quote
from .curves import ois_quote

__all__ = [
    "YieldCurve",
    "flat_curve",
    "discount_curve",
    "zero_curve",
    "bootstrap",
    "forward_price",
    "futures_basis",
    "fit_discount_curve",
    "deposit_quote",
    "swap_quote",
    "ois_quote",
]
