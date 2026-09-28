"""Scalar/vector empirical laws from supplied values and optional masses."""

from .._backends.empirical import empirical_law
from ..interfaces import Distribution


def empirical(values, weights=None) -> Distribution:
    """Finite law on (n,) scalar or (n,d) vector values; repeated atoms add mass.

    Weights are nonnegative relative masses. Zero-weight rows are discarded.
    Quantiles are inverse ECDFs with no interpolation or inferred tails.
    """
    return empirical_law(values, weights)
