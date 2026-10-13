"""Concrete specifications and convenience functions for volatility.

Ordinary workflows may import these names directly from convexity.math.volatility.
See docs/math/reference/volatility.md for public signatures and mathematical conventions.
"""
from .arch import (ARCHModelForecaster, ARCHForecaster, GARCHForecaster,
                   GJRGARCHForecaster, EGARCHForecaster, ConstantVarianceForecaster, EWMAForecaster)
from .har import HARForecaster
from .innovations import innovation_distribution
from .realized import realized_variance, realized_volatility

__all__ = ['ARCHModelForecaster', 'ARCHForecaster', 'GARCHForecaster', 'GJRGARCHForecaster',
           'EGARCHForecaster', 'ConstantVarianceForecaster', 'EWMAForecaster', 'HARForecaster',
           'innovation_distribution', 'realized_variance', 'realized_volatility']
