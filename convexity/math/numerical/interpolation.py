"""Public interpolation namespace; implementation adapters remain private."""
from .builtins.interpolation import (
    Interpolation,
    interpolate,
)
from .builtins.approximation import (
    Spline,
    MultivariateInterpolation,
    PolynomialApproximation,
    spline,
    regular_grid,
    scattered,
    polynomial_approximation,
)

__all__ = ['Interpolation', 'interpolate', 'Spline', 'MultivariateInterpolation', 'PolynomialApproximation', 'spline', 'regular_grid', 'scattered', 'polynomial_approximation']
