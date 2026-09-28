"""Public integrate namespace; implementation adapters remain private."""
from .builtins.integrate import (
    MultidimensionalQuadrature,
    GaussianQuadrature,
    GaussianRule,
    nquad,
    gauss_quad,
    quadrature_rule,
    integrate_density,
    AdaptiveQuadrature,
    quad,
)

__all__ = ['MultidimensionalQuadrature', 'GaussianQuadrature', 'GaussianRule', 'nquad', 'gauss_quad', 'quadrature_rule', 'integrate_density', 'AdaptiveQuadrature', 'quad']
