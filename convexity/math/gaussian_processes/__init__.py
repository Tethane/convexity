"""Kernels and exact Gaussian-process regression, independent of applications.

    from convexity.math import gaussian_processes as gp
    fitted = gp.GaussianProcessRegressor(gp.RBF(), alpha=.01).fit(x, y)
    law = fitted.posterior(query)

Only NumPy loads on family import. SciPy and scikit-learn load on use.
"""

from . import builtins, interfaces
from .builtins import (
    Constant,
    DotProduct,
    GaussianProcessRegressor,
    Matern,
    Periodic,
    RBF,
    RationalQuadratic,
    WhiteNoise,
    gaussian_process_regression,
)
from .interfaces import (
    ConvergenceError,
    Diagnostics,
    FittedGaussianProcess,
    GaussianPrediction,
    Kernel,
    Regressor,
    from_sklearn,
)

__all__ = [
    "builtins",
    "interfaces",
    "Constant",
    "DotProduct",
    "GaussianProcessRegressor",
    "Matern",
    "Periodic",
    "RBF",
    "RationalQuadratic",
    "WhiteNoise",
    "gaussian_process_regression",
    "ConvergenceError",
    "Diagnostics",
    "FittedGaussianProcess",
    "GaussianPrediction",
    "Kernel",
    "Regressor",
    "from_sklearn",
]
