"""Reusable, immutable GP regression specifications."""

from dataclasses import dataclass

from .. import _validation as v
from ..interfaces import ConvergenceError, Kernel


@dataclass(frozen=True)
class GaussianProcessRegressor:
    """Exact scalar-response Gaussian regression using scikit-learn.

    alpha is fixed training diagonal variance, in normalized target units when
    normalize_y=True. It is never automatically added to future uncertainty.
    Kernel hyperparameters are fit by marginal likelihood when optimize=True.
    """

    kernel: Kernel | None = None
    alpha: object = 1e-10
    normalize_y: bool = False
    optimize: bool = True
    restarts: int = 0
    seed: int = 0
    tolerance: float = 1e-8
    maxiter: int = 1000
    check_convergence: bool = True

    def __post_init__(self):
        if self.kernel is not None and not isinstance(self.kernel, Kernel):
            raise TypeError("kernel must be a Kernel; use from_sklearn for custom kernels")
        object.__setattr__(self, "alpha", v.variance(self.alpha, "alpha"))
        for name in ("normalize_y", "optimize", "check_convergence"):
            v.boolean(getattr(self, name), name)
        v.integer(self.restarts, "restarts")
        v.integer(self.seed, "seed")
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32")
        if self.restarts and not self.optimize:
            raise ValueError("restarts require optimize=True")
        v.integer(self.maxiter, "maxiter", 1)
        object.__setattr__(self, "tolerance", v.scalar(self.tolerance, "tolerance", minimum=0, strict=True))

    def fit(self, features, response):
        from .._backends.sklearn import fit
        from .kernels import Constant, RBF

        x = v.features(features)
        y = v.array(response, "response", ndim=1)
        if y.shape != (len(x),):
            raise ValueError("response must align with training rows")
        alpha = v.variance(self.alpha, "alpha", len(x))
        kernel = self.kernel if self.kernel is not None else Constant() * RBF()
        result = fit(self, kernel, x, y, alpha)
        if self.check_convergence and not result.diagnostics.converged:
            raise ConvergenceError(result)
        return result


def gaussian_process_regression(features, response, **kwargs):
    return GaussianProcessRegressor(**kwargs).fit(features, response)
