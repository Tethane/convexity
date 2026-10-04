"""Reusable, immutable GP regression specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, Unpack

from numpy.typing import ArrayLike

from .. import _validation as v
from ..interfaces import ConvergenceError, Kernel
from ..interfaces.regression import FittedGaussianProcess


class GaussianProcessRegressionOptions(TypedDict, total=False):
    """Optional keyword settings for gaussian_process_regression."""

    kernel: Kernel | None
    alpha: object
    normalize_y: bool
    optimize: bool
    restarts: int
    seed: int
    tolerance: float
    maxiter: int
    check_convergence: bool


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

    def __post_init__(self) -> None:
        if self.kernel is not None and not isinstance(self.kernel, Kernel):
            raise TypeError(
                "kernel must be a Kernel; use from_sklearn for custom kernels"
            )
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
        object.__setattr__(
            self,
            "tolerance",
            v.scalar(self.tolerance, "tolerance", minimum=0, strict=True),
        )

    def fit(self, features: ArrayLike, response: ArrayLike) -> FittedGaussianProcess:
        from .._backends.sklearn import fit
        from .kernels import RBF, Constant

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


def gaussian_process_regression(
    features: ArrayLike,
    response: ArrayLike,
    **kwargs: Unpack[GaussianProcessRegressionOptions],
) -> FittedGaussianProcess:
    """Fit GaussianProcessRegressor to features (observation,feature) and aligned response.

    Exact scalar-response Gaussian regression using scikit-learn.
    Return the checked fitted predictor; no feature scaling is implicit.
    Optional settings: kernel, alpha, normalize_y, optimize, restarts, seed, tolerance, maxiter, check_convergence."""
    return GaussianProcessRegressor(**kwargs).fit(features, response)
