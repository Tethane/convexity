"""Linear coefficient estimation: explicit design columns, including intercepts."""

from dataclasses import dataclass

from .. import _validation as v
from .._backends import sklearn, statsmodels
from ..interfaces import LinearProblem, fit


@dataclass(frozen=True)
class LinearLeastSquares:
    covariance_type: str = "nonrobust"

    def __post_init__(self):
        v.choice(self.covariance_type, ("nonrobust", "HC0", "HC1", "HC2", "HC3"), "covariance_type")

    def __call__(self, problem: LinearProblem):
        return statsmodels.linear(problem, covariance_type=self.covariance_type)


@dataclass(frozen=True)
class RobustLinear:
    loss: str = "huber"
    tuning: float | None = None
    covariance_type: str = "H1"
    tolerance: float = 1e-8
    maxiter: int = 200

    def __post_init__(self):
        v.choice(self.loss, ("huber", "tukey"), "loss")
        tuning = (1.345 if self.loss == "huber" else 4.685) if self.tuning is None else self.tuning
        object.__setattr__(self, "tuning", v.scalar(tuning, "tuning", positive=True))
        v.choice(self.covariance_type, ("H1", "H2", "H3"), "covariance_type")
        v.scalar(self.tolerance, "tolerance", positive=True)
        v.integer(self.maxiter, "maxiter", minimum=2)

    def __call__(self, problem: LinearProblem):
        return statsmodels.robust_linear(problem, loss=self.loss, tuning=self.tuning,
                                         covariance_type=self.covariance_type,
                                         tolerance=self.tolerance, maxiter=self.maxiter)


@dataclass(frozen=True)
class RegularizedLinear:
    """SSE/(2n) + alpha * [l1_ratio*||theta||_1 + (1-l1_ratio)*||theta||²/2].

    Every supplied column is penalized, including an explicit constant column.
    No intercept is inserted, centered, or exempted; no features are rescaled.
    """

    alpha: float = 1.
    l1_ratio: float = 0.
    tolerance: float = 1e-8
    maxiter: int = 10000

    def __post_init__(self):
        v.scalar(self.alpha, "alpha", positive=True)
        ratio = v.scalar(self.l1_ratio, "l1_ratio")
        if not 0 <= ratio <= 1:
            raise ValueError("l1_ratio must lie in [0,1]")
        v.scalar(self.tolerance, "tolerance", positive=True)
        v.integer(self.maxiter, "maxiter")

    def __call__(self, problem: LinearProblem):
        return sklearn.regularized_linear(problem, alpha=self.alpha, l1_ratio=self.l1_ratio,
                                          tolerance=self.tolerance, maxiter=self.maxiter)


def linear_least_squares(design, response, *, names=None, weights=None, sigma=None,
                         covariance_type="nonrobust"):
    """OLS, WLS (precision weights) or GLS (relative observation covariance)."""
    return fit(LinearProblem(design, response, names, weights, sigma), LinearLeastSquares(covariance_type))


def robust_linear(design, response, *, names=None, loss="huber", tuning=None, covariance_type="H1",
                   tolerance=1e-8, maxiter=200, check_convergence=True):
    return fit(LinearProblem(design, response, names),
               RobustLinear(loss, tuning, covariance_type, tolerance, maxiter),
               check_convergence=check_convergence)


def regularized_linear(design, response, *, names=None, alpha=1., l1_ratio=0.,
                        tolerance=1e-8, maxiter=10000, check_convergence=True):
    return fit(LinearProblem(design, response, names), RegularizedLinear(alpha, l1_ratio, tolerance, maxiter),
               check_convergence=check_convergence)


def ridge(design, response, *, alpha=1., names=None):
    return regularized_linear(design, response, alpha=alpha, l1_ratio=0., names=names)


def lasso(design, response, *, alpha=1., names=None, tolerance=1e-8, maxiter=10000):
    return regularized_linear(design, response, alpha=alpha, l1_ratio=1., names=names,
                               tolerance=tolerance, maxiter=maxiter)


def elastic_net(design, response, *, alpha=1., l1_ratio=.5, names=None, tolerance=1e-8, maxiter=10000):
    return regularized_linear(design, response, alpha=alpha, l1_ratio=l1_ratio, names=names,
                               tolerance=tolerance, maxiter=maxiter)
