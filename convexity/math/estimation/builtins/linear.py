"""Linear coefficient estimation: explicit design columns, including intercepts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Estimate


from dataclasses import dataclass

from .. import _validation as v
from .._backends import sklearn, statsmodels
from ..interfaces import LinearProblem, fit


@dataclass(frozen=True)
class LinearLeastSquares:
    """Fit a linear design to observations, retaining coefficient covariance and fit diagnostics when available."""

    covariance_type: str = "nonrobust"

    def __post_init__(self) -> None:
        v.choice(
            self.covariance_type,
            ("nonrobust", "HC0", "HC1", "HC2", "HC3"),
            "covariance_type",
        )

    def __call__(self, problem: LinearProblem):
        return statsmodels.linear(problem, covariance_type=self.covariance_type)


@dataclass(frozen=True)
class RobustLinear:
    """Fit a linear response with the selected robust loss; robustness does not remove identification requirements."""

    loss: str = "huber"
    tuning: float | None = None
    covariance_type: str = "H1"
    tolerance: float = 1e-8
    maxiter: int = 200

    def __post_init__(self) -> None:
        v.choice(self.loss, ("huber", "tukey"), "loss")
        tuning = (
            (1.345 if self.loss == "huber" else 4.685)
            if self.tuning is None
            else self.tuning
        )
        object.__setattr__(self, "tuning", v.scalar(tuning, "tuning", positive=True))
        v.choice(self.covariance_type, ("H1", "H2", "H3"), "covariance_type")
        v.scalar(self.tolerance, "tolerance", positive=True)
        v.integer(self.maxiter, "maxiter", minimum=2)

    def __call__(self, problem: LinearProblem):
        return statsmodels.robust_linear(
            problem,
            loss=self.loss,
            tuning=self.tuning,
            covariance_type=self.covariance_type,
            tolerance=self.tolerance,
            maxiter=self.maxiter,
        )


@dataclass(frozen=True)
class RegularizedLinear:
    """SSE/(2n) + alpha * [l1_ratio*||theta||_1 + (1-l1_ratio)*||theta||²/2].

    Every supplied column is penalized, including an explicit constant column.
    No intercept is inserted, centered, or exempted; no features are rescaled.
    """

    alpha: float = 1.0
    l1_ratio: float = 0.0
    tolerance: float = 1e-8
    maxiter: int = 10000

    def __post_init__(self) -> None:
        v.scalar(self.alpha, "alpha", positive=True)
        ratio = v.scalar(self.l1_ratio, "l1_ratio")
        if not 0 <= ratio <= 1:
            raise ValueError("l1_ratio must lie in [0,1]")
        v.scalar(self.tolerance, "tolerance", positive=True)
        v.integer(self.maxiter, "maxiter")

    def __call__(self, problem: LinearProblem):
        return sklearn.regularized_linear(
            problem,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            tolerance=self.tolerance,
            maxiter=self.maxiter,
        )


def linear_least_squares(
    design: ArrayLike,
    response: ArrayLike,
    *,
    names: Sequence[str] | None = None,
    weights: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    covariance_type: str = "nonrobust",
) -> Estimate:
    """OLS, WLS (precision weights) or GLS (relative observation covariance)."""
    return fit(
        LinearProblem(design, response, names, weights, sigma),
        LinearLeastSquares(covariance_type),
    )


def robust_linear(
    design: ArrayLike,
    response: ArrayLike,
    *,
    names: Sequence[str] | None = None,
    loss: str = "huber",
    tuning: float | None = None,
    covariance_type: str = "H1",
    tolerance: float = 1e-8,
    maxiter: int = 200,
    check_convergence: bool = True,
) -> Estimate:
    """Fit a robust M-estimator to design (n,p) and response (n,).

    No intercept or scaling is implicit. loss/tuning select the residual influence
    function; covariance_type selects the asymptotic covariance estimator."""
    return fit(
        LinearProblem(design, response, names),
        RobustLinear(loss, tuning, covariance_type, tolerance, maxiter),
        check_convergence=check_convergence,
    )


def regularized_linear(
    design: ArrayLike,
    response: ArrayLike,
    *,
    names: Sequence[str] | None = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.0,
    tolerance: float = 1e-8,
    maxiter: int = 10000,
    check_convergence: bool = True,
) -> Estimate:
    """Fit a penalized linear design (n,p) against scalar response (n,).

    No intercept or scaling is added. l1_ratio interpolates ridge and lasso.
    Return Estimate; inspect diagnostics for convergence and uncertainty semantics."""
    return fit(
        LinearProblem(design, response, names),
        RegularizedLinear(alpha, l1_ratio, tolerance, maxiter),
        check_convergence=check_convergence,
    )


def ridge(
    design: ArrayLike,
    response: ArrayLike,
    *,
    alpha: float = 1.0,
    names: Sequence[str] | None = None,
) -> Estimate:
    """Fit y=X@theta with an L2 penalty and no implicit intercept/scaling.

    Design is (n,p), response (n,); alpha>0. Return a regularized Estimate."""
    return regularized_linear(design, response, alpha=alpha, l1_ratio=0.0, names=names)


def lasso(
    design: ArrayLike,
    response: ArrayLike,
    *,
    alpha: float = 1.0,
    names: Sequence[str] | None = None,
    tolerance: float = 1e-8,
    maxiter: int = 10000,
) -> Estimate:
    """Fit y=X@theta with an L1 penalty and no implicit intercept/scaling.

    Design is (n,p), response (n,); alpha>0. Return a regularized Estimate."""
    return regularized_linear(
        design,
        response,
        alpha=alpha,
        l1_ratio=1.0,
        names=names,
        tolerance=tolerance,
        maxiter=maxiter,
    )


def elastic_net(
    design: ArrayLike,
    response: ArrayLike,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    names: Sequence[str] | None = None,
    tolerance: float = 1e-8,
    maxiter: int = 10000,
) -> Estimate:
    """Fit y=X@theta with an elastic-net penalty and no implicit intercept/scaling.

    Design is (n,p), response (n,); alpha>0 and l1_ratio in [0,1].
    Return Estimate; penalized curvature is not unpenalized sampling covariance."""
    return regularized_linear(
        design,
        response,
        alpha=alpha,
        l1_ratio=l1_ratio,
        names=names,
        tolerance=tolerance,
        maxiter=maxiter,
    )
