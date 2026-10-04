"""Nonlinear least squares and explicit robust residual losses."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import Estimator, LeastSquaresProblem


import numpy as np

if TYPE_CHECKING:
    from .. import Estimate, ParameterSpace


from dataclasses import dataclass

from .. import _validation as v
from .._backends import scipy
from ..interfaces import LeastSquaresProblem, fit
from ..interfaces.parameters import as_space


@dataclass(frozen=True)
class LeastSquares:
    """Estimate parameters by minimizing supplied residuals; initial parameters, bounds and covariance interpretation are explicit."""

    method: str = "trf"
    loss: str = "linear"
    f_scale: float = 1.0
    absolute_sigma: bool = False
    tolerance: float = 1e-9
    max_evaluations: int = 2000

    def __post_init__(self) -> None:
        v.choice(self.method, ("trf", "dogbox", "lm"), "method")
        v.choice(self.loss, ("linear", "soft_l1", "huber", "cauchy", "arctan"), "loss")
        v.scalar(self.f_scale, "f_scale", positive=True)
        v.boolean(self.absolute_sigma, "absolute_sigma")
        if v.scalar(self.tolerance, "tolerance", positive=True) <= np.finfo(float).eps:
            raise ValueError("tolerance must exceed machine epsilon")
        v.integer(self.max_evaluations, "max_evaluations")

    def __call__(self, problem: LeastSquaresProblem):
        return scipy.least_squares(
            problem,
            method=self.method,
            loss=self.loss,
            f_scale=self.f_scale,
            absolute_sigma=self.absolute_sigma,
            tolerance=self.tolerance,
            max_evaluations=self.max_evaluations,
        )


def least_squares(
    residuals: Callable[[FloatArray], ArrayLike],
    initial: ParameterSpace | ArrayLike,
    *,
    jacobian: Callable[[FloatArray], ArrayLike] | None = None,
    loss: str = "linear",
    f_scale: float = 1.0,
    absolute_sigma: bool = False,
    estimator: Estimator[LeastSquaresProblem] | None = None,
    check_convergence: bool = True,
) -> Estimate:
    """Fit r(theta). Divide residuals by known SDs and set absolute_sigma=True.

    Otherwise covariance is scaled by SSE/(n-p). Robust losses return point
    estimates without attaching ordinary least-squares covariance.
    """
    if estimator is not None and (
        loss != "linear" or f_scale != 1.0 or absolute_sigma is not False
    ):
        raise ValueError(
            "configure loss, f_scale and absolute_sigma on the supplied estimator"
        )
    tool = (
        LeastSquares(loss=loss, f_scale=f_scale, absolute_sigma=absolute_sigma)
        if estimator is None
        else estimator
    )
    return fit(
        LeastSquaresProblem(residuals, as_space(initial), jacobian),
        tool,
        check_convergence=check_convergence,
    )
