"""Weighted inverse problems with explicit parameter and observation axes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike

from ..._finance import array, integer, scalar
from ..._typing import FloatArray
from ..._validation import boolean


@dataclass(frozen=True)
class CalibrationProblem:
    """Market targets, model response and parameter specification for numerical calibration."""

    predict: Callable[[FloatArray], ArrayLike]
    observations: FloatArray
    initial: FloatArray
    weights: FloatArray = cast("FloatArray", None)
    lower: FloatArray = cast("FloatArray", -np.inf)
    upper: FloatArray = cast("FloatArray", np.inf)
    regularization: float = 0.0
    prior: FloatArray = cast("FloatArray", None)

    def __post_init__(self) -> None:
        y = array(self.observations, "observations", ndim=1)
        x = array(self.initial, "initial", ndim=1)
        w = (
            np.ones_like(y)
            if self.weights is None
            else array(self.weights, "weights", ndim=1)
        )
        lo = np.broadcast_to(array(self.lower, "lower", finite=False), x.shape).copy()
        hi = np.broadcast_to(array(self.upper, "upper", finite=False), x.shape).copy()
        if (
            w.shape != y.shape
            or np.any(w < 0)
            or not np.any(w > 0)
            or np.any(lo >= hi)
            or np.any(x < lo)
            or np.any(x > hi)
        ):
            raise ValueError("invalid quote weights or parameter bounds/initial point")
        prior = x if self.prior is None else array(self.prior, "prior", ndim=1)
        if prior.shape != x.shape or not callable(self.predict):
            raise ValueError("prediction callback and aligned prior required")
        scalar(self.regularization, "regularization", minimum=0)
        for n, v in [
            ("observations", y),
            ("initial", x),
            ("weights", w),
            ("lower", lo),
            ("upper", hi),
            ("prior", prior),
        ]:
            object.__setattr__(self, n, array(v, n, finite=n not in ("lower", "upper")))

    def __init__(
        self,
        predict: Callable[[FloatArray], ArrayLike],
        observations: ArrayLike,
        initial: ArrayLike,
        weights: ArrayLike | None = None,
        lower: ArrayLike = -float("inf"),
        upper: ArrayLike = float("inf"),
        regularization: float = 0.0,
        prior: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "predict", predict)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "regularization", regularization)
        object.__setattr__(self, "prior", prior)
        self.__post_init__()


@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated parameters and residual/solver diagnostics; a good fit does not prove identification."""

    parameters: FloatArray
    residuals: FloatArray
    jacobian: FloatArray
    singular_values: FloatArray
    rank: int
    covariance: FloatArray | None
    success: bool
    message: str
    evaluations: int
    active_bounds: FloatArray

    def __post_init__(self) -> None:
        for n in (
            "parameters",
            "residuals",
            "jacobian",
            "singular_values",
            "active_bounds",
        ):
            object.__setattr__(self, n, array(getattr(self, n), n))
        n, m = self.parameters.size, self.residuals.size
        if (
            self.parameters.ndim != 1
            or self.residuals.ndim != 1
            or self.jacobian.shape != (m, n)
            or self.singular_values.shape != (min(m, n),)
            or self.active_bounds.shape != (n,)
            or not np.isin(self.active_bounds, [-1, 0, 1]).all()
            or np.any(self.singular_values < 0)
        ):
            raise ValueError("calibration result axes or diagnostics are inconsistent")
        if integer(self.rank, "rank") > min(m, n):
            raise ValueError("rank exceeds Jacobian dimensions")
        integer(self.evaluations, "evaluations")
        boolean(self.success, "success")
        if self.covariance is not None:
            from ..._covariance import psd

            cov = psd(self.covariance, "parameter covariance")
            if cov.shape != (n, n):
                raise ValueError("parameter covariance must have shape (n,n)")
            object.__setattr__(self, "covariance", cov)

    @property
    def identified(self):
        return self.rank == len(self.parameters)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))

    def __init__(
        self,
        parameters: ArrayLike,
        residuals: ArrayLike,
        jacobian: ArrayLike,
        singular_values: ArrayLike,
        rank: int,
        covariance: ArrayLike | None,
        success: bool,
        message: str,
        evaluations: int,
        active_bounds: ArrayLike,
    ) -> None:
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "active_bounds", active_bounds)
        self.__post_init__()


class Calibrator(Protocol):
    """Solve a calibration problem and return owned parameters, residuals and numerical diagnostics."""

    def fit(self, problem: CalibrationProblem) -> CalibrationResult: ...
