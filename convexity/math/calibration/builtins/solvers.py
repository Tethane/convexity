"""SciPy inverse problems and native QuantLib calibration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike

if TYPE_CHECKING:
    from .. import NativeCalibrationResult
if TYPE_CHECKING:
    from ..._finance_types import (
        NativeCalibratedModel,
        NativeCalibrationHelper,
        NativeDate,
    )


if TYPE_CHECKING:
    from .. import CalibrationProblem, CalibrationResult, Calibrator


from dataclasses import dataclass

import numpy as np

from ..._finance import integer, positive
from .._backends import quantlib as _quantlib
from .._backends import scipy as _scipy
from ..interfaces import CalibrationResult


@dataclass(frozen=True)
class LeastSquaresCalibration:
    """Calibrate parameters by weighted residual minimization with explicit solver controls."""

    tolerance: float = 1e-10
    max_evaluations: int = 3000
    rank_tolerance: float = 1e-07

    def __post_init__(self) -> None:
        positive(self.tolerance, "tolerance")
        integer(self.max_evaluations, "max evaluations", 1)
        positive(self.rank_tolerance, "rank tolerance")
        if self.rank_tolerance >= 1:
            raise ValueError("relative rank tolerance must be less than one")

    def fit(self, problem):
        return _scipy.fit(self, problem)


def calibrate(
    problem: CalibrationProblem,
    *,
    solver: Calibrator | None = None,
    require_success: bool = True,
) -> CalibrationResult:
    """Fit CalibrationProblem residuals using a built-in or custom solver.

    Return parameters, residual/Jacobian diagnostics, rank and optional covariance.
    require_success=True rejects failed solves; local convergence is not global optimality."""
    result = (solver or LeastSquaresCalibration()).fit(problem)
    if not isinstance(result, CalibrationResult):
        raise TypeError("calibrator must return CalibrationResult")
    if (
        result.parameters.shape != problem.initial.shape
        or result.residuals.shape != problem.observations.shape
    ):
        raise ValueError(
            "calibration result parameter and observation axes must match the problem"
        )
    if np.any(result.parameters < problem.lower) or np.any(
        result.parameters > problem.upper
    ):
        raise ValueError("calibration result violates parameter bounds")
    if require_success and (not result.success):
        raise RuntimeError(result.message)
    return result


def quantlib_calibrate(
    model: NativeCalibratedModel,
    helpers: Sequence[NativeCalibrationHelper],
    valuation_date: DateLike | NativeDate,
    *,
    weights: ArrayLike | None = None,
    fixed_parameters: Sequence[bool] | None = None,
    max_iterations: int = 1000,
) -> NativeCalibrationResult:
    """Mutate a supplied QL model using native helper errors and LM calibration.

    Returns parameters, helper residuals and native stopping status. Native
    helpers define relative-price/price/implied-vol errors themselves.
    """
    return _quantlib.quantlib_calibrate(
        model,
        helpers,
        valuation_date,
        weights=weights,
        fixed_parameters=fixed_parameters,
        max_iterations=max_iterations,
    )
