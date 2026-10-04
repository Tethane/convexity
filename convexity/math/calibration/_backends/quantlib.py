"""Private quantlib execution for calibration."""

from ..interfaces import NativeCalibrationResult
import numpy as np
import QuantLib as ql
from ..._finance import array, integer, evaluation_date


def quantlib_calibrate(
    model,
    helpers,
    valuation_date,
    *,
    weights=None,
    fixed_parameters=None,
    max_iterations=1000,
):
    """Mutate a supplied QL model using native helper errors and LM calibration.

    Returns parameters, helper residuals and native stopping status. Native
    helpers define relative-price/price/implied-vol errors themselves.
    """
    helpers = list(helpers)
    if not helpers:
        raise ValueError("calibration helpers required")
    w = np.ones(len(helpers)) if weights is None else array(weights, "weights", ndim=1)
    if w.shape != (len(helpers),) or np.any(w < 0) or (not np.any(w > 0)):
        raise ValueError("weights must align and have positive mass")
    n = integer(max_iterations, "max iterations", 20)
    with evaluation_date(valuation_date):
        model.calibrate(
            helpers,
            ql.LevenbergMarquardt(),
            ql.EndCriteria(n, min(100, n // 2), 1e-08, 1e-08, 1e-08),
            ql.NoConstraint(),
            w.tolist(),
            [] if fixed_parameters is None else list(fixed_parameters),
        )
        return NativeCalibrationResult(
            parameters=array(list(model.params()), "parameters"),
            residuals=array([h.calibrationError() for h in helpers], "residuals"),
            end_criteria=model.endCriteria(),
            success=model.endCriteria()
            in (
                ql.EndCriteria.StationaryPoint,
                ql.EndCriteria.StationaryFunctionValue,
                ql.EndCriteria.StationaryFunctionAccuracy,
                ql.EndCriteria.ZeroGradientNorm,
            ),
        )
