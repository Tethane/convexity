"""Private scipy execution for curves."""

import numpy as np
from ..._finance import array


def fit_discount_curve(
    reference_date, cashflow_times, cashflows, prices, initial, *, model
):
    """Fit a supplied positive discount-factor model(times, parameters) with SciPy."""
    from ...calibration import CalibrationProblem, calibrate
    from ..._finance import qdate
    from ..interfaces import CurveFitResult

    reference = qdate(reference_date).to_date()

    t, c, p = (
        array(cashflow_times, "times", ndim=1),
        array(cashflows, "cashflows", ndim=2),
        array(prices, "prices", ndim=1),
    )
    if c.shape != (len(p), len(t)) or np.any(t < 0):
        raise ValueError("cashflows must have shape (instrument,time)")

    def predict(x):
        d = array(model(t, x), "discounts", ndim=1)
        if d.shape != t.shape or np.any(d <= 0):
            raise ValueError("model must return positive discounts at each time")
        return c @ d

    result = calibrate(CalibrationProblem(predict, p, initial))
    return CurveFitResult(reference, result)
