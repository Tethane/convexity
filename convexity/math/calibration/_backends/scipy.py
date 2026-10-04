"""Private scipy execution for calibration."""

import numpy as np
from scipy.optimize import least_squares
from ..._finance import array
from ..interfaces import CalibrationResult


def fit(self, problem):
    p = problem

    def data_residual(x):
        values = array(p.predict(x), "model predictions", ndim=1)
        if values.shape != p.observations.shape:
            raise ValueError("model predictions must align with observations")
        return np.sqrt(p.weights) * (values - p.observations)

    def residual(x):
        data = data_residual(x)
        return (
            np.r_[data, np.sqrt(p.regularization) * (x - p.prior)]
            if p.regularization
            else data
        )

    fit = least_squares(
        residual,
        p.initial,
        bounds=(p.lower, p.upper),
        max_nfev=self.max_evaluations,
        xtol=self.tolerance,
        ftol=self.tolerance,
        gtol=self.tolerance,
    )
    jac = least_squares(
        data_residual, fit.x, jac="3-point", bounds=(p.lower, p.upper), max_nfev=1
    ).jac
    _, singular, vh = np.linalg.svd(jac, full_matrices=False)
    rank = int(np.count_nonzero(singular > self.rank_tolerance * singular[0]))
    dof = int(np.count_nonzero(p.weights)) - len(fit.x)
    covariance = None
    if (
        fit.success
        and rank == len(fit.x)
        and (dof > 0)
        and (not p.regularization)
        and (not np.any(fit.active_mask))
    ):
        covariance = (
            vh.T
            / singular**2
            @ vh
            * (data_residual(fit.x) @ data_residual(fit.x))
            / dof
        )
    return CalibrationResult(
        fit.x,
        p.predict(fit.x) - p.observations,
        jac,
        singular,
        rank,
        covariance,
        bool(fit.success),
        fit.message,
        fit.nfev,
        fit.active_mask,
    )
