"""scikit-learn conditioning, kernel algebra and marginal-likelihood fitting."""

from copy import deepcopy
from typing import Any
import warnings

import numpy as np
from sklearn.gaussian_process.kernels import ExpSineSquared

from ..interfaces import Diagnostics, FittedGaussianProcess


class OneDimensionalPeriodic(ExpSineSquared):
    def __call__(self, X, Y=None, eval_gradient=False):
        if X.shape[1] != 1 or (Y is not None and Y.shape[1] != 1):
            raise ValueError("Periodic requires one-dimensional features")
        return super().__call__(X, Y, eval_gradient=eval_gradient)

    def diag(self, X):
        if X.shape[1] != 1:
            raise ValueError("Periodic requires one-dimensional features")
        return super().diag(X)


def periodic_kernel(length, period, length_bounds, period_bounds):
    return OneDimensionalPeriodic(length, period, length_bounds, period_bounds)


def fit(spec, kernel, x, y, alpha):
    from scipy.optimize import minimize
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor

    # Validate dimensions before sklearn's optimizer sees the covariance.
    kernel.diag(x)
    if spec.optimize and kernel.theta.size:
        bounds = kernel.bounds
        if bounds.shape != (len(kernel.theta), 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("free kernel parameters require finite increasing log bounds")
        if np.any(kernel.theta < bounds[:, 0]) or np.any(kernel.theta > bounds[:, 1]):
            raise ValueError("initial kernel theta must lie within its bounds")
    runs = []

    def optimizer(objective, initial_theta, bounds):
        result = minimize(objective, initial_theta, jac=True, bounds=bounds,
                          method="L-BFGS-B", options={"maxiter": spec.maxiter,
                                                    "ftol": spec.tolerance,
                                                    "gtol": spec.tolerance})
        runs.append(result)
        return result.x, result.fun

    optimizer_option: Any = optimizer if spec.optimize else None
    model = GaussianProcessRegressor(kernel=deepcopy(kernel._kernel), alpha=deepcopy(alpha),
                                     normalize_y=spec.normalize_y,
                                     optimizer=optimizer_option,
                                     n_restarts_optimizer=spec.restarts,
                                     random_state=spec.seed, copy_X_train=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(np.array(x), np.array(y))
    messages = tuple(str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning))
    for warning in caught:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
    best = min(runs, key=lambda r: r.fun) if runs else None
    converged = best is None or bool(best.success and np.isfinite(best.fun))
    diagnostics = Diagnostics(converged=converged, optimized=bool(runs),
                              message="Fixed hyperparameters" if best is None else str(best.message),
                              iterations=0 if best is None else int(best.nit),
                              optimizer_runs=len(runs), warnings=messages)
    # Retain fitted numeric state without the optimizer closure and its run history.
    model.set_params(optimizer=None)
    return FittedGaussianProcess(model, diagnostics)
