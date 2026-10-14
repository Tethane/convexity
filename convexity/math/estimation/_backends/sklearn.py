"""One objective convention across scikit-learn ridge, lasso and elastic net."""

import warnings

import numpy as np

from ..interfaces import Diagnostics, Estimate


def regularized_linear(problem, *, alpha, l1_ratio, tolerance, maxiter):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    if problem.weights is not None or problem.sigma is not None:
        raise ValueError("regularized_linear accepts an unweighted design")
    n = len(problem.response)
    if l1_ratio == 0:
        model = Ridge(alpha=n * alpha, fit_intercept=False, solver="svd")
    elif l1_ratio == 1:
        model = Lasso(alpha=alpha, fit_intercept=False, tol=tolerance, max_iter=maxiter)
    else:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            tol=tolerance,
            max_iter=maxiter,
            selection="cyclic",
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(problem.design, problem.response)
    failures = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    for warning in caught:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
    theta = model.coef_
    residuals = np.asarray(problem.response) - np.asarray(problem.design) @ theta
    objective = (
        float(residuals @ residuals) / (2 * n)
        + alpha * l1_ratio * float(np.abs(theta).sum())
        + alpha * (1 - l1_ratio) * float(theta @ theta) / 2
    )
    return Estimate(
        theta,
        names=problem.names,
        objective=objective,
        residuals=residuals,
        nobs=n,
        diagnostics=Diagnostics(
            converged=not failures,
            message="; ".join(str(w.message) for w in failures),
            backend=f"sklearn.{type(model).__name__}",
            iterations=getattr(model, "n_iter_", None),
            rank=int(np.linalg.matrix_rank(problem.design)),
            uncertainty_note="Penalized coefficients have no automatic frequentist covariance; use explicit resampling.",
        ),
    )
