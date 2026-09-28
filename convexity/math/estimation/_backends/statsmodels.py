"""Classical and robust coefficient estimation through statsmodels."""

import numpy as np

from ..interfaces import Diagnostics, Estimate


def linear(problem, *, covariance_type):
    from statsmodels.regression.linear_model import GLS, OLS, WLS

    if problem.sigma is not None:
        model = GLS(problem.response, problem.design, sigma=problem.sigma)
    elif problem.weights is not None:
        model = WLS(problem.response, problem.design, weights=problem.weights)
    else:
        model = OLS(problem.response, problem.design)
    raw = model.fit(cov_type=covariance_type, use_t=covariance_type == "nonrobust")
    rank = int(np.linalg.matrix_rank(model.wexog))
    covariance = None
    df = None
    note = "Gaussian errors with relative observation covariance estimated up to scale."
    if rank < len(raw.params) or raw.df_resid <= 0:
        note = "Covariance unavailable: full column rank and positive residual degrees of freedom required."
    else:
        candidate = raw.cov_params()
        if np.all(np.isfinite(candidate)):
            covariance = candidate
            if covariance_type == "nonrobust":
                df = float(raw.df_resid)
            else:
                note = "Heteroskedasticity-consistent covariance; asymptotic normal intervals."
        else:
            note = "Backend covariance is nonfinite (for example, unit leverage)."
    return Estimate(raw.params, names=problem.names, objective=float(raw.ssr) / 2,
                    covariance=covariance, covariance_kind=covariance_type if covariance is not None else None,
                    nobs=len(problem.response), df=df, residuals=raw.resid,
                    diagnostics=Diagnostics(backend=f"statsmodels.{type(model).__name__}", rank=rank,
                                            uncertainty_note=note))


def robust_linear(problem, *, loss, tuning, covariance_type, tolerance, maxiter):
    from statsmodels.robust.norms import HuberT, TukeyBiweight
    from statsmodels.robust.robust_linear_model import RLM

    if problem.weights is not None or problem.sigma is not None:
        raise ValueError("robust_linear accepts an unweighted design; supply whitened inputs explicitly")
    norm = HuberT(t=tuning) if loss == "huber" else TukeyBiweight(c=tuning)
    model = RLM(problem.response, problem.design, M=norm)
    raw = model.fit(maxiter=maxiter, tol=tolerance, conv="coefs", cov=covariance_type)
    history = raw.fit_history
    iterations = int(history["iteration"])
    params = history["params"]
    converged = bool(len(params) >= 3 and np.all(np.abs(np.asarray(params[-1]) - params[-2]) <= tolerance))
    # A zero residual scale is an exact fit; robust scale-based uncertainty is undefined.
    exact = bool(raw.scale == 0 and np.allclose(raw.resid, 0))
    converged = converged or exact
    rank = int(np.linalg.matrix_rank(problem.design))
    covariance = None
    note = "Asymptotic robust M-estimator covariance; independent observations assumed."
    if converged and rank == len(raw.params) and raw.df_resid > 0 and raw.scale > 0:
        candidate = raw.cov_params()
        if np.all(np.isfinite(candidate)):
            covariance = candidate
    if covariance is None:
        note = "Robust covariance unavailable: check convergence, rank, residual degrees of freedom and scale."
    # RLM estimates scale jointly; its standardized loss is not comparable to SSE or a log likelihood.
    return Estimate(raw.params, names=problem.names, nobs=len(problem.response), residuals=raw.resid,
                    covariance=covariance, covariance_kind=f"robust_{covariance_type}" if covariance is not None else None,
                    diagnostics=Diagnostics(converged=converged, backend=f"statsmodels.RLM:{loss}",
                                            iterations=iterations, rank=rank, uncertainty_note=note,
                                            message="IRLS converged" if converged else "IRLS iteration limit reached"))
