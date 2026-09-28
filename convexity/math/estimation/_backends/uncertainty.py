"""SciPy reference quantiles and paired-row resampling."""

import numpy as np
from scipy import stats

from .. import _validation as v
from ..interfaces import BootstrapResult, ConvergenceError, Diagnostics, Estimate, Interval


def interval(estimate, *, level, credible):
    if not estimate.diagnostics.converged:
        raise ConvergenceError(estimate)
    if credible != (estimate.covariance_kind == "laplace"):
        raise ValueError("use laplace_interval for a Laplace posterior covariance, wald_interval for frequentist covariance")
    errors = estimate.standard_errors
    use_t = not credible and estimate.df is not None
    quantile = stats.t.isf((1 - level) / 2, estimate.df) if use_t else stats.norm.isf((1 - level) / 2)
    return Interval(estimate.parameters - quantile * errors, estimate.parameters + quantile * errors,
                    level, "Laplace normal" if credible else "Wald Student-t" if use_t else "Wald normal",
                    "credible" if credible else "confidence", estimate.names)


def bootstrap(statistic, samples, *, rng, level, method, n_resamples, batch, names):
    def evaluate(rows):
        value = statistic(*(v.array(sample[rows.astype(int)], "resampled observations") for sample in samples))
        if isinstance(value, Estimate):
            if not value.diagnostics.converged:
                raise ConvergenceError(value)
            if value.names != result_names:
                raise ValueError("bootstrap estimator changed parameter names")
            value = value.parameters
        output = v.array(value, "bootstrap statistic")
        if output.ndim > 1:
            raise ValueError("bootstrap statistic must be a scalar, vector or Estimate")
        output = np.atleast_1d(output)
        if output.shape != point.shape:
            raise ValueError("bootstrap statistic changed shape")
        return output

    original = statistic(*samples)
    if isinstance(original, Estimate):
        if not original.diagnostics.converged:
            raise ConvergenceError(original)
        point = original.parameters
        result_names = original.names
        if names is not None and tuple(names) != result_names:
            raise ValueError("names must match the estimator's parameter names")
    else:
        raw = v.array(original, "statistic")
        if raw.ndim > 1:
            raise ValueError("bootstrap statistic must be a scalar, vector or Estimate")
        point = np.atleast_1d(raw)
        result_names = v.names(names, len(point))
    # Resampling indices lets SciPy own BCa and its jackknife while preserving
    # entire vector rows and pairing across arbitrary arrays (e.g. X and y).
    raw = stats.bootstrap((np.arange(len(samples[0])),), evaluate, vectorized=False,
                          confidence_level=level, method=method, n_resamples=n_resamples,
                          batch=batch, rng=rng)
    draws = raw.bootstrap_distribution.T
    if not np.all(np.isfinite(raw.confidence_interval)):
        raise ValueError("bootstrap interval is undefined (for example, degenerate BCa); choose a method explicitly")
    estimate = Estimate(point, names=result_names, covariance=np.atleast_2d(np.cov(draws, rowvar=False, ddof=1)),
                        covariance_kind="bootstrap", nobs=len(samples[0]),
                        diagnostics=Diagnostics(backend="scipy.stats.bootstrap",
                                                uncertainty_note="Independent identically distributed paired rows; all resamples must fit successfully."))
    bounds = Interval(raw.confidence_interval.low, raw.confidence_interval.high,
                      level, f"bootstrap {method}", names=result_names)
    return BootstrapResult(estimate, bounds, draws)
