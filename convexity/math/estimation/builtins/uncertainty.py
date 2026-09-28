"""Intervals, uncertainty propagation and resampling of compatible estimators."""

import numpy as np

from .. import _validation as v
from .._backends import uncertainty
from ..interfaces import ConvergenceError, Diagnostics, Estimate


def wald_interval(estimate: Estimate, *, level=.95):
    """Marginal two-sided confidence intervals. No clipping to parameter bounds."""
    if not isinstance(estimate, Estimate):
        raise TypeError("estimate must be Estimate")
    return uncertainty.interval(estimate, level=v.level(level), credible=False)


def laplace_interval(estimate: Estimate, *, level=.95):
    """Marginal approximate credible intervals from local posterior curvature."""
    if not isinstance(estimate, Estimate):
        raise TypeError("estimate must be Estimate")
    return uncertainty.interval(estimate, level=v.level(level), credible=True)


def delta_method(estimate: Estimate, function, *, jacobian=None, names=None):
    """Propagate covariance as J C J.T for a smooth scalar/vector transformation.

    Nonlinear transformations use asymptotic normal inference, not exact t
    inference. A Laplace posterior covariance retains its credible meaning.
    """
    if not isinstance(estimate, Estimate):
        raise TypeError("estimate must be Estimate")
    if not estimate.diagnostics.converged:
        raise ConvergenceError(estimate)
    if estimate.covariance is None:
        raise ValueError("delta method requires parameter covariance")
    v.function(function, "function")

    def evaluate(theta):
        value = v.array(function(v.array(theta, "parameters", ndim=1)), "transformed parameters")
        if value.ndim > 1:
            raise ValueError("function must return a scalar or vector")
        return np.atleast_1d(value)

    value = evaluate(estimate.parameters)
    if jacobian is None:
        from ...numerical import jacobian as differentiate

        matrix = differentiate(evaluate, estimate.parameters).value
    else:
        v.function(jacobian, "Jacobian")
        matrix = v.array(jacobian(estimate.parameters), "Jacobian", ndim=2)
    if matrix.shape != (len(value), len(estimate.parameters)):
        raise ValueError("Jacobian must have shape (output parameters,input parameters)")
    return Estimate(value, names=names, covariance=matrix @ estimate.covariance @ matrix.T,
                    covariance_kind="laplace" if estimate.covariance_kind == "laplace" else f"delta:{estimate.covariance_kind}",
                    nobs=estimate.nobs, diagnostics=Diagnostics(backend="delta method",
                                                               uncertainty_note="First-order covariance propagation; normal approximation."))


def bootstrap(statistic, data, *, rng, level=.95, method="BCa", n_resamples=999,
              batch=256, names=None):
    """Bootstrap independent rows, preserving pairing across a tuple of arrays.

    statistic(*data) returns a scalar, vector or converged Estimate. Each array
    has n rows; trailing dimensions are preserved. No failed fit is discarded.
    RNG must be a numpy.random.Generator. This is not a time-series bootstrap.
    """
    v.function(statistic, "statistic")
    if not isinstance(data, tuple) or not data:
        raise TypeError("data must be a nonempty tuple of observation arrays")
    samples = tuple(v.array(sample, "sample") for sample in data)
    if any(sample.ndim == 0 for sample in samples):
        raise ValueError("each sample must have an observation axis")
    n = len(samples[0])
    if n < 2 or any(len(sample) != n for sample in samples):
        raise ValueError("paired samples must have equal row counts >= 2")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    v.choice(method, ("percentile", "basic", "BCa"), "method")
    return uncertainty.bootstrap(statistic, samples, rng=rng, level=v.level(level), method=method,
                                  n_resamples=v.integer(n_resamples, "n_resamples", minimum=2),
                                  batch=v.integer(batch, "batch"), names=names)
