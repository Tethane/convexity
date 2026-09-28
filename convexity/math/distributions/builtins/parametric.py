"""Validated scalar and vector constructors. Parameters describe laws, not fits."""

import numpy as np

from .. import _validation as v
from .._backends.scipy import EllipticalAdapter, scalar_law
from ..interfaces import Distribution


def normal(loc=0., scale=1.) -> Distribution:
    """Gaussian with location loc and strictly positive standard deviation scale."""
    return scalar_law("normal", loc=v.scalar(loc, "loc"), scale=v.scalar(scale, "scale", positive=True))


def student_t(df, loc=0., scale=1.) -> Distribution:
    """Student-t; scale is not its standard deviation. df must be finite and > 0."""
    return scalar_law("student_t", df=v.scalar(df, "df", positive=True),
                      loc=v.scalar(loc, "loc"), scale=v.scalar(scale, "scale", positive=True))


def lognormal(log_loc=0., log_scale=1.) -> Distribution:
    """exp(Normal(log_loc, log_scale)); both parameters describe log(X)."""
    mu = v.scalar(log_loc, "log_loc")
    sigma = v.scalar(log_scale, "log_scale", positive=True)
    with np.errstate(over="ignore", under="ignore"):
        scale = np.exp(mu)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("exp(log_loc) must be representable as a positive finite float")
    return scalar_law("lognormal", s=sigma, scale=scale)


def _probability(p):
    p = v.scalar(p, "p")
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    return p


def bernoulli(p) -> Distribution:
    return scalar_law("bernoulli", p=_probability(p))


def binomial(n, p) -> Distribution:
    return scalar_law("binomial", n=v.integer(n, "n"), p=_probability(p))


def poisson(rate) -> Distribution:
    return scalar_law("poisson", mu=v.scalar(rate, "rate", nonnegative=True))


def multivariate_normal(mean, covariance) -> Distribution:
    """Gaussian with a positive-definite covariance; singular laws are excluded."""
    location, matrix = v.location_matrix(mean, covariance, "covariance")
    return Distribution(EllipticalAdapter(location, matrix))


def multivariate_student_t(df, loc, shape) -> Distribution:
    """Elliptical Student-t with positive-definite shape (not covariance).

    For df > 2, covariance = shape * df / (df - 2). This is a dependent
    multivariate t, including when shape is diagonal, not independent t marginals.
    """
    location, matrix = v.location_matrix(loc, shape, "shape")
    return Distribution(EllipticalAdapter(location, matrix, v.scalar(df, "df", positive=True)))
