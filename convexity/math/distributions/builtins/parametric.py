"""Validated scalar and vector constructors. Parameters describe laws, not fits."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import ScalarLike
from .. import _validation as v
from .._backends.scipy import EllipticalAdapter, scalar_law
from ..interfaces import Distribution


def normal(loc: ScalarLike = 0.0, scale: ScalarLike = 1.0) -> Distribution:
    """Gaussian with location loc and strictly positive standard deviation scale."""
    return scalar_law(
        "normal",
        loc=v.scalar(loc, "loc"),
        scale=v.scalar(scale, "scale", positive=True),
    )


def student_t(df: float, loc: float = 0.0, scale: float = 1.0) -> Distribution:
    """Student-t; scale is not its standard deviation. df must be finite and > 0."""
    return scalar_law(
        "student_t",
        df=v.scalar(df, "df", positive=True),
        loc=v.scalar(loc, "loc"),
        scale=v.scalar(scale, "scale", positive=True),
    )


def lognormal(log_loc: float = 0.0, log_scale: float = 1.0) -> Distribution:
    """exp(Normal(log_loc, log_scale)); both parameters describe log(X)."""
    mu = v.scalar(log_loc, "log_loc")
    sigma = v.scalar(log_scale, "log_scale", positive=True)
    with np.errstate(over="ignore", under="ignore"):
        scale = np.exp(mu)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(
            "exp(log_loc) must be representable as a positive finite float"
        )
    return scalar_law("lognormal", s=sigma, scale=scale)


def _probability(p):
    p = v.scalar(p, "p")
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    return p


def bernoulli(p: float) -> Distribution:
    """Return a scalar Bernoulli law with success probability p in [0,1].

    Support is {0,1}; density denotes probability mass and variance is p*(1-p)."""
    return scalar_law("bernoulli", p=_probability(p))


def binomial(n: int, p: float) -> Distribution:
    """Return the count of successes in n independent Bernoulli(p) trials.

    n is a nonnegative integer, p in [0,1]; support is the integers from 0 to n."""
    return scalar_law("binomial", n=v.integer(n, "n"), p=_probability(p))


def poisson(rate: float) -> Distribution:
    """Return a scalar Poisson count law with finite nonnegative rate.

    Mean and variance equal rate; density is probability mass, not continuous density."""
    return scalar_law("poisson", mu=v.scalar(rate, "rate", nonnegative=True))


def multivariate_normal(mean: ArrayLike, covariance: ArrayLike) -> Distribution:
    """Gaussian with a positive-definite covariance; singular laws are excluded."""
    location, matrix = v.location_matrix(mean, covariance, "covariance")
    return Distribution(EllipticalAdapter(location, matrix))


def multivariate_student_t(df: float, loc: ArrayLike, shape: ArrayLike) -> Distribution:
    """Elliptical Student-t with positive-definite shape (not covariance).

    For df > 2, covariance = shape * df / (df - 2). This is a dependent
    multivariate t, including when shape is diagonal, not independent t marginals.
    """
    location, matrix = v.location_matrix(loc, shape, "shape")
    return Distribution(
        EllipticalAdapter(location, matrix, v.scalar(df, "df", positive=True))
    )
