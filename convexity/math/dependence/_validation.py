"""Local array, matrix and random-state contracts."""

from numbers import Integral
from typing import Any

from .._validation import array, boolean, choice, generator, integer

import numpy as np


def scalar(value, name, *, positive=False):
    result = float(array(value, name, ndim=0))
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def samples(value, minimum=2):
    x = array(value, "samples", ndim=2)
    if len(x) < minimum:
        raise ValueError(f"samples need at least {minimum} rows")
    return x


def coordinates(value, dimension, *, finite=True):
    x = array(value, "values", finite=finite, empty=True)
    if x.ndim < 1 or x.shape[-1] != dimension:
        raise ValueError(f"values must have shape (..., {dimension})")
    return x


def symmetric(value, name="matrix"):
    x = array(value, name, ndim=2)
    if x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be square")
    scale = max(float(np.max(np.abs(x))), np.finfo(float).tiny)
    if np.max(np.abs(x - x.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    return array(x / 2 + x.T / 2, name)


def psd(value, name="covariance", *, definite=False):
    from .._covariance import psd as checked

    return checked(value, name, definite=definite)


def correlation(value, *, definite=False):
    x = psd(value, "correlation", definite=definite)
    if not np.allclose(np.diag(x), 1, rtol=0, atol=1e-10) or np.any(
        np.abs(x) > 1 + 1e-10
    ):
        raise ValueError("correlation must have unit diagonal and entries in [-1,1]")
    return elliptical_shape(x, "correlation") if definite else x


def elliptical_shape(value, name="shape"):
    """Use SciPy's numerical rank convention for full-dimensional densities."""
    from scipy.stats import multivariate_normal

    matrix = psd(value, name, definite=True)
    generator: Any = multivariate_normal
    try:
        generator(cov=matrix, allow_singular=False)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"{name} must be numerically positive definite for SciPy densities"
        ) from error
    return matrix


def probabilities(value, *, interior=False):
    x = array(value, "probabilities", empty=True)
    bad = (x <= 0) | (x >= 1) if interior else (x < 0) | (x > 1)
    if np.any(bad):
        raise ValueError(
            f"probabilities must lie in {'(0,1)' if interior else '[0,1]'}"
        )
    return x


def sample_shape(size):
    if isinstance(size, Integral):
        return (integer(size, "size"),)
    if not isinstance(size, tuple):
        raise TypeError("size must be a nonnegative integer or tuple of integers")
    return tuple(integer(n, "size") for n in size)
