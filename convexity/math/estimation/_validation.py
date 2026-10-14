"""Shared ownership, shape and real-valued contracts for estimation."""

from numbers import Integral

from .._validation import array, boolean, choice

import numpy as np


def scalar(value, name, *, finite=True, positive=False, nonnegative=False):
    result = float(array(value, name, ndim=0, finite=finite))
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def function(value, name):
    if not callable(value):
        raise TypeError(f"{name} must be callable")


def names(value, size):
    result = tuple(f"theta_{i}" for i in range(size)) if value is None else tuple(value)
    if (
        len(result) != size
        or any(not isinstance(x, str) or not x for x in result)
        or len(set(result)) != size
    ):
        raise ValueError("names must contain one unique nonempty string per parameter")
    return result


def broadcast(value, size, name, *, finite=True):
    result = array(value, name, finite=finite)
    if result.ndim > 1 or (result.ndim == 1 and result.shape != (size,)):
        raise ValueError(f"{name} must be scalar or shape ({size},)")
    return array(np.broadcast_to(result, (size,)), name, finite=finite)


def bounds(lower, upper, size, *, strict=False):
    lo = broadcast(lower, size, "lower", finite=False)
    hi = broadcast(upper, size, "upper", finite=False)
    if (
        np.any(lo > hi)
        or (strict and np.any(lo == hi))
        or np.any(np.isposinf(lo))
        or np.any(np.isneginf(hi))
    ):
        raise ValueError("bounds must be ordered with finite feasible points")
    return lo, hi


def matrix(value, size, name, *, positive=False):
    from .._covariance import psd

    result = psd(value, name, definite=positive)
    if result.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    return result


def level(value):
    result = scalar(value, "level")
    if not 0 < result < 1:
        raise ValueError("level must lie strictly between zero and one")
    return result
