"""Array ownership and dimensions local to the transforms family."""

from .._validation import array, boolean, choice, integer

import numpy as np


def scalar(value, name, *, positive=False):
    result = float(array(value, name, ndim=0))
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def samples(value, *, minimum=1):
    result = array(value, "samples", ndim=2)
    if len(result) < minimum:
        raise ValueError(f"samples need at least {minimum} rows")
    return result


def coordinates(value, size, name="coordinates"):
    result = array(value, name, empty=size == 0)
    if (
        result.ndim < 1
        or result.shape[-1] != size
        or any(n == 0 for n in result.shape[:-1])
    ):
        raise ValueError(f"{name} must have shape (..., {size})")
    return result


def series(value):
    result = array(value, "series")
    if result.ndim not in (1, 2):
        raise ValueError("series must have shape (n,) or (n,d)")
    return result


def names(value, size):
    result = tuple(value)
    if (
        len(result) != size
        or any(not isinstance(x, str) or not x for x in result)
        or len(set(result)) != size
    ):
        raise ValueError("columns must contain one unique nonempty name per column")
    return result
