"""Local array and parameter contracts for kernels and scalar-response GPs."""

from numbers import Integral
from typing import Any

from .._validation import array, boolean, integer, scalar

import numpy as np


def features(value, dimension=None):
    x = array(value, "features", ndim=2)
    if dimension is not None and x.shape[1] != dimension:
        raise ValueError(f"features must have {dimension} columns")
    return x


def variance(value, name, n=None):
    a = array(value, name)
    if a.ndim not in (0, 1) or np.any(a < 0):
        raise ValueError(f"{name} must be a nonnegative scalar or vector")
    if n is not None and a.ndim == 1 and a.shape != (n,):
        raise ValueError(f"{name} must align with rows")
    return float(a) if a.ndim == 0 else a


def parameter(
    value, bounds, name, *, vector=False, allow_zero=False
) -> tuple[Any, Any]:
    a = array(value, name)
    if a.ndim not in ((0, 1) if vector else (0,)) or np.any(
        a < 0 if allow_zero else a <= 0
    ):
        raise ValueError(
            f"{name} must be {'nonnegative' if allow_zero else 'positive'} "
            f"and {'scalar or a vector' if vector else 'scalar'}"
        )
    if isinstance(bounds, str):
        if bounds != "fixed":
            raise ValueError(f"{name}_bounds must be 'fixed' or positive ordered pairs")
        checked = bounds
    else:
        b = array(bounds, f"{name}_bounds")
        if (
            b.shape not in ((2,), (a.size, 2))
            or np.any(b <= 0)
            or np.any(b[..., 0] >= b[..., 1])
        ):
            raise ValueError(f"{name}_bounds must contain positive increasing pairs")
        if np.any(a < b[..., 0]) or np.any(a > b[..., 1]):
            raise ValueError(f"{name} must lie within its bounds")
        checked = b
    return (float(a) if a.ndim == 0 else a), checked


def sample_shape(size):
    if isinstance(size, Integral) and not isinstance(size, (bool, np.bool_)):
        size = (integer(size, "size"),)
    if not isinstance(size, tuple):
        raise ValueError("size must be an integer or tuple of integers")
    return tuple(integer(s, "size") for s in size)
