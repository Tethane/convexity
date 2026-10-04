"""Shared array checks for risk and decision mathematics (NumPy only)."""

import numpy as np
from ._validation import array, scalar, integer, boolean, choice, generator

__all__ = [
    "array",
    "scalar",
    "integer",
    "boolean",
    "choice",
    "generator",
    "probability",
    "weights",
    "vector",
    "psd",
    "mask",
    "indices",
]


def probability(value, name="probability", *, open_interval=False):
    x = scalar(value, name)
    if not (0 < x < 1 if open_interval else 0 <= x <= 1):
        raise ValueError(
            f"{name} must lie in " + ("(0,1)" if open_interval else "[0,1]")
        )
    return x


def weights(value, n):
    w = np.ones(n) if value is None else array(value, "weights", ndim=1)
    if w.shape != (n,) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError(
            "weights must align with scenarios and have positive nonnegative mass"
        )
    w = w / w.max()
    return array(w / w.sum(), "weights")


def vector(value, n, name, *, finite=True):
    x = array(value, name, finite=finite)
    return array(np.broadcast_to(x, (n,)), name, finite=finite)


def psd(value, name, *, definite=False):
    from ._covariance import psd as checked
    return checked(value, name, definite=definite)


def mask(value, shape, name="mask"):
    x = np.asarray(value)
    if x.dtype.kind != "b" or x.shape != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    x = x.copy()
    x.setflags(write=False)
    return x


def indices(value, shape, name):
    x = np.asarray(value)
    if x.dtype.kind not in "iu" or x.shape != shape:
        raise ValueError(f"{name} must be integer with shape {shape}")
    x = x.astype(np.int64, copy=True)
    x.setflags(write=False)
    return x
