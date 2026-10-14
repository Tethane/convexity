"""Forecasting shapes and causal missing-value policies; NumPy only."""

import numpy as np

from .._validation import array, boolean, choice, generator, integer, scalar


def matrix(value, name, *, missing=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2) or not raw.size:
        raise ValueError(f"{name} must be a nonempty real (n,) or (n,k) array")
    out = np.array(raw, dtype=float).reshape(len(raw), -1)
    if np.isinf(out).any() or (not missing and np.isnan(out).any()):
        raise ValueError(f"{name} contains unsupported missing or infinite values")
    out.setflags(write=False)
    return out


def index(value, n, name="timestamps"):
    out = np.array(value, copy=True)
    if out.shape != (n,) or out.dtype.kind not in "iufM":
        raise ValueError(f"{name} must be a numeric or datetime64 vector of length {n}")
    if not np.isfinite(out).all() or np.any(out[1:] <= out[:-1]):
        raise ValueError(f"{name} must be finite and strictly increasing")
    out.setflags(write=False)
    return out


def levels(value):
    out = array(value, "quantile levels", ndim=1)
    if np.any((out <= 0) | (out >= 1)) or np.any(np.diff(out) <= 0):
        raise ValueError("quantile levels must increase strictly inside (0,1)")
    return out


def probability(value):
    out = scalar(value, "probability")
    if not 0 < out < 1:
        raise ValueError("probability must lie strictly inside (0,1)")
    return out


def seed(value):
    out = integer(value, "seed")
    if out >= 2**32:
        raise ValueError("seed must be less than 2**32")
    return out


def covariance(value, n):
    from .._covariance import psd

    out = psd(value)
    if out.shape != (n, n):
        raise ValueError("covariance must align with flattened forecasts")
    return out
