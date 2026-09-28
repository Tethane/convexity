"""Array and option contracts shared by predictive tools."""

from .._validation import array, boolean, choice, integer, scalar

import numpy as np


def features(value):
    return array(value, "features", ndim=2)


def coordinates(value, dimension):
    x = array(value, "features", empty=True)
    if x.ndim < 1 or x.shape[-1] != dimension:
        raise ValueError(f"features must have shape (..., {dimension})")
    return x


def response(value, n, *, multi=True):
    y = array(value, "response")
    if y.ndim not in ((1, 2) if multi else (1,)) or len(y) != n:
        raise ValueError("response must align with training rows")
    return y


def labels(value, *, n=None):
    raw = np.asarray(value)
    if raw.ndim != 1 or not raw.size or (n is not None and len(raw) != n):
        raise ValueError("labels must be a nonempty vector aligned with rows")
    if raw.dtype.kind in "iu":
        pass
    elif raw.dtype.kind == "f":
        if not np.isfinite(raw).all() or np.any(raw != np.floor(raw)):
            raise ValueError("numeric class labels must be finite integers")
    elif raw.dtype.kind == "U":
        if np.any(raw == ""):
            raise ValueError("class labels must be nonempty strings")
    else:
        raise ValueError("class labels must be integers or strings")
    result = np.array(raw, copy=True)
    result.setflags(write=False)
    return result


def weights(value, n):
    if value is None:
        return None
    w = array(value, "sample_weight", ndim=1)
    if len(w) != n or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("sample_weight must align with rows, be nonnegative and have positive finite sum")
    return w


def training(x, y, sample_weight=None, *, classification=False, multi=True):
    x = features(x)
    y = labels(y, n=len(x)) if classification else response(y, len(x), multi=multi)
    w = weights(sample_weight, len(x))
    if w is not None:
        # Zero weight must not affect class discovery, tree splits or regularization.
        keep = w > 0
        x, y, w = x[keep], y[keep], w[keep]
        for a in (x, y, w):
            a.setflags(write=False)
    if classification and len(np.unique(y)) < 2:
        raise ValueError("classification requires at least two positive-weight classes")
    return x, y, w


def levels(value):
    q = array(value, "quantiles", ndim=1)
    if np.any((q <= 0) | (q >= 1)) or np.any(np.diff(q) <= 0):
        raise ValueError("quantiles must be strictly increasing inside (0,1)")
    return q


def seed(value):
    result = integer(value, "seed")
    if result >= 2**32:
        raise ValueError("seed must be less than 2**32")
    return result
