"""History grids start at the model's fixed time origin, zero."""

import numpy as np
from .._validation import array, boolean, generator, integer, scalar
from ..processes._validation import covariance, label, shaped, times as ordered_times


def times(value):
    if any(
        isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat
    ):
        raise ValueError("times must be numeric, not boolean")
    out = ordered_times(value)
    if out[0] != 0:
        raise ValueError("times must start at zero; past history cannot be discarded")
    return out


def nonnegative(value, name):
    out = array(value, name)
    if np.any(out < 0):
        raise ValueError(f"{name} must be nonnegative")
    return out


def hurst(value, *, rough=False):
    out = scalar(value, "hurst", minimum=0, strict=True)
    if (rough and out > 0.5) or (not rough and out >= 1):
        raise ValueError(
            "hurst must be in (0, .5]" if rough else "hurst must be in (0, 1)"
        )
    return out


def interval(lower, upper):
    lower = scalar(lower, "lower", minimum=0)
    upper = scalar(upper, "upper", minimum=lower)
    return lower, upper


def overlap(t, s, until):
    t, s = scalar(t, "t", minimum=0), scalar(s, "s", minimum=0)
    end = min(t, s) if until is None else scalar(until, "until", minimum=0)
    if end > min(t, s):
        raise ValueError("until must not exceed either maturity")
    return t, s, end
