"""Process states, time units and owned numeric results."""

import numpy as np
from .._validation import array, boolean, choice, generator, integer, scalar


def shape(value):
    if not isinstance(value, tuple) or len(value) > 1:
        raise ValueError("event_shape must be () or (dimension,)")
    if value:
        integer(value[0], "dimension", 1)
    return value


def shaped(value, expected, name):
    result = array(value, name, empty=0 in expected)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {result.shape}")
    return result


def label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def covariance(value, event_shape):
    from .._covariance import psd

    out = shaped(value, event_shape * 2, "covariance")
    if not event_shape:
        if out < 0:
            raise ValueError("covariance must be nonnegative")
        return out
    return psd(out)


def interval(time, dt):
    time, dt = scalar(time, "time"), scalar(dt, "dt", minimum=0)
    if not np.isfinite(time + dt) or (dt > 0 and time + dt == time):
        raise ValueError("end time must be finite and distinguishable from start")
    return time, dt


def times(value):
    result = array(value, "times", ndim=1)
    if np.any(np.diff(result) <= 0) or not np.all(np.isfinite(np.diff(result))):
        raise ValueError("times must be strictly increasing with finite intervals")
    return result


def sample_shape(size):
    if isinstance(size, (int, np.integer)) and not isinstance(size, (bool, np.bool_)):
        size = (int(size),)
    if not isinstance(size, tuple):
        raise ValueError("size must be an integer or tuple of integers")
    return tuple(integer(x, "sample dimension") for x in size)
