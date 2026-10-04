"""Probability, regime and event-time checks."""
import numpy as np
from .._validation import array, boolean, choice, generator, integer, scalar


def probabilities(value, name='probabilities', *, ndim=1):
    x = array(value, name, ndim=ndim)
    if np.any(x < 0) or not np.allclose(x.sum(axis=-1), 1, rtol=1e-10, atol=1e-12):
        raise ValueError(f'{name} must be nonnegative and sum to one along the last axis')
    return array(x / x.sum(axis=-1, keepdims=True), name)


def transition(value):
    x = probabilities(value, 'transition', ndim=2)
    if x.shape[0] != x.shape[1]:
        raise ValueError('transition must be square')
    return x


def state(value, size):
    value = integer(value, 'regime')
    if value >= size:
        raise ValueError('regime index out of range')
    return value


def interval(start, end):
    start, end = scalar(start, 'start'), scalar(end, 'end')
    if end < start:
        raise ValueError('end must be >= start')
    return start, end


def times(value, *, strict=True):
    x = array(value, 'times', ndim=1, empty=True)
    if np.any(np.diff(x) <= 0 if strict else np.diff(x) < 0):
        raise ValueError('times must be increasing (strictly for point processes)')
    return x


def law(value, shape=None):
    from ..distributions import Distribution
    if not isinstance(value, Distribution):
        raise TypeError('law must be a Distribution')
    if shape is not None and value.event_shape != shape:
        raise ValueError(f'law must have event shape {shape}')
    return value


def moment(value):
    """Owned moments retain NaN/Inf for undefined/divergent expectations."""
    out = np.array(value, dtype=float)
    out.setflags(write=False)
    return out
