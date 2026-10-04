"""Bayesian parameter, probability, and sample shape checks."""
import numpy as np
from .._validation import array, boolean, choice, generator, integer, scalar


def names(value, size):
    result = tuple(f'theta{i}' for i in range(size)) if value is None else tuple(value)
    if (len(result) != size or any(not isinstance(n, str) or not n for n in result)
            or len(set(result)) != size):
        raise ValueError('names must contain one unique nonempty string per coordinate')
    return result


def probability(value, name='level', *, closed=False):
    result = scalar(value, name)
    if not (0 <= result <= 1 if closed else 0 < result < 1):
        raise ValueError(f'{name} must lie in {"[0,1]" if closed else "(0,1)"}')
    return result


def positive(value, name):
    return scalar(value, name, minimum=0, strict=True)


def logs(value, name, shape=()):
    result = array(value, name, finite=False)
    if result.shape != shape or np.any(np.isposinf(result)):
        raise ValueError(f'{name} must have shape {shape}, allowing -inf but not +inf')
    return result


def counts(value, name='counts'):
    result = array(value, name, ndim=1, empty=True)
    if np.any(result < 0) or np.any(result != np.floor(result)):
        raise ValueError(f'{name} must contain nonnegative integer counts')
    return result


def covariance(value, size):
    from .._covariance import psd
    result = psd(value, definite=True)
    if result.shape != (size, size):
        raise ValueError('covariance must have shape (p,p)')
    return result
