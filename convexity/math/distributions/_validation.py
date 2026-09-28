"""Local probability contracts; no dependency on other mathematical families."""

from numbers import Integral

from .._validation import array, generator, integer

import numpy as np


def scalar(value, name, *, positive=False, nonnegative=False):
    result = float(array(value, name, ndim=0))
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def sample_shape(size):
    if isinstance(size, Integral):
        return (integer(size, "size"),)
    if not isinstance(size, tuple):
        raise ValueError("size must be a nonnegative integer or tuple of integers")
    return tuple(integer(n, "size entry") for n in size)


def weights(value, n):
    raw = np.ones(n) if value is None else array(value, "weights", ndim=1)
    if raw.shape != (n,) or np.any(raw < 0) or not np.any(raw > 0):
        raise ValueError("weights must align with outcomes, be nonnegative and have positive total mass")
    # Scaling before summing avoids overflow; retain unnormalized mass for ECDFs.
    mass = raw / raw.max()
    mass.setflags(write=False)
    return mass


def indices(value, dimension, *, proper=False):
    raw = np.asarray(value)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iu":
        raise ValueError("indices must be a nonempty sequence of integers")
    if len(np.unique(raw)) != raw.size or np.any(raw < 0) or np.any(raw >= dimension):
        raise ValueError("indices must be distinct and in range")
    if proper and raw.size == dimension:
        raise ValueError("conditioning must leave at least one unobserved coordinate")
    return raw.astype(int)


def location_matrix(location, matrix, name):
    location = array(location, "location", ndim=1)
    matrix = array(matrix, name, ndim=2)
    if matrix.shape != (location.size, location.size):
        raise ValueError(f"{name} must have shape (dimension, dimension)")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    # Make the accepted roundoff-level asymmetry exact before the backend sees it.
    matrix = array(matrix / 2 + matrix.T / 2, name)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return location, matrix
