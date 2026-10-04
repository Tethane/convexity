"""Exact joint Gaussian grid law for supplied affine Gaussian transitions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from ...processes._backends.laws import gaussian
from .. import _validation as v
from ..interfaces import PathDistribution


def gaussian_path_distribution(
    initial: ArrayLike,
    times: ArrayLike,
    coefficients: Callable[[float, float], tuple[ArrayLike, ArrayLike, ArrayLike]],
    *,
    initial_covariance: ArrayLike | None = None,
) -> PathDistribution:
    """Law of X[j+1]=F X[j]+b+epsilon, epsilon~N(0,Q), independent by step.

    coefficients(time,dt) returns (F,b,Q). Scalar initial uses scalar F,b,Q;
    vector initial uses (d,d),(d,),(d,d). `initial` is the initial mean;
    omitted initial_covariance means a deterministic initial condition.
    The callback must describe exact transitions to claim an exact process law.
    """
    initial = v.array(initial, "initial")
    shape = v.shape(initial.shape)
    times = v.times(times)
    if not callable(coefficients):
        raise TypeError("coefficients must be callable")
    d, n = initial.size, len(times)
    covariance = (
        np.zeros(shape * 2)
        if initial_covariance is None
        else v.covariance(initial_covariance, shape)
    )
    mean = np.empty((n, d))
    joint = np.zeros((n * d, n * d))
    mean[0] = initial.reshape(d)
    joint[:d, :d] = np.asarray(covariance).reshape(d, d)
    for j, dt in enumerate(np.diff(times), start=1):
        matrix, offset, noise = coefficients(times[j - 1], dt)
        matrix = v.shaped(matrix, shape * 2, "transition matrix").reshape(d, d)
        offset = v.shaped(offset, shape, "transition offset").reshape(d)
        noise = v.covariance(noise, shape).reshape(d, d)
        mean[j] = matrix @ mean[j - 1] + offset
        current, previous = slice(j * d, (j + 1) * d), slice((j - 1) * d, j * d)
        joint[current, : j * d] = matrix @ joint[previous, : j * d]
        joint[: j * d, current] = joint[current, : j * d].T
        joint[current, current] = matrix @ joint[previous, previous] @ matrix.T + noise
    return PathDistribution(times, shape, gaussian(mean.ravel(), joint))
