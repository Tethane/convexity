"""Exact elapsed-time linear diffusions and useful local-state models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import LinearGaussianTransition


from functools import lru_cache

import numpy as np

from .. import _validation as v
from ..interfaces.models import StateSpaceModel
from ..interfaces.results import GaussianState
from .models import LinearGaussianObservation, LinearGaussianTransition


def continuous_linear_transition(
    drift: ArrayLike,
    diffusion_covariance: ArrayLike,
    *,
    forcing: ArrayLike | None = None,
) -> LinearGaussianTransition:
    """Exact transition of dx=(A x + b)dt + L dW, diffusion_covariance=L L.T.

    SciPy's block matrix exponential integrates both drift and covariance,
    including singular drift matrices. Noise is a rate per unit elapsed time.
    """
    a = v.array(drift, "drift", ndim=2)
    d = len(a)
    a = v.shaped(a, (d, d), "drift")
    diffusion = v.covariance(diffusion_covariance, d, "diffusion_covariance")
    b = np.zeros(d) if forcing is None else v.shaped(forcing, (d,), "forcing")

    @lru_cache(maxsize=128)
    def discretize(dt: float):
        from scipy.linalg import expm

        dt = v.scalar(dt, "dt", minimum=0)
        # Van Loan's lower block contains -A.T; for stable A it can overflow
        # across a long gap even though the desired transition is bounded.
        # Discretize a short interval and compose its exact semigroup instead.
        magnitude = float(np.linalg.norm(a, 1)) * dt
        if not np.isfinite(magnitude):
            raise ValueError("drift times elapsed duration must be representable")
        squarings = max(0, int(np.ceil(np.log2(magnitude)))) if magnitude > 1 else 0
        interval = dt / 2.0**squarings
        block = np.zeros((2 * d, 2 * d))
        block[:d, :d], block[:d, d:], block[d:, d:] = a, diffusion, -a.T
        exponential = expm(block * interval)
        matrix = exponential[:d, :d]
        noise = v.covariance(exponential[:d, d:] @ matrix.T, d)
        affine = np.zeros((d + 1, d + 1))
        affine[:d, :d], affine[:d, d] = a, b
        offset = expm(affine * interval)[:d, d]
        for _ in range(squarings):
            offset = matrix @ offset + offset
            noise = matrix @ noise @ matrix.T + noise
            matrix = matrix @ matrix
        matrix = v.shaped(matrix, (d, d), "discrete transition")
        noise = v.covariance(noise, d)
        offset = v.shaped(offset, (d,), "discrete offset")
        return matrix, noise, offset

    return LinearGaussianTransition(
        lambda step: discretize(step.dt)[0],
        lambda step: discretize(step.dt)[1],
        lambda step: discretize(step.dt)[2],
        d,
    )


def local_level(
    process_variance: float,
    observation_variance: float,
    *,
    initial: GaussianState | None = None,
    time_mode: str = "discrete",
    time_unit: np.timedelta64 | None = None,
) -> StateSpaceModel:
    """Random walk observed in Gaussian noise; elapsed process variance is a rate."""
    q = v.scalar(process_variance, "process_variance", minimum=0)
    r = v.scalar(observation_variance, "observation_variance", minimum=0, strict=True)
    return StateSpaceModel(
        GaussianState([0.0], [[1.0]]) if initial is None else initial,
        LinearGaussianTransition([[1.0]], lambda step: [[q * step.dt]]),
        LinearGaussianObservation([[1.0]], [[r]]),
        time_mode,
        time_unit,
        ("level",),
    )


def local_linear_trend(
    acceleration_variance: float,
    observation_variance: float,
    *,
    initial: GaussianState | None = None,
    time_unit: np.timedelta64 | None = None,
) -> StateSpaceModel:
    """Integrated Brownian velocity: Q=q*[[dt^3/3,dt^2/2],[dt^2/2,dt]].

    Always elapsed-time. The two states are position and velocity; numeric
    timestamps supply units, datetime timestamps require an explicit time_unit.
    """
    q = v.scalar(acceleration_variance, "acceleration_variance", minimum=0)
    r = v.scalar(observation_variance, "observation_variance", minimum=0, strict=True)
    transition = continuous_linear_transition(
        [[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, q]]
    )
    return StateSpaceModel(
        GaussianState([0.0, 0.0], np.eye(2)) if initial is None else initial,
        transition,
        LinearGaussianObservation([[1.0, 0.0]], [[r]]),
        "elapsed",
        time_unit,
        ("position", "velocity"),
    )
