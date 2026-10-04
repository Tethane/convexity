"""Kernel composition, explicit Ito integration and exact multi-asset GBM."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import BrownianIncrements
if TYPE_CHECKING:
    from ...processes import ItoProcess

if TYPE_CHECKING:
    from .. import PathScenarios
if TYPE_CHECKING:
    from ...processes import StochasticProcess


if TYPE_CHECKING:
    from .. import PathSampler


import numpy as np

from .. import _validation as v
from ..interfaces import BrownianIncrements, PathScenarios


def simulate(
    process: StochasticProcess,
    initial: ArrayLike,
    times: ArrayLike,
    *,
    draws: int = 1,
    rng: np.random.Generator,
    method: str = "exact",
    substeps: int = 1,
) -> PathScenarios:
    """Compose family-13 transition kernels, including CIR, jumps, QE and SRI2.

    Exactness/backend labels come from the kernels. This sampler is IID MC;
    use explicit Brownian increments for QMC, coupling or antithetic schemes.
    """
    from ...processes import sample_paths

    result = sample_paths(
        process, initial, times, draws=draws, rng=rng, method=method, substeps=substeps
    )
    return PathScenarios.from_process_paths(result, sampling="iid")


def sample_paths(
    sampler: PathSampler,
    initial: ArrayLike,
    times: ArrayLike,
    *,
    draws: int = 1,
    rng: np.random.Generator,
) -> PathScenarios:
    """Checked dispatch for a user PathSampler; no registry or inheritance."""
    times = v.times(times)
    draws = v.integer(draws, "draws", 1)
    v.generator(rng)
    result = sampler.sample(initial, times, draws=draws, rng=rng)
    if not isinstance(result, PathScenarios):
        raise TypeError("sampler.sample must return PathScenarios")
    if len(result.values) != draws or not np.array_equal(result.times, times):
        raise ValueError("sampled paths must match requested draws and times")
    return result


def integrate(
    process: ItoProcess,
    initial: ArrayLike,
    increments: BrownianIncrements,
    *,
    method: str = "euler",
    diffusion_derivative: Callable[[float, FloatArray], ArrayLike] | None = None,
) -> PathScenarios:
    """Ito Euler or scalar one-driver Milstein on supplied increments.

    Milstein requires the analytic scalar derivative d sigma(t,x)/dx. Vector
    Milstein, Levy areas, state clipping and automatic domain repair are excluded.
    Refinement is explicit: generate a fine driver and coarsen it for comparisons.
    """
    if not isinstance(increments, BrownianIncrements):
        raise TypeError("increments must be BrownianIncrements")
    v.choice(method, ("euler", "milstein"), "method")
    shape = v.shape(process.event_shape)
    m = v.integer(process.noise_dimension, "noise_dimension", 1)
    initial = v.shaped(initial, shape, "initial")
    if increments.values.shape[2] != m:
        raise ValueError("driver dimension must match process.noise_dimension")
    if method == "milstein" and (shape or m != 1 or not callable(diffusion_derivative)):
        raise ValueError(
            "Milstein requires scalar state, one driver and diffusion_derivative(time,state)"
        )
    if method == "euler" and diffusion_derivative is not None:
        raise ValueError("diffusion_derivative is only used by Milstein")
    times, noise = increments.times, increments.values
    values = np.empty((len(noise), len(times)) + shape)
    values[:, 0] = initial
    # Validate coefficients even for a grid containing only the initial state.
    v.shaped(process.drift(times[0], initial), shape, "drift")
    v.shaped(process.diffusion(times[0], initial), shape + (m,), "diffusion")
    for j, dt in enumerate(np.diff(times)):
        for i in range(len(noise)):
            state = v.shaped(values[i, j], shape, "state")
            drift = v.shaped(process.drift(times[j], state), shape, "drift")
            sigma = v.shaped(
                process.diffusion(times[j], state), shape + (m,), "diffusion"
            )
            update = state + drift * dt + sigma @ noise[i, j]
            if method == "milstein":
                assert diffusion_derivative is not None
                derivative = v.scalar(
                    diffusion_derivative(times[j], state), "diffusion_derivative"
                )
                update += 0.5 * sigma[0] * derivative * (noise[i, j, 0] ** 2 - dt)
            values[i, j + 1] = v.shaped(update, shape, "updated state")
    return PathScenarios(times, values, sampling=increments.sampling, method=method)


def geometric_brownian_paths(
    initial: ArrayLike,
    drift: ArrayLike,
    loading: ArrayLike,
    increments: BrownianIncrements,
) -> PathScenarios:
    """Exact vector GBM on a grid, with correlated log returns via loading B.

    dS_i/S_i = drift_i dt + sum_j B_ij dW_j. Initial (d,), drift (d,),
    loading (d,m); zero initial levels stay zero. No probability measure inferred.
    """
    if not isinstance(increments, BrownianIncrements):
        raise TypeError("increments must be BrownianIncrements")
    initial = v.array(initial, "initial", ndim=1)
    drift = v.shaped(drift, initial.shape, "drift")
    loading = v.shaped(loading, (len(initial), increments.values.shape[2]), "loading")
    if np.any(initial < 0):
        raise ValueError("initial levels must be nonnegative")
    times = increments.times
    dt = np.diff(times)
    log_changes = (drift - 0.5 * np.sum(loading**2, axis=1)) * dt[
        :, None
    ] + increments.values @ loading.T
    values = np.empty((len(increments.values), len(times), len(initial)))
    values[:, 0] = initial
    # Avoid 0*exp(overflow) for absorbing zero coordinates.
    positive = initial > 0
    values[:, 1:] = 0
    values[:, 1:, positive] = np.exp(
        np.log(initial[positive]) + np.cumsum(log_changes[:, :, positive], axis=1)
    )
    return PathScenarios(
        times, values, sampling=increments.sampling, method="exact-gbm", exact=True
    )
