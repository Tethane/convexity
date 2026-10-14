"""Structural process contracts and checked entry points for user models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from .results import ProcessPaths, Transition


class StochasticProcess(Protocol):
    """Conditional-transition interface with declared event_shape; the returned Transition states available sampling and law capabilities."""

    @property
    def event_shape(self) -> tuple[int, ...]: ...

    def transition(
        self,
        state: ArrayLike,
        dt: float,
        *,
        time: float = 0.0,
        method: str = "exact",
        substeps: int = 1,
    ) -> Transition: ...


class ItoProcess(StochasticProcess, Protocol):
    """Ito drift/diffusion extension; state, time and driving-noise coordinates follow the process contract."""

    @property
    def noise_dimension(self) -> int: ...

    def drift(self, time: float, state: ArrayLike) -> FloatArray: ...
    def diffusion(self, time: float, state: ArrayLike) -> FloatArray: ...


def transition(
    process: StochasticProcess,
    state: ArrayLike,
    dt: float,
    *,
    time: float = 0.0,
    method: str = "exact",
    substeps: int = 1,
) -> Transition:
    """Construct a conditional process transition from state over nonnegative dt.

    time and dt share process units. exact requires an analytic supported transition;
    Euler substeps are approximate. Sample-only kernels never fabricate a density."""
    shape = v.shape(process.event_shape)
    state = v.shaped(state, shape, "state")
    time, dt = v.interval(time, dt)
    substeps = v.integer(substeps, "substeps", 1)
    result = process.transition(state, dt, time=time, method=method, substeps=substeps)
    if not isinstance(result, Transition) or result.event_shape != shape:
        raise TypeError(
            "process.transition must return a Transition with matching event_shape"
        )
    return result


def diffusion_generator(
    process: ItoProcess,
    time: float,
    state: ArrayLike,
    gradient: ArrayLike | Callable[[FloatArray], ArrayLike],
    hessian: ArrayLike | Callable[[FloatArray], ArrayLike],
) -> float:
    """Spatial Ito generator b·grad(f) + 1/2 tr(B B.T Hess(f)).

    Derivatives are supplied explicitly. This is the continuous part only for
    a jump process; the model's generator adds its jump expectation separately.
    """
    shape = v.shape(process.event_shape)
    time = v.scalar(time, "time")
    x = v.shaped(state, shape, "state")
    m = v.integer(process.noise_dimension, "noise_dimension", 1)
    b = v.shaped(process.drift(time, x), shape, "drift")
    sigma = v.shaped(process.diffusion(time, x), shape + (m,), "diffusion")
    grad = v.shaped(gradient(x) if callable(gradient) else gradient, shape, "gradient")
    hess = v.shaped(hessian(x) if callable(hessian) else hessian, shape * 2, "hessian")
    if shape:
        value = b @ grad + 0.5 * np.sum((sigma @ sigma.T) * hess)
    else:
        value = b * grad + 0.5 * np.sum(sigma**2) * hess
    return v.scalar(value, "generator")


def sample_paths(
    process: StochasticProcess,
    initial: ArrayLike,
    times: ArrayLike,
    *,
    draws: int = 1,
    rng: np.random.Generator,
    method: str = "exact",
    substeps: int = 1,
) -> ProcessPaths:
    """Minimal sequential kernel composition; times use caller-selected units."""
    times = v.times(times)
    shape = v.shape(process.event_shape)
    initial = v.shaped(initial, shape, "initial")
    draws = v.integer(draws, "draws", 1)
    substeps = v.integer(substeps, "substeps", 1)
    v.generator(rng)
    # Validate the method and domain even when there are no outgoing intervals.
    first = transition(
        process, initial, 0.0, time=times[0], method=method, substeps=substeps
    )
    values = np.empty((draws, len(times)) + shape)
    values[:, 0] = initial
    exact = first.exact
    methods, backends = [], []

    def record(kernel):
        if kernel.method not in methods:
            methods.append(kernel.method)
        if kernel.backend not in backends:
            backends.append(kernel.backend)

    for j in range(1, len(times)):
        if j == 1:
            kernel = transition(
                process,
                initial,
                times[j] - times[j - 1],
                time=times[j - 1],
                method=method,
                substeps=substeps,
            )
            values[:, j] = kernel.sample(draws, rng=rng)
            exact = exact and kernel.exact
            record(kernel)
            continue
        for i in range(draws):
            kernel = transition(
                process,
                values[i, j - 1],
                times[j] - times[j - 1],
                time=times[j - 1],
                method=method,
                substeps=substeps,
            )
            values[i, j] = kernel.sample(rng=rng)
            exact = exact and kernel.exact
            record(kernel)
    if not methods:
        record(first)
    method = methods[0] if len(methods) == 1 else "mixed"
    backend = backends[0] if len(backends) == 1 else "mixed"
    return ProcessPaths(
        times, values, method, backend, exact, tuple(methods), tuple(backends)
    )
