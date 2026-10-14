"""particles Feynman-Kac adapter, ESS resampling, and FFBS path smoothing."""

from contextlib import contextmanager
from dataclasses import replace
from threading import RLock
from typing import Any

import numpy as np

from .. import _validation as v
from ..interfaces.results import FilterResult, StateSequence

_RANDOM_LOCK = RLock()


class ParticleDegeneracyError(RuntimeError):
    """The supplied observation assigns zero likelihood to every particle."""


@contextmanager
def _backend_random(rng):
    """Isolate the backend's legacy global RNG; our kernels use the Generator."""
    v.generator(rng)
    with _RANDOM_LOCK:
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**32, dtype=np.uint64)))
        try:
            yield
        finally:
            np.random.set_state(state)


def _logs(value, n, name):
    out = v.array(value, name, ndim=1, finite=False)
    if out.shape != (n,) or np.any(np.isposinf(out)):
        raise ValueError(f"{name} must return ({n},) values without +inf")
    if not np.any(np.isfinite(out)):
        raise ParticleDegeneracyError(f"{name}: all particles have zero likelihood")
    return out.copy()  # particles.Weights sanitizes its private log-weight buffer.


def run(model, observations, specification, *, rng):
    v.generator(rng)
    try:
        import particles
    except ImportError as exc:
        raise ImportError(
            "particle inference requires convexity[state-space] (particles)"
        ) from exc
    series = v.data(observations, model.observation_dimension)
    steps = model.steps(series)
    d = model.state_dimension

    class Kernel(particles.FeynmanKac):
        def __init__(self):
            super().__init__(T=len(series))

        def M0(self, N):
            return v.shaped(
                model.initial_distribution().sample(N, rng=rng),
                (N, d),
                "initial samples",
            )

        def M(self, t, xp):
            return v.shaped(
                model.transition.sample(xp, steps[t], rng=rng),
                xp.shape,
                "transition samples",
            )

        def logG(self, t, xp, x):
            if np.all(np.isnan(series.values[t])):
                return np.zeros(len(x))
            return _logs(
                model.observation.log_density(series.values[t], x, steps[t]),
                len(x),
                "observation log_density",
            )

        def logpt(self, t, xp, x):
            if not callable(getattr(model.transition, "log_density", None)):
                raise NotImplementedError(
                    "backward smoothing requires transition log_density; use ancestral tracing"
                )
            return _logs(
                model.transition.log_density(x, xp, steps[t]),
                len(xp),
                "transition log_density",
            )

    with _backend_random(rng):
        engine: Any = particles.SMC(
            fk=Kernel(),
            N=specification.particles,
            resampling=specification.resampling,
            ESSrmin=specification.ess_threshold,
            store_history=True,
        )
        for _ in engine:
            if not np.isfinite(engine.logLt) or not np.all(np.isfinite(engine.W)):
                raise ParticleDegeneracyError(
                    "particle weights collapsed; increase particles or revise the proposal"
                )
    history = engine.hist
    values = np.array(history.X)
    weights = np.array([w.W for w in history.wgts])
    likelihoods = np.diff(np.r_[0.0, engine.summaries.logLts])
    return FilterResult(
        series,
        StateSequence.from_particles(values, weights),
        likelihoods,
        "particles",
        "bootstrap",
        effective_sample_size=engine.summaries.ESSs,
        resampled=np.asarray(engine.summaries.rs_flags, dtype=bool),
        state_names=model.state_names,
        approximation="weighted empirical Monte Carlo",
        _history=history,
    )


def smooth_result(result, paths, method, *, rng):
    v.generator(rng)
    count = v.integer(paths, "paths", 1)
    v.choice(method, ("backward", "ancestral"), "smoothing method")
    if (
        not isinstance(result, FilterResult)
        or result.backend != "particles"
        or result._history is None
    ):
        raise ValueError(
            "smoothing requires a particle FilterResult with retained history"
        )
    history = result._history
    with _backend_random(rng):
        if method == "backward":
            raw = np.asarray(history.backward_sampling_ON2(count))
            draws = (
                raw[:, None, :].transpose(1, 0, 2)
                if count == 1
                else raw.transpose(1, 0, 2)
            )
        else:
            draws = np.array([history.extract_one_trajectory() for _ in range(count)])
    marginals = StateSequence.from_particles(
        draws.transpose(1, 0, 2), np.full((len(result.observations), count), 1 / count)
    )
    return replace(
        result, smoothed=marginals, paths=draws, method=f"bootstrap/{method}"
    )
