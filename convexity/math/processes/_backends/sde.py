"""Explicit NumPy generators passed through to sdeint, never global seeds."""

import numpy as np


def integrate(model, state, time, dt, substeps, method, size, rng):
    try:
        import sdeint
    except ImportError as exc:
        raise ImportError(
            "custom SDE integration requires convexity[processes] (sdeint)"
        ) from exc
    from .. import _validation as v

    shape = model.event_shape
    d = shape[0] if shape else 1
    m = model.noise_dimension

    def drift(x, t):
        checked = v.shaped(np.asarray(x).reshape(shape), shape, "state")
        return v.shaped(model.drift(time + t, checked), shape, "drift").reshape(d)

    def diffusion(x, t):
        checked = v.shaped(np.asarray(x).reshape(shape), shape, "state")
        return v.shaped(
            model.diffusion(time + t, checked), shape + (m,), "diffusion"
        ).reshape(d, m)

    # Absolute linspace endpoints can round dt (e.g. 1e15 + .3 - 1e15
    # equals .25). Integrate elapsed time and restore the origin for callbacks.
    grid = np.linspace(0.0, dt, substeps + 1)
    if np.any(np.diff(grid) <= 0):
        raise ValueError("integration substeps must be representable in the time unit")
    solver = sdeint.itoEuler if method == "euler" else sdeint.itoSRI2
    count = int(np.prod(size))
    result = [
        solver(drift, diffusion, np.atleast_1d(state), grid, generator=rng)[-1]
        for _ in range(count)
    ]
    return np.asarray(result).reshape(size + shape)
