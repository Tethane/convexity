"""IID/pair/replicate error estimates and fixed-coefficient control variates."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import MonteCarloEstimate, PathScenarios


import numpy as np

from .. import _validation as v
from .._backends.random import normals
from ..interfaces import PathScenarios
from ..interfaces.estimates import MonteCarloEstimate
from .drivers import _design


def estimate(values: ArrayLike, *, sampling: str = "iid") -> MonteCarloEstimate:
    """Average (draw,...) values; errors require independent units, finite variance.

    Antithetic halves are paired before estimating error. A single QMC net or
    unspecified dependence supplies a mean and no standard error. With one
    independent unit the standard error is also unavailable (None), never zero.
    """
    values = v.array(values, "values")
    if values.ndim < 1:
        raise ValueError("values require a leading draw axis")
    v.sampling(sampling)
    if sampling not in ("iid", "antithetic"):
        return MonteCarloEstimate(values.mean(axis=0), None, 0, sampling)
    if sampling == "antithetic":
        if len(values) % 2:
            raise ValueError("antithetic values require an even number of draws")
        n = len(values) // 2
        values = 0.5 * values[:n] + 0.5 * values[n:]
    n = len(values)
    error = None if n < 2 else values.std(axis=0, ddof=1) / np.sqrt(n)
    return MonteCarloEstimate(values.mean(axis=0), error, n, sampling)


def estimate_scenarios(
    scenarios: PathScenarios, function: Callable[[FloatArray], ArrayLike]
) -> MonteCarloEstimate:
    """Evaluate one path at a time; unequal weights have no automatic MC error."""
    if not isinstance(scenarios, PathScenarios):
        raise TypeError("scenarios must be PathScenarios")
    positive = scenarios.weights > 0
    values = v.array(
        [function(x) for x in scenarios.values[positive]], "function values"
    )
    weights = scenarios.weights[positive]
    if np.all(positive) and np.all(weights == weights[0]):
        return estimate(values, sampling=scenarios.sampling)
    return MonteCarloEstimate(
        np.tensordot(weights, values, axes=(0, 0)), None, 0, "weighted"
    )


def control_variates(
    values: ArrayLike, controls: ArrayLike, expected: ArrayLike, coefficients: ArrayLike
) -> FloatArray:
    """Y - (C-E[C]) beta for a scalar response and (draw,control) controls.

    Coefficients must be fixed in advance or fitted on an independent pilot.
    This function does not fit on the estimation sample or assume known means.
    Evaluate the adjusted rows with estimate(), retaining the sampling design.
    """
    values = v.array(values, "values", ndim=1)
    controls = v.array(controls, "controls", ndim=2)
    if len(controls) != len(values):
        raise ValueError("controls must have one row per value")
    shape = (controls.shape[1],)
    expected = v.shaped(expected, shape, "expected")
    coefficients = v.shaped(coefficients, shape, "coefficients")
    return v.array(values - (controls - expected) @ coefficients, "adjusted values")


def qmc_expectation(
    function: Callable[[FloatArray], ArrayLike],
    dimension: int,
    *,
    draws: int,
    replicates: int,
    rng: np.random.Generator,
    method: str = "sobol",
) -> MonteCarloEstimate:
    """Integrate a vectorized function of standard-normal rows using fresh nets.

    function receives (draw,dimension), returns (draw,...). Standard error is
    computed across independent scramble means, never across points in a net.
    Transform the normal rows inside the callback for another target law.
    """
    dimension = v.integer(dimension, "dimension", 1)
    replicates = v.integer(replicates, "replicates", 2)
    v.choice(method, ("sobol", "halton"), "method")
    draws = v.integer(draws, "draws", 1)
    _design(draws, method, False, rng)
    means = []
    for _ in range(replicates):
        z = v.array(normals(draws, dimension, method, rng), "normal points")
        values = v.array(function(z), "function values")
        if values.ndim < 1 or len(values) != draws:
            raise ValueError("function must return one value per normal point")
        means.append(values.mean(axis=0))
    result = estimate(means)
    return MonteCarloEstimate(
        result.value, result.standard_error, replicates, f"randomized-{method}"
    )
