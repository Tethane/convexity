"""Posterior prediction by conditional laws; integrates both uncertainty sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast, overload

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...distributions import Distribution

from types import MappingProxyType

import numpy as np

from .. import _validation as v
from ..interfaces import PosteriorSamples


@overload
def posterior_predictive(
    posterior: PosteriorSamples,
    conditional: Callable[[Mapping[str, FloatArray | np.float64]], Distribution],
    *,
    draws: int = 1000,
    rng: np.random.Generator,
) -> Distribution: ...


@overload
def posterior_predictive(
    posterior: Distribution,
    conditional: Callable[[FloatArray | np.float64], Distribution],
    *,
    draws: int = 1000,
    rng: np.random.Generator,
) -> Distribution: ...


def posterior_predictive(
    posterior: Distribution | PosteriorSamples,
    conditional: Callable[[Mapping[str, FloatArray | np.float64]], Distribution]
    | Callable[[FloatArray | np.float64], Distribution],
    *,
    draws: int = 1000,
    rng: np.random.Generator,
) -> Distribution:
    """Return a finite joint predictive Distribution.

    posterior is a Distribution or PosteriorSamples. conditional(theta) returns
    one scalar/vector Distribution; for samples theta is a name->event mapping.
    Each draw uses one joint parameter draw and one conditional observation draw.
    """
    from ...distributions import Distribution, empirical

    v.generator(rng)
    v.integer(draws, "draws", 1)
    if not callable(conditional):
        raise TypeError("conditional must be callable")
    if isinstance(posterior, PosteriorSamples):
        indices = rng.integers(posterior.chains * posterior.draws, size=draws)
        values = {
            n: a.reshape((-1,) + a.shape[2:]) for n, a in posterior.samples.items()
        }
        parameters = (
            MappingProxyType({n: a[i] for n, a in values.items()}) for i in indices
        )
    elif isinstance(posterior, Distribution):
        parameters = iter(posterior.sample(draws, rng=rng))
    else:
        raise TypeError("posterior must be Distribution or PosteriorSamples")
    shape = None
    samples = []
    for theta in parameters:
        # The overloads pair each posterior representation with its callback input.
        law = cast(Callable[..., Distribution], conditional)(theta)
        if not isinstance(law, Distribution):
            raise TypeError("conditional must return Distribution")
        if shape is not None and law.event_shape != shape:
            raise ValueError("conditional event shape must remain constant")
        shape = law.event_shape
        samples.append(law.sample(rng=rng))
    return empirical(np.asarray(samples))
