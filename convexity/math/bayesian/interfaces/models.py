"""Proper priors, scalar likelihoods, and extensible inference contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import BayesianProblem, PosteriorSamples, PyMCModel


from typing import cast

import numpy as np

if TYPE_CHECKING:
    from pymc import Model

    from ...distributions import Distribution
    from ...estimation import Likelihood
    from .results import PosteriorSamples
from dataclasses import dataclass
from typing import Protocol

from .. import _validation as v


@dataclass(frozen=True)
class BayesianProblem:
    """Vector parameter prior and a complete log-likelihood callback.

    likelihood may also be an estimation.Likelihood. Constants must be retained
    for evidence calculations. Coordinates and Jacobians are the caller's choice.
    """

    prior: Distribution
    likelihood: Callable[[FloatArray], float] | Likelihood
    names: tuple[str, ...] | None = None
    initial: FloatArray | None = cast("FloatArray | None", None)

    def __post_init__(self) -> None:
        from ...distributions import Distribution

        if not isinstance(self.prior, Distribution) or len(self.prior.event_shape) != 1:
            raise TypeError(
                "prior must be a vector Distribution (use independent for scalar parameters)"
            )
        if not callable(self.likelihood) and not callable(
            getattr(self.likelihood, "log_likelihood", None)
        ):
            raise TypeError("likelihood must be callable or provide log_likelihood")
        object.__setattr__(
            self, "names", v.names(self.names, self.prior.event_shape[0])
        )
        if self.initial is not None:
            initial = self.point(self.initial)
            if not np.isfinite(self.log_posterior(initial)):
                raise ValueError("initial must have finite posterior log density")
            object.__setattr__(self, "initial", initial)

    def point(self, theta: ArrayLike):
        point = v.array(theta, "parameters", ndim=1)
        if point.shape != self.prior.event_shape:
            raise ValueError("parameters must match prior event shape")
        return point

    def log_likelihood(self, theta: ArrayLike) -> float:
        fn = (
            self.likelihood
            if callable(self.likelihood)
            else self.likelihood.log_likelihood
        )
        return float(v.logs(fn(self.point(theta)), "log likelihood"))

    def log_posterior(self, theta: ArrayLike):
        point = self.point(theta)
        prior = float(v.logs(self.prior.log_density(point), "log prior"))
        return -np.inf if prior == -np.inf else prior + self.log_likelihood(point)

    def __init__(
        self,
        prior: Distribution,
        likelihood: Callable[[FloatArray], float] | Likelihood,
        names: tuple[str, ...] | None = None,
        initial: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "likelihood", likelihood)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "initial", initial)
        self.__post_init__()


@dataclass(frozen=True)
class PyMCModel:
    """builder() returns a fresh pymc.Model with named priors and observations.

    Keep captured inputs stable. Native symbolic models support automatic NUTS,
    discrete compound steps, VI, and joint latent state/parameter posteriors.
    """

    builder: Callable[[], Model]

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError("builder must be callable")


class InferenceMethod[ModelT = object](Protocol):
    """Infer a posterior from a built-in or custom model, preserving the model type and caller RNG contract."""
    def fit(self, model: ModelT, *, rng: np.random.Generator) -> PosteriorSamples: ...


def infer[ModelT](
    model: ModelT, method: InferenceMethod[ModelT], *, rng: np.random.Generator
) -> PosteriorSamples:
    """Run built-in or user inference and validate the shared sample result."""
    from .results import PosteriorSamples

    v.generator(rng)
    result = method.fit(model, rng=rng)
    if not isinstance(result, PosteriorSamples):
        raise TypeError("inference must return PosteriorSamples")
    if isinstance(model, BayesianProblem):
        if tuple(result.samples) != model.names or any(
            x.ndim != 2 for x in result.samples.values()
        ):
            raise ValueError("inference must preserve scalar parameter names and order")
    return result
