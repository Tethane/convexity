"""Checked copulas and composition with continuous scalar marginals."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _validation as v
from .base import CopulaAdapter, Diagnostics, diagnostics


@dataclass(frozen=True)
class Copula:
    adapter: CopulaAdapter

    def __post_init__(self):
        v.integer(self.adapter.dimension, "dimension", 2)
        if not callable(self.adapter.evaluate):
            raise TypeError("adapter must implement evaluate")
        if not isinstance(self.adapter.capabilities, frozenset):
            raise TypeError("capabilities must be a frozenset")

    @property
    def dimension(self):
        return self.adapter.dimension

    @property
    def capabilities(self):
        return self.adapter.capabilities

    def _call(self, operation, *args, **kwargs):
        if operation not in self.capabilities:
            raise NotImplementedError(f"copula does not support {operation}")
        return self.adapter.evaluate(operation, *args, **kwargs)

    def log_density(self, uniforms):
        u = v.probabilities(v.coordinates(uniforms, self.dimension), interior=True)
        result = v.array(self._call("log_density", u), "log density", finite=False, empty=True)
        if result.shape != u.shape[:-1]:
            raise ValueError("copula density must preserve leading dimensions")
        return result

    def density(self, uniforms):
        return v.array(np.exp(self.log_density(uniforms)), "density", finite=False, empty=True)

    def cdf(self, uniforms, *, rng=None):
        """Elliptical adapters require rng for randomized SciPy integration."""
        u = v.probabilities(v.coordinates(uniforms, self.dimension))
        if rng is not None:
            v.generator(rng)
        result = v.probabilities(self._call("cdf", u, rng=rng))
        if result.shape != u.shape[:-1]:
            raise ValueError("copula CDF must preserve leading dimensions")
        return result

    def sample(self, size=(), *, rng):
        shape = v.sample_shape(size)
        v.generator(rng)
        result = v.probabilities(self._call("sample", shape, rng=rng))
        if result.shape != shape + (self.dimension,):
            raise ValueError("copula samples must have size + (dimension,) shape")
        return result

    def tail_dependence(self, i=0, j=1):
        """Pairwise (lower, upper) limiting conditional exceedance probabilities."""
        i, j = v.integer(i, "i"), v.integer(j, "j")
        if i == j or max(i, j) >= self.dimension:
            raise ValueError("tail dependence requires two distinct valid coordinates")
        result = v.probabilities(self._call("tail_dependence", i, j))
        if result.shape != (2,):
            raise ValueError("tail dependence must return (lower, upper)")
        return tuple(map(float, result))


@dataclass(frozen=True)
class CopulaFit:
    copula: Copula
    nobs: int
    log_likelihood: float
    repair_distance: float = 0.
    diagnostics: Diagnostics = Diagnostics()

    def __post_init__(self):
        if not isinstance(self.copula, Copula):
            raise TypeError("copula must be Copula")
        v.integer(self.nobs, "nobs", 2)
        object.__setattr__(self, "log_likelihood", v.scalar(self.log_likelihood, "log_likelihood"))
        distance = v.scalar(self.repair_distance, "repair_distance")
        if distance < 0:
            raise ValueError("repair_distance must be nonnegative")
        object.__setattr__(self, "repair_distance", distance)
        diagnostics(self.diagnostics)


@dataclass(frozen=True, init=False)
class CopulaDistribution:
    """Sklar composition. Continuous marginals supply cdf, quantile, log_density."""

    copula: Copula
    marginals: tuple[Any, ...]

    def __init__(self, copula, marginals):
        if not isinstance(copula, Copula):
            raise TypeError("copula must be Copula")
        laws = tuple(marginals)
        if len(laws) != copula.dimension:
            raise ValueError("one marginal is required per copula coordinate")
        for law in laws:
            if law.event_shape != () or law.measure != "continuous":
                raise ValueError("marginals must be continuous scalar laws")
            if any(not callable(getattr(law, method, None)) for method in ("cdf", "quantile", "log_density")):
                raise TypeError("marginals must implement cdf, quantile and log_density")
        object.__setattr__(self, "copula", copula)
        object.__setattr__(self, "marginals", laws)

    @property
    def event_shape(self):
        return (self.copula.dimension,)

    @property
    def measure(self):
        return "continuous"

    def _marginal_values(self, values, operation):
        x = v.coordinates(values, self.copula.dimension, finite=False)
        columns = []
        for j, law in enumerate(self.marginals):
            column = v.array(getattr(law, operation)(x[..., j]), operation, finite=False, empty=True)
            if column.shape != x.shape[:-1]:
                raise ValueError("marginals must preserve point shapes")
            columns.append(column)
        return np.stack(columns, axis=-1)

    def sample(self, size=(), *, rng):
        uniforms = self.copula.sample(size, rng=rng)
        return v.array(self._marginal_values(uniforms, "quantile"), "samples", empty=True)

    def cdf(self, values, *, rng=None):
        return self.copula.cdf(self._marginal_values(values, "cdf"), rng=rng)

    def log_density(self, values):
        logs = self._marginal_values(values, "log_density")
        uniforms = v.probabilities(self._marginal_values(values, "cdf"))
        leading = logs.shape[:-1]
        flat_logs = logs.reshape(-1, self.copula.dimension)
        flat_u = uniforms.reshape(-1, self.copula.dimension)
        result = np.full(len(flat_logs), -np.inf)
        active = ~np.any(np.isneginf(flat_logs), axis=1)
        # At saturated marginal CDFs the density ratio is not numerically
        # identified. Do not replace it with an arbitrary clipped density.
        if np.any(active):
            result[active] = self.copula.log_density(flat_u[active]) + flat_logs[active].sum(axis=1)
        return v.array(result.reshape(leading), "log density", finite=False, empty=True)

    def density(self, values):
        return v.array(np.exp(self.log_density(values)), "density", finite=False, empty=True)
