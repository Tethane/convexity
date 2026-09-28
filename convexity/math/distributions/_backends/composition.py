"""Probability algebra composed through the public law interface."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import optimize, special

from ..interfaces import Distribution


@dataclass(frozen=True)
class IndependentAdapter:
    components: tuple[Distribution, ...]

    @property
    def event_shape(self):
        return (len(self.components),)

    @property
    def measure(self):
        return self.components[0].measure

    @property
    def capabilities(self):
        shared = set.intersection(*(set(c.capabilities) for c in self.components))
        supported = shared & {"log_density", "sample", "cdf", "mean", "variance", "moment"}
        if "variance" in shared:
            supported.add("covariance")
        return frozenset(supported | {"marginal", "condition"})

    def evaluate(self, operation, *args, **kwargs) -> Any:
        if operation == "log_density":
            return np.sum([c.log_density(args[0][..., i]) for i, c in enumerate(self.components)], axis=0)
        if operation == "cdf":
            return np.prod([c.cdf(args[0][..., i], **kwargs) for i, c in enumerate(self.components)], axis=0)
        if operation == "sample":
            return np.stack([c.sample(args[0], **kwargs) for c in self.components], axis=-1)
        if operation in {"mean", "variance", "moment"}:
            return np.stack([getattr(c, operation)(*args) for c in self.components])
        if operation == "covariance":
            return np.diag(self.evaluate("variance"))
        if operation == "marginal":
            return Distribution(IndependentAdapter(tuple(self.components[i] for i in args[0])))
        if operation == "condition":
            observed, values = args
            if any(np.isneginf(self.components[i].log_density(x)) for i, x in zip(observed, values)):
                raise ValueError("observed values lie outside the support")
            return Distribution(IndependentAdapter(tuple(c for i, c in enumerate(self.components) if i not in observed)))
        raise NotImplementedError(operation)


@dataclass(frozen=True)
class MixtureAdapter:
    components: tuple[Distribution, ...]
    weights: Any

    @property
    def event_shape(self):
        return self.components[0].event_shape

    @property
    def measure(self):
        return self.components[0].measure

    @property
    def capabilities(self):
        shared = set.intersection(*(set(c.capabilities) for c in self.components))
        supported = shared & {"log_density", "sample", "cdf", "survival", "mean",
                              "variance", "covariance", "moment", "expect", "support", "marginal"}
        if {"condition", "marginal"} <= shared:
            supported.add("condition")
        # Inversion is for continuous scalar CDFs only. Never interpolate atoms.
        if not self.event_shape and self.measure == "continuous" and {"quantile", "cdf", "support"} <= shared:
            supported.add("quantile")
        if "mean" not in shared:
            supported -= {"variance", "covariance"}
        return frozenset(supported)

    def _weighted(self, values):
        return np.tensordot(self.weights, np.stack(values), axes=(0, 0))

    def evaluate(self, operation, *args, **kwargs) -> Any:
        if operation == "log_density":
            densities = np.stack([c.log_density(args[0]) for c in self.components])
            weights = np.log(self.weights).reshape((-1,) + (1,) * (densities.ndim - 1))
            return special.logsumexp(densities + weights, axis=0)
        if operation in {"cdf", "survival", "mean", "moment", "expect"}:
            value = self._weighted([getattr(c, operation)(*args, **kwargs) for c in self.components])
            return np.clip(value, 0., 1.) if operation in {"cdf", "survival"} else value
        if operation in {"variance", "covariance"}:
            means = np.stack([c.mean() for c in self.components])
            centered = means - self._weighted(means)
            between = centered ** 2 if operation == "variance" else centered[..., :, None] * centered[..., None, :]
            within = np.stack([getattr(c, operation)() for c in self.components])
            return self._weighted(within + between)
        if operation == "sample":
            size, rng = args[0], kwargs["rng"]
            count = int(np.prod(size, dtype=int))
            labels = rng.choice(len(self.components), size=count, p=self.weights)
            result = np.empty((count,) + self.event_shape)
            for i, component in enumerate(self.components):
                selected = labels == i
                if np.any(selected):
                    result[selected] = component.sample(int(selected.sum()), rng=rng)
            return result.reshape(size + self.event_shape)
        if operation == "support":
            supports = [c.support() for c in self.components]
            return min(s[0] for s in supports), max(s[1] for s in supports)
        if operation == "quantile":
            def inverse(p):
                if p == 0 or p == 1:
                    return self.evaluate("support")[int(p)]
                bounds = [float(c.quantile(p)) for c in self.components]
                lo, hi = min(bounds), max(bounds)
                if not np.isfinite([lo, hi]).all():
                    raise ValueError("component quantile bounds are not representable")
                if lo == hi or self.evaluate("cdf", lo, rng=None) >= p:
                    return lo
                # Bisection of the monotone predicate also handles a flat CDF:
                # return the LEFT end of a quantile plateau, not an arbitrary root.
                return optimize.bisect(lambda x: 1. if self.evaluate("cdf", x, rng=None) >= p else -1.,
                                       lo, hi, xtol=np.nextafter(0., 1.),
                                       rtol=4*np.finfo(float).eps, maxiter=2048)
            p = args[0]
            return np.array([inverse(float(q)) for q in p.flat]).reshape(p.shape)
        if operation == "marginal":
            return Distribution(MixtureAdapter(tuple(c.marginal(args[0]) for c in self.components), self.weights))
        if operation == "condition":
            observed, values = args
            likelihoods = np.array([float(c.marginal(observed).log_density(values)) for c in self.components])
            log_weights = np.log(self.weights) + likelihoods
            total = float(cast(Any, special.logsumexp(log_weights)))
            if not np.isfinite(total):
                raise ValueError("conditioning requires a finite, positive marginal density or mass")
            weights = np.exp(log_weights - total)
            positive = weights > 0
            components = tuple(c.condition(observed, values) for c, keep in zip(self.components, positive) if keep)
            weights = weights[positive]
            weights /= weights.sum()
            weights.setflags(write=False)
            return Distribution(MixtureAdapter(components, weights))
        raise NotImplementedError(operation)
