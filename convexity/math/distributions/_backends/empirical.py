"""Exact finite probability algebra and NumPy sampling; no smoothing or fitting."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _validation as v
from ..interfaces import Distribution, Measure


@dataclass(frozen=True)
class EmpiricalAdapter:
    values: Any
    mass: Any
    measure: Measure = "discrete"

    @property
    def event_shape(self):
        return self.values.shape[1:]

    @property
    def capabilities(self):
        common = {
            "log_density",
            "sample",
            "cdf",
            "mean",
            "variance",
            "moment",
            "expect",
            "probability",
            "given",
            "map",
        }
        return frozenset(
            common
            | (
                {"covariance", "marginal", "condition"}
                if self.event_shape
                else {"survival", "quantile", "support"}
            )
        )

    @property
    def weights(self):
        return self.mass / self.mass.sum()

    def _mask(self, event):
        mask = np.asarray([event(x) for x in self.values])
        if mask.shape != (len(self.values),) or mask.dtype.kind != "b":
            raise ValueError("event must return one boolean per event")
        return mask

    def evaluate(self, operation, *args, **kwargs):
        if operation in {"log_density", "cdf", "survival"}:
            x = args[0]
            shape = x.shape[:-1] if self.event_shape else x.shape
            points = x.reshape((-1,) + self.event_shape)
            # Iterate queries to avoid allocating queries x observations x dimension.
            result = []
            for point in points:
                if operation == "log_density":
                    mask = self.values == point
                elif operation == "cdf":
                    mask = self.values <= point
                else:
                    mask = self.values > point
                if self.event_shape:
                    mask = np.all(mask, axis=-1)
                result.append(self.mass[mask].sum() / self.mass.sum())
            result = np.asarray(result).reshape(shape)
            if operation == "log_density":
                with np.errstate(divide="ignore"):
                    result = np.log(result)
            return result
        if operation == "sample":
            indices = kwargs["rng"].choice(
                len(self.values), size=args[0], p=self.weights
            )
            return self.values[indices]
        if operation == "support":
            return self.values.min(), self.values.max()
        if operation == "quantile":
            order = np.argsort(self.values, kind="stable")
            # Accumulate unnormalized masses: equal masses retain ECDF boundaries.
            cumulative = np.cumsum(self.mass[order])
            cumulative /= cumulative[-1]
            indices = np.searchsorted(cumulative, args[0], side="left")
            return self.values[order[indices]]
        if operation == "mean":
            return np.tensordot(self.weights, self.values, axes=(0, 0))
        if operation in {"variance", "covariance"}:
            centered = self.values - self.evaluate("mean")
            if operation == "variance":
                return np.tensordot(self.weights, centered**2, axes=(0, 0))
            return (centered.T * self.weights) @ centered
        if operation == "moment":
            return np.tensordot(self.weights, self.values ** args[0], axes=(0, 0))
        if operation == "expect":
            if kwargs:
                raise TypeError("finite expectations do not accept integration options")
            values = np.asarray(
                [v.array(args[0](x), "expectation callback") for x in self.values]
            )
            return np.tensordot(self.weights, values, axes=(0, 0))
        if operation == "probability":
            return self.mass[self._mask(args[0])].sum() / self.mass.sum()
        if operation == "given":
            mask = self._mask(args[0])
            return empirical_law(self.values[mask], self.mass[mask])
        if operation == "map":
            return empirical_law([args[0](x) for x in self.values], self.mass)
        if operation == "marginal":
            return empirical_law(self.values[:, args[0]], self.mass)
        if operation == "condition":
            observed, values = args
            mask = np.all(self.values[:, observed] == values, axis=1)
            remaining = np.setdiff1d(np.arange(self.event_shape[0]), observed)
            return empirical_law(self.values[mask][:, remaining], self.mass[mask])
        raise NotImplementedError(operation)


def empirical_law(values, weights=None):
    values = v.array(values, "empirical values")
    if values.ndim not in {1, 2} or (values.ndim == 2 and values.shape[1] == 0):
        raise ValueError("empirical values must have shape (n,) or (n,d)")
    mass = v.weights(weights, len(values))
    positive = mass > 0
    # Zero mass cannot affect support, callback evaluation, or moments.
    values, mass = values[positive], mass[positive]
    values.setflags(write=False)
    mass.setflags(write=False)
    return Distribution(EmpiricalAdapter(values, mass))
