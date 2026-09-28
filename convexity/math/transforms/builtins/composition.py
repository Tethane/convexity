"""Sequential fitting and composition of row-preserving coordinate maps."""

from dataclasses import dataclass
from typing import Any

from .. import _validation as v
from ..interfaces import checked_transform, fit


@dataclass(frozen=True)
class FittedPipeline:
    steps: tuple[Any, ...]

    def __post_init__(self):
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("pipeline must have at least one step")
        for step in steps:
            v.integer(step.input_dimension, "input_dimension", 1)
            v.integer(step.output_dimension, "output_dimension", 1)
            if not callable(step.transform):
                raise TypeError("each step must have a transform method")
        if any(a.output_dimension != b.input_dimension for a, b in zip(steps, steps[1:])):
            raise ValueError("adjacent pipeline dimensions must match")
        object.__setattr__(self, "steps", steps)

    @property
    def input_dimension(self):
        return self.steps[0].input_dimension

    @property
    def output_dimension(self):
        return self.steps[-1].output_dimension

    @property
    def is_invertible(self):
        return all(getattr(step, "is_invertible", False) for step in self.steps)

    def transform(self, values):
        current = v.coordinates(values, self.input_dimension, "values")
        for step in self.steps:
            current = checked_transform(step, current)
        return current

    def _decode(self, coordinates, method):
        current = v.coordinates(coordinates, self.output_dimension)
        for step in reversed(self.steps):
            callback = getattr(step, method, None)
            if not callable(callback):
                raise ValueError(f"pipeline step does not support {method}")
            decoded = v.coordinates(callback(current), step.input_dimension, "decoded values")
            if decoded.shape[:-1] != current.shape[:-1]:
                raise ValueError("decoding must preserve all leading dimensions")
            current = decoded
        return current

    def reconstruct(self, coordinates):
        return self._decode(coordinates, "reconstruct")

    def inverse_transform(self, coordinates):
        if not self.is_invertible:
            raise ValueError("pipeline has no exact inverse; use reconstruct")
        return self._decode(coordinates, "inverse_transform")

    def residuals(self, values):
        x = v.coordinates(values, self.input_dimension, "values")
        return v.array(x - self.reconstruct(self.transform(x)), "residuals")


@dataclass(frozen=True)
class Pipeline:
    """Fit each step only on the preceding step's transformed training rows."""

    steps: tuple[Any, ...]

    def __post_init__(self):
        steps = tuple(self.steps)
        if not steps or any(not callable(getattr(step, "fit", None)) for step in steps):
            raise TypeError("pipeline requires one or more fitters")
        object.__setattr__(self, "steps", steps)

    def fit(self, samples):
        current = v.samples(samples)
        fitted = []
        for step in self.steps:
            result = fit(current, step)
            fitted.append(result)
            current = checked_transform(result, current)
        return FittedPipeline(tuple(fitted))
