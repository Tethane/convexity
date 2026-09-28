"""Public protocols for fitted, row-preserving coordinate maps."""

from dataclasses import dataclass
from typing import Any, Protocol

from numpy.typing import NDArray

from .. import _validation as v


class Transform(Protocol):
    @property
    def input_dimension(self) -> int: ...

    @property
    def output_dimension(self) -> int: ...

    def transform(self, values: Any) -> NDArray[Any]: ...


class TransformFitter(Protocol):
    def fit(self, samples: Any) -> Transform: ...


@dataclass(frozen=True)
class Diagnostics:
    backend: str = "user"
    converged: bool = True
    message: str = ""
    iterations: int | None = None

    def __post_init__(self):
        v.boolean(self.converged, "converged")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a nonempty string")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if self.iterations is not None:
            v.integer(self.iterations, "iterations")


class ConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(result.diagnostics.message or "Transform fit did not converge")


def checked_transform(transform, values):
    """Check a user-written map at the composition boundary."""
    d = v.integer(transform.input_dimension, "input_dimension", 1)
    k = v.integer(transform.output_dimension, "output_dimension", 1)
    x = v.coordinates(values, d, "values")
    z = v.coordinates(transform.transform(x), k, "transformed values")
    if z.shape[:-1] != x.shape[:-1]:
        raise ValueError("transform must preserve all leading dimensions")
    return z


def fit(samples, fitter: TransformFitter) -> Transform:
    """Fit a user or built-in tool to an explicit (n,d) training matrix."""
    x = v.samples(samples)
    result = fitter.fit(x)
    if result.input_dimension != x.shape[1]:
        raise ValueError("fitted input_dimension must match training columns")
    checked_transform(result, x)
    return result
