"""Public protocols for fitted, row-preserving coordinate maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decomposition import FactorAnalysisResult


from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import Transform


from dataclasses import dataclass
from typing import Protocol

from .. import _validation as v


class Transform(Protocol):
    """Coordinate map with declared input/output dimensions; transform preserves leading observation axes."""

    @property
    def input_dimension(self) -> int: ...

    @property
    def output_dimension(self) -> int: ...

    def transform(self, values: ArrayLike) -> FloatArray: ...


class TransformFitter(Protocol):
    """Fit a coordinate transform from observation rows without implicitly changing target semantics."""

    def fit(self, samples: ArrayLike) -> Transform: ...


@dataclass(frozen=True)
class Diagnostics:
    """Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable."""

    backend: str = "user"
    converged: bool = True
    message: str = ""
    iterations: int | None = None

    def __post_init__(self) -> None:
        v.boolean(self.converged, "converged")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a nonempty string")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if self.iterations is not None:
            v.integer(self.iterations, "iterations")


class ConvergenceError(RuntimeError):
    """Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics."""

    def __init__(self, result: FactorAnalysisResult) -> None:
        self.result = result
        super().__init__(result.diagnostics.message or "Transform fit did not converge")


def checked_transform(transform: Transform, values: ArrayLike) -> FloatArray:
    """Check a user-written map at the composition boundary."""
    d = v.integer(transform.input_dimension, "input_dimension", 1)
    k = v.integer(transform.output_dimension, "output_dimension", 1)
    x = v.coordinates(values, d, "values")
    z = v.coordinates(transform.transform(x), k, "transformed values")
    if z.shape[:-1] != x.shape[:-1]:
        raise ValueError("transform must preserve all leading dimensions")
    return z


def fit(samples: ArrayLike, fitter: TransformFitter) -> Transform:
    """Fit a user or built-in tool to an explicit (n,d) training matrix."""
    x = v.samples(samples)
    result = fitter.fit(x)
    if result.input_dimension != x.shape[1]:
        raise ValueError("fitted input_dimension must match training columns")
    checked_transform(result, x)
    return result
