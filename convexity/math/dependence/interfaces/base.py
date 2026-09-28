"""Structural extension contracts and fitting diagnostics."""

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

from .. import _validation as v

if TYPE_CHECKING:
    from .covariance import CovarianceEstimate


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


def diagnostics(value):
    if value is None:
        return Diagnostics()
    if not isinstance(value, Diagnostics):
        raise TypeError("diagnostics must be Diagnostics")
    return value


class ConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(result.diagnostics.message or "Dependence fit did not converge")


def checked_convergence(result, check):
    v.boolean(check, "check_convergence")
    if check and not result.diagnostics.converged:
        raise ConvergenceError(result)
    return result


class CovarianceEstimator(Protocol):
    def fit(self, samples: Any) -> "CovarianceEstimate": ...


class CopulaAdapter(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    def evaluate(self, operation: str, *args: Any, **kwargs: Any) -> Any: ...


class ContinuousMarginal(Protocol):
    @property
    def event_shape(self) -> tuple[int, ...]: ...

    @property
    def measure(self) -> str: ...

    def cdf(self, values: Any) -> Any: ...
    def quantile(self, probability: Any) -> Any: ...
    def log_density(self, values: Any) -> Any: ...
