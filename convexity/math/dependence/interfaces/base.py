"""Structural extension contracts and fitting diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import CovarianceEstimate, MatrixRepair, StudentTResult, TailFit

from dataclasses import dataclass
from typing import Protocol

from numpy.typing import ArrayLike

from .. import _validation as v

if TYPE_CHECKING:
    from .covariance import CovarianceEstimate


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


def diagnostics(value):
    if value is None:
        return Diagnostics()
    if not isinstance(value, Diagnostics):
        raise TypeError("diagnostics must be Diagnostics")
    return value


class ConvergenceError(RuntimeError):
    """Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics."""

    def __init__(
        self, result: CovarianceEstimate | StudentTResult | TailFit | MatrixRepair
    ) -> None:
        self.result = result
        super().__init__(
            result.diagnostics.message or "Dependence fit did not converge"
        )


def checked_convergence[
    ResultT: CovarianceEstimate | StudentTResult | TailFit | MatrixRepair
](result: ResultT, check: bool) -> ResultT:
    v.boolean(check, "check_convergence")
    if check and not result.diagnostics.converged:
        raise ConvergenceError(result)
    return result


class CovarianceEstimator(Protocol):
    """Fit location/covariance from observation rows, returning a checked CovarianceEstimate."""

    def fit(self, samples: ArrayLike) -> "CovarianceEstimate": ...


class CopulaAdapter(Protocol):
    """Dependence adapter declaring dimension and supported operations on unit-interval coordinates."""

    @property
    def dimension(self) -> int: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    def evaluate(
        self, operation: str, *args: object, **kwargs: object
    ) -> ArrayLike: ...


class ContinuousMarginal(Protocol):
    """Scalar marginal law providing CDF, quantile and log density for copula composition."""

    @property
    def event_shape(self) -> tuple[int, ...]: ...

    @property
    def measure(self) -> str: ...

    def cdf(self, values: ArrayLike, /) -> ArrayLike: ...
    def quantile(self, probability: ArrayLike, /) -> ArrayLike: ...
    def log_density(self, values: ArrayLike, /) -> ArrayLike: ...
