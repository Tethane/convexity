"""Immutable estimates and structural contracts for user-written estimators."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _validation as v


@dataclass(frozen=True)
class Diagnostics:
    converged: bool = True
    message: str = ""
    backend: str = "user"
    iterations: int | None = None
    evaluations: int | None = None
    rank: int | None = None
    constraint_violation: float = 0.
    uncertainty_note: str = ""

    def __post_init__(self):
        v.boolean(self.converged, "converged")
        for name in ("iterations", "evaluations", "rank"):
            if getattr(self, name) is not None:
                v.integer(getattr(self, name), name, minimum=0)
        v.scalar(self.constraint_violation, "constraint_violation", nonnegative=True)


@dataclass(frozen=True, init=False)
class Estimate:
    """One parameter vector; missing uncertainty stays None.

    Covariance is in the coordinates of parameters. df controls the Student-t
    reference for frequentist intervals; None means an asymptotic normal law.
    A Laplace covariance describes local posterior curvature, not coverage.
    """

    parameters: NDArray[Any]
    names: tuple[str, ...]
    diagnostics: Diagnostics
    objective: float | None
    covariance: NDArray[Any] | None
    covariance_kind: str | None
    df: float | None
    nobs: int | None
    residuals: NDArray[Any] | None

    def __init__(self, parameters: ArrayLike, *, names=None, diagnostics=None,
                 objective=None, covariance=None, covariance_kind=None, df=None,
                 nobs=None, residuals=None):
        theta = v.array(parameters, "parameters", ndim=1)
        report = Diagnostics() if diagnostics is None else diagnostics
        if not isinstance(report, Diagnostics):
            raise TypeError("diagnostics must be Diagnostics")
        if (covariance is None) != (covariance_kind is None):
            raise ValueError("covariance and covariance_kind must be supplied together")
        if covariance_kind is not None and (not isinstance(covariance_kind, str) or not covariance_kind):
            raise ValueError("covariance_kind must be a nonempty string")
        values = dict(parameters=theta, names=v.names(names, len(theta)), diagnostics=report,
                      objective=None if objective is None else v.scalar(
                          objective, "objective", finite=report.converged),
                      covariance=None if covariance is None else v.matrix(covariance, len(theta), "covariance"),
                      covariance_kind=covariance_kind,
                      df=None if df is None else v.scalar(df, "df", positive=True),
                      nobs=None if nobs is None else v.integer(nobs, "nobs"),
                      residuals=None if residuals is None else v.array(residuals, "residuals", ndim=1))
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def named_parameters(self):
        return MappingProxyType(dict(zip(self.names, map(float, self.parameters))))

    @property
    def standard_errors(self):
        if self.covariance is None:
            raise ValueError(self.diagnostics.uncertainty_note or "covariance is unavailable")
        return v.array(np.sqrt(np.maximum(np.diag(self.covariance), 0)), "standard errors")


class EstimationProblem(Protocol):
    def validate_result(self, result: Any) -> None: ...


P = TypeVar("P", bound=EstimationProblem)
Input = TypeVar("Input", contravariant=True)


class Estimator(Protocol[Input]):
    """A callable accepting one problem and returning a checked Estimate."""

    def __call__(self, problem: Input, /) -> Estimate: ...


class ConvergenceError(RuntimeError):
    def __init__(self, result: Estimate):
        self.result = result
        super().__init__(result.diagnostics.message or "Estimation did not converge")


def fit(problem: P, estimator: Estimator[P], *, check_convergence=True) -> Estimate:
    v.function(estimator, "estimator")
    v.boolean(check_convergence, "check_convergence")
    result = estimator(problem)
    problem.validate_result(result)
    if check_convergence and not result.diagnostics.converged:
        raise ConvergenceError(result)
    return result


def check_estimate(result, names):
    if not isinstance(result, Estimate):
        raise TypeError("estimator must return Estimate")
    if result.parameters.shape != (len(names),) or result.names != names:
        raise ValueError("estimate must preserve parameter shape, names and order")
