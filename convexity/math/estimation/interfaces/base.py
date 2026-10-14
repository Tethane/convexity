"""Immutable estimates and structural contracts for user-written estimators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v


@dataclass(frozen=True)
class Diagnostics:
    """Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable."""

    converged: bool = True
    message: str = ""
    backend: str = "user"
    iterations: int | None = None
    evaluations: int | None = None
    rank: int | None = None
    constraint_violation: float = 0.0
    uncertainty_note: str = ""

    def __post_init__(self) -> None:
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

    parameters: FloatArray
    names: tuple[str, ...]
    diagnostics: Diagnostics
    objective: float | None
    covariance: FloatArray | None
    covariance_kind: str | None
    df: float | None
    nobs: int | None
    residuals: FloatArray | None

    def __init__(
        self,
        parameters: ArrayLike,
        *,
        names: Sequence[str] | None = None,
        diagnostics: Diagnostics | None = None,
        objective: float | None = None,
        covariance: ArrayLike | None = None,
        covariance_kind: str | None = None,
        df: float | None = None,
        nobs: int | None = None,
        residuals: ArrayLike | None = None,
    ) -> None:
        theta = v.array(parameters, "parameters", ndim=1)
        report = Diagnostics() if diagnostics is None else diagnostics
        if not isinstance(report, Diagnostics):
            raise TypeError("diagnostics must be Diagnostics")
        if (covariance is None) != (covariance_kind is None):
            raise ValueError("covariance and covariance_kind must be supplied together")
        if covariance_kind is not None and (
            not isinstance(covariance_kind, str) or not covariance_kind
        ):
            raise ValueError("covariance_kind must be a nonempty string")
        values = dict(
            parameters=theta,
            names=v.names(names, len(theta)),
            diagnostics=report,
            objective=None
            if objective is None
            else v.scalar(objective, "objective", finite=report.converged),
            covariance=None
            if covariance is None
            else v.matrix(covariance, len(theta), "covariance"),
            covariance_kind=covariance_kind,
            df=None if df is None else v.scalar(df, "df", positive=True),
            nobs=None if nobs is None else v.integer(nobs, "nobs"),
            residuals=None
            if residuals is None
            else v.array(residuals, "residuals", ndim=1),
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def named_parameters(self):
        return MappingProxyType(dict(zip(self.names, map(float, self.parameters))))

    @property
    def standard_errors(self) -> FloatArray:
        if self.covariance is None:
            raise ValueError(
                self.diagnostics.uncertainty_note or "covariance is unavailable"
            )
        return v.array(
            np.sqrt(np.maximum(np.diag(self.covariance), 0)), "standard errors"
        )


class EstimationProblem(Protocol):
    """Structural parameter-estimation problem; names defines parameter coordinate order."""

    def validate_result(self, result: Estimate) -> None: ...


P = TypeVar("P", bound=EstimationProblem)
Input = TypeVar("Input", contravariant=True)


class Estimator(Protocol[Input]):
    """A callable accepting one problem and returning a checked Estimate."""

    def __call__(self, problem: Input, /) -> Estimate: ...


class ConvergenceError(RuntimeError):
    """Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics."""

    def __init__(self, result: Estimate) -> None:
        self.result = result
        super().__init__(result.diagnostics.message or "Estimation did not converge")


def fit(
    problem: P, estimator: Estimator[P], *, check_convergence: bool = True
) -> Estimate:
    """Estimate a checked problem using a compatible built-in or user Estimator.

    Return Estimate with named parameter coordinates, objective diagnostics and
    optional uncertainty. check_convergence=True raises ConvergenceError on failure."""
    v.function(estimator, "estimator")
    v.boolean(check_convergence, "check_convergence")
    result = estimator(problem)
    problem.validate_result(result)
    if check_convergence and not result.diagnostics.converged:
        raise ConvergenceError(result)
    return result


def check_estimate(result, names: Sequence[str]):
    if not isinstance(result, Estimate):
        raise TypeError("estimator must return Estimate")
    if result.parameters.shape != (len(names),) or result.names != names:
        raise ValueError("estimate must preserve parameter shape, names and order")
