"""Scalar objectives, residual models, likelihoods and linear designs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .. import _validation as v
from .base import check_estimate
from .parameters import LinearConstraint, NonlinearConstraint, ParameterSpace


@dataclass(frozen=True)
class ObjectiveProblem:
    """Minimize a scalar criterion. curvature labels its inverse Hessian meaning.

    +inf is a permitted trial objective outside likelihood/prior support; NaN
    and -inf are rejected. Derivatives are of this entire minimized objective.
    """

    objective: Callable[[FloatArray], float]
    space: ParameterSpace
    gradient: Callable[[FloatArray], ArrayLike] | None = None
    hessian: Callable[[FloatArray], ArrayLike] | None = None
    constraints: Sequence[LinearConstraint | NonlinearConstraint] = ()
    curvature: str | None = None
    nobs: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.space, ParameterSpace):
            raise TypeError("space must be ParameterSpace")
        v.function(self.objective, "objective")
        for name in ("gradient", "hessian"):
            if getattr(self, name) is not None:
                v.function(getattr(self, name), name)
        v.choice(self.curvature, (None, "observed_information", "laplace"), "curvature")
        if self.nobs is not None:
            v.integer(self.nobs, "nobs")
        constraints = tuple(self.constraints)
        for constraint in constraints:
            if not isinstance(constraint, (LinearConstraint, NonlinearConstraint)):
                raise TypeError(
                    "constraints must be LinearConstraint or NonlinearConstraint"
                )
            if (
                isinstance(constraint, LinearConstraint)
                and np.shape(constraint.matrix)[1] != self.space.size
            ):
                raise ValueError("constraint matrix must have p columns")
            constraint.evaluate(self.space.initial)
        object.__setattr__(self, "constraints", constraints)

    def evaluate(self, theta: ArrayLike):
        value = v.scalar(
            self.objective(self.space.point(theta)), "objective", finite=False
        )
        if value == -np.inf:
            raise ValueError("objective must not be -inf")
        return value

    def derivative(self, theta: ArrayLike):
        assert self.gradient is not None
        value = v.array(self.gradient(self.space.point(theta)), "gradient", ndim=1)
        if value.shape != (self.space.size,):
            raise ValueError("gradient must have shape (p,)")
        return value

    def second_derivative(self, theta: ArrayLike):
        assert self.hessian is not None
        value = v.array(self.hessian(self.space.point(theta)), "Hessian", ndim=2)
        if value.shape != (self.space.size, self.space.size) or not np.allclose(
            value, value.T
        ):
            raise ValueError("Hessian must be symmetric with shape (p,p)")
        return value

    def violation(self, theta: ArrayLike):
        result = self.space.violation(theta)
        for constraint in self.constraints:
            value = constraint.evaluate(theta)
            result = max(
                result,
                float(np.max(constraint.lower - value)),
                float(np.max(value - constraint.upper)),
            )
        return result

    def validate_result(self, result):
        check_estimate(result, self.space.names)
        if result.diagnostics.converged and self.violation(result.parameters) > 1e-6:
            raise ValueError("converged estimate violates parameter constraints")


@dataclass(frozen=True)
class LeastSquaresProblem:
    """Residual vector r(theta), optionally already whitened by known noise.

    The criterion is half the sum of squared residuals for linear loss. An
    analytic Jacobian must differentiate the same (possibly whitened) residuals.
    """

    residuals: Callable[[FloatArray], ArrayLike]
    space: ParameterSpace
    jacobian: Callable[[FloatArray], ArrayLike] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.space, ParameterSpace):
            raise TypeError("space must be ParameterSpace")
        v.function(self.residuals, "residuals")
        if self.jacobian is not None:
            v.function(self.jacobian, "Jacobian")

    def evaluate(self, theta: ArrayLike) -> FloatArray:
        return v.array(self.residuals(self.space.point(theta)), "residuals", ndim=1)

    def derivative(self, theta: ArrayLike):
        assert self.jacobian is not None
        value = v.array(self.jacobian(self.space.point(theta)), "Jacobian", ndim=2)
        if value.shape != (len(self.evaluate(theta)), self.space.size):
            raise ValueError("residual Jacobian must have shape (n,p)")
        return value

    def validate_result(self, result):
        check_estimate(result, self.space.names)
        if (
            result.diagnostics.converged
            and self.space.violation(result.parameters) > 1e-6
        ):
            raise ValueError("converged estimate violates parameter bounds")


@dataclass(frozen=True)
class Likelihood:
    """Independent observation contributions log_density(theta, data) -> (n,).

    Rows are observations. Optional integer frequency weights mean literal
    replication; they are not precision, survey or importance weights. Zero
    frequency rows are removed before calling the model. Constants are retained.
    """

    data: FloatArray
    log_density: Callable[[FloatArray, FloatArray], ArrayLike]
    frequency_weights: FloatArray = cast("FloatArray", None)

    def __post_init__(self) -> None:
        data = v.array(self.data, "data")
        if data.ndim == 0:
            raise ValueError("data must have an observation axis")
        v.function(self.log_density, "log_density")
        weights = (
            np.ones(len(data))
            if self.frequency_weights is None
            else v.array(self.frequency_weights, "frequency_weights", ndim=1)
        )
        if (
            weights.shape != (len(data),)
            or np.any(weights < 0)
            or np.any(weights != np.floor(weights))
            or not np.any(weights > 0)
            or not np.isfinite(weights.sum())
            or weights.sum() > 2**53
        ):
            raise ValueError(
                "frequency_weights must be nonnegative integer counts with a positive representable sum"
            )
        keep = weights > 0
        object.__setattr__(self, "data", v.array(data[keep], "data"))
        object.__setattr__(
            self, "frequency_weights", v.array(weights[keep], "frequency_weights")
        )

    @property
    def nobs(self) -> int:
        return int(np.asarray(self.frequency_weights).sum())

    def contributions(self, theta: ArrayLike):
        value = v.array(
            self.log_density(v.array(theta, "parameters", ndim=1), self.data),
            "log likelihood contributions",
            ndim=1,
            finite=False,
        )
        if value.shape != (np.shape(self.data)[0],) or np.any(np.isposinf(value)):
            raise ValueError("log_density must return (n,) values without +inf")
        return value

    def log_likelihood(self, theta: ArrayLike) -> float:
        return float(
            np.dot(np.asarray(self.frequency_weights), self.contributions(theta))
        )

    def negative_log_likelihood(self, theta: ArrayLike):
        return -self.log_likelihood(theta)

    def __init__(
        self,
        data: ArrayLike,
        log_density: Callable[[FloatArray, FloatArray], ArrayLike],
        frequency_weights: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "log_density", log_density)
        object.__setattr__(self, "frequency_weights", frequency_weights)
        self.__post_init__()


@dataclass(frozen=True)
class LinearProblem:
    """y = X @ theta + error, with no implicit intercept or standardization.

    weights are strictly positive relative precisions, or sigma is a positive
    definite relative observation covariance. The two are mutually exclusive.
    """

    design: FloatArray
    response: FloatArray
    names: tuple[str, ...] = cast("tuple[str, ...]", None)
    weights: FloatArray | None = cast("FloatArray | None", None)
    sigma: FloatArray | None = cast("FloatArray | None", None)

    def __post_init__(self) -> None:
        design = v.array(self.design, "design", ndim=2)
        response = v.array(self.response, "response", ndim=1)
        if design.shape[0] != len(response):
            raise ValueError("design rows must match response length")
        if self.weights is not None and self.sigma is not None:
            raise ValueError("supply either weights or sigma")
        weights = None
        sigma = None
        if self.weights is not None:
            weights = v.array(self.weights, "weights", ndim=1)
            if weights.shape != response.shape or np.any(weights <= 0):
                raise ValueError("weights must be strictly positive with shape (n,)")
        if self.sigma is not None:
            sigma = v.matrix(self.sigma, len(response), "sigma", positive=True)
        for name, value in dict(
            design=design,
            response=response,
            names=v.names(self.names, design.shape[1]),
            weights=weights,
            sigma=sigma,
        ).items():
            object.__setattr__(self, name, value)

    def validate_result(self, result):
        check_estimate(result, self.names)
        if result.residuals is not None and result.residuals.shape != np.shape(
            self.response
        ):
            raise ValueError("linear residuals must have shape (n,)")

    def __init__(
        self,
        design: ArrayLike,
        response: ArrayLike,
        names: Sequence[str] | None = None,
        weights: ArrayLike | None = None,
        sigma: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "sigma", sigma)
        self.__post_init__()
