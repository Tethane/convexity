"""Convenient scalar estimation, likelihood fitting and MAP constructors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from .. import Estimator, ObjectiveProblem


if TYPE_CHECKING:
    from .. import Estimate, LinearConstraint, NonlinearConstraint, ParameterSpace


from dataclasses import dataclass

from .. import _validation as v
from .._backends import scipy
from ..interfaces import Likelihood, ObjectiveProblem, fit
from ..interfaces.parameters import as_space


@dataclass(frozen=True)
class ScipyMinimizer:
    """Local optimization. auto selects SLSQP for bounds/constraints, L-BFGS-B otherwise.

    hessian in ObjectiveProblem supplies uncertainty and, for trust-constr, the
    optimizer. Other methods use only the objective and optional gradient.
    """

    method: str = "auto"
    tolerance: float = 1e-8
    maxiter: int = 1000

    def __post_init__(self) -> None:
        v.choice(
            self.method,
            ("auto", "L-BFGS-B", "BFGS", "SLSQP", "trust-constr", "Powell"),
            "method",
        )
        v.scalar(self.tolerance, "tolerance", positive=True)
        v.integer(self.maxiter, "maxiter")

    def __call__(self, problem: ObjectiveProblem):
        return scipy.minimize(
            problem, method=self.method, tolerance=self.tolerance, maxiter=self.maxiter
        )


def minimize_estimate(
    objective: Callable[[FloatArray], float],
    initial: ParameterSpace | ArrayLike,
    *,
    gradient: Callable[[FloatArray], ArrayLike] | None = None,
    hessian: Callable[[FloatArray], ArrayLike] | None = None,
    constraints: Sequence[LinearConstraint | NonlinearConstraint] = (),
    estimator: Estimator[ObjectiveProblem] | None = None,
    check_convergence: bool = True,
) -> Estimate:
    """Estimate parameters by minimizing a user criterion; no automatic covariance."""
    problem = ObjectiveProblem(
        objective, as_space(initial), gradient, hessian, constraints
    )
    return fit(
        problem,
        ScipyMinimizer() if estimator is None else estimator,
        check_convergence=check_convergence,
    )


class LogDensityLaw(Protocol):
    """Scalar/vector observation law; one log density per observation row."""

    def log_density(self, values: ArrayLike, /) -> ArrayLike: ...


def likelihood(
    factory: Callable[[FloatArray], LogDensityLaw],
    data: ArrayLike,
    *,
    frequency_weights: ArrayLike | None = None,
) -> Likelihood:
    """Compose a theta -> Distribution factory with scalar or vector observations.

    Any object with log_density works; the probability family needs no changes.
    For conditional models use Likelihood(data, log_density) directly.
    """
    v.function(factory, "distribution factory")
    return Likelihood(
        data,
        lambda theta, observations: factory(theta).log_density(observations),
        frequency_weights,
    )


def maximum_likelihood(
    model: Likelihood,
    initial: ParameterSpace | ArrayLike,
    *,
    gradient: Callable[[FloatArray], ArrayLike] | None = None,
    hessian: Callable[[FloatArray], ArrayLike] | None = None,
    constraints: Sequence[LinearConstraint | NonlinearConstraint] = (),
    uncertainty: bool = True,
    estimator: Estimator[ObjectiveProblem] | None = None,
    check_convergence: bool = True,
) -> Estimate:
    """Minimize the summed negative log likelihood; derivatives use that sign.

    uncertainty requests inverse observed information for an interior fit.
    """
    if not isinstance(model, Likelihood):
        raise TypeError("model must be Likelihood")
    v.boolean(uncertainty, "uncertainty")
    problem = ObjectiveProblem(
        model.negative_log_likelihood,
        as_space(initial),
        gradient,
        hessian,
        constraints,
        "observed_information" if uncertainty else None,
        model.nobs,
    )
    return fit(
        problem,
        ScipyMinimizer() if estimator is None else estimator,
        check_convergence=check_convergence,
    )


def maximum_a_posteriori(
    model: Likelihood,
    log_prior: Callable[[FloatArray], float],
    initial: ParameterSpace | ArrayLike,
    *,
    gradient: Callable[[FloatArray], ArrayLike] | None = None,
    hessian: Callable[[FloatArray], ArrayLike] | None = None,
    constraints: Sequence[LinearConstraint | NonlinearConstraint] = (),
    uncertainty: bool = True,
    estimator: Estimator[ObjectiveProblem] | None = None,
    check_convergence: bool = True,
) -> Estimate:
    """Minimize -log likelihood - log prior in the supplied parameter coordinates.

    Derivatives are of the whole negative log posterior. uncertainty requests a
    local Laplace posterior approximation, valid only at a smooth interior mode.
    """
    if not isinstance(model, Likelihood):
        raise TypeError("model must be Likelihood")
    v.function(log_prior, "log_prior")
    v.boolean(uncertainty, "uncertainty")
    from .penalties import IndependentPrior

    if (
        uncertainty
        and isinstance(log_prior, IndependentPrior)
        and log_prior.family == "laplace"
    ):
        raise ValueError(
            "a Laplace prior is nonsmooth; set uncertainty=False and use an appropriate optimizer"
        )

    def objective(theta):
        prior = v.scalar(log_prior(theta), "log prior", finite=False)
        if prior == float("inf"):
            raise ValueError("log prior must not be +inf")
        return model.negative_log_likelihood(theta) - prior

    problem = ObjectiveProblem(
        objective,
        as_space(initial),
        gradient,
        hessian,
        constraints,
        "laplace" if uncertainty else None,
        model.nobs,
    )
    return fit(
        problem,
        ScipyMinimizer() if estimator is None else estimator,
        check_convergence=check_convergence,
    )
