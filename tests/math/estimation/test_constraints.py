import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import estimation as est


def test_box_and_linear_equality_constraints():
    space = est.ParameterSpace([.5, .5], names=("a", "b"), lower=0, upper=1)
    constraint = est.LinearConstraint([[1., 1.]], lower=1., upper=1.)
    result = est.minimize_estimate(lambda t: np.sum((t - [.8, .7])**2), space, constraints=[constraint])
    assert_allclose(result.parameters, [.55, .45], atol=1e-6)
    assert result.diagnostics.constraint_violation < 1e-7


def test_nonlinear_constraint_and_jacobian():
    constraint = est.NonlinearConstraint(lambda t: [t @ t], lower=-np.inf, upper=1., jacobian=lambda t: [2 * t])
    result = est.minimize_estimate(lambda t: np.sum((t - [2., 0.])**2), [.5, .2],
                                   gradient=lambda t: 2 * (t - [2., 0.]), constraints=[constraint])
    assert_allclose(result.parameters, [1., 0.], atol=2e-5)


def test_constrained_mle_does_not_supply_unconstrained_covariance():
    model = est.Likelihood([1., 2.], lambda t, y: -(y - t[0])**2 / 2)
    result = est.maximum_likelihood(model, [0.], constraints=[est.LinearConstraint([[1.]], upper=1.)])
    assert_allclose(result.parameters, [1.], atol=1e-6)
    assert result.covariance is None
    assert "constraints" in result.diagnostics.uncertainty_note


def test_explicit_solver_capability_rejection():
    space = est.ParameterSpace([1.], lower=0)
    with pytest.raises(ValueError, match="BFGS"):
        est.minimize_estimate(lambda t: t @ t, space, estimator=est.ScipyMinimizer("BFGS"))
    with pytest.raises(ValueError, match="constraints"):
        est.minimize_estimate(lambda t: t @ t, [1.], constraints=[est.LinearConstraint([[1.]], lower=0)],
                               estimator=est.ScipyMinimizer("L-BFGS-B"))


def test_failure_is_retained_and_raised_by_default():
    fun = lambda t: 100 * (t[1] - t[0]**2)**2 + (1 - t[0])**2
    tool = est.ScipyMinimizer(maxiter=1)
    with pytest.raises(est.ConvergenceError) as error:
        est.minimize_estimate(fun, [-1.2, 1.], estimator=tool)
    assert not error.value.result.diagnostics.converged
    result = est.minimize_estimate(fun, [-1.2, 1.], estimator=tool, check_convergence=False)
    assert_allclose(result.parameters, error.value.result.parameters)


def test_custom_estimator_and_result_contract():
    problem = est.ObjectiveProblem(lambda t: (t[0] - 3)**2, est.ParameterSpace([0.], names=("mu",)))
    result = est.fit(problem, lambda p: est.Estimate([3.], names=p.space.names, objective=0.))
    assert result.named_parameters == {"mu": 3.}
    with pytest.raises(ValueError, match="names"):
        est.fit(problem, lambda p: est.Estimate([3.]))
    with pytest.raises(TypeError, match="Estimate"):
        est.fit(problem, lambda p: [3.])  # type: ignore[arg-type] -- test invalid custom output
    constrained = est.ObjectiveProblem(lambda t: t @ t, est.ParameterSpace([0.], lower=0.))
    with pytest.raises(ValueError, match="violates"):
        est.fit(constrained, lambda p: est.Estimate([-1.]))
