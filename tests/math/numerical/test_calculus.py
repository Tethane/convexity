import math
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import numerical as nm


@pytest.mark.parametrize("function,a,b,expected", [
    (lambda x: x**4, 0, 1, .2),
    (math.sin, 0, np.pi, 2),
    (lambda x: math.exp(-x*x), -np.inf, np.inf, np.sqrt(np.pi)),
    (math.sin, np.pi, 0, -2),
    (math.sin, 1, 1, 0),
])
def test_quadrature_known_integrals(function, a, b, expected):
    result = nm.quad(function, a, b)
    assert_allclose(result.value, expected, atol=1e-10)
    assert result.diagnostics.evaluations is not None
    assert result.diagnostics.evaluations >= 0
    assert np.all(np.asarray(result.diagnostics.error_estimate) >= 0)


def test_quadrature_failure_exposes_partial_result():
    with pytest.raises(nm.ConvergenceError) as caught:
        nm.quad(lambda x: np.sin(1000*x), 0, 1, tool=nm.AdaptiveQuadrature(limit=1))
    assert isinstance(caught.value.result, nm.ArrayResult)
    assert not caught.value.result.diagnostics.converged
    assert isinstance(caught.value.result, nm.ArrayResult)
    assert caught.value.result.diagnostics.message


def test_scalar_derivative_with_nonvectorized_callback_and_one_sided_boundary():
    result = nm.derivative(math.sin, .4)
    assert_allclose(result.value, math.cos(.4), rtol=1e-9)

    def domain_limited(x):
        if x < 0:
            raise ValueError("outside domain")
        return math.log1p(x)

    assert_allclose(nm.derivative(domain_limited, 0., tool=nm.FiniteDifference(direction=1)).value,
                    1., atol=1e-6)


def test_gradient_hessian_known_quadratic_and_callback_shapes():
    a = np.array([[3., .5], [.5, 2.]])
    x = np.array([.2, -.7])
    seen = []

    def objective(point):
        assert point.shape == (2,)
        seen.append(point.shape)
        return .5 * point @ a @ point + np.array([1., -1.]) @ point

    assert_allclose(nm.gradient(objective, x).value, a @ x + [1., -1.], atol=1e-8)
    assert_allclose(nm.hessian(objective, x).value, a, atol=1e-7)
    assert seen


def test_rectangular_jacobian_orientation():
    def function(x):
        return [x[0] * x[1], x[0]**2, math.sin(x[1])]

    expected = [[3., 2.], [4., 0.], [0., math.cos(3.)]]
    assert_allclose(nm.jacobian(function, [2., 3.]).value, expected, atol=1e-8)


def test_finite_difference_accuracy_improves_with_refinement():
    coarse = nm.derivative(math.exp, 1., tool=nm.FiniteDifference(order=2, maxiter=2),
                           check_convergence=False)
    fine = nm.derivative(math.exp, 1., tool=nm.FiniteDifference(order=2, maxiter=12))
    assert not coarse.diagnostics.converged
    assert abs(float(fine.value) - math.e) < abs(float(coarse.value) - math.e) / 100
    assert np.asarray(fine.diagnostics.error_estimate) < np.asarray(coarse.diagnostics.error_estimate)


def test_derivative_nonconvergence_and_function_contracts():
    with pytest.raises(nm.ConvergenceError):
        nm.derivative(math.sin, .4, tool=nm.FiniteDifference(maxiter=1))
    with pytest.raises(ValueError, match="dimensions"):
        nm.gradient(lambda x: x, [1., 2.])
    with pytest.raises(ValueError, match="finite"):
        nm.derivative(lambda x: float("nan"), 1.)
    with pytest.raises(ValueError, match="real"):
        nm.quad(cast(Any, lambda x: 1j), 0, 1)
    with pytest.raises(ValueError, match="scalar derivatives"):
        nm.hessian(lambda x: x @ x, [1, 2], tool=nm.FiniteDifference(direction=1))


@pytest.mark.parametrize("factory", [
    lambda: nm.FiniteDifference(maxiter=0), lambda: nm.FiniteDifference(order=cast(Any, 1.5)),
    lambda: nm.FiniteDifference(initial_step=-1), lambda: nm.FiniteDifference(atol=0, rtol=0),
    lambda: nm.AdaptiveQuadrature(atol=0, rtol=1e-18),
])
def test_invalid_numerical_controls(factory):
    with pytest.raises(ValueError):
        factory()
