from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import numerical as nm


def test_custom_callable_uses_same_validated_interface_and_builtin_entry_point():
    calls = []

    def diagonal_solver(problem: nm.LinearSystem) -> nm.ArrayResult:
        calls.append(problem)
        a, b = np.asarray(problem.a), np.asarray(problem.b)
        if not np.array_equal(a, np.diag(np.diag(a))) or b.ndim != 1:
            raise ValueError("this tool only accepts diagonal systems with vector RHS")
        x = b / np.diag(a)
        return nm.ArrayResult(x, nm.Diagnostics(residual_norm=float(np.linalg.norm(a @ x-b))))

    problem = nm.LinearSystem([[2, 0], [0, 4]], [6, 8])
    assert_allclose(nm.run(problem, diagonal_solver).value, [3, 2])
    assert_allclose(nm.solve(problem.a, problem.b, tool=diagonal_solver).value, [3, 2])
    with pytest.raises(ValueError):
        nm.solve([[1, 2]], [1], tool=diagonal_solver)
    assert len(calls) == 2  # Invalid input never reaches the custom tool.


def test_custom_differentiator_class_and_interpolator():
    @dataclass
    class AnalyticGradient:
        def __call__(self, problem: nm.DifferentiationProblem) -> nm.ArrayResult:
            return nm.ArrayResult(2*np.asarray(problem.x))

    assert_allclose(nm.gradient(lambda x: x @ x, [1, 2], tool=AnalyticGradient()).value, [2, 4])

    def constant_interpolator(problem: nm.InterpolationProblem) -> nm.Interpolant:
        x, y = np.asarray(problem.x), np.asarray(problem.y)
        return nm.Interpolant(lambda points: np.full(points.shape, y[0]),
                              (x[0], x[-1]), extrapolate=problem.extrapolate)

    assert_allclose(nm.interpolate([0, 1], [2, 2], tool=constant_interpolator)([.2, .8]), [2, 2])


def test_contract_rejects_bad_custom_results_and_failure_claims():
    problem = nm.LinearSystem(np.eye(2), [1, 2])
    with pytest.raises(TypeError, match="ArrayResult"):
        nm.run(problem, lambda p: np.ones(2))
    with pytest.raises(ValueError, match="shape"):
        nm.run(problem, lambda p: nm.ArrayResult([1, 2, 3]))
    with pytest.raises(ValueError, match="finite"):
        nm.run(problem, lambda p: nm.ArrayResult([np.nan, np.inf]))
    failed = nm.ArrayResult([1, 2], nm.Diagnostics(converged=False, message="budget exhausted"))
    with pytest.raises(nm.ConvergenceError, match="budget exhausted") as caught:
        nm.run(problem, lambda p: failed)
    assert caught.value.result is failed
    assert nm.run(problem, lambda p: failed, check_convergence=False) is failed


def test_input_and_output_snapshots_are_not_aliases():
    a, b = np.eye(2), np.array([1., 2.])
    problem = nm.LinearSystem(a, b)
    result = nm.ArrayResult(b)
    a[:] = 0
    b[:] = 0
    assert isinstance(problem.a, np.ndarray)
    assert_allclose(problem.a, np.eye(2))
    assert_allclose(np.asarray(problem.b), [1, 2])
    assert_allclose(result.value, [1, 2])
    with pytest.raises(ValueError):
        result.value[0] = 4


@pytest.mark.parametrize("factory", [
    lambda: nm.ArrayResult(["x"]), lambda: nm.Diagnostics(evaluations=-1),
    lambda: nm.Diagnostics(iterations=cast(Any, 1.5)), lambda: nm.Diagnostics(residual_norm=-1),
    lambda: nm.Diagnostics(error_estimate=-1), lambda: nm.Diagnostics(converged=cast(Any, "yes")),
    lambda: nm.ArrayResult([1, 2], nm.Diagnostics(error_estimate=[0.])),
])
def test_result_diagnostics_are_checked(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()
