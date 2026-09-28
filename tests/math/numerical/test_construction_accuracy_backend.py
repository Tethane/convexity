import numpy as np
from typing import Any, cast
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from convexity.math import numerical as n


def test_grids_mesh_and_cartesian_order():
    assert_allclose(n.linspace(0, 1, 5), [0, .25, .5, .75, 1])
    assert_allclose(n.logspace(0, 2, 3), [1, 10, 100])
    assert_allclose(n.geomspace(1, 100, 3), [1, 10, 100])
    assert_allclose(n.arange(1, -1, -.5), [1, .5, 0, -.5])
    a, b = n.meshgrid([1, 2], [3, 4, 5])
    assert a.shape == b.shape == (2, 3)
    assert_allclose(n.cartesian_product([1, 2], [3, 4, 5]), np.stack([a.ravel(), b.ravel()], axis=-1))
    small = n.meshgrid([1, 2], [3, 4, 5], sparse=True)
    assert small[0].shape == (2, 1) and small[1].shape == (1, 3)
    assert_allclose(n.linspace(0, 1, 4, endpoint=False), [0, .25, .5, .75])


@pytest.mark.parametrize("order", [1, 2])
def test_nonuniform_derivative_matrices(order):
    x = np.linspace(0, 1, 20)**1.4
    matrix = n.differentiation_matrix(x, order=order, accuracy=4)
    assert sparse.issparse(matrix)
    expected = 3*x*x if order == 1 else 6*x
    assert_allclose(matrix @ x**3, expected, atol=1e-9)


def test_derivative_operator_refinement_and_periodicity():
    errors = []
    for size in (21, 41, 81):
        x = np.linspace(0, 2*np.pi, size, endpoint=False)
        matrix = n.differentiation_matrix(x, accuracy=2, periodic=True)
        errors.append(np.max(np.abs(matrix @ np.sin(x)-np.cos(x))))
    assert errors[1] < errors[0]/3
    assert errors[2] < errors[1]/3


def test_mixed_tensor_derivative_and_laplacian_primitives():
    x, y = np.linspace(0, 1, 7), np.linspace(-1, 1, 8)
    xx, yy = n.meshgrid(x, y)
    dxy = n.partial_derivative_matrix((x, y), (1, 1))
    assert_allclose((dxy @ (xx*yy).ravel()).reshape(xx.shape), np.ones(xx.shape), atol=1e-12)
    for boundary in ("neumann", "periodic"):
        lap = n.laplacian_matrix((5, 6), spacing=.1, boundary=boundary)
        assert_allclose(lap @ np.ones(30), 0, atol=1e-12)
        assert_allclose(lap.toarray(), lap.toarray().T)
    lap = n.laplacian_matrix((20,), spacing=1/21, boundary="dirichlet")
    x = np.arange(1, 21)/21
    assert_allclose(lap @ (x*(1-x)), -2, atol=1e-12)


@pytest.mark.parametrize("factory", [
    lambda:n.linspace(0, 1, 0), lambda:n.geomspace(-1, 1, 3), lambda:n.arange(0, 1, -1),
    lambda:n.meshgrid(), lambda:n.differentiation_matrix([0, 1]),
    lambda:n.differentiation_matrix([0, .2, .8, 1], periodic=True),
    lambda:n.laplacian_matrix((3,), boundary="unknown"),
])
def test_construction_invalid_inputs(factory):
    with pytest.raises(ValueError):
        factory()


def test_accuracy_tolerances_and_nonfinite_policy():
    assert_allclose(n.machine_precision(np.float32).epsilon, np.finfo(np.float32).eps)
    extended = n.machine_precision(np.longdouble)
    assert np.isfinite(extended.largest) and extended.smallest_subnormal > 0
    assert n.allclose([1., 2.], [1., 2.+1e-9])
    assert not n.allclose([1., 2.], [1., 2.01])
    assert n.allclose([np.inf], [np.inf])
    assert not n.allclose([np.nan], [np.nan])
    assert n.allclose([np.nan], [np.nan], equal_nan=True)
    assert not n.allclose([1.], [1.+1e-15], tolerances=n.Tolerances(0, 0))
    with pytest.raises(ValueError, match="finite"):
        n.require_finite([1., np.inf])
    with pytest.raises(ValueError, match="shapes"):
        n.allclose([1, 2], [[1, 2]])


def test_residual_dense_sparse_matrix_free_and_zero_rhs():
    a = np.array([[2., 1.], [0., 3.]])
    for matrix in (a, n.sparse_matrix(a), n.linear_operator((2, 2), lambda x:a @ x)):
        result = n.residual(matrix, [1., 2.], [4., 6.])
        assert_allclose(result.value, 0)
        assert result.absolute == result.relative == 0
    result = n.residual(a, [1., 2.], [0., 0.])
    assert result.relative == np.inf
    assert_allclose(result.absolute, np.sqrt(52))


def test_backend_routing_overrides_and_missing_capabilities():
    backend = n.scipy_backend()
    problem = n.LinearSystem([[2., 0.], [0., 4.]], [6., 8.])
    assert_allclose(backend.run("solve", problem).value, [3, 2])
    calls = []
    def custom(p):
        calls.append(p)
        return n.ArrayResult([3., 2.])
    replacement = backend.with_tools(solve=custom)
    assert_allclose(n.solve(problem.a, problem.b, tool=replacement.tool("solve")).value, [3, 2])
    assert len(calls) == 1 and backend.tool("solve") is not custom
    with pytest.raises(TypeError):
        cast(Any, backend.tools)["solve"] = custom
    with pytest.raises(NotImplementedError, match="does not support"):
        n.torch_backend().tool("solve")
    with pytest.raises(ValueError, match="shape"):
        n.solve(problem.a, problem.b, tool=backend.with_tools(solve=lambda p:n.ArrayResult([1])).tool("solve"))


def test_custom_ode_tool_uses_public_checked_schema():
    def exact_decay(problem):
        t = np.asarray(problem.t_eval)
        solution = n.DenseSolution(lambda times:np.exp(-times)[None, ...], (0., 1.), 1)
        return n.ODEResult(t, solution(t), solution)
    result = n.solve_ivp(lambda t,y:-y, [1.], (0, 1), t_eval=[0., .5, 1.], tool=exact_decay)
    assert_allclose(result.y[0], np.exp(-np.array([0., .5, 1.])))


def test_public_namespaces():
    assert_allclose(n.linalg.eigh(np.eye(2)).values, [1, 1])
    assert_allclose(n.roots.solve(lambda x:x-1, [0.]).value, [1])
    assert n.integrate.quad(lambda x:x, 0, 1).value == .5
    assert callable(n.interpolation.spline)
    assert callable(n.calculus.jvp)
    assert callable(n.operators.laplacian_matrix)
    assert callable(n.grids.linspace)
    assert_allclose(n.linalg.norm([3, 4]).value, 5.)
