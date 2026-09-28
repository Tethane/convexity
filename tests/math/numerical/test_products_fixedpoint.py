import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import numerical as n


@pytest.mark.parametrize("autodiff", [False, True])
def test_derivative_products_rectangular_and_adjoint_identity(autodiff):
    if autodiff:
        torch = pytest.importorskip("torch")
        function = lambda x:torch.stack([x[0]*x[1], x[0]**2, x[1]**3])
        tool = n.AutoDiffProduct()
    else:
        function = lambda x:np.array([x[0]*x[1], x[0]**2, x[1]**3])
        tool = n.FiniteDifferenceProduct()
    jac = np.array([[3., 2.], [4., 0.], [0., 27.]])
    v, w = np.array([.5, 2.]), np.array([2., 3., 4.])
    jv = n.jvp(function, [2., 3.], v, tool=tool).value
    vj = n.vjp(function, [2., 3.], w, tool=tool).value
    assert_allclose(jv, jac @ v, atol=1e-7)
    assert_allclose(vj, w @ jac, atol=1e-7)
    assert_allclose(w @ jv, vj @ v, atol=1e-7)


def test_derivative_check_and_invalid_products():
    f = lambda x:x @ x
    assert n.check_derivative(f, [1., 2.], [2., 4.], kind="gradient").passed
    result = n.check_derivative(f, [1., 2.], [2., 5.], kind="gradient")
    assert not result.passed
    assert result.scaled_error[1] > 1
    with pytest.raises(ValueError):
        n.jvp(lambda x:x, [1., 2.], [1])
    with pytest.raises(ValueError, match="output shape"):
        n.vjp(lambda x:x, [1., 2.], [1])
    with pytest.raises(n.ConvergenceError):
        n.check_derivative(np.sin, .3, 1., kind="derivative", reference_tool=n.FiniteDifference(maxiter=1))


@pytest.mark.parametrize("method", ["iteration", "del2"])
def test_scalar_vector_fixed_points(method):
    tool = n.FixedPointSolver(method=method)
    scalar = n.fixed_point(np.cos, 1., tool=tool)
    assert_allclose(scalar.value, .7390851332151607, atol=1e-9)
    a = np.diag([.2, .4])
    result = n.fixed_point(lambda x:a @ x + 1, [0., 0.], tool=tool)
    assert_allclose(result.value, [1.25, 1/.6], atol=1e-9)
    with pytest.raises(ValueError, match="shape"):
        n.fixed_point(lambda x:np.ones(3), [0., 0.], tool=tool)


@pytest.mark.parametrize("method", ["broyden1", "broyden2", "anderson", "krylov"])
def test_nonlinear_system_methods(method):
    result = n.root(lambda x:np.array([x[0]**2-2, x[1]-3]), [1., 1.],
                    tool=n.SystemRootSolver(method=method, maxiter=200))
    # Unbracketed methods may converge to either valid square root.
    assert_allclose([result.value[0]**2, result.value[1]], [2., 3.], atol=1e-7)


def test_fixed_point_and_quasi_newton_failures():
    with pytest.raises(RuntimeError):
        n.fixed_point(lambda x:x+1, 0., tool=n.FixedPointSolver(method="iteration", maxiter=2))
    with pytest.raises(ValueError, match="Jacobians"):
        n.root(lambda x:x, [1.], jacobian=lambda x:[[1.]], tool=n.SystemRootSolver(method="broyden1"))
    with pytest.raises(n.ConvergenceError):
        n.root(lambda x:x*x-2, [10.], tool=n.SystemRootSolver(method="broyden1", maxiter=1))
