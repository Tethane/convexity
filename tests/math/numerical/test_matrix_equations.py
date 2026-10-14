import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import numerical as nm


@pytest.mark.parametrize("time", ["continuous", "discrete"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_lyapunov_recovers_constructed_solution(time, complex_values):
    a = (
        np.array([[-2.0, 0.3], [0.0, -1.0]])
        if time == "continuous"
        else np.array([[0.2, 0.1], [0.0, 0.4]])
    )
    expected = np.array([[2.0, 0.2], [0.2, 1.0]])
    if complex_values:
        a = a + np.diag([0.1j, -0.2j])
        expected = expected + np.array([[0, 0.1j], [-0.1j, 0]])
    q = (
        -(a @ expected + expected @ a.conj().T)
        if time == "continuous"
        else expected - a @ expected @ a.conj().T
    )
    result = nm.solve_lyapunov(a, q, time=time)
    assert_allclose(result.value, expected, atol=1e-13)
    assert result.diagnostics.residual_norm is not None
    assert result.diagnostics.residual_norm < 1e-12


def test_lyapunov_sign_convention_and_scalar_solutions():
    assert_allclose(nm.solve_lyapunov([[-2]], [[8]]).value, [[2]])
    assert_allclose(nm.solve_lyapunov([[0.5]], [[3]], time="discrete").value, [[4]])


@pytest.mark.parametrize("time", ["continuous", "discrete"])
def test_riccati_analytic_scalar_solution(time):
    result = nm.solve_riccati([[1]], [[1]], [[1]], [[1]], time=time)
    expected = 1 + np.sqrt(2) if time == "continuous" else (1 + np.sqrt(5)) / 2
    assert_allclose(result.value, [[expected]], atol=1e-13)
    assert result.diagnostics.residual_norm is not None
    assert result.diagnostics.residual_norm < 1e-12


@pytest.mark.parametrize("time", ["continuous", "discrete"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_riccati_residual_and_stabilizing_solution(time, complex_values):
    a = (
        np.array([[0.0, 1.0], [-1.0, -0.2]])
        if time == "continuous"
        else np.array([[1.0, 0.1], [0.0, 1.0]])
    )
    if complex_values:
        a = a + np.diag([0.1j, -0.1j])
    b, q, r = np.array([[0.0], [1.0]]), np.eye(2), np.array([[1.0]])
    x = nm.solve_riccati(a, b, q, r, time=time).value
    assert_allclose(x, x.conj().T, atol=1e-13)
    assert np.linalg.eigvalsh(x).min() > 0
    if time == "continuous":
        gain = np.linalg.solve(r, b.conj().T @ x)
        residual = a.conj().T @ x + x @ a - x @ b @ gain + q
        assert np.max(np.real(np.linalg.eigvals(a - b @ gain))) < 0
    else:
        gain = np.linalg.solve(r + b.conj().T @ x @ b, b.conj().T @ x @ a)
        residual = a.conj().T @ x @ a - x - a.conj().T @ x @ b @ gain + q
        assert np.max(np.abs(np.linalg.eigvals(a - b @ gain))) < 1
    assert_allclose(residual, np.zeros((2, 2)), atol=1e-12)


def test_matrix_equation_dimensions_and_unsolvable_riccati():
    with pytest.raises(ValueError, match="identical"):
        nm.solve_lyapunov(np.eye(2), np.eye(3))
    with pytest.raises(ValueError, match="Hermitian"):
        nm.solve_lyapunov(np.eye(2), [[1, 1], [0, 1]])
    with pytest.raises(ValueError, match="expected"):
        nm.solve_riccati(np.eye(2), np.ones((3, 1)), np.eye(2), [[1]])
    with pytest.raises(np.linalg.LinAlgError):
        nm.solve_riccati([[1]], [[0]], [[1]], [[1]])
