import numpy as np
import pytest
from typing import Any, cast
from numpy.testing import assert_allclose
from scipy import sparse

from convexity.math import numerical as nm


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("multiple_rhs", [False, True])
def test_dense_solve_recovers_known_solution(complex_values, multiple_rhs):
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    if complex_values:
        a = a + 1j * np.array([[0.0, 1.0], [-1.0, 0.0]])
    x = np.array([[1.0, 2.0], [-2.0, 3.0]]) if multiple_rhs else np.array([1.0, -2.0])
    result = nm.solve(a, a @ x)
    assert_allclose(result.value, x, atol=1e-14)
    assert result.diagnostics.residual_norm is not None
    assert result.diagnostics.residual_norm < 1e-13


@pytest.mark.parametrize(
    "a", [np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([[2, 1j], [-1j, 3]])]
)
@pytest.mark.parametrize("hermitian", [False, True])
def test_eigenpairs(a, hermitian):
    if hermitian and not np.allclose(a, a.conj().T):
        with pytest.raises(ValueError, match="Hermitian"):
            nm.eig(a, hermitian=True)
        return
    result = nm.eig(a, hermitian=hermitian)
    assert_allclose(a @ result.vectors, result.vectors * result.values, atol=1e-13)
    if hermitian:
        assert_allclose(result.vectors.conj().T @ result.vectors, np.eye(2), atol=1e-13)
        assert np.all(np.diff(result.values) >= 0)


@pytest.mark.parametrize("shape", [(4, 2), (2, 4), (3, 3)])
@pytest.mark.parametrize("full", [False, True])
def test_svd_reconstruction_and_orthogonality(shape, full):
    rng = np.random.default_rng(42)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    result = nm.svd(a, full_matrices=full)
    k = min(shape)
    assert_allclose((result.u[:, :k] * result.s) @ result.vh[:k], a, atol=1e-13)
    assert_allclose(result.u.conj().T @ result.u, np.eye(result.u.shape[1]), atol=1e-13)
    assert_allclose(
        result.vh @ result.vh.conj().T, np.eye(result.vh.shape[0]), atol=1e-13
    )
    assert np.all(np.diff(result.s) <= 0)


def test_matrix_exponential_rotation_and_inverse():
    theta = 0.7
    a = np.array([[0, -theta], [theta, 0]])
    expected = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    assert_allclose(nm.expm(a).value, expected, atol=1e-14)
    assert_allclose(nm.expm(a).value @ nm.expm(-a).value, np.eye(2), atol=1e-14)
    assert_allclose(nm.expm(np.diag([1j, -1j])).value, np.diag(np.exp([1j, -1j])))


@pytest.mark.parametrize("method", ["direct", "cg", "gmres"])
def test_sparse_solvers_do_not_densify(method, monkeypatch):
    a = sparse.diags_array(
        [-np.ones(19), 3 * np.ones(20), -np.ones(19)],
        offsets=cast(Any, [-1, 0, 1]),
        format="csr",
    )
    expected = np.arange(20, dtype=float)

    def forbid_dense(*args, **kwargs):
        raise AssertionError("sparse input was densified")

    monkeypatch.setattr(sparse.csr_array, "toarray", forbid_dense)
    result = nm.solve(a, a @ expected, tool=nm.SparseSolve(method=method))
    assert_allclose(result.value, expected, atol=1e-7)
    assert result.diagnostics.converged


@pytest.mark.parametrize("columns", [1, 3])
def test_sparse_direct_multiple_rhs_preserves_shape(columns):
    a = nm.sparse_matrix([[2, 0], [0, 4]], format="csc")
    b = np.ones((2, columns))
    result = nm.solve(a, b)
    assert result.value.shape == b.shape
    assert_allclose(a @ result.value, b)


def test_sparse_coordinate_construction_duplicates_and_arithmetic():
    a = nm.sparse_matrix(([1, 2, 4], ([0, 0, 1], [0, 0, 1])), shape=(2, 2))
    assert sparse.issparse(a)
    assert_allclose(a @ [1, 2], [3, 8])
    assert_allclose((a + a).toarray(), [[6, 0], [0, 8]])


def test_sparse_nonconvergence_is_visible():
    a = sparse.diags_array(np.arange(1, 101, dtype=float), format="csr")
    tool = nm.SparseSolve(method="cg", maxiter=1, rtol=1e-14)
    with pytest.raises(nm.ConvergenceError) as caught:
        nm.solve(a, np.ones(100), tool=tool)
    assert isinstance(caught.value.result, nm.ArrayResult)
    assert caught.value.result.diagnostics.iterations == 1
    result = nm.solve(a, np.ones(100), tool=tool, check_convergence=False)
    assert not result.diagnostics.converged
    assert result.diagnostics.residual_norm is not None
    assert result.diagnostics.residual_norm > 1


@pytest.mark.parametrize(
    "a,b",
    [
        ([[1, 2, 3]], [1]),
        ([[1, 0], [0, 1]], [1]),
        ([[np.nan]], [1]),
        ([[1]], [np.inf]),
        ([], []),
    ],
)
def test_invalid_linear_systems(a, b):
    with pytest.raises(ValueError):
        nm.solve(a, b)


@pytest.mark.parametrize("make", [np.asarray, nm.sparse_matrix])
def test_singular_solve_raises(make):
    with pytest.raises(np.linalg.LinAlgError):
        nm.solve(make([[1.0, 1.0], [1.0, 1.0]]), [1.0, 2.0])


def test_sparse_method_requirements():
    with pytest.raises(ValueError, match="shape"):
        nm.solve(
            nm.sparse_matrix(np.eye(2)),
            np.ones((2, 1)),
            tool=nm.SparseSolve(method="cg"),
        )
    with pytest.raises(ValueError, match="Hermitian"):
        nm.solve(
            nm.sparse_matrix([[2.0, 1.0], [0.0, 2.0]]),
            [1, 1],
            tool=nm.SparseSolve(method="cg"),
        )
