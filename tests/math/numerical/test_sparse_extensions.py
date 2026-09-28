import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from convexity.math import numerical as n


@pytest.mark.parametrize("method", ["cg", "gmres", "bicgstab"])
def test_matrix_free_preconditioned_solve(method):
    diagonal = np.arange(1., 11.)
    calls = []
    def matvec(x):
        calls.append(x.shape)
        return diagonal*x
    a = n.linear_operator((10, 10), matvec, rmatvec=matvec)
    preconditioner = n.linear_operator((10, 10), lambda x:x/diagonal)
    result = n.solve(a, diagonal, tool=n.SparseSolve(method=method, preconditioner=preconditioner))
    assert_allclose(result.value, np.ones(10), atol=1e-9)
    assert calls and all(shape == (10,) for shape in calls)


def test_default_matrix_free_solver_and_bad_callback_shapes():
    a = n.linear_operator((2, 2), lambda x:2*x)
    assert_allclose(n.solve(a, [2., 4.]).value, [1, 2])
    with pytest.raises(ValueError, match="explicit"):
        n.solve(a, [1, 2], tool=n.SparseSolve())
    with pytest.raises(ValueError, match="shape"):
        n.solve(n.linear_operator((2, 2), lambda x:np.ones(3)), [1, 2])


@pytest.mark.parametrize("hermitian", [False, True])
def test_sparse_eigenpairs_without_dense_conversion(hermitian, monkeypatch):
    a = sparse.diags_array(np.arange(1., 11.), format="csr")
    def forbid(*args, **kwargs):
        raise AssertionError("unexpected dense conversion")
    monkeypatch.setattr(sparse.csr_array, "toarray", forbid)
    result = n.sparse_eig(a, 2, hermitian=hermitian)
    assert_allclose(a @ result.vectors, result.vectors*result.values, atol=1e-9)
    assert_allclose(np.sort(np.real(result.values)), [9, 10], atol=1e-9)


def test_sparse_svd_and_adjoint_on_rectangular_operator():
    a = np.arange(1., 25.).reshape(6, 4)
    op = n.linear_operator(a.shape, lambda x:a @ x, rmatvec=lambda x:a.T @ x)
    result = n.sparse_svd(op, 2)
    assert_allclose(a @ result.vh.T, result.u*result.s, atol=1e-9)
    assert_allclose(result.s, np.linalg.svd(a, compute_uv=False)[:2], atol=1e-9)


def test_sparse_spectral_invalid_requests():
    a = n.sparse_matrix(np.eye(3))
    with pytest.raises(ValueError, match="k"):
        n.sparse_eig(a, 3)
    with pytest.raises(ValueError, match="k"):
        n.sparse_eig(a, 2, hermitian=False)
    with pytest.raises(ValueError, match="Hermitian"):
        n.sparse_eig(n.sparse_matrix([[1, 1, 0], [0, 2, 0], [0, 0, 3]]))


def test_real_operator_supports_complex_vectors_and_complex_adjoint():
    a = np.array([[0., -1.], [1., 0.]])
    real_operator = n.linear_operator(a.shape, lambda x:a @ x, rmatvec=lambda x:a.T @ x)
    vector = np.array([1+2j, 3-1j])
    assert_allclose(real_operator @ vector, a @ vector)
    complex_a = a + np.diag([1j, -1j])
    complex_operator = n.linear_operator(a.shape, lambda x:complex_a @ x,
                                         rmatvec=lambda x:complex_a.conj().T @ x, dtype=np.complex128)
    assert_allclose(complex_operator.H @ vector, complex_a.conj().T @ vector)
