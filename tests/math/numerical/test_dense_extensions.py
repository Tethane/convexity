import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import numerical as n


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (3, 3)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_least_squares_normal_equations_and_minimum_norm(shape, complex_input):
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=shape), rng.normal(size=(shape[0], 2))
    if complex_input:
        a = a + 1j * rng.normal(size=shape)
    result = n.lstsq(a, b)
    assert_allclose(a.conj().T @ (a @ result.value - b), 0, atol=1e-12)
    assert_allclose(result.value, np.linalg.pinv(a) @ b, atol=1e-12)
    assert result.rank == min(shape)
    assert result.singular_values.shape == (min(shape),)


def test_rank_deficient_least_squares_and_pseudoinverse():
    a = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    result = n.lstsq(a, [1.0, 2.0, 3.0])
    assert result.rank == 1
    assert_allclose(result.value, [0.2, 0.4])
    p = n.pinv(a).value
    assert_allclose(a @ p @ a, a)
    assert_allclose(p @ a @ p, p)
    assert n.matrix_rank(a).value == 1


@pytest.mark.parametrize("hermitian", [False, True])
def test_generalized_eigenpairs(hermitian):
    a = np.array([[2.0, 1j], [-1j, 4.0]])
    b = np.diag([1.0, 2.0])
    result = n.generalized_eig(a, b, hermitian=hermitian)
    assert_allclose(
        (a @ result.vectors) * result.beta,
        (b @ result.vectors) * result.alpha,
        atol=1e-13,
    )
    if hermitian:
        assert_allclose(
            result.vectors.conj().T @ b @ result.vectors, np.eye(2), atol=1e-13
        )


def test_singular_generalized_pencil_and_invalid_positive_definite_b():
    result = n.generalized_eig(np.diag([2.0, 3.0]), np.diag([1.0, 0.0]))
    assert np.count_nonzero(result.beta == 0) == 1
    assert np.count_nonzero(np.isinf(result.values)) == 1
    with pytest.raises(np.linalg.LinAlgError):
        n.generalized_eig(np.eye(2), np.diag([1.0, -1.0]), hermitian=True)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (3, 3)])
@pytest.mark.parametrize("pivoting", [False, True])
def test_qr_and_lu_reconstruction(shape, pivoting):
    rng = np.random.default_rng(5)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    result = n.qr(a, pivoting=pivoting)
    q, r = result.factors["q"], result.factors["r"]
    assert_allclose(q @ r, a[:, result.permutation], atol=1e-13)
    assert_allclose(q.conj().T @ q, np.eye(min(shape)), atol=1e-13)
    factors = n.lu(a).factors
    assert_allclose(factors["p"] @ factors["l"] @ factors["u"], a, atol=1e-13)


def test_cholesky_schur_and_matrix_functions():
    a = np.array([[3.0, 1j], [-1j, 2.0]])
    l = n.cholesky(a).factors["l"]
    assert_allclose(l @ l.conj().T, a, atol=1e-13)
    t, z = (n.schur(a).factors[key] for key in ("t", "z"))
    assert_allclose(z @ t @ z.conj().T, a, atol=1e-13)
    root = n.sqrtm(a).value
    assert_allclose(root @ root, a, atol=1e-12)
    assert_allclose(n.expm(n.logm(a).value).value, a, atol=1e-12)
    assert_allclose(n.sqrtm([[-1.0]]).value, [[1j]], atol=1e-13)
    with pytest.raises(np.linalg.LinAlgError):
        n.cholesky([[1, 2], [2, 1]])
    with pytest.raises(np.linalg.LinAlgError):
        n.logm([[0.0]])


def test_determinants_condition_and_norms():
    a = np.diag([-2.0, 3.0])
    assert_allclose(n.det(a).value, -6.0)
    result = n.slogdet(a)
    assert result.sign == -1
    assert_allclose(result.logabsdet, np.log(6))
    large = n.slogdet(np.diag([1e200, 1e200]))
    assert_allclose(large.logabsdet, 400 * np.log(10))
    singular = n.slogdet(np.zeros((2, 2)))
    assert singular.sign == 0 and singular.logabsdet == -np.inf
    assert n.condition_number(np.diag([1.0, 0.0])).value == np.inf
    assert_allclose(n.condition_number(a).value, 1.5)
    assert_allclose(n.norm(a).value, np.sqrt(13))
    assert_allclose(n.norm(a, order=1).value, 3)


def test_dense_tools_are_replaceable_and_checked():
    custom = lambda p: n.FactorizationResult(
        {"q": np.eye(2), "r": np.asarray(p.a)}, np.arange(2)
    )
    assert_allclose(
        n.qr(np.diag([2.0, 3.0]), tool=custom).factors["r"], np.diag([2.0, 3.0])
    )
    with pytest.raises(ValueError, match="factor"):
        n.qr(np.eye(2), tool=lambda p: n.FactorizationResult({"q": np.eye(2)}))
    with pytest.raises(ValueError):
        n.lstsq(np.ones((2, 3)), [1])
