"""Moment identities, backend parity, factor recovery and matrix constraints."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import dependence as dep


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("dimension", [1, 4])
@pytest.mark.parametrize("assume_centered", [False, True])
def test_sample_moments(ddof, dimension, assume_centered):
    x = np.random.default_rng(7).normal(size=(40, dimension)) + 3
    result = dep.sample_covariance(x, ddof=ddof, assume_centered=assume_centered)
    location = np.zeros(dimension) if assume_centered else x.mean(axis=0)
    centered = x - location
    assert_allclose(result.location, location)
    assert_allclose(result.covariance, centered.T @ centered / (len(x) - ddof))
    assert result.nobs == result.effective_nobs == 40
    assert_allclose(result.precision @ result.covariance, np.eye(dimension), atol=2e-14)
    assert_allclose(
        result.mahalanobis(x),
        np.einsum("ni,ij,nj->n", centered, result.precision, centered),
    )
    weights = np.arange(dimension * 6.0).reshape(2, 3, dimension)
    assert_allclose(
        result.variance(weights),
        np.einsum("...i,ij,...j->...", weights, result.covariance, weights),
    )


@pytest.mark.parametrize("decay", [0.7, 0.94, 1.0])
@pytest.mark.parametrize("ddof", [0, 1])
def test_ewma_weighted_covariance(decay, ddof):
    x = np.random.default_rng(8).normal(size=(50, 3)) + [2.0, 3.0, -1.0]
    result = dep.ewma_covariance(x, decay=decay, ddof=ddof)
    weights = decay ** np.arange(49, -1, -1.0)
    weights /= weights.sum()
    assert_allclose(result.location, weights @ x)
    assert_allclose(
        result.covariance, np.cov(x, rowvar=False, aweights=weights, ddof=ddof)
    )
    assert result.effective_nobs is not None
    assert_allclose(result.effective_nobs, 1 / np.sum(weights**2))
    centered = dep.ewma_covariance(x, decay=decay, ddof=ddof, assume_centered=True)
    assert_allclose(
        centered.covariance, (x.T * weights) @ x / (1 - ddof * np.sum(weights**2))
    )
    if decay == 1:
        assert_allclose(
            result.covariance, dep.sample_covariance(x, ddof=ddof).covariance
        )


def test_ewma_orientation_short_history_and_degenerate_weights():
    result = dep.ewma_covariance([[1.0], [8.0]], decay=0.25)
    assert_allclose(result.location, [6.6])
    assert_allclose(dep.ewma_covariance([[2.0]], ddof=0).covariance, [[0.0]])
    with pytest.raises(ValueError, match="effective observation"):
        dep.ewma_covariance([[1.0], [2.0]], decay=1e-25, ddof=1)


@pytest.mark.parametrize("name", ["LedoitWolf", "OAS", "ShrunkCovariance"])
@pytest.mark.parametrize("centered", [False, True])
def test_shrinkage_delegates_to_sklearn(name, centered):
    from sklearn import covariance

    x = np.random.default_rng(9).normal(size=(15, 20)) + 2
    result = getattr(dep, name)(assume_centered=centered).fit(x)
    reference = getattr(covariance, name)(assume_centered=centered).fit(x)
    assert_allclose(result.covariance, reference.covariance_)
    assert_allclose(result.location, reference.location_)
    alpha = reference.shrinkage if name == "ShrunkCovariance" else reference.shrinkage_
    assert_allclose(result.shrinkage, alpha)
    empirical = dep.sample_covariance(x, ddof=0, assume_centered=centered).covariance
    assert_allclose(
        result.covariance,
        (1 - alpha) * empirical + alpha * np.trace(empirical) / 20 * np.eye(20),
    )
    assert np.linalg.eigvalsh(result.covariance)[0] > 0


def test_singular_covariance_and_constant_columns_are_explicit():
    fitted = dep.sample_covariance([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    assert fitted.rank == 1
    assert_allclose(fitted.correlation, np.ones((2, 2)))
    with pytest.raises(ValueError, match="positive definite"):
        _ = fitted.precision
    constant = dep.sample_covariance([[1.0, 3.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="zero-variance"):
        _ = constant.correlation
    zero = dep.LedoitWolf().fit(np.zeros((4, 2)))
    assert_allclose(zero.covariance, 0)


def test_factor_covariance_and_observed_factor_recovery():
    rng = np.random.default_rng(10)
    b = np.array([[1.0, 2.0], [2.0, -0.3], [0.4, 1.0]])
    f = np.array([[2.0, 0.3], [0.3, 1.0]])
    noise = np.array([0.1, 0.2, 0.3])
    model = dep.FactorCovariance(b, f, noise, location=[2, -1, 3])
    assert_allclose(model.covariance, b @ f @ b.T + np.diag(noise))
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = dep.FactorCovariance(b @ rotation, rotation.T @ f @ rotation, noise)
    assert_allclose(rotated.covariance, model.covariance)
    factors = rng.multivariate_normal([1.0, -2.0], f, size=12000)
    x = (
        factors @ b.T
        + rng.normal(size=(len(factors), 3)) * np.sqrt(noise)
        + model.location
    )
    fitted = dep.fit_factor_covariance(x, factors)
    assert_allclose(fitted.loadings, b, atol=0.015)
    assert_allclose(fitted.covariance, model.covariance, rtol=0.04, atol=0.025)
    assert_allclose(np.diag(fitted.residual_covariance), noise, rtol=0.025)
    full = dep.fit_factor_covariance(x, factors, residual="full", ddof=0)
    assert_allclose(
        full.covariance, dep.sample_covariance(x, ddof=0).covariance, atol=1e-13
    )


def test_factor_composition_with_completed_transforms():
    from convexity.math.transforms import GaussianFactorModel

    factors = GaussianFactorModel(
        [1.0, 2.0, 3.0], [[1.0], [2.0], [-1.0]], [0.2, 0.3, 0.1]
    )
    result = dep.FactorCovariance(
        factors.loadings, [[1.0]], factors.noise_variance, location=factors.location
    )
    assert_allclose(result.covariance, factors.covariance)


def test_correlations_and_covariance_conversion():
    from scipy import stats

    x = np.random.default_rng(11).normal(size=(100, 3))
    x[:, 1] = np.round(x[:, 1], 1)  # tied ranks
    for method in ("pearson", "spearman", "kendall"):
        matrix = dep.correlation(x, method=method)
        for i in range(3):
            for j in range(3):
                reference = {
                    "pearson": stats.pearsonr,
                    "spearman": stats.spearmanr,
                    "kendall": stats.kendalltau,
                }[method](x[:, i], x[:, j]).statistic
                assert_allclose(matrix[i, j], reference, atol=1e-14)
    cov = dep.sample_covariance(x)
    assert_allclose(
        dep.correlation_to_covariance(cov.correlation, cov.standard_deviation),
        cov.covariance,
    )
    assert_allclose(dep.correlation(x[:, :1]), [[1.0]])


def test_psd_projection_is_spectral_frobenius_projection():
    a = np.array([[1.0, 2.0], [2.0, 1.0]])
    constraint = dep.PositiveSemidefinite(min_eigenvalue=0.2)
    assert not constraint.contains(a)
    result = constraint.project(a)
    assert constraint.contains(result.matrix)
    assert_allclose(np.linalg.eigvalsh(result.matrix), [0.2, 3.0])
    assert_allclose(result.distance, 1.2)
    assert_allclose(constraint.project(result.matrix).matrix, result.matrix)
    assert_allclose(result.min_eigenvalue_before, -1)
    tiny = np.diag([1e-20, -1e-20])
    assert not dep.PositiveSemidefinite().contains(tiny)
    with pytest.raises(ValueError):
        dep.CovarianceEstimate([0, 0], tiny)


def test_nearest_correlation_preserves_diagonal_and_reports_failure():
    a = np.array([[1.0, 1.2], [1.2, 1.0]])
    result = dep.nearest_correlation(a, min_eigenvalue=1e-6)
    assert_allclose(np.diag(result.matrix), 1)
    assert_allclose(result.matrix[0, 1], 1 - 1e-6, atol=1e-12)
    assert result.diagnostics.converged
    assert_allclose(dep.nearest_correlation(np.eye(3)).matrix, np.eye(3))
    with pytest.raises(dep.ConvergenceError) as caught:
        dep.nearest_correlation(a, maxiter=1)
    assert not caught.value.result.diagnostics.converged
    failed = dep.nearest_correlation(a, maxiter=1, check_convergence=False)
    assert not failed.diagnostics.converged


@pytest.mark.parametrize(
    "bad", [np.eye(2) * -1, [[1, 0.2], [0.3, 1]], [[1, 2]], [[0, 1], [1, 0]]]
)
def test_invalid_covariance_matrices(bad):
    with pytest.raises(ValueError):
        dep.CovarianceEstimate([0, 0], bad)
