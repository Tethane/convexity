import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import transforms as tx


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("center,scale", [(True, True), (True, False), (False, True), (False, False)])
def test_scaling_formula_constants_batches_and_held_out_rows(ddof, center, scale):
    x = np.array([[1., 7., 8.], [3., 7., 4.], [8., 7., 9.]])
    fit = tx.standardize(x, center=center, scale=scale, ddof=ddof)
    expected_scale = x.std(axis=0, ddof=ddof) if scale else np.ones(3)
    expected_scale[expected_scale == 0] = 1
    assert_allclose(fit.scale, expected_scale)
    assert_allclose(fit.location, x.mean(axis=0) if center else np.zeros(3))
    held_out = np.array([[[40., 9., -2.], [0., 4., 1.]]])
    z = fit.transform(held_out)
    assert_allclose(z, (held_out - fit.location) / expected_scale)
    assert_allclose(fit.inverse_transform(z), held_out, atol=1e-14)
    assert fit.is_invertible


def test_robust_and_minmax_scaling_reference_formulas():
    x = np.array([[0., 3.], [2., 3.], [4., 3.], [6., 3.], [100., 3.]])
    robust = tx.robust_scale(x)
    assert_allclose(robust.location, [4., 3.])
    assert_allclose(robust.scale, [4., 1.])
    assert_allclose(robust.inverse_transform(robust.transform(x)), x)
    bounded = tx.minmax_scale(x, lower=-2, upper=2)
    z = bounded.transform(x)
    assert_allclose(z[0], [-2, -2])
    assert_allclose(z[-1], [2, -2])
    assert bounded.transform([200., 3.])[0] > 2  # no clipping of future inputs
    assert_allclose(bounded.inverse_transform(bounded.transform(x)), x)


@pytest.mark.parametrize("ddof", [0, 1])
def test_pca_spectrum_reconstruction_optimal_error_and_orthogonal_residuals(ddof):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(90, 5)) @ np.diag([4., 3., 2., 1., .5]) + np.arange(5)
    result = tx.pca(x, 2, ddof=ddof)
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    scores = result.transform(x)
    residual = result.residuals(x)
    assert_allclose(result.explained_variance, singular[:2]**2 / (len(x) - ddof))
    assert_allclose(result.singular_values, singular[:2])
    assert_allclose(scores.T @ scores / (len(x) - ddof), np.diag(result.explained_variance), atol=1e-13)
    assert_allclose(result.reconstruct(scores) + residual, x)
    assert_allclose(residual @ result.components, 0, atol=1e-13)
    assert_allclose(np.sum(residual**2), np.sum(singular[2:]**2))
    assert_allclose(result.explained_variance_ratio, singular[:2]**2 / np.sum(singular**2))
    assert not result.is_invertible
    with pytest.raises(ValueError, match="exact inverse"):
        result.inverse_transform(scores)
    full = tx.pca(x)
    assert_allclose(full.inverse_transform(full.transform(x)), x, atol=1e-13)


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("method", ["pca", "zca"])
def test_whitening_covariance_inverse_and_zca_symmetry(method, ddof):
    x = np.random.default_rng(9).normal(size=(80, 4)) @ np.array(
        [[1., 2., 0., 0.], [0., 1., 1., 0.], [0., 0., 2., 1.], [1., 0., 0., 2.]])
    result = tx.whitening(x, method=method, ddof=ddof)
    z = result.transform(x)
    assert_allclose(z.mean(axis=0), 0, atol=1e-14)
    assert_allclose(z.T @ z / (len(x) - ddof), np.eye(4), atol=1e-12)
    assert_allclose(result.inverse_transform(z), x, atol=1e-12)
    if method == "zca":
        assert_allclose(result.encoder, result.encoder.T, atol=1e-14)


def test_rank_deficient_and_wide_pca_whitening_is_explicit():
    t = np.arange(12.)
    x = np.column_stack((t, 2*t, np.ones(12)))
    full = tx.pca(x)
    assert_allclose(full.inverse_transform(full.transform(x)), x, atol=1e-13)
    with pytest.raises(ValueError, match="zero-variance"):
        tx.whitening(x)
    reduced = tx.whitening(x, method="pca", n_components=1)
    assert_allclose(np.var(reduced.transform(x), ddof=1), 1)
    wide = np.random.default_rng(3).normal(size=(3, 8))
    assert tx.pca(wide).output_dimension == 3
    with pytest.raises(ValueError):
        tx.whitening(wide)
    with pytest.raises(ValueError, match="nonzero"):
        tx.pca(np.ones((5, 3)))


def test_pipeline_training_only_composition_and_lossy_reconstruction():
    x = np.random.default_rng(8).normal(size=(100, 4)) * [1, 2, 3, 4] + 3
    pipeline = tx.Pipeline((tx.Standardize(), tx.PCA(2))).fit(x[:70])
    scale, pca = pipeline.steps
    assert_allclose(scale.location, x[:70].mean(axis=0))
    assert_allclose(pipeline.transform(x[70:]), pca.transform(scale.transform(x[70:])))
    assert_allclose(pipeline.reconstruct(pipeline.transform(x)) + pipeline.residuals(x), x)
    with pytest.raises(ValueError):
        pipeline.inverse_transform(pipeline.transform(x))
    exact = tx.Pipeline((tx.RobustScale(), tx.PCA(whiten=True))).fit(x)
    assert_allclose(exact.inverse_transform(exact.transform(x.reshape(10, 10, 4))), x.reshape(10, 10, 4), atol=1e-13)


def test_gaussian_factor_posterior_matches_gaussian_conditioning():
    b = np.array([[1., .2], [2., -.3], [.5, 1.], [0., 2.]])
    noise = np.array([.3, .4, .5, .2])
    model = tx.GaussianFactorModel([1., 2., 3., 4.], b, noise)
    x = np.random.default_rng(2).normal(size=(2, 3, 4))
    covariance = b @ b.T + np.diag(noise)
    encoder = np.linalg.solve(covariance, b)
    assert_allclose(model.transform(x), (x - model.location) @ encoder)
    assert_allclose(model.posterior_covariance, np.eye(2) - b.T @ encoder)
    assert_allclose(model.covariance, covariance)
    assert_allclose(model.reconstruct(model.transform(x)) + model.residuals(x), x)


@pytest.mark.parametrize("rotation", [None, "varimax", "quartimax"])
def test_factor_analysis_reference_scores_and_covariance_recovery(rotation):
    from sklearn.decomposition import FactorAnalysis
    rng = np.random.default_rng(123)
    b = np.array([[2., 0.], [1., .1], [0., 2.], [.1, 1.5], [1., 1.]])
    x = rng.normal(size=(2500, 2)) @ b.T + rng.normal(size=(2500, 5)) * .4
    result = tx.factor_analysis(x, 2, rotation=rotation)
    reference = FactorAnalysis(2, rotation=rotation, svd_method="lapack", tol=1e-6).fit(x)
    assert result.diagnostics.converged
    assert_allclose(result.transform(x[:8]), reference.transform(x[:8]), atol=1e-12)
    assert_allclose(result.covariance, reference.get_covariance(), atol=1e-12)
    assert_allclose(result.covariance, b @ b.T + np.eye(5) * .16, atol=.16)


def test_factor_nonconvergence_is_retained_and_opt_out_is_explicit():
    x = np.random.default_rng(9).normal(size=(50, 4))
    with pytest.raises(tx.ConvergenceError) as failure:
        tx.factor_analysis(x, 2, maxiter=1)
    assert not failure.value.result.diagnostics.converged
    result = tx.factor_analysis(x, 2, maxiter=1, check_convergence=False)
    assert not result.diagnostics.converged
    assert result.log_likelihood.shape == (1,)
