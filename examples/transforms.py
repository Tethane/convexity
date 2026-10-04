"""Family 4 problem book: generated matrices, sequences and exact identities."""

import numpy as np
from numpy.testing import assert_allclose

from convexity.math import transforms as tx


def main():
    rng = np.random.default_rng(42)
    loadings = np.array([[2., 0.], [1., .2], [0., 2.], [.2, 1.5], [1., 1.]])
    x = rng.normal(size=(1800, 2)) @ loadings.T + .35 * rng.normal(size=(1800, 5))
    training, held_out = x[:1500], x[1500:]

    coordinates = tx.Pipeline((tx.Standardize(), tx.PCA(2))).fit(training)
    scores = coordinates.transform(held_out)
    assert_allclose(coordinates.reconstruct(scores) + coordinates.residuals(held_out), held_out)
    assert isinstance(coordinates.steps[-1], tx.PCAResult)
    print("Retained variance ratios:", coordinates.steps[-1].explained_variance_ratio)
    print("Held-out reconstruction RMSE:", np.sqrt(np.mean(coordinates.residuals(held_out)**2)))

    full = tx.Pipeline((tx.RobustScale(), tx.PCA())).fit(training)
    assert_allclose(full.inverse_transform(full.transform(held_out)), held_out, atol=1e-12)
    whitened = tx.whitening(training, method="zca")
    assert_allclose(np.cov(whitened.transform(training).T), np.eye(5), atol=1e-12)
    assert_allclose(whitened.inverse_transform(whitened.transform(held_out)), held_out, atol=1e-12)
    print("Full-coordinate inversion and ZCA identity covariance verified.")

    t = np.arange(90.)
    y = .02 * t**2 + np.sin(2*np.pi*t/12)
    delta = tx.difference(y[:60], lag=12, order=2)
    assert_allclose(delta.inverse_transform(), y[:60], atol=1e-12)
    future = tx.difference(y, lag=12, order=2).values[60-delta.row_offset:]
    assert_allclose(delta.inverse_transform(future, continuation=True), y[60:], atol=1e-12)
    lags = tx.lag_features(y, (0, 1, 12))
    assert_allclose(lags.values[:, 0], y[lags.rows])
    periodic = tx.fourier_features(30, 12, order=3, start=60)
    assert_allclose(periodic.values[:, 0], np.sin(2*np.pi*t[60:]/12), atol=1e-14)
    print("Seasonal differencing, continuation and lag alignment verified.")

    factors = tx.factor_analysis(training, 2, rotation="varimax")
    assert factors.diagnostics.converged
    assert_allclose(factors.reconstruct(factors.transform(held_out)) + factors.residuals(held_out), held_out)
    print("Fitted factor noise variances:", factors.noise_variance)

    source = tx.subspace(loadings)
    target = tx.Subspace(source.basis[:, ::-1])
    aligned = tx.align_subspaces(source, target)
    assert_allclose(aligned.aligned.basis, target.basis, atol=1e-12)
    z = source.coordinates(held_out)
    assert_allclose(aligned.aligned.reconstruct(aligned.transform(z)), source.project(held_out), atol=1e-12)
    assert_allclose(source.residuals(held_out) @ source.basis, 0., atol=1e-12)
    print("Subspace alignment error:", aligned.error)


if __name__ == "__main__":
    main()
