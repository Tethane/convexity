"""Joint sampling, singular support and pointwise interval contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import gaussian_processes as gp


def test_joint_sampling_reproduces_covariance_and_uses_explicit_rng():
    mean = np.array([1.0, -2.0])
    covariance = np.array([[0.4, 0.3], [0.3, 0.7]])
    law = gp.GaussianPrediction(mean, covariance)
    draws = law.sample(70000, rng=np.random.default_rng(42))
    assert_allclose(draws.mean(axis=0), mean, atol=0.01)
    assert_allclose(np.cov(draws.T), covariance, atol=0.01)
    assert_allclose(
        law.sample((2, 3), rng=np.random.default_rng(8)),
        law.sample((2, 3), rng=np.random.default_rng(8)),
    )
    assert law.sample(rng=np.random.default_rng(0)).shape == (2,)
    assert law.sample(0, rng=np.random.default_rng(0)).shape == (0, 2)
    assert law.sample((0, 3), rng=np.random.default_rng(0)).shape == (0, 3, 2)
    with pytest.raises(TypeError, match="Generator"):
        law.sample(2, rng=1)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.


def test_density_intervals_and_marginal_order():
    from scipy.stats import multivariate_normal, norm

    law = gp.GaussianPrediction([1, 2], [[4, 1], [1, 9]])
    x = np.array([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 0.0], [3.0, 4.0]]])
    assert_allclose(
        law.log_density(x),
        multivariate_normal(mean=law.mean, cov=law.covariance).logpdf(x),  # pyright: ignore[reportArgumentType] -- SciPy accepts covariance matrices.
    )
    assert_allclose(
        law.interval(0.9), law.mean[:, None] + law.std[:, None] * norm.ppf([0.05, 0.95])
    )
    reordered = law.marginal([1, 0])
    assert_allclose(reordered.mean, [2, 1])
    assert_allclose(reordered.covariance, [[9, 1], [1, 4]])
    single = law.marginal([1])
    assert single.log_density([2.0]).shape == ()
    assert single.log_density(np.zeros((2, 3, 1))).shape == (2, 3)
    assert law.log_density(np.empty((0, 2))).shape == (0,)


def test_singular_laws_and_exact_deterministic_predictions():
    law = gp.GaussianPrediction([1, 1], [[0.2, 0.2], [0.2, 0.2]])
    samples = law.sample(100, rng=np.random.default_rng(42))
    assert_allclose(samples[:, 0], samples[:, 1], atol=1e-14)
    assert np.isfinite(law.log_density([1, 1]))
    assert np.isneginf(law.log_density([1, 2]))
    point = gp.GaussianPrediction([1, 2], np.zeros((2, 2)))
    assert_allclose(point.sample(3, rng=np.random.default_rng(42)), [[1, 2]] * 3)
    assert_allclose(point.interval(), [[1, 1], [2, 2]])
    assert point.log_density([1, 2]) == 0
    assert np.isneginf(point.log_density([1, 3]))


def test_roundoff_psd_correction_and_material_indefiniteness():
    law = gp.GaussianPrediction([0, 0], [[1, 1 + 1e-15], [1 + 1e-15, 1]])
    assert np.linalg.eigvalsh(law.covariance).min() >= -1e-15
    with pytest.raises(ValueError, match="positive semidefinite"):
        gp.GaussianPrediction([0, 0], [[1, 1.01], [1.01, 1]])
    with pytest.raises(ValueError, match="symmetric"):
        gp.GaussianPrediction([0, 0], [[1, 0.1], [0.2, 1]])


@pytest.mark.parametrize("indices", [[], [True], [0.0, 1.0], [0, 0], [-1], [2]])
def test_invalid_marginal_indices(indices):
    with pytest.raises(ValueError):
        gp.GaussianPrediction([1, 2], np.eye(2)).marginal(indices)
