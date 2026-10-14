"""Independent dense Gaussian conditioning references, not backend snapshots."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from convexity.math import state_space as s


def joint_reference(model, series):
    steps = model.steps(series)
    n, d = len(series), model.state_dimension
    means = np.zeros((n, d))
    covariance = np.zeros((n, d, n, d))
    means[0], covariance[0, :, 0, :] = model.initial.mean, model.initial.covariance
    for i in range(1, n):
        f, b = model.transition.coefficients(steps[i])
        means[i] = f @ means[i - 1] + b
        for j in range(i):
            covariance[i, :, j, :] = f @ covariance[i - 1, :, j, :]
            covariance[j, :, i, :] = covariance[i, :, j, :].T
        covariance[i, :, i, :] = f @ covariance[
            i - 1, :, i - 1, :
        ] @ f.T + model.transition.covariance(steps[i])
    cov = covariance.reshape(n * d, n * d)
    h = np.asarray(
        block_diag(*[model.observation.coefficients(step)[0] for step in steps])
    )
    offset = np.concatenate([model.observation.coefficients(step)[1] for step in steps])
    r = np.asarray(block_diag(*[model.observation.covariance(step) for step in steps]))
    return means.ravel(), cov, h @ means.ravel() + offset, h @ cov @ h.T + r, cov @ h.T


def affine_problem():
    model = s.linear_gaussian(
        s.GaussianState([0.3, -0.2], [[0.8, 0.1], [0.1, 0.5]]),
        lambda step: [[1.0, 0.2 * step.dt], [0.0, 0.9]],
        lambda step: [[1.0, 0.1 * step.index], [0.2, 1.0]],
        lambda step: np.diag([0.1, 0.05]) * step.dt,
        [[0.4, 0.16], [0.16, 0.3]],
        transition_offset=lambda step: [0.1 * step.dt, -0.03],
        observation_offset=lambda step: [0.2, -0.1 * step.index],
        time_mode="elapsed",
        observation_dimension=2,
    )
    series = s.TimeSeries(
        [[np.nan, np.nan], [1.0, -0.5], [np.nan, 0.4], [0.7, np.nan], [0.3, 0.6]],
        [0.0, 0.2, 1.0, 2.7, 3.0],
    )
    return model, series


def test_dense_conditioning_with_partial_missing_correlated_observations_and_irregular_times():
    model, series = affine_problem()
    result = s.KalmanFilter().smooth(model, series)
    assert result.smoothed is not None and result.predicted is not None
    mx, px, my, py, pxy = joint_reference(model, series)
    seen = np.flatnonzero(~np.isnan(series.values.ravel()))
    dy = series.values.ravel()[seen] - my[seen]
    gain = np.linalg.solve(py[np.ix_(seen, seen)], pxy[:, seen].T).T
    assert_allclose(result.smoothed.mean.ravel(), mx + gain @ dy, atol=2e-12)
    posterior_cov = px - gain @ pxy[:, seen].T
    for t in range(len(series)):
        ix = np.arange(2 * t, 2 * t + 2)
        assert_allclose(
            result.smoothed.covariance[t], posterior_cov[np.ix_(ix, ix)], atol=2e-12
        )
        prefix = seen[seen < 2 * (t + 1)]
        if len(prefix):
            gain_t = np.linalg.solve(
                py[np.ix_(prefix, prefix)], pxy[np.ix_(ix, prefix)].T
            ).T
            expected = mx[ix] + gain_t @ (series.values.ravel()[prefix] - my[prefix])
            cov = px[np.ix_(ix, ix)] - gain_t @ pxy[np.ix_(ix, prefix)].T
        else:
            expected, cov = mx[ix], px[np.ix_(ix, ix)]
        assert_allclose(result.filtered.mean[t], expected, atol=2e-12)
        assert_allclose(result.filtered.covariance[t], cov, atol=2e-12)
    expected_ll = multivariate_normal.logpdf(
        series.values.ravel()[seen],
        my[seen],
        py[np.ix_(seen, seen)],  # pyright: ignore[reportArgumentType] -- SciPy accepts covariance matrices.
    )
    assert_allclose(result.log_likelihood, expected_ll, atol=2e-12)
    assert result.log_likelihoods[0] == 0
    assert_allclose(result.smoothed.mean[-1], result.filtered.mean[-1])
    assert_allclose(result.smoothed.covariance[-1], result.filtered.covariance[-1])
    assert np.isnan(result.innovations[0]).all()


def test_initial_timing_and_all_missing_predictions():
    model = s.local_level(100.0, 3.0, initial=s.GaussianState([2.0], [[4.0]]))
    result = s.KalmanFilter().smooth(model, [5.0])
    assert result.smoothed is not None and result.predicted is not None
    assert_allclose(result.filtered.mean, [[2 + 4 / 7 * 3]])
    assert_allclose(result.filtered.covariance, [[[12 / 7]]])
    assert_allclose(result.predicted.covariance, [[[4.0]]])
    missing = s.KalmanFilter().smooth(model, [np.nan] * 4)
    assert missing.smoothed is not None
    assert_allclose(missing.filtered.mean[:, 0], 2)
    assert_allclose(missing.filtered.covariance[:, 0, 0], [4, 104, 204, 304])
    assert_allclose(missing.smoothed.covariance, missing.filtered.covariance)
    assert missing.log_likelihood == 0


def test_prefix_causality_and_distribution_composition():
    model, series = affine_problem()
    full = s.filter_states(model, series, s.KalmanFilter())
    assert full.smoothed is None
    for stop in range(1, len(series) + 1):
        prefix = s.KalmanFilter().filter(model, series.slice(0, stop))
        assert_allclose(prefix.filtered.mean, full.filtered.mean[:stop])
    law = full.filtered.distribution(2)
    assert_allclose(law.mean(), full.filtered.mean[2])
    assert_allclose(law.covariance(), full.filtered.covariance[2])
    assert law.sample(10, rng=np.random.default_rng(4)).shape == (10, 2)
    assert_allclose(law.marginal([1]).mean(), full.filtered.mean[2, 1:])
    assert_array_equal(full.timestamps, series.timestamps)


def test_singular_state_covariance_with_noisy_measurements():
    model = s.linear_gaussian(
        s.GaussianState([0.0, 2.0], [[1.0, 0.0], [0.0, 0.0]]),
        np.eye(2),
        [[1.0, 0.0]],
        np.zeros((2, 2)),
        [[0.2]],
    )
    result = s.KalmanFilter().smooth(model, [1.0, 2.0, 3.0])
    assert_allclose(result.filtered.mean[:, 1], 2)
    assert_allclose(result.filtered.covariance[:, 1], 0)
    assert np.isfinite(
        result.filtered.distribution(0).log_density(result.filtered.mean[0])
    )
    deterministic = s.GaussianState([1.0, 2.0], np.zeros((2, 2))).distribution()
    assert deterministic.measure == "discrete"
    assert_array_equal(
        deterministic.sample(3, rng=np.random.default_rng(0)), [[1.0, 2.0]] * 3
    )
