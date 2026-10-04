import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import state_space as s
from .test_linear import affine_problem

pytest.importorskip('filterpy')


@pytest.mark.parametrize('estimator', [s.ExtendedKalmanFilter(), s.UnscentedKalmanFilter()])
def test_linear_reduction_including_process_noise_offsets_partial_missing_and_smoother(estimator):
    model, series = affine_problem()
    expected = s.KalmanFilter().smooth(model, series)
    result = estimator.smooth(model, series)
    for name in ('filtered', 'predicted', 'smoothed'):
        actual, reference = getattr(result, name), getattr(expected, name)
        assert_allclose(actual.mean, reference.mean, atol=2e-12)
        assert_allclose(actual.covariance, reference.covariance, atol=2e-12)
    assert_allclose(result.log_likelihoods, expected.log_likelihoods, atol=2e-12)
    assert expected.observation_covariance is not None and result.observation_covariance is not None
    assert_allclose(result.observation_covariance, expected.observation_covariance, atol=2e-12)


def test_unscented_squared_observation_exact_first_two_moments():
    mu, p, r = 1.2, .8, .3
    model = s.StateSpaceModel(s.GaussianState([mu], [[p]]),
        s.GaussianTransition(1, lambda x, step: x, [[.5]]),
        s.GaussianObservation(1, 1, lambda x, step: x**2, [[r]]))
    result = s.UnscentedKalmanFilter().filter(model, [3.])
    assert result.observation_mean is not None
    assert_allclose(result.observation_mean[0], [mu**2+p], atol=1e-12)
    variance = 4*mu**2*p + 2*p**2 + r
    assert result.observation_covariance is not None
    assert_allclose(result.observation_covariance[0], [[variance]], atol=1e-12)
    assert_allclose(result.filtered.mean[0], [mu+2*mu*p/variance*(3-mu**2-p)], atol=1e-12)


@pytest.mark.parametrize('estimator', [s.ExtendedKalmanFilter(), s.UnscentedKalmanFilter()])
def test_nonlinear_generated_recovery_and_smoothing(estimator):
    model = s.StateSpaceModel(s.GaussianState([.5], [[.1]]),
        s.GaussianTransition(1, lambda x, step: x-.2*np.sin(x)*step.dt,
                             lambda step: [[.03*step.dt]], lambda x, step: [[1-.2*np.cos(x[0])*step.dt]]),
        s.GaussianObservation(1, 1, lambda x, step: x+.05*x**3, [[.2]],
                              lambda x, step: [[1+.15*x[0]**2]]), time_mode='elapsed')
    times = np.cumsum(np.random.default_rng(4).uniform(.2, 1, 120))
    generated = s.simulate(model, times, rng=np.random.default_rng(12))
    y = generated.observations.values.copy()
    y[::9] = np.nan
    result = estimator.smooth(model, s.TimeSeries(y, times))
    assert result.smoothed is not None
    filtered_error = np.mean((result.filtered.mean - generated.states)**2)
    smoothed_error = np.mean((result.smoothed.mean - generated.states)**2)
    assert filtered_error < .1
    assert smoothed_error < filtered_error
    assert np.all(result.smoothed.covariance <= result.filtered.covariance+1e-12)
    assert_allclose(result.smoothed.mean[-1], result.filtered.mean[-1])
    assert_allclose(result.log_likelihoods[::9], 0)


def test_missing_jacobians_are_explicit_and_ukf_accepts_semidefinite_states():
    model = s.StateSpaceModel(s.GaussianState([0.], [[0.]]),
        s.GaussianTransition(1, lambda x, step: x, [[.2]]),
        s.GaussianObservation(1, 1, lambda x, step: x, [[.3]]))
    with pytest.raises(NotImplementedError, match='derivative'):
        s.ExtendedKalmanFilter().filter(model, [1., 2.])
    result = s.UnscentedKalmanFilter().smooth(model, [1., 2.])
    assert_allclose(result.filtered.mean[0], [0.])
    assert_allclose(result.filtered.mean[1], [.8])
    assert result.smoothed is not None
    assert_allclose(result.smoothed.covariance[0], [[0.]])
