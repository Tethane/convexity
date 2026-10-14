import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import state_space as s


def test_exact_scalar_ou_discretization_and_semigroup():
    rate, level, variance_rate = 0.7, 1.5, 0.4
    transition = s.continuous_linear_transition(
        [[-rate]], [[variance_rate]], forcing=[rate * level]
    )
    for dt in [0.0, 0.01, 0.7, 3.0, 3000.0]:
        step = s.TimeStep(1, dt, dt)
        f, b = transition.coefficients(step)
        expected_f = np.exp(-rate * dt)
        assert_allclose(f, [[expected_f]], atol=1e-14)
        assert_allclose(b, [level * (1 - expected_f)], atol=1e-14)
        assert_allclose(
            transition.covariance(step),
            [[variance_rate * (1 - expected_f**2) / (2 * rate)]],
            atol=1e-14,
        )
    a, b, ab = [s.TimeStep(1, dt, dt) for dt in (0.2, 0.9, 1.1)]
    fa, ca = transition.coefficients(a)
    fb, cb = transition.coefficients(b)
    fab, cab = transition.coefficients(ab)
    assert_allclose(fab, fb @ fa)
    assert_allclose(cab, fb @ ca + cb)
    assert_allclose(
        transition.covariance(ab),
        fb @ transition.covariance(a) @ fb.T + transition.covariance(b),
    )


def test_integrated_velocity_exact_covariance_and_missing_gap():
    model = s.local_linear_trend(0.4, 0.3)
    assert isinstance(model.transition, s.LinearGaussianTransition)
    dt = 2.3
    step = s.TimeStep(1, dt, dt)
    assert_allclose(model.transition.coefficients(step)[0], [[1.0, dt], [0.0, 1.0]])
    assert_allclose(
        model.transition.covariance(step),
        0.4 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]),
    )
    sparse = s.KalmanFilter().filter(model, s.TimeSeries([1.0, -0.5], [0.0, 4.0]))
    grid = s.KalmanFilter().filter(
        model, s.TimeSeries([1.0, np.nan, np.nan, np.nan, -0.5], np.arange(5.0))
    )
    assert_allclose(sparse.filtered.mean[-1], grid.filtered.mean[-1], atol=1e-12)
    assert_allclose(
        sparse.filtered.covariance[-1], grid.filtered.covariance[-1], atol=1e-12
    )
    assert_allclose(sparse.log_likelihood, grid.log_likelihood, atol=1e-12)


def test_datetime_units_equal_numeric_times_and_are_not_implicit():
    numeric = np.array([0, 1, 3, 7])
    dates = np.datetime64("2026-01-01") + numeric * np.timedelta64(1, "D")
    y = [1.0, np.nan, 0.2, -0.5]
    model = s.local_level(0.1, 0.3, time_mode="elapsed")
    expected = s.KalmanFilter().smooth(model, s.TimeSeries(y, numeric))
    with pytest.raises(ValueError, match="time_unit"):
        s.KalmanFilter().filter(model, s.TimeSeries(y, dates))
    dated = s.local_level(
        0.1, 0.3, time_mode="elapsed", time_unit=np.timedelta64(1, "D")
    )
    result = s.KalmanFilter().smooth(dated, s.TimeSeries(y, dates))
    assert_allclose(result.filtered.mean, expected.filtered.mean)
    assert_allclose(result.filtered.covariance, expected.filtered.covariance)
    assert_array_equal(result.timestamps, dates)
    with pytest.raises(ValueError, match="irregular"):
        s.KalmanFilter().filter(s.local_level(0.1, 0.3), s.TimeSeries(y, dates))
    with pytest.raises(ValueError, match="numeric"):
        s.KalmanFilter().filter(dated, s.TimeSeries(y, numeric))


def test_simulation_reproducibility_and_empirical_gaussian_moments():
    model = s.local_level(0.2, 0.3, time_mode="elapsed")
    first = s.simulate(model, [0.0, 0.1, 2.0], rng=np.random.default_rng(8))
    second = s.simulate(model, [0.0, 0.1, 2.0], rng=np.random.default_rng(8))
    assert_array_equal(first.states, second.states)
    assert_array_equal(first.observations.values, second.observations.values)
    n = 60000
    states = np.full((n, 1), 1.2)
    step = s.TimeStep(1, 2.0, 2.0)
    draws = model.transition.sample(states, step, rng=np.random.default_rng(3))
    assert_allclose(draws.mean(), 1.2, atol=0.01)
    assert_allclose(draws.var(), 0.4, atol=0.01)
    residual = (
        model.observation.sample(draws, step, rng=np.random.default_rng(4)) - draws
    )
    assert_allclose(residual.var(), 0.3, atol=0.01)


@pytest.mark.parametrize(
    "estimator", [s.KalmanFilter(), s.ExtendedKalmanFilter(), s.UnscentedKalmanFilter()]
)
def test_user_callbacks_receive_destination_time_and_duration(estimator):
    if not isinstance(estimator, s.KalmanFilter):
        pytest.importorskip("filterpy")
    called = []

    def matrix(step):
        called.append((step.index, step.time, step.dt))
        return [[1.0]]

    model = s.linear_gaussian(
        s.GaussianState([0.0], [[1.0]]),
        matrix,
        [[1.0]],
        [[0.1]],
        [[0.2]],
        time_mode="elapsed",
    )
    estimator.filter(model, s.TimeSeries([1.0, 2.0, 3.0], [10.0, 10.5, 12.0]))
    assert set(called) == {(1, 10.5, 0.5), (2, 12.0, 1.5)}
