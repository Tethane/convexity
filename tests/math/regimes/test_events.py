"""Analytic intensity, compensator and compound-jump references."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from convexity.math import regimes as r, distributions as d


def test_poisson_laws_likelihood_fit_and_piecewise_gaps():
    process = r.PoissonProcess(r.ConstantIntensity(2))
    assert_allclose(process.count_distribution(1, 4).mean(), 6)
    assert_allclose(process.event_probability(1, 4), 1-np.exp(-6))
    assert_allclose(process.log_likelihood([2, 3], 1, 4), 2*np.log(2)-6)
    fitted = r.PoissonProcess.fit([2, 3], 1, 4)
    assert isinstance(fitted.rate, r.ConstantIntensity)
    assert_allclose(fitted.rate.rate, 2/3)
    empty = r.PoissonProcess.fit([], 0, 1)
    assert isinstance(empty.rate, r.ConstantIntensity)
    assert empty.rate.rate == 0
    assert r.PoissonProcess(r.ConstantIntensity(0)).count_distribution(1, 2).mean() == 0
    intensity = r.PiecewiseConstantIntensity([0, 1, 2], [0, 3, 0, 5])
    assert intensity.intensity(1) == 0
    assert_allclose(intensity.integrated(-1, 3), 8)
    process = r.PoissonProcess(intensity)
    rng = np.random.default_rng(6)
    counts, first = [], []
    for _ in range(2500):
        times = process.sample(-1, 3, rng=rng)
        assert np.all(((times > 0) & (times < 1)) | ((times > 2) & (times <= 3)))
        counts.append(len(times))
        first.append(np.sum(times < 1))
    assert abs(np.mean(counts)-8) < .2
    assert abs(np.var(counts)-8) < .5
    assert abs(np.mean(first)-3) < .12
    assert process.log_likelihood([1.5], -1, 3) == -np.inf
    assert len(process.sample(2, 2, rng=rng)) == 0


def test_custom_intensity_time_change():
    class LinearRate:
        def intensity(self, time):
            return 2*time
        def integrated(self, start, end):
            return end**2-start**2
    process = r.PoissonProcess(LinearRate())
    draws = process.sample(0, 100, rng=np.random.default_rng(7))
    assert abs(np.mean((draws/100)**2)-.5) < .015
    assert_allclose(process.log_likelihood([1, 2], 0, 3), np.log(2)+np.log(4)-9)


def test_hawkes_compensator_and_likelihood():
    process = r.ExponentialHawkes(.7, .6, 1.5)
    history, times = np.array([-1., 0.]), np.array([.2, .7, 1.1])
    full = np.r_[history, times]
    integral = quad(lambda t: process.intensity(t, full), 0, 2, points=times)[0]
    assert_allclose(process.integrated(0, 2, full), integral, atol=1e-11)
    expected = sum(np.log(process.intensity(t, full)) for t in times) - integral
    assert_allclose(process.log_likelihood(times, 0, 2, history=history), expected)
    assert_allclose(process.stationary_rate, .7/(1-.6/1.5))
    assert process.intensity(.2, [.2]) == process.baseline  # predictable left limit
    poisson = r.ExponentialHawkes(2, 0, 1)
    assert_allclose(poisson.log_likelihood([1, 2], 0, 3), 2*np.log(2)-6)
    rng = np.random.default_rng(8)
    counts = [len(poisson.sample(0, 3, rng=rng)) for _ in range(2000)]
    assert abs(np.mean(counts)-6) < .15
    # Empty-history expected count solves m'(t)=baseline + (alpha-beta)*m(t)+baseline*beta*t.
    a, b, mu, end = .6, 1.5, .7, 3
    expected_count = mu*b/(b-a)*end - mu*a/(b-a)**2*(1-np.exp(-(b-a)*end))
    counts = [len(process.sample(0, end, rng=rng)) for _ in range(4000)]
    assert abs(np.mean(counts)-expected_count) < .13
    assert len(r.ExponentialHawkes(0, 0, 1).sample(0, 10, rng=rng)) == 0
    assert r.ExponentialHawkes(0, .3, 1).log_likelihood([1], 0, 2) == -np.inf


def test_scheduled_surprise_conditional_impact_and_boundaries():
    event = r.ScheduledEvent(2, d.normal(1, 2), lambda surprise: d.normal(3*surprise, .5), 'release')
    surprises, impacts = event.sample(draws=15000, rng=np.random.default_rng(9))
    assert abs(impacts.mean()-3) < .15
    assert abs(impacts.var()-(36+.25)) < 1.2
    assert np.corrcoef(surprises, impacts)[0, 1] > .99
    schedule = r.EventSchedule([r.ScheduledEvent(2, d.empirical([3])), r.ScheduledEvent(1, d.empirical([2]))])
    rng = np.random.default_rng(10)
    assert_allclose(schedule.sample(0, 1, rng=rng).total, 2)
    assert_allclose(schedule.sample(1, 2, rng=rng).total, 3)
    assert_allclose(schedule.sample(0, 2, rng=rng).total, 5)
    assert_allclose(schedule.sample(2, 3, rng=rng).total, 0)
    vector = r.EventSchedule([r.ScheduledEvent(1, d.empirical([[1, -1]]))], (2,))
    assert_allclose(vector.sample(0, 1, rng=rng).total, [1, -1])
    assert_allclose(vector.sample(1, 2, rng=rng).total, [0, 0])


def test_compound_poisson_moments_and_empirical_atom():
    process = r.MarkedPointProcess(r.PoissonProcess(r.ConstantIntensity(1.3)), d.normal(2, .5))
    mean, variance = process.moments(0, 2)
    assert_allclose(mean, 2.6*2)
    assert_allclose(variance, 2.6*(.25+4))
    law = process.aggregate(0, 2, draws=5000, rng=np.random.default_rng(11))
    assert law.measure == 'discrete'
    assert abs(float(law.mean())-mean) < .17
    assert abs(float(law.variance())-variance) < .5
    assert abs(float(law.density(0))-np.exp(-2.6)) < .015
    vector = r.MarkedPointProcess(process.process, d.multivariate_normal([1, -2], [[1, .3], [.3, 2]]))
    mean, covariance = vector.moments(0, 2)
    assert_allclose(mean, 2.6*np.array([1, -2]))
    assert_allclose(covariance, 2.6*np.array([[2, -1.7], [-1.7, 6]]))
    zero_mean, zero_covariance = vector.moments(2, 2)
    assert_allclose(zero_mean, np.zeros(2))
    assert_allclose(zero_covariance, np.zeros((2, 2)))


def test_event_seeds_and_custom_process():
    class Clock:
        def sample(self, start, end, *, rng):
            return [end] if end > start else []
    marked = r.MarkedPointProcess(Clock(), d.normal())
    assert_array_equal(marked.sample(0, 1, rng=np.random.default_rng(1)).times, [1])
    for process in [r.ExponentialHawkes(.7, .5, 1), r.PoissonProcess(r.ConstantIntensity(2))]:
        assert_array_equal(process.sample(0, 5, rng=np.random.default_rng(3)),
                           process.sample(0, 5, rng=np.random.default_rng(3)))
