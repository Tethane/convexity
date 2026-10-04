import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import distributions as dist, forecasting as f, state_space as s


def test_existing_contract_identity_and_user_estimator():
    assert s.TimeSeries is f.TimeSeries

    class Estimator:
        def filter(self, model, observations):
            return s.FilterResult(observations, s.StateSequence(np.zeros((len(observations), 1)),
                np.ones((len(observations), 1, 1))), np.zeros(len(observations)), 'user', 'custom')

    result = s.filter_states(s.local_level(.1, .2), [1., 2.], Estimator())
    assert isinstance(result.filtered.distribution(0), dist.Distribution)
    assert result.backend == 'user'


def test_user_kernels_without_backend_inheritance():
    pytest.importorskip('particles')

    class Transition:
        dimension = 1

        def sample(self, states, step, *, rng):
            return .8*states + rng.normal(size=states.shape)*np.sqrt(.1*step.dt)

        def log_density(self, values, states, step):
            from scipy.stats import norm
            return norm.logpdf(np.asarray(values)[..., 0], .8*states[:, 0], np.sqrt(.1*step.dt))

    class Observation:
        dimension = state_dimension = 1

        def sample(self, states, step, *, rng):
            return states + rng.standard_t(5, size=states.shape)*.3

        def log_density(self, value, states, step):
            from scipy.stats import t
            return t.logpdf(value[0]-states[:, 0], df=5, scale=.3)

    model = s.StateSpaceModel(s.GaussianState([0.], [[1.]]), Transition(), Observation(), time_mode='elapsed')
    data = s.simulate(model, [0., .2, 1.], rng=np.random.default_rng(0))
    result = s.filter_states(model, data.observations, s.ParticleFilter(300), rng=np.random.default_rng(1))
    assert result.filtered.mean.shape == (3, 1)
    assert np.isfinite(result.log_likelihood)


def test_input_and_result_ownership_and_weighted_distribution_moments():
    mean, cov, matrix = np.array([1.]), np.array([[2.]]), np.array([[1.]])
    state = s.GaussianState(mean, cov)
    kernel = s.LinearGaussianTransition(matrix, cov)
    mean[:] = 99
    cov[:] = 99
    matrix[:] = 99
    assert_allclose(state.mean, [1.])
    assert_allclose(state.covariance, [[2.]])
    assert_allclose(kernel.mean([1.], s.TimeStep(1, 1., 1.)), [1.])
    seq = s.StateSequence.from_particles([[[0., 1.], [2., 3.], [4., 5.]]], [[1., 2., 1.]])
    assert_allclose(seq.mean, [[2., 3.]])
    assert_allclose(seq.covariance, [[[2., 2.], [2., 2.]]])
    law = seq.distribution(0)
    assert_allclose(law.mean(), seq.mean[0])
    assert_allclose(law.covariance(), seq.covariance[0])
    for value in (state.mean, state.covariance, kernel.matrix, seq.mean, seq.covariance, seq.particles, seq.weights):
        assert value is not None
        assert not value.flags.writeable


@pytest.mark.parametrize('factory', [
    lambda: s.GaussianState([0.], [[-1.]]),
    lambda: s.GaussianState([0., 0.], [[1., .5], [0., 1.]]),
    lambda: s.GaussianState([0.], [[np.nan]]),
    lambda: s.LinearGaussianTransition([[1., 0.]], [[1.]]),
    lambda: s.LinearGaussianTransition(lambda step: [[1.]], [[1.]]),
    lambda: s.LinearGaussianObservation([[1., 0.]], [[1.]], state_dimension=3),
    lambda: s.ParticleFilter(True),
    lambda: s.ParticleFilter(1),
    lambda: s.ParticleFilter(10, ess_threshold=1.1),
    lambda: s.UnscentedKalmanFilter(alpha=0.),
    lambda: s.StateSequence([[0.]], [[[-1.]]]),
    lambda: s.StateSequence([[0.]], [[[1.]]], [[[1.], [2.]]], [[.5, .5]]),
    lambda: s.StateSequence.from_particles([[[0.], [1.]]], [[0., 0.]]),
    lambda: s.StateSequence.from_particles([[[0.], [1.]]], [[-.2, 1.2]]),
    lambda: s.StateSpaceModel(dist.normal(), s.LinearGaussianTransition([[1.]], [[1.]]), s.LinearGaussianObservation([[1.]], [[1.]])),
    lambda: s.StateSpaceModel(s.GaussianState([0., 0.], np.eye(2)), s.LinearGaussianTransition([[1.]], [[1.]]), s.LinearGaussianObservation([[1.]], [[1.]])),
])
def test_reject_invalid_shapes_covariances_parameters_and_laws(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_validation_rejects_bad_data_callback_outputs_and_unsupported_capabilities():
    model = s.local_level(.1, .2)
    for data in ([], [1., np.inf], [[1., 2.]]):
        with pytest.raises(ValueError):
            s.KalmanFilter().filter(model, data)
    with pytest.raises(ValueError, match='covariates'):
        s.KalmanFilter().filter(model, s.TimeSeries([1., 2.], covariates=f.Covariates(static=[1.])))
    with pytest.raises(NotImplementedError):
        s.KalmanFilter().filter(model, [1., 2.]).path_distribution()
    with pytest.raises(TypeError, match='GaussianState'):
        nong = s.StateSpaceModel(dist.empirical([[0.], [1.]]), model.transition, model.observation)
        s.KalmanFilter().filter(nong, [1.])
    bad = s.linear_gaussian(s.GaussianState([0.], [[1.]]), lambda step: [[1., 2.]], [[1.]], [[.1]], [[.2]])
    with pytest.raises(ValueError, match='shape'):
        s.KalmanFilter().filter(bad, [1., 2.])
    law = s.DistributionTransition(1, lambda x, step: dist.normal())
    with pytest.raises(ValueError, match='vector Distribution'):
        law.sample([[0.]], s.TimeStep(1, 1., 1.), rng=np.random.default_rng(0))


def test_observed_singular_innovations_are_rejected():
    model = s.linear_gaussian(s.GaussianState([0.], [[0.]]), [[1.]], [[1.]], [[0.]], [[0.]])
    with pytest.raises(ValueError, match='positive definite'):
        s.KalmanFilter().filter(model, [1.])


def test_malformed_result_probabilities_and_nonfixed_time_units_are_rejected():
    observed = s.TimeSeries([1.])
    marginal = s.StateSequence([[0.]], [[[1.]]])
    with pytest.raises(ValueError, match='empirical paths'):
        s.FilterResult(observed, marginal, [0.], 'user', 'test', smoothed=marginal, paths=[[[0.]]])
    particles = s.StateSequence.from_particles([[[0.], [1.]]], [[.5, .5]])
    with pytest.raises(ValueError, match='filtered weights'):
        s.FilterResult(observed, particles, [0.], 'user', 'test', effective_sample_size=[1.])
    dated = s.local_level(.1, .3, time_mode='elapsed', time_unit=np.timedelta64(1, 'M'))
    with pytest.raises(ValueError, match='fixed elapsed'):
        s.KalmanFilter().filter(dated, s.TimeSeries([1., 2.], np.array(['2026-01-01', '2026-02-01'], dtype='datetime64[D]')))
    with pytest.raises(ValueError):
        s.TimeStep(1, np.inf, .1)
    with pytest.raises(ValueError):
        s.TimeStep(1, 1., -.1)


def test_lazy_imports_and_public_exports():
    script = """
import sys
from convexity.math import state_space as s
for name in ('scipy', 'statsmodels', 'filterpy', 'particles', 'sklearn', 'torch'):
    assert name not in sys.modules, name
for name in s.__all__:
    assert getattr(s, name) is not None, name
"""
    subprocess.run([sys.executable, '-c', script], check=True)
