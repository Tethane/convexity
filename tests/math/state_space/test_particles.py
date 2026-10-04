from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from convexity.math import distributions as dist, state_space as s

pytest.importorskip('particles')


@pytest.mark.parametrize('scheme', ['systematic', 'stratified', 'multinomial', 'residual'])
def test_bootstrap_gaussian_reference_weights_ess_likelihood_and_missing(scheme):
    model = s.local_level(.2, .4, time_mode='elapsed')
    data = s.TimeSeries([np.nan, .1, -.2, np.nan, .5, .2], [0., .3, 1., 1.4, 3., 4.])
    exact = s.KalmanFilter().filter(model, data)
    result = s.ParticleFilter(10000, resampling=scheme).filter(model, data, rng=np.random.default_rng(0))
    assert result.filtered.weights is not None
    assert_allclose(result.filtered.mean, exact.filtered.mean, atol=.035)
    assert_allclose(result.filtered.covariance, exact.filtered.covariance, atol=.04)
    assert_allclose(result.log_likelihood, exact.log_likelihood, atol=.1)
    assert_allclose(result.log_likelihoods[[0, 3]], 0, atol=1e-14)
    assert_allclose(result.filtered.weights.sum(axis=1), 1)
    assert result.effective_sample_size is not None and result.filtered.weights is not None
    assert_allclose(result.effective_sample_size, 1/np.sum(result.filtered.weights**2, axis=1))
    assert result.resampled is not None
    assert not result.resampled[0]
    assert result.resampled is not None
    assert result.resampled.any()
    assert result.predicted is None and result.smoothed is None
    assert result.filtered.distribution(2).measure == 'discrete'


def test_backward_smoothing_converges_to_kalman_and_reuses_forward_history():
    model = s.local_level(.2, .4)
    y = [0., .3, -.5, .7, .8]
    exact = s.KalmanFilter().smooth(model, y)
    estimator = s.ParticleFilter(1500)
    filtered = estimator.filter(model, y, rng=np.random.default_rng(4))
    result = estimator.smooth_result(filtered, rng=np.random.default_rng(6), paths=1200)
    assert result.smoothed is not None and exact.smoothed is not None
    assert_allclose(result.smoothed.mean, exact.smoothed.mean, atol=.065)
    assert_allclose(result.smoothed.covariance, exact.smoothed.covariance, atol=.055)
    assert result.paths is not None
    assert result.paths.shape == (1200, 5, 1)
    assert_allclose(result.path_distribution().mean(), result.smoothed.mean.ravel())
    assert_array_equal(result.filtered.particles, filtered.filtered.particles)
    assert filtered.paths is None
    # Smoothing marginals do not turn independent filtered draws into paths.
    with pytest.raises(NotImplementedError):
        filtered.path_distribution()


def test_non_gaussian_distribution_kernels_against_exact_enumeration():
    # Tiny finite latent chain gives an independent, exact nonlinear/non-Gaussian reference.
    transition = np.array([[.8, .2], [.3, .7]])
    emission = np.array([[.85, .15], [.25, .75]])
    laws = [dist.empirical([[0.], [1.]], weights=row) for row in transition]
    observations = [dist.empirical([[0.], [1.]], weights=row) for row in emission]
    model = s.StateSpaceModel(dist.empirical([[0.], [1.]], [.6, .4]),
        s.DistributionTransition(1, lambda x, step: laws[int(x[0])]),
        s.DistributionObservation(1, 1, lambda x, step: observations[int(x[0])]))
    y = np.array([1., np.nan, 0., 1.])
    sequences = np.array(list(product([0, 1], repeat=len(y))))
    weights = []
    for sequence in sequences:
        mass = [.6, .4][sequence[0]]
        for t, state in enumerate(sequence):
            if t:
                mass *= transition[sequence[t-1], state]
            if not np.isnan(y[t]):
                mass *= emission[state, int(y[t])]
        weights.append(mass)
    weights = np.array(weights)
    evidence = weights.sum()
    weights /= evidence
    result = s.ParticleFilter(1200).smooth(model, y, rng=np.random.default_rng(2), paths=350, method='ancestral')
    assert result.smoothed is not None
    assert_allclose(result.smoothed.mean[:, 0], weights @ sequences, atol=.07)
    assert_allclose(result.log_likelihood, np.log(evidence), atol=.08)
    # DistributionTransition density participates in actual backward simulation.
    backward = s.ParticleFilter().smooth_result(result, rng=np.random.default_rng(1), paths=40)
    assert backward.paths is not None
    assert backward.paths.shape == (40, 4, 1)
    assert backward.paths is not None
    assert set(backward.paths.ravel()) == {0., 1.}


@pytest.mark.parametrize('method', ['ancestral', 'backward'])
def test_particle_randomness_reproducibility_single_path_and_global_rng_isolation(method):
    model = s.local_level(.1, .2)
    state = np.random.get_state(legacy=False)
    a = s.ParticleFilter(200).smooth(model, [1., np.nan, .5], rng=np.random.default_rng(4), paths=1, method=method)
    b = s.ParticleFilter(200).smooth(model, [1., np.nan, .5], rng=np.random.default_rng(4), paths=1, method=method)
    assert_array_equal(a.paths, b.paths)
    assert_array_equal(a.filtered.particles, b.filtered.particles)
    after = np.random.get_state(legacy=False)
    assert_equal(state, after)
    assert a.paths is not None
    assert a.paths.shape == (1, 3, 1)
    assert a.smoothed is not None
    assert_allclose(a.smoothed.covariance, 0)


def test_degenerate_weights_fail_explicitly_and_restore_rng():
    model = s.StateSpaceModel(dist.empirical([[0.]]),
        s.DistributionTransition(1, lambda x, step: dist.empirical([[0.]])),
        s.DistributionObservation(1, 1, lambda x, step: dist.empirical([[0.]])))
    state = np.random.get_state(legacy=False)
    with pytest.raises(s.ParticleDegeneracyError, match='zero likelihood'):
        s.ParticleFilter(10).filter(model, [1.], rng=np.random.default_rng(0))
    assert_equal(np.random.get_state(legacy=False), state)
    with pytest.raises(TypeError, match='Generator'):
        s.ParticleFilter(10).filter(model, [0.], rng=1)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.


def test_partial_missing_likelihood_marginalizes_correlated_noise():
    obs = s.LinearGaussianObservation([[1.], [2.]], [[1., .7], [.7, 2.]])
    generic = s.DistributionObservation(1, 2, lambda x, step: dist.multivariate_normal([x[0], 2*x[0]], [[1., .7], [.7, 2.]]))
    states = np.array([[-1.], [0.], [1.]])
    step = s.TimeStep(0, 0., 0.)
    for y in ([np.nan, .2], [.2, np.nan], [.2, .3], [np.nan, np.nan]):
        assert_allclose(obs.log_density(y, states, step), generic.log_density(y, states, step))
    model = s.StateSpaceModel(s.GaussianState([0.], [[1.]]), s.LinearGaussianTransition([[1.]], [[.1]]), obs)
    data = [[np.nan, .2], [.1, np.nan], [.3, .6]]
    exact = s.KalmanFilter().filter(model, data)
    result = s.ParticleFilter(6000).filter(model, data, rng=np.random.default_rng(3))
    assert_allclose(result.filtered.mean, exact.filtered.mean, atol=.025)


def test_all_missing_and_no_resampling_and_one_row_smoothing():
    model = s.local_level(.1, .3)
    result = s.ParticleFilter(100, ess_threshold=0).smooth(model, [np.nan], rng=np.random.default_rng(2), paths=10)
    assert result.filtered.weights is not None
    assert_allclose(result.filtered.weights, .01)
    assert result.effective_sample_size is not None
    assert_allclose(result.effective_sample_size, 100)
    assert result.log_likelihood == 0
    assert result.resampled is not None
    assert not result.resampled.any()
    assert result.paths is not None
    assert result.paths.shape == (10, 1, 1)


def test_sample_only_transition_has_explicit_ancestral_smoothing_capability():
    class Deterministic:
        dimension = 1

        def sample(self, states, step, *, rng):
            return states + .1*step.dt

    model = s.StateSpaceModel(dist.empirical([[0.], [1.]]), Deterministic(),
                              s.LinearGaussianObservation([[1.]], [[.3]]))
    estimator = s.ParticleFilter(100)
    result = estimator.smooth(model, [.1, .5, .3], rng=np.random.default_rng(0), paths=20, method='ancestral')
    assert result.paths is not None
    assert_allclose(np.diff(result.paths, axis=1), .1)
    with pytest.raises(NotImplementedError, match='transition log_density'):
        estimator.smooth_result(result, rng=np.random.default_rng(1), paths=2, method='backward')
