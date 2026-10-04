"""Joint SMC against an exactly enumerable parameter/hidden-state model."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
pytest.importorskip('particles')
from convexity.math import bayesian as b, distributions as d, state_space as ss


@dataclass(frozen=True)
class Transition:
    theta: float
    dimension: int = 1

    def sample(self, states, step, *, rng):
        probability = np.where(states[:, 0] == 1, self.theta, 1-self.theta)
        return rng.binomial(1, probability)[:, None]


@dataclass(frozen=True)
class Observation:
    state_dimension: int = 1
    dimension: int = 1

    def sample(self, states, step, *, rng):
        return rng.binomial(1, .15+.7*states)

    def log_density(self, value, states, step):
        p = .15+.7*states[:, 0]
        return np.log(p if value[0] == 1 else 1-p)


def factory(theta):
    prior = d.independent([d.bernoulli(theta[0])])
    return ss.StateSpaceModel(prior, Transition(theta[0]), Observation(), time_mode='elapsed')


def specification():
    return b.StateParameterModel(d.empirical([[.2], [.8]], [.4, .6]), factory, [.5], ('persistence',))


def exact(observations):
    theta = [.2, .8]
    weights = np.array([[.4*.8, .4*.2], [.6*.2, .6*.8]])
    result, logz = [], 0.
    atoms = np.array([[.2, 0], [.2, 1], [.8, 0], [.8, 1]])
    for i, y in enumerate(observations):
        if i:
            weights = np.array([w@np.array([[t, 1-t], [1-t, t]]) for w, t in zip(weights, theta)])
        if not np.isnan(y):
            weights *= np.array([.15, .85]) if y else np.array([.85, .15])
        z = weights.sum()
        logz += np.log(z)
        weights /= z
        result.append(d.empirical(atoms, weights.ravel()))
    return result, logz


def test_joint_filter_exact_prefixes_evidence_and_static_ancestry():
    observations = ss.TimeSeries([1., 1, np.nan, 0], [0., .4, 1.2, 3.])
    result = b.JointParticleFilter(particles=6000, ess_threshold=.8).smooth(
        specification(), observations, paths=200, rng=np.random.default_rng(20))
    reference, logz = exact(observations.values[:, 0])
    for i, law in enumerate(reference):
        assert_allclose(result.joint_posterior(i).mean(), law.mean(), atol=.018)
        assert_allclose(result.joint_posterior(i).covariance(), law.covariance(), atol=.016)
        assert_allclose(result.parameter_posterior(i).mean(), law.mean()[:1], atol=.018)
        assert_allclose(result.state_posterior(i).mean(), law.mean()[1:], atol=.018)
    assert_allclose(result.filtering.log_likelihood, logz, atol=.04)
    assert result.filtering.log_likelihoods[2] == pytest.approx(0, abs=1e-12)
    # Every smoothed trajectory retains exactly one static parameter.
    assert result.filtering.paths is not None
    assert np.all(result.filtering.paths[:, :, 0] == result.filtering.paths[:, :1, 0])
    assert result.state_posterior(0, smoothed=True).event_shape == (1,)
    assert result.filtering.paths is not None
    assert not result.filtering.paths.flags.writeable


def test_reproducibility_global_rng_and_input_validation():
    engine = b.JointParticleFilter(particles=100)
    np.random.seed(124)
    before = np.random.get_state(legacy=False)
    a = engine.filter(specification(), [1, 0], rng=np.random.default_rng(42))
    after = np.random.get_state(legacy=False)
    assert_equal(before, after)
    c = engine.filter(specification(), [1, 0], rng=np.random.default_rng(42))
    assert a.filtering.filtered.particles is not None and c.filtering.filtered.particles is not None
    assert_allclose(a.filtering.filtered.particles, c.filtering.filtered.particles, rtol=0, atol=0)
    assert a.filtering.filtered.weights is not None and c.filtering.filtered.weights is not None
    assert_allclose(a.filtering.filtered.weights, c.filtering.filtered.weights, rtol=0, atol=0)
    with pytest.raises(ValueError):
        a.state_posterior(0, smoothed=True)
    with pytest.raises(ValueError):
        b.StateParameterModel(d.independent([d.normal()]), factory, [0, 1])
    with pytest.raises(TypeError):
        b.StateParameterModel(d.independent([d.normal()]), lambda theta: 1, [0])  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(TypeError):
        bad: Any = None
        engine.filter(bad, [1], rng=np.random.default_rng())
