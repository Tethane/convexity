"""Finite path enumeration gives independent exact hidden-regime references."""
import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from convexity.math import distributions as d
from convexity.math import regimes as r
from convexity.math.forecasting import TimeSeries


def enumerate_paths(model, y):
    paths = np.array(list(itertools.product(range(model.chain.size), repeat=len(y))))
    weights = []
    for path in paths:
        w = model.initial[path[0]]
        for t, state in enumerate(path):
            if t:
                w *= model.chain.transition[path[t-1], state]
            if not np.isnan(y[t]):
                w *= float(model.emissions[state].density(y[t]))
        weights.append(w)
    weights = np.array(weights)
    evidence = weights.sum()
    return paths, weights/evidence, evidence


def model():
    return r.HiddenMarkovModel(r.MarkovChain([[.85, .15], [.3, .7]]),
                               [d.normal(-1, .8), d.normal(2, 1.2)], [.4, .6])


def test_chain_identities_estimation_and_ownership():
    raw = np.array([[.8, .2], [.4, .6]])
    chain = r.MarkovChain(raw)
    raw[:] = 0
    assert_allclose(chain.stationary(), [2/3, 1/3])
    assert_allclose(chain.stationary() @ chain.transition, chain.stationary())
    assert_allclose(chain.matrix(0), np.eye(2))
    assert_allclose(chain.matrix(5), chain.matrix(2) @ chain.matrix(3))
    assert_allclose(chain.durations(), [5, 2.5])
    assert np.isinf(r.MarkovChain([[1]]).durations()[0])
    with pytest.raises(ValueError, match='unique'):
        r.MarkovChain(np.eye(2)).stationary()
    assert_allclose(r.MarkovChain([[0, 1], [1, 0]]).stationary(), [.5, .5])
    path = chain.sample([1, 0], 60000, rng=np.random.default_rng(1))[0]
    fitted = r.MarkovChain.fit(path, size=2)
    assert_allclose(fitted.transition, chain.transition, atol=.008)
    assert not chain.transition.flags.writeable
    assert_array_equal(chain.sample([1, 0], 3, rng=np.random.default_rng(2)),
                       chain.sample([1, 0], 3, rng=np.random.default_rng(2)))
    assert r.MarkovChain.fit([0], size=2, pseudocount=1).transition.shape == (2, 2)


def test_forward_backward_viterbi_and_pairs_against_enumeration():
    m, y = model(), [-.8, np.nan, 1.3, 2.1]
    result = m.smooth(y)
    paths, weights, evidence = enumerate_paths(m, y)
    assert_allclose(result.log_likelihood, np.log(evidence), atol=1e-12)
    for t in range(len(y)):
        marginal = [weights[paths[:, t] == j].sum() for j in range(2)]
        assert result.smoothed is not None
        assert_allclose(result.smoothed[t], marginal, atol=1e-12)
        prefix, pw, _ = enumerate_paths(m, y[:t+1])
        assert_allclose(result.filtered[t], [pw[prefix[:, -1] == j].sum() for j in range(2)])
        if t:
            expected = [[weights[(paths[:, t-1] == i) & (paths[:, t] == j)].sum()
                         for j in range(2)] for i in range(2)]
            assert result.pair_probabilities is not None
            assert_allclose(result.pair_probabilities[t-1], expected, atol=1e-12)
    assert_allclose(result.predicted[0], m.initial)
    assert_allclose(result.predicted[1:], result.filtered[:-1] @ m.chain.transition)
    assert abs(result.log_likelihoods[1]) < 1e-14
    assert_array_equal(m.decode(y), paths[weights.argmax()])
    samples = m.posterior_paths(y, draws=25000, rng=np.random.default_rng(3))
    for path, weight in zip(paths, weights):
        assert abs(np.mean(np.all(samples == path, axis=1))-weight) < .012
    assert result.smoothed is not None
    assert_allclose(result.distribution(2, kind='smoothed').mean(), result.smoothed[2, 1])


def test_missing_vector_observations_and_stable_extreme_likelihoods():
    m = r.HiddenMarkovModel(r.MarkovChain([[1, 0], [0, 1]]),
        [d.multivariate_normal([0, 0], [[1, .5], [.5, 2]]),
         d.multivariate_normal([2, 2], [[1, .5], [.5, 2]])], [.5, .5])
    result = m.smooth(TimeSeries([[np.nan, np.nan], [np.nan, 2]], [2, 9]))
    assert_allclose(result.filtered[0], [.5, .5])
    weights = np.array([float(d.normal(0, np.sqrt(2)).density(2)), float(d.normal(2, np.sqrt(2)).density(2))])
    assert_allclose(result.filtered[1], weights/weights.sum())
    assert_array_equal(result.timestamps, [2, 9])
    sharp = r.HiddenMarkovModel(r.MarkovChain(np.eye(2)), [d.normal(-30), d.normal(30)], [.5, .5])
    # The first filter underflows the second state, which later evidence revives.
    result = sharp.smooth([-30, 30])
    assert result.smoothed is not None
    assert_allclose(result.smoothed, .5, atol=1e-12)
    assert result.pair_probabilities is not None
    assert result.pair_probabilities[0, 0, 1] == 0
    one = model().smooth([0])
    assert one.pair_probabilities is not None
    assert one.pair_probabilities.shape == (0, 2, 2)
    assert one.smoothed is not None
    assert_allclose(one.smoothed, one.filtered)


def test_prediction_and_switching_dynamics():
    m = model()
    p = [.9, .1]
    expected_weights = np.array(p) @ m.chain.matrix(3)
    law = m.predictive(p, steps=3)
    assert_allclose(law.mean(), expected_weights @ [-1, 2])
    paths = m.sample(4, draws=4000, rng=np.random.default_rng(4))
    assert paths.values.shape == (4000, 4)
    assert np.cov(paths.values.T)[0, 1] > .4
    dynamics = r.SwitchingDynamics(r.MarkovChain([[0, 1], [1, 0]]),
                                  lambda regime, state, step: d.empirical([state+regime+step]))
    result = dynamics.sample(0., 0, 3, draws=3, rng=np.random.default_rng(5))
    assert_array_equal(result.regimes, [[0, 1, 0, 1]]*3)
    assert_allclose(result.values, [[0, 2, 4, 8]]*3)
    assert_allclose(dynamics.distribution(0., 0).mean(), 2)
