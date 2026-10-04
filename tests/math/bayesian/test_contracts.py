"""Validation, ownership, extensibility and import isolation."""
import subprocess
import sys
from dataclasses import FrozenInstanceError
from typing import Any
import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import bayesian as b, distributions as d, estimation as e


@pytest.mark.parametrize('factory,kwargs', [
    (b.BetaBinomial, {'alpha': 0}), (b.GammaPoisson, {'rate': -1}),
    (b.NormalMean, {'mean': np.nan}), (b.NormalMean, {'variance': 0}),
    (b.NormalInverseGamma, {'precision': True}), (b.NormalInverseGamma, {'scale': np.inf}),
    (b.MCMC, {'draws': 0}), (b.MCMC, {'tune': -1}), (b.MCMC, {'chains': True}),
    (b.MCMC, {'method': 'bad'}), (b.MCMC, {'target_accept': 1}),
    (b.VariationalInference, {'iterations': 0}), (b.VariationalInference, {'learning_rate': 0}),
    (b.VariationalInference, {'method': 'bad'}), (b.JointParticleFilter, {'particles': 1}),
    (b.JointParticleFilter, {'ess_threshold': 1.1}), (b.JointParticleFilter, {'resampling': 'bad'})])
def test_bad_specifications(factory, kwargs):
    with pytest.raises((TypeError, ValueError)):
        factory(**kwargs)


@pytest.mark.parametrize('action', [
    lambda: b.BetaBinomial().update([2]), lambda: b.BetaBinomial().update([1], [1, 2]),
    lambda: b.BetaBinomial().update([.5]), lambda: b.GammaPoisson().update([-1]),
    lambda: b.GammaPoisson().update([2], 0), lambda: b.NormalMean().update([np.nan]),
    lambda: b.NormalMean().update([[2]]), lambda: b.GaussianLinearRegression([0], [[0]]),
    lambda: b.GaussianLinearRegression([0], [[1]]).update([[1]], [1, 2]),
    lambda: b.BayesianProblem(d.normal(), lambda t: 0),
    lambda: b.PosteriorSamples({'x': [1, 2]}),
    lambda: b.PosteriorSamples({'x': np.zeros((1, 2)), 'y': np.zeros((1, 3))}),
    lambda: b.PosteriorSamples({'x': [[np.nan]]}),
    lambda: b.PosteriorSamples({}), lambda: invalid_builder(),
])
def test_bad_inputs(action):
    with pytest.raises((TypeError, ValueError)):
        action()


def invalid_builder():
    bad: Any = 2
    return b.PyMCModel(bad)


def test_likelihood_prior_and_support_compose_with_estimation():
    likelihood = e.Likelihood([1., 2.], lambda theta, y: d.normal(theta[0], 1).log_density(y))
    model = b.BayesianProblem(d.independent([b.gamma(2)]), likelihood, ('rate',), [1.])
    assert_allclose(model.log_posterior([2]), b.gamma(2).log_density(2)+np.sum(d.normal(2).log_density([1, 2])))
    assert model.log_posterior([-1]) == -np.inf
    for bad in (np.nan, np.inf, [1]):
        with pytest.raises(ValueError):
            b.BayesianProblem(d.independent([d.normal()]), lambda t: bad).log_likelihood([0])  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(ValueError):
        b.BayesianProblem(d.independent([d.normal()]), lambda t: 0, initial=[0, 1])


def test_samples_ownership_joint_dependence_and_extensibility():
    raw = np.arange(12.).reshape(2, 6)
    result: Any = b.PosteriorSamples({'a': raw, 'b': -raw})
    raw[:] = 100
    assert_allclose(result.mean('a'), 5.5)
    draws = result.distribution().sample(20, rng=np.random.default_rng(1))
    assert_allclose(draws.sum(axis=1), 0)
    assert result.interval('a').shape == (2,)
    assert not result.samples['a'].flags.writeable
    with pytest.raises(TypeError):
        result.samples['c'] = raw
    with pytest.raises(FrozenInstanceError):
        result.samples = {}
    for names in ([], ['a', 'a'], ['missing']):
        with pytest.raises(ValueError):
            result.distribution(names)
    class UserMethod:
        def fit(self, model, *, rng):
            return result
    assert b.infer(None, UserMethod(), rng=np.random.default_rng()) is result
    with pytest.raises(TypeError):
        b.infer(None, UserMethod(), rng=1)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    predictive = b.posterior_predictive(result, lambda theta: d.empirical([theta['a']+theta['b']]),
                                         draws=20, rng=np.random.default_rng(1))
    assert_allclose(predictive.mean(), 0)


def test_lazy_imports_and_exports():
    code = '''
import sys
from convexity.math import bayesian as b
for name in ('pymc', 'pytensor', 'arviz', 'particles', 'scipy', 'statsmodels', 'torch'):
    assert name not in sys.modules, name
for name in b.__all__:
    assert hasattr(b, name), name
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_prediction_callback_shape_and_type_errors():
    rng = np.random.default_rng(2)
    with pytest.raises(TypeError, match='conditional'):
        b.posterior_predictive(d.normal(), lambda theta: theta, draws=1, rng=rng)  # pyright: ignore[reportArgumentType, reportCallIssue] # Exercise runtime rejection.
    laws = iter([d.normal(), d.independent([d.normal()])])
    with pytest.raises(ValueError, match='shape'):
        b.posterior_predictive(d.normal(), lambda theta: next(laws), draws=2, rng=rng)
    with pytest.raises(ValueError):
        b.posterior_predictive(d.normal(), lambda theta: d.normal(), draws=0, rng=rng)
    with pytest.raises(ValueError, match='native'):
        b.PosteriorSamples({'x': [[0.]]}).posterior_predictive(rng=rng)


def test_custom_inference_rejects_wrong_parameters():
    class BadNames:
        def fit(self, model, *, rng):
            return b.PosteriorSamples({'wrong': [[0.]]})
    model = b.BayesianProblem(d.independent([d.normal()]), lambda theta: 0, ('mean',))
    with pytest.raises(ValueError, match='names'):
        b.infer(model, BadNames(), rng=np.random.default_rng(2))


def test_diagnostic_arrays_are_owned():
    values = np.array([1., np.nan])
    loss = np.array([2., 1.])
    report = b.InferenceDiagnostics('mcmc', rhat={'x': values}, loss=loss)
    values[:] = 100
    loss[:] = 100
    assert report.rhat is not None
    assert_allclose(report.rhat['x'], [1., np.nan])
    assert report.loss is not None
    assert_allclose(report.loss, [2., 1.])
    assert not report.rhat['x'].flags.writeable
