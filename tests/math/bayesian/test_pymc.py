"""Actual PyMC execution, checked against conjugate reference posteriors."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pm = pytest.importorskip("pymc")
from convexity.math import bayesian as b, distributions as d


def normal_model():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 2)
        pm.Normal("y", mu, 1, observed=np.array([1.0, 2, 3]))
    return model


def test_nuts_conjugate_reference_prediction_and_diagnostics():
    result = b.infer(
        b.PyMCModel(normal_model),
        b.MCMC(draws=700, tune=400, chains=2),
        rng=np.random.default_rng(15),
    )
    reference = b.NormalMean(0, 4, 1).update([1, 2, 3])
    assert_allclose(result.mean("mu"), reference.mean, atol=0.07)
    assert_allclose(result.samples["mu"].var(), reference.variance, rtol=0.18)
    assert result.samples["mu"].shape == (2, 700)
    assert result.diagnostics.rhat is not None
    assert result.diagnostics.ess_bulk is not None
    assert result.diagnostics.rhat["mu"] < 1.05
    assert result.diagnostics.ess_bulk["mu"] > 100
    assert result.diagnostics.divergences == 0
    prediction = result.posterior_predictive(rng=np.random.default_rng(16))
    assert prediction.samples["y"].shape == (2, 700, 3)
    assert_allclose(prediction.samples["y"].var(), reference.variance + 1, rtol=0.1)

    # Out-of-sample prediction shares the same inferred mu with a new event shape.
    def future():
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 2)
            pm.Normal("future", mu, 1, shape=2)
        return model

    future_samples = result.posterior_predictive(
        model=b.PyMCModel(future), names=["future"], rng=np.random.default_rng(17)
    )
    assert future_samples.samples["future"].shape == (2, 700, 2)
    with pytest.raises(ValueError):
        result.posterior_predictive(names=["mu"], rng=np.random.default_rng())


def test_black_box_metropolis_and_reproducibility():
    problem = b.BayesianProblem(
        d.independent([d.normal(0, 2)]),
        lambda t: float(np.sum(d.normal(t[0], 1).log_density([1.0, 2.0, 3.0]))),
        ("mu",),
        [0.0],
    )
    spec = b.MCMC(draws=1200, tune=600, chains=2)
    result = spec.fit(problem, rng=np.random.default_rng(18))
    assert_allclose(result.mean("mu"), 6 / 3.25, atol=0.1)
    assert result.diagnostics.method == "metropolis"
    short = b.MCMC(draws=15, tune=10, chains=1)
    a = short.fit(problem, rng=np.random.default_rng(7))
    c = short.fit(problem, rng=np.random.default_rng(7))
    assert_allclose(a.samples["mu"], c.samples["mu"], rtol=0, atol=0)
    assert a.diagnostics.rhat is None
    with pytest.raises(ValueError, match="symbolic"):
        b.MCMC(method="nuts").fit(problem, rng=np.random.default_rng())
    with pytest.raises(TypeError):
        b.VariationalInference().fit(problem, rng=np.random.default_rng())


@pytest.mark.parametrize("method", ["advi", "fullrank_advi"])
def test_vi_against_conjugate_reference(method):
    result = b.VariationalInference(iterations=6000, draws=4000, method=method).fit(
        b.PyMCModel(normal_model), rng=np.random.default_rng(19)
    )
    reference = b.NormalMean(0, 4, 1).update([1, 2, 3])
    assert_allclose(result.mean("mu"), reference.mean, atol=0.15)
    assert_allclose(result.samples["mu"].var(), reference.variance, rtol=0.3)
    assert result.diagnostics.loss is not None
    assert result.diagnostics.loss.shape == (6000,)
    assert result.diagnostics.rhat is None
    assert result.diagnostics.ess_bulk is None


def test_native_joint_state_parameter_model():
    observations = np.array([0.2, 0.3, 0.1, 0.5])

    def build():
        with pm.Model() as model:
            offset = pm.Normal("offset", 0, 0.5)
            states = pm.GaussianRandomWalk(
                "states", sigma=0.2, init_dist=pm.Normal.dist(0, 0.4), shape=4
            )
            pm.Normal("y", offset + states, 0.3, observed=observations)
        return model

    result = b.MCMC(draws=600, tune=400, chains=2).fit(
        b.PyMCModel(build), rng=np.random.default_rng(23)
    )
    # Dense Gaussian conditioning on offset and all latent states.
    state_cov = 0.16 + 0.04 * np.minimum.outer(np.arange(4), np.arange(4))
    prior_cov = np.zeros((5, 5))
    prior_cov[0, 0] = 0.25
    prior_cov[1:, 1:] = state_cov
    design = np.column_stack([np.ones(4), np.eye(4)])
    gain = (
        prior_cov
        @ design.T
        @ np.linalg.inv(design @ prior_cov @ design.T + 0.09 * np.eye(4))
    )
    law = result.distribution(["offset", "states"])
    assert_allclose(law.mean(), gain @ observations, atol=0.07)
    assert_allclose(law.covariance(), prior_cov - gain @ design @ prior_cov, atol=0.025)


def test_discrete_auto_steps_and_vi_rejection():
    def build():
        with pm.Model() as model:
            theta = pm.Bernoulli("theta", 0.6)
            pm.Bernoulli("y", 0.2 + 0.6 * theta, observed=[1])
        return model

    result = b.MCMC(draws=800, tune=100, chains=1).fit(
        b.PyMCModel(build), rng=np.random.default_rng(2)
    )
    assert_allclose(result.mean("theta"), 0.48 / 0.56, atol=0.06)
    with pytest.raises(ValueError, match="continuous"):
        b.VariationalInference().fit(b.PyMCModel(build), rng=np.random.default_rng())


def test_numerical_discrete_prior_rejected_and_native_builder_checked():
    model = b.BayesianProblem(
        d.independent([d.bernoulli(0.5)]), lambda theta: 0, initial=[0]
    )
    with pytest.raises(ValueError, match="continuous prior"):
        b.MCMC(draws=1).fit(model, rng=np.random.default_rng())
    with pytest.raises(TypeError, match="pymc.Model"):
        b.MCMC(draws=1).fit(b.PyMCModel(lambda: 3), rng=np.random.default_rng())  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(ValueError, match="free random"):
        b.MCMC(draws=1).fit(b.PyMCModel(pm.Model), rng=np.random.default_rng())
