"""Analytic posterior identities, sequential updates, and predictive uncertainty."""
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from convexity.math import bayesian as b


def test_beta_binomial_sequential_and_predictive():
    prior = b.BetaBinomial(2, 3)
    posterior = prior.update([3, 1, 4], [4, 2, 5])
    sequential = prior.update([3], 4).update([1, 4], [2, 5])
    assert posterior == sequential == b.BetaBinomial(10, 6)
    assert prior == b.BetaBinomial(2, 3)
    assert_allclose(posterior.posterior.mean(), 10/16)
    assert_allclose(posterior.predictive(7).density(np.arange(8)), stats.betabinom.pmf(np.arange(8), 7, 10, 6))
    assert_allclose(posterior.predictive(0).sample(3, rng=np.random.default_rng(1)), 0)
    assert prior.update([]) == prior


def test_gamma_poisson_exposure_and_predictive():
    p = b.GammaPoisson(2, 3).update([2, 5], [1, 2])
    assert p == b.GammaPoisson(9, 6)
    assert p == b.GammaPoisson(2, 3).update([2], 1).update([5], 2)
    assert_allclose(p.posterior.mean(), 1.5)
    assert_allclose(p.predictive(2).mean(), 3)
    assert_allclose(p.predictive(2).variance(), 3+4*9/36)
    assert b.GammaPoisson().update([]) == b.GammaPoisson()


def test_normal_mean_and_predictive_total_variance():
    prior = b.NormalMean(1, 4, 2)
    y = np.array([2, 3, 1, 4.])
    p = prior.update(y)
    expected_var = 1/(.25+2)
    assert_allclose(p.posterior.mean(), expected_var*(.25+5))
    assert_allclose(p.posterior.variance(), expected_var)
    assert_allclose(p.predictive().variance(), expected_var+2)
    assert_allclose(prior.update(y[:2]).update(y[2:]).mean, p.mean)
    mc = b.posterior_predictive(p.posterior, lambda mu: b.NormalMean(float(mu), 1, 2).predictive(),
                                draws=12000, rng=np.random.default_rng(1))
    # The callback above adds its own variance 1+2; integration adds p.variance.
    assert_allclose(mc.variance(), p.variance+3, rtol=.04)


def test_normal_inverse_gamma_joint_moments_and_updates():
    y = np.array([1., 2, -1, 3, 5])
    prior = b.NormalInverseGamma(.3, 2, 3, 4)
    posterior = prior.update(y)
    split = prior.update(y[:2]).update(y[2:])
    assert_allclose([posterior.mean, posterior.precision, posterior.shape, posterior.scale],
                    [split.mean, split.precision, split.shape, split.scale])
    draws = posterior.posterior.sample(40000, rng=np.random.default_rng(2))
    assert_allclose(draws.mean(axis=0), posterior.posterior.mean(), rtol=.02)
    assert_allclose(draws.var(axis=0), posterior.posterior.variance(), rtol=.07)
    variance = 1.4
    point = [2., variance]
    expected = stats.invgamma.logpdf(variance, posterior.shape, scale=posterior.scale)+stats.norm.logpdf(
        2, posterior.mean, np.sqrt(variance/posterior.precision))
    assert_allclose(posterior.posterior.log_density(point), expected)
    assert posterior.posterior.log_density([0, -1]) == -np.inf
    assert posterior.posterior.sample((2, 3), rng=np.random.default_rng(4)).shape == (2, 3, 2)
    assert_allclose(posterior.predictive().variance(), posterior.scale/(posterior.shape-1)*(1+1/posterior.precision))
    assert_allclose(posterior.mean_posterior.variance(), posterior.scale/((posterior.shape-1)*posterior.precision))


def test_regression_matches_dense_gaussian_conditioning_and_preserves_query_dependence():
    x = np.array([[1., 0], [1, 1], [1, 2]])
    mean = np.array([.2, -.4])
    covariance = np.array([[2., .3], [.3, 1]])
    y = np.array([1, 2, 2.5])
    prior = b.GaussianLinearRegression(mean, covariance, .5)
    p = prior.update(x, y)
    gain = covariance@x.T@np.linalg.inv(x@covariance@x.T+.5*np.eye(3))
    assert_allclose(p.mean, mean+gain@(y-x@mean))
    assert_allclose(p.covariance, covariance-gain@x@covariance)
    split = prior.update(x[:1], y[:1]).update(x[1:], y[1:])
    assert_allclose(split.mean, p.mean)
    assert_allclose(split.covariance, p.covariance)
    query = np.array([[1., 3], [1, 3]])
    latent = p.predictive(query, include_noise=False)
    observed = p.predictive(query)
    assert_allclose(observed.covariance()-latent.covariance(), .5*np.eye(2), atol=1e-14)
    samples = latent.sample(10, rng=np.random.default_rng(4))
    assert_allclose(samples[:, 0], samples[:, 1], atol=1e-7)
    mean[:] = 8
    covariance[:] = 8
    assert_allclose(prior.mean, [.2, -.4])
