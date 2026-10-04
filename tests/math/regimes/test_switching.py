"""Fit recovery and independent enumeration of centered switching AR futures."""
import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from convexity.math import regimes as r, distributions as d
from convexity.math.forecasting import TimeSeries, ForecastHorizon


@pytest.fixture(scope='module')
def fitted_regression():
    rng = np.random.default_rng(20)
    model = r.HiddenMarkovModel(r.MarkovChain([[.95, .05], [.1, .9]]),
                               [d.normal(-2, .5), d.normal(2, .8)], [.5, .5])
    y = model.sample(900, rng=rng).values[0]
    return r.SwitchingRegression(switching_variance=True).fit(y)


def test_switching_regression_recovery_and_hmm_likelihood(fitted_regression):
    fitted = fitted_regression
    order = np.argsort(fitted.coefficients[:, 0])
    assert fitted.converged
    assert_allclose(fitted.coefficients[order, 0], [-2, 2], atol=.15)
    assert_allclose(fitted.variances[order], [.25, .64], atol=.15)
    assert_allclose(fitted.chain.transition[np.ix_(order, order)], [[.95, .05], [.1, .9]], atol=.035)
    laws = [d.normal(mu, np.sqrt(var)) for mu, var in zip(fitted.coefficients[:, 0], fitted.variances)]
    hmm = r.HiddenMarkovModel(fitted.chain, laws, fitted.inference.predicted[0])
    expected = hmm.smooth(fitted.history)
    assert_allclose(fitted.inference.log_likelihood, expected.log_likelihood, atol=1e-8)
    assert_allclose(fitted.inference.filtered, expected.filtered, atol=1e-10)
    assert expected.smoothed is not None and fitted.inference.smoothed is not None
    assert_allclose(fitted.inference.smoothed, expected.smoothed, atol=1e-10)
    assert_allclose(fitted.aic, 2*len(fitted.parameters)-2*expected.log_likelihood)
    assert_allclose(fitted.bic, np.log(len(fitted.history))*len(fitted.parameters)-2*expected.log_likelihood)
    law = fitted.marginal(2)
    paths = fitted.sample(2, draws=30000, rng=np.random.default_rng(21))
    assert abs(paths.values[:, 1].mean()-float(law.mean())) < .04
    assert abs(paths.values[:, 1].var()-float(law.variance())) < .06
    assert np.cov(paths.values.T)[0, 1] > .05
    forecast = fitted.predict(ForecastHorizon([1, 3]), draws=100, rng=np.random.default_rng(22))
    assert forecast.paths.shape == (100, 2, 1)
    assert_array_equal(forecast.horizon.timestamps, [900, 902])
    assert_array_equal(forecast.paths[:, :, 0], fitted.sample(3, draws=100, rng=np.random.default_rng(22)).values[:, [0, 2]])


def test_regression_with_exog_and_common_parameters():
    rng = np.random.default_rng(23)
    states = r.MarkovChain([[.95, .05], [.08, .92]]).sample([.5, .5], 599, rng=rng)[0]
    x = rng.normal(size=(600, 1))
    y = np.where(states == 0, -2., 2.) + 1.7*x[:, 0] + rng.normal(scale=.3, size=600)
    fitted = r.SwitchingRegression(switching_exog=False).fit(y, exog=x)
    assert fitted.converged
    assert_allclose(fitted.coefficients[:, 1], 1.7, atol=.06)
    assert_allclose(sorted(fitted.coefficients[:, 0]), [-2, 2], atol=.1)
    assert_allclose(fitted.marginal(2, exog=[[0], [1]]).mean() - fitted.marginal(2, exog=[[0], [0]]).mean(),
                    fitted.coefficients[0, 1])
    with pytest.raises(ValueError, match='exog'):
        fitted.sample(2, rng=rng)
    with pytest.raises(ValueError, match='exog'):
        fitted.sample(2, exog=[[0]], rng=rng)


def test_autoregression_backend_recovery_and_alignment():
    rng = np.random.default_rng(24)
    states = r.MarkovChain([[.97, .03], [.05, .95]]).sample([.5, .5], 899, rng=rng)[0]
    y = np.empty(900)
    means = np.array([-2., 2.])
    residual = 0.
    for t, state in enumerate(states):
        residual = .45*residual + rng.normal(scale=.4)
        y[t] = means[state]+residual
    spec = r.SwitchingAutoregression(order=1, switching_ar=False)
    # Explicit valid transformed starting values avoid a local mode in this synthetic fit.
    fitted = spec.fit(y, start_params=[.96, .04, -2, 2, .16, .4])
    assert fitted.converged
    assert_allclose(sorted(fitted.coefficients[:, 0]), [-2, 2], atol=.12)
    assert_allclose(fitted.autoregressive, .45, atol=.1)
    assert_allclose(fitted.variances, .16, atol=.04)
    assert_array_equal(fitted.inference.timestamps, np.arange(1, 900))
    assert fitted.sample(3, draws=10, rng=rng).values.shape == (10, 3)
    with pytest.raises(NotImplementedError):
        fitted.marginal()


def test_ar2_joint_lag_history_and_future_covariance_against_enumeration():
    chain = r.MarkovChain([[.8, .2], [.3, .7]])
    joint = np.zeros((2, 2, 2))
    joint[0, 1, 0], joint[1, 0, 1] = .65, .35
    prior = joint.sum(axis=(1, 2))[None, :]
    fit = r.FittedSwitching(r.SwitchingAutoregression(order=2), TimeSeries([1., -.5, .7]),
        chain, r.RegimeResult(prior, prior, [0]), [0.], ('placeholder',),
        [[-2.], [2.]], [.2, .4], [[.5, -.2], [.3, .25]], True, 0., 0., np.ones((3, 1)), joint)
    means, covariances, weights = [], [], []
    beta, phi = fit.coefficients[:, 0], fit.autoregressive
    for a, b, c, s, u in itertools.product(range(2), repeat=5):
        weight = joint[a, b, c]*chain.transition[a, s]*chain.transition[s, u]
        residual = phi[s] @ [fit.history.values[-1, 0]-beta[a], fit.history.values[-2, 0]-beta[b]]
        m1 = beta[s]+residual
        m2 = beta[u]+phi[u, 0]*residual+phi[u, 1]*(fit.history.values[-1, 0]-beta[a])
        covariance = np.array([[fit.variances[s], phi[u, 0]*fit.variances[s]],
                               [phi[u, 0]*fit.variances[s], phi[u, 0]**2*fit.variances[s]+fit.variances[u]]])
        weights.append(weight)
        means.append([m1, m2])
        covariances.append(covariance)
    weights, means = np.array(weights), np.array(means)
    expected_mean = weights @ means
    delta = means-expected_mean
    expected_cov = np.einsum('n,nij->ij', weights, covariances) + (delta.T*weights) @ delta
    paths = fit.sample(2, draws=90000, rng=np.random.default_rng(25)).values
    assert_allclose(paths.mean(axis=0), expected_mean, atol=.035)
    assert_allclose(np.cov(paths.T, ddof=0), expected_cov, atol=.065)
