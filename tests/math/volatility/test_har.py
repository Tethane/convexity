from convexity.math.volatility._backends.har import HARAdapter
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import volatility as v


def test_log_har_recovery_and_statsmodels_equivalence(realized):
    from statsmodels.regression.linear_model import OLS
    fitted = v.HARForecaster().fit(realized)
    z = np.log(realized)
    x = np.array([[1, z[t-1], z[t-5:t].mean(), z[t-22:t].mean()] for t in range(22, len(z))])
    native = OLS(z[22:], x).fit(cov_type='HC0')
    assert_allclose(np.asarray(fitted.parameters['coefficients'], dtype=float), native.params)
    assert_allclose(np.asarray(fitted.parameters['coefficients'], dtype=float), [-.1, .35, .25, .2], atol=.11)
    assert_allclose(np.asarray(fitted.parameters['parameter_covariance'], dtype=float), native.cov_params())
    log_mean = np.array([1, z[-1], z[-5:].mean(), z[-22:].mean()]) @ native.params
    var = native.mse_resid
    out = fitted.forecast(1)
    assert_allclose(out.point, np.exp(log_mean+.5*var))
    assert_allclose(out.variance(), np.expm1(var)*np.exp(2*log_mean+var))
    assert_allclose(out.conditional_variance, out.point)
    assert out.target_kind == 'realized_variance'
    assert_allclose(out.variance_forecast().variance(), out.variance())
    assert fitted.diagnostics.residuals is not None
    assert np.isnan(fitted.diagnostics.residuals[:22]).all()


def test_log_har_multistep_jensen_and_joint_paths(realized):
    spec = v.HARForecaster(simulations=60000)
    fit = spec.fit(realized)
    analytic = fit.forecast(6)
    assert isinstance(fit.adapter, HARAdapter)
    sim = replace(fit, adapter=replace(fit.adapter, spec=replace(spec, method='simulation')))
    out = sim.forecast(6, rng=np.random.default_rng(31))
    assert out.paths is not None
    assert np.all(out.paths > 0)
    assert_allclose(out.mean(), analytic.mean(), rtol=.004)
    assert_allclose(out.variance(), analytic.variance(), rtol=.025)
    assert out.covariance()[0, 1] > 0
    assert_array_equal(out.paths, out.conditional_variance_paths)
    assert_array_equal(out.paths, sim.forecast(6, rng=np.random.default_rng(31)).paths)
    assert out.interval().shape == (6, 1, 2)


def test_analytic_har_exposes_correlated_lognormal_law_and_exact_intervals(realized):
    from scipy.special import ndtri
    fitted = v.HARForecaster().fit(realized)
    analytic = fitted.forecast(6)
    draws = analytic.sample(80000, rng=np.random.default_rng(38))
    flat = draws[:, :, 0]
    median = analytic.quantile(.5)[:, 0]
    log_variance = np.log1p(analytic.variance()[:, 0] / analytic.mean()[:, 0]**2)
    assert_allclose(median, analytic.mean()[:, 0]*np.exp(-.5*log_variance))
    assert_allclose(analytic.quantile(.9)[:, 0], median*np.exp(ndtri(.9)*np.sqrt(log_variance)))
    assert_allclose(flat.mean(axis=0), analytic.mean()[:, 0], rtol=.004)
    assert_allclose(np.cov(flat, rowvar=False, ddof=0), analytic.covariance(), rtol=.07, atol=.0001)
    assert analytic.covariance()[0, 1] > 0
    sparse = fitted.forecast([1, 3, 6])
    assert_allclose(sparse.covariance(), analytic.covariance()[np.ix_([0, 2, 5], [0, 2, 5])])
    assert_allclose(sparse.interval(), analytic.interval()[[0, 2, 5]])
    log_covariance = np.log1p(analytic.covariance()/np.outer(analytic.mean().ravel(), analytic.mean().ravel()))
    logs = np.log(flat[:5])
    centered = logs - np.log(median)
    quadratic = np.sum(centered * np.linalg.solve(log_covariance, centered.T).T, axis=1)
    expected = -.5*(6*np.log(2*np.pi) + np.linalg.slogdet(log_covariance)[1] + quadratic) - logs.sum(axis=1)
    assert_allclose(analytic.log_density(draws[:5]), expected, atol=1e-10)
    assert np.isneginf(analytic.log_density(np.zeros((6, 1))))
    assert_allclose(analytic.variance_forecast().covariance(), analytic.covariance())


def test_har_deterministic_log_recursion_is_an_empirical_atom(realized):
    fitted = v.HARForecaster().fit(realized)
    assert isinstance(fitted.adapter, HARAdapter)
    deterministic = replace(fitted, adapter=replace(fitted.adapter, innovation_variance=0.))
    result = deterministic.forecast(3)
    assert result.distribution().measure == 'discrete'
    assert_allclose(result.variance(), 0)
    assert_allclose(result.quantile(.1), result.point)
    assert_allclose(result.sample(4, rng=np.random.default_rng(0)), np.broadcast_to(result.point, (4, 3, 1)))


def test_level_har_recursion_covariates_and_positivity(realized):
    x = np.random.default_rng(10).uniform(size=(len(realized), 1))
    fit = v.HARForecaster(lags=(1, 5), transform='level').fit(realized, covariates=v.Covariates(known_future=x))
    future = np.array([[0.], [1.], [.5]])
    out = fit.forecast(3, covariates=future)
    p = np.asarray(fit.parameters['coefficients'], dtype=float)
    values = list(realized)
    expected = []
    for h in range(3):
        new = p @ [1, values[-1], np.mean(values[-5:]), future[h, 0]]
        values.append(new)
        expected.append(new)
    assert_allclose(out.point[:, 0], expected)
    assert out.capabilities == {'point'}
    assert isinstance(fit.adapter, HARAdapter)
    broken = replace(fit, adapter=replace(fit.adapter, coefficients=np.array([-1., 0, 0, 0])))
    with pytest.raises(ValueError, match='negative realized variance'):
        broken.forecast(1, covariates=[[0]])


def test_realized_measures_units_and_validation():
    r = np.array([[.01, -.02, .03], [-.02, .01, .01]])
    assert_allclose(v.realized_variance(r), [.0014, .0006])
    assert_allclose(v.realized_volatility(r)**2, v.realized_variance(r))
    assert_allclose(v.realized_variance(r.T, axis=0), v.realized_variance(r))
    assert_allclose(v.realized_variance(100*r), 10000*v.realized_variance(r))
    assert v.realized_variance(r[0]).shape == ()
    for invalid in ([np.nan, 1], [np.inf, 1], [], 1.):
        with pytest.raises(ValueError):
            v.realized_variance(invalid)
    with pytest.raises(ValueError):
        v.realized_variance(r, axis=True)


def test_har_edge_cases(realized):
    for values in (np.ones(100), realized[:24], np.zeros(100), -realized, np.column_stack([realized, realized])):
        with pytest.raises(ValueError):
            v.HARForecaster().fit(values)
    cases: tuple[dict[str, Any], ...] = ({'lags': ()}, {'lags': (5, 1)}, {'lags': (True,)},
                    {'transform': 'level', 'method': 'simulation'}, {'covariance': 'typo'})
    for options in cases:
        with pytest.raises(ValueError):
            v.HARForecaster(**options)
    updated = v.HARForecaster().fit(realized[:-3]).update(realized[-3:])
    assert_allclose(updated.forecast(3).point, v.HARForecaster().fit(realized).forecast(3).point)
