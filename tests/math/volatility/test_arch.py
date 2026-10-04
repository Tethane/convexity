from convexity.math.volatility._backends.arch import ARCHAdapter
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import volatility as v


@pytest.mark.parametrize('spec', [v.ARCHForecaster(2), v.GARCHForecaster(2, 2),
    v.GJRGARCHForecaster(), v.EGARCHForecaster(), v.ConstantVarianceForecaster(), v.EWMAForecaster()])
def test_catalog_native_backend_and_units(spec, returns):
    fitted = spec.fit(returns)
    assert isinstance(fitted, v.FittedVolatilityForecaster)
    assert fitted.diagnostics.converged
    assert fitted.dependency_versions['arch']
    out = fitted.forecast(1)
    assert isinstance(out, v.VolatilityForecastResult)
    assert isinstance(fitted.adapter, ARCHAdapter)
    native = fitted.adapter.model.forecast(horizon=1)
    scale = np.asarray(fitted.parameters['scale'], dtype=float)
    assert_allclose(out.point[:, 0], native.mean.iloc[-1] / scale)
    assert_allclose(out.variance()[:, 0], native.variance.iloc[-1] / scale**2)
    assert_allclose(out.conditional_variance[:, 0], native.residual_variance.iloc[-1] / scale**2)
    assert_allclose(out.volatility()**2, out.conditional_variance)
    assert_allclose(out.volatility(periods_per_year=252)**2, out.conditional_variance*252)
    assert np.isfinite(fitted.forecast(4, rng=np.random.default_rng(0)).conditional_variance).all()


def test_garch_parameter_recovery_and_analytic_recursion(returns):
    fitted = v.GARCHForecaster(mean='zero').fit(returns)
    p = fitted.parameters['coefficients']
    assert isinstance(p, dict)
    assert_allclose([p['omega'], p['alpha[1]'], p['beta[1]']], [.08, .12, .8], atol=.055)
    h = np.asarray(fitted.parameters['conditional_volatility'], dtype=float)[-1]**2
    expected = []
    for t in range(8):
        h = p['omega'] + (p['alpha[1]']*returns[-1]**2 + p['beta[1]']*h if t == 0
                          else (p['alpha[1]'] + p['beta[1]'])*h)
        expected.append(h)
    out = fitted.forecast(8)
    assert_allclose(out.conditional_variance[:, 0], expected, rtol=1e-12)
    assert_allclose(out.variance(), out.conditional_variance)
    assert 'distribution' not in out.capabilities
    assert 'variance' not in out.variance_forecast().capabilities
    with pytest.raises(NotImplementedError):
        out.quantile(.95)


@pytest.mark.parametrize('spec', [v.ARCHForecaster(), v.GARCHForecaster(), v.GJRGARCHForecaster(),
                                 v.EGARCHForecaster(), v.EWMAForecaster()])
def test_simulation_preserves_path_recursions(spec, returns):
    fitted = replace(spec, method='simulation', simulations=1000, mean='zero').fit(returns)
    out = fitted.forecast(4, rng=np.random.default_rng(31))
    h = out.conditional_variance_paths[..., 0]
    e = out.paths[..., 0]
    p = fitted.parameters['coefficients']
    assert isinstance(p, dict)
    for t in range(1, 4):
        if spec.model == 'egarch':
            z = e[:, t-1]/np.sqrt(h[:, t-1])
            expected = np.exp(p['omega'] + p['alpha[1]']*(np.abs(z)-np.sqrt(2/np.pi))
                              + p['gamma[1]']*z + p['beta[1]']*np.log(h[:, t-1]))
        elif spec.model == 'ewma':
            expected = spec.decay*h[:, t-1] + (1-spec.decay)*e[:, t-1]**2
        else:
            expected = (p['omega'] + p['alpha[1]']*e[:, t-1]**2 + p.get('beta[1]', 0)*h[:, t-1]
                        + p.get('gamma[1]', 0)*e[:, t-1]**2*(e[:, t-1] < 0))
        assert_allclose(h[:, t], expected, rtol=1e-10)
    assert_allclose(h.mean(axis=0), out.conditional_variance[:, 0])
    assert_allclose(out.variance_forecast().mean(), out.conditional_variance)
    assert np.ptp(h[:, 0]) == 0
    assert_array_equal(out.paths, fitted.forecast(4, rng=np.random.default_rng(31)).paths)
    assert out.variance_forecast().sample(3, rng=np.random.default_rng(0)).shape == (3, 4, 1)


def test_monte_carlo_matches_analytic_multistep(returns):
    spec = v.GARCHForecaster(mean='zero', simulations=60000)
    fitted = spec.fit(returns)
    analytic = fitted.forecast(5)
    assert isinstance(fitted.adapter, ARCHAdapter)
    simulator = replace(fitted, adapter=replace(fitted.adapter, spec=replace(spec, method='simulation')))
    simulated = simulator.forecast(5, rng=np.random.default_rng(7))
    assert_allclose(simulated.conditional_variance, analytic.conditional_variance, rtol=.015)
    assert_allclose(simulated.variance(), analytic.variance(), rtol=.035)


@pytest.mark.parametrize('innovation', ['normal', 'student_t', 'skew_student_t', 'ged'])
@pytest.mark.parametrize('factory', [v.ARCHForecaster, v.GARCHForecaster, v.GJRGARCHForecaster,
                                     v.EGARCHForecaster, v.ConstantVarianceForecaster, v.EWMAForecaster])
def test_every_innovation_fits_and_forecasts(innovation, factory, returns):
    fitted = factory(innovation=innovation, method='simulation', simulations=600).fit(returns)
    assert fitted.forecast(3, rng=np.random.default_rng(4)).paths.shape == (600, 3, 1)
    assert isinstance(fitted.adapter, ARCHAdapter)
    analytic = replace(fitted, adapter=replace(fitted.adapter, spec=replace(fitted.specification, method='analytic')))
    out = analytic.forecast(1)
    assert_allclose(out.marginal(0).cdf(out.quantile(.91)[0, 0]), .91, atol=1e-8)
    assert np.isfinite(out.log_density(out.point)).all()


def test_skewed_gjr_and_egarch_require_correct_multistep_method(returns):
    for spec in (v.GJRGARCHForecaster(innovation='skew_student_t'), v.EGARCHForecaster()):
        fitted = spec.fit(returns)
        assert fitted.forecast(1).shape == (1, 1)
        with pytest.raises(TypeError, match='Generator'):
            fitted.forecast(3)
        assert fitted.forecast(3, rng=np.random.default_rng(4)).paths is not None
        assert isinstance(fitted.adapter, ARCHAdapter)
        fitted = replace(fitted, adapter=replace(fitted.adapter, spec=replace(spec, method='analytic')))
        with pytest.raises(NotImplementedError, match='multi-step'):
            fitted.forecast(3)


def test_bootstrap_reproducible_historical_standardized_shocks(returns):
    fitted = v.GARCHForecaster(method='bootstrap', mean='zero', simulations=100).fit(returns)
    out = fitted.forecast(4, rng=np.random.default_rng(20))
    assert out.conditional_variance_paths is not None and out.paths is not None
    z = out.paths / np.sqrt(out.conditional_variance_paths)
    empirical = np.asarray(fitted.parameters['standardized_residuals'], dtype=float)
    assert np.max(np.min(np.abs(z.ravel()[:, None]-empirical), axis=1)) < 1e-12
    assert_array_equal(out.paths, fitted.forecast(4, rng=np.random.default_rng(20)).paths)


def test_return_and_residual_variance_differ_for_ar_mean(returns):
    y = returns.copy()
    for t in range(1, len(y)):
        y[t] += .6*y[t-1]
    fitted = v.GARCHForecaster(mean='ar', lags=1).fit(y)
    out = fitted.forecast(5)
    coefficients = fitted.parameters['coefficients']
    assert isinstance(coefficients, dict)
    phi = coefficients['y[1]']
    expected = [sum(phi**(2*j)*out.conditional_variance[h-j, 0] for j in range(h+1)) for h in range(5)]
    assert_allclose(out.variance()[:, 0], expected)
    assert np.all(out.variance()[1:] > out.conditional_variance[1:])


def test_rescaling_preserves_input_units(returns):
    y = returns * .0003
    fitted = v.GARCHForecaster().fit(y)
    scale = np.asarray(fitted.parameters['scale'], dtype=float)
    assert scale > 1
    native = v.GARCHForecaster(rescale=False).fit(y * scale)
    assert_allclose(fitted.forecast(4).variance()*scale**2, native.forecast(4).variance(), rtol=1e-8)
    assert_allclose(fitted.forecast(1).quantile(.1)*scale, native.forecast(1).quantile(.1), rtol=1e-8)
    assert_allclose(np.asarray(fitted.parameters['loglikelihood'], dtype=float), np.asarray(native.parameters['loglikelihood'], dtype=float)+len(y)*np.log(scale))


def test_known_covariates_and_refit(returns):
    x = np.random.default_rng(1).normal(size=(len(returns), 2))
    y = .4 + x @ np.array([2., -1.]) + returns*.1
    fitted = v.GARCHForecaster().fit(y, covariates=v.Covariates(known_future=x))
    out = fitted.forecast(3, covariates=[[0, 0], [1, 0], [0, 1]])
    assert_allclose(out.point[:, 0], [.4, 2.4, -.6], atol=.025)
    with pytest.raises(ValueError, match='required'):
        fitted.forecast(3)
    updated = fitted.update([.3, 2.3], covariates=v.Covariates(known_future=[[0, 0], [1, 0]]))
    assert updated.nobs == len(y)+2
    assert fitted.nobs == len(y)


def test_convergence_failure_retains_result(returns):
    spec = v.GARCHForecaster(fit_options={'options': {'maxiter': 0}})
    with pytest.raises(v.ConvergenceError) as error:
        spec.fit(returns)
    assert not error.value.result.diagnostics.converged
    assert error.value.result.diagnostics.warnings
    assert not replace(spec, check_convergence=False).fit(returns).diagnostics.converged
