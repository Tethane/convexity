import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import forecasting as f


def series(seed=10, n=160):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + rng.normal(scale=0.3)
    return y


def test_baseline_formulas():
    y = np.column_stack((np.arange(1.0, 11), np.arange(1.0, 11) * 2))
    assert_allclose(f.NaiveForecaster().fit(y).predict(3), [[10, 20]] * 3)
    assert_allclose(
        f.DriftForecaster().fit(y).predict(3), [[11, 22], [12, 24], [13, 26]]
    )
    assert_allclose(f.MovingMeanForecaster(window=4).fit(y).predict(2), [[8.5, 17]] * 2)
    assert_allclose(
        f.SeasonalNaiveForecaster(period=3).fit(y).predict(5)[:, 0], [8, 9, 10, 8, 9]
    )
    assert_allclose(
        f.SeasonalMeanForecaster(period=3).fit(y).predict(4)[:, 0], [5, 6, 5.5, 5]
    )
    with pytest.raises(ValueError):
        f.DriftForecaster().fit([1])


def test_ar1_closed_form_and_backend_equivalence():
    from statsmodels.tsa.arima.model import ARIMA

    y = series(n=500)
    fitted = f.ARForecaster(trend="n").fit(y)
    native = ARIMA(y, order=(1, 0, 0), trend="n").fit()
    out = fitted.forecast(5)
    phi, sigma2 = native.params
    assert abs(phi - 0.7) < 0.1
    assert_allclose(out.mean()[:, 0], y[-1] * phi ** np.arange(1, 6), atol=1e-10)
    assert_allclose(
        out.variance()[:, 0], sigma2 * np.cumsum(phi ** (2 * np.arange(5))), atol=1e-8
    )
    assert_allclose(out.point[:, 0], native.forecast(5), atol=1e-10)
    assert "sample" in out.capabilities


@pytest.mark.parametrize(
    "model",
    [
        f.ARMAForecaster(),
        f.ARIMAForecaster((1, 1, 1)),
        f.SARIMAForecaster(seasonal_order=(1, 0, 0, 4)),
        f.SARIMAXForecaster(),
        f.ExponentialSmoothingForecaster(),
        f.HoltForecaster(),
        f.HoltWintersForecaster(period=4),
        f.ETSForecaster(trend="add"),
        f.StructuralForecaster(),
    ],
)
def test_univariate_catalog_runs(model):
    y = 20 + np.arange(120) * 0.1 + np.tile([0, 1, 0, -1], 30) + series(n=120)
    fitted = model.fit(y)
    out = fitted.forecast(4)
    assert out.shape == (4, 1)
    assert np.isfinite(out.point).all()
    assert fitted.diagnostics.backend.startswith("statsmodels.")
    assert fitted.parameters
    assert fitted.diagnostics.residuals.shape[1] == 1


def test_exogenous_forecasts_native_missing_and_refit_updates():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(120, 1))
    y = 2 + 3 * x[:, 0] + rng.normal(size=120) * 0.1
    y[20:25] = np.nan
    fitted = f.ARIMAForecaster(
        (0, 0, 0), missing="native", fit_options={"method_kwargs": {"method": "powell"}}
    ).fit(y, covariates=f.Covariates(known_future=x))
    forecast = fitted.forecast(3, covariates=[[0], [1], [2]])
    assert_allclose(forecast.point[:, 0], [2, 5, 8], atol=0.1)
    updated = fitted.update([2, 5], covariates=f.Covariates(known_future=[[0], [1]]))
    assert updated.nobs == 122
    assert fitted.nobs == 120
    with pytest.raises(ValueError):
        f.ETSForecaster(missing="native")


def test_var_joint_covariance_and_vecm():
    from statsmodels.tsa.vector_ar.var_model import VAR

    rng = np.random.default_rng(9)
    y = np.zeros((400, 2))
    a = np.array([[0.6, 0.2], [0, 0.5]])
    for i in range(1, len(y)):
        y[i] = a @ y[i - 1] + rng.normal(size=2)
    out = f.VARForecaster().fit(y).forecast(3)
    native = VAR(y).fit(maxlags=1)
    assert_allclose(out.point, native.forecast(y[-1:], 3))
    assert_allclose(out.covariance()[:2, :2], native.sigma_u)
    assert_allclose(out.covariance()[:2, 2:4], native.sigma_u @ native.coefs[0].T)
    assert_allclose(
        out.variance(), np.diagonal(native.forecast_cov(3), axis1=1, axis2=2)
    )
    draws = out.sample(30000, rng=rng)
    assert_allclose(
        np.cov(draws.reshape(len(draws), -1), rowvar=False), out.covariance(), atol=0.04
    )
    trend = np.cumsum(rng.normal(size=250))
    cointegrated = np.column_stack((trend, trend + rng.normal(size=250) * 0.2))
    vecm = f.VECMForecaster().fit(cointegrated).forecast(4)
    assert vecm.shape == (4, 2)
    assert "sample" in vecm.capabilities
    assert_allclose(vecm.mean()[:, 0], vecm.mean()[:, 1], atol=0.4)


@pytest.mark.parametrize(
    "model",
    [
        f.VARMAXForecaster(fit_options={"maxiter": 300}),
        f.DynamicFactorForecaster(fit_options={"maxiter": 300}),
    ],
)
def test_multivariate_state_space_catalog(model):
    rng = np.random.default_rng(2)
    latent = series(seed=2, n=160)
    y = np.column_stack((latent, latent * 0.7)) + rng.normal(size=(160, 2)) * 0.15
    fitted = model.fit(y)
    out = fitted.forecast(3)
    assert out.shape == (3, 2)
    assert np.all(out.variance() > 0)
    assert "sample" in out.capabilities


def test_convergence_failure_is_inspectable():
    model = f.ARIMAForecaster((2, 0, 2), fit_options={"method_kwargs": {"maxiter": 0}})
    with pytest.raises(f.ConvergenceError) as error:
        model.fit(series())
    assert not error.value.result.diagnostics.converged
    assert error.value.result.diagnostics.warnings


def test_fit_options_are_not_mutated_and_target_units_cannot_change():
    options = {"method_kwargs": {"method": "powell"}}
    spec = f.ARIMAForecaster((1, 0, 0), fit_options=options)
    fitted = spec.fit(series(n=50))
    assert spec.fit_options == options
    fitted.update([0.2, 0.3])
    assert spec.fit_options == options
    with pytest.raises(ValueError, match="target units"):
        f.SARIMAXForecaster((1, 1, 0), model_options={"simple_differencing": True})
    with pytest.raises(ValueError, match="return_params"):
        f.ARForecaster(fit_options={"return_params": True})
