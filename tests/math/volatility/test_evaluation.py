import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import forecasting as f, volatility as v


def test_qlike_math_shapes_and_zero_proxy():
    actual = np.array([0.0, 1.0, 4.0])
    predicted = np.array([1.0, 2.0, 3.0])
    assert_allclose(
        v.qlike(actual, predicted, axis=()), np.log(predicted) + actual / predicted
    )
    a = np.ones((10, 2)) * 3
    assert v.qlike(a, a) < v.qlike(a, a * 0.5)
    assert v.qlike(a, a) < v.qlike(a, a * 2)
    for a, b in (([-1.0], [1.0]), ([1.0], [0.0]), ([1.0], [[1.0]])):
        with pytest.raises(ValueError):
            v.qlike(a, b)


def test_variance_backtest_reuses_temporal_contract_and_no_proxy_leakage(returns):
    model = v.GARCHForecaster(mean="zero")
    data = returns[:500]
    result = v.variance_backtest(
        model, data, variance_proxy=data**2, origins=[350, 400], horizon=[1, 3]
    )
    assert result.origins == (350, 400)
    assert result.score(by_horizon=True).shape == (2, 1)
    assert np.isfinite(result.score())
    assert result.score("rmse") >= 0
    with pytest.raises(ValueError, match="training units"):
        result.score("mase")
    for record in result.records:
        assert_allclose(
            record.actual[:, 0], data[record.origin + np.array([0, 2])] ** 2
        )
        assert_allclose(
            record.forecast.point,
            model.fit(data[: record.origin]).forecast([1, 3]).conditional_variance,
        )
    altered = data**2 + 100
    other = v.variance_backtest(
        model, data, variance_proxy=altered, origins=[350, 400], horizon=[1, 3]
    )
    for a, b in zip(result.records, other.records):
        assert_array_equal(a.forecast.point, b.forecast.point)
    with pytest.raises(ValueError, match="explicit variance_proxy"):
        v.variance_backtest(model, data, origins=[350])
    with pytest.raises(ValueError, match="timestamps"):
        v.variance_backtest(
            model,
            data,
            variance_proxy=v.TimeSeries(data**2, np.arange(500) + 1),
            origins=[350],
        )
    # The generic evaluator still evaluates return targets and one-step laws.
    assert np.isfinite(f.backtest(model, data, origins=[350]).score("log_score"))


def test_har_backtest_actuals_and_future_covariates(realized):
    cov = np.random.default_rng(9).normal(size=(len(realized), 1))
    data = v.TimeSeries(realized, covariates=v.Covariates(known_future=cov))
    model = v.HARForecaster()
    result = v.variance_backtest(model, data, origins=[400, 500], horizon=[1, 4])
    for record in result.records:
        assert_array_equal(record.actual, data.values[record.origin + np.array([0, 3])])
        fitted = model.fit(data.slice(0, record.origin))
        assert_allclose(
            record.forecast.point,
            fitted.forecast(
                [1, 4], covariates=cov[record.origin : record.origin + 4]
            ).point,
        )


def test_volatility_diagnostics_match_statsmodels(returns):
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

    fitted = v.GARCHForecaster(mean="zero").fit(returns)
    z = np.asarray(fitted.parameters["standardized_residuals"], dtype=float)
    out = v.volatility_diagnostics(z, lags=8, ddof=3)
    lm = het_arch(z, nlags=8, ddof=3, store=True)
    assert_allclose(out["arch_lm_statistic"], lm[0])
    assert_allclose(out["arch_lm_pvalue"], lm[1])
    assert_allclose(
        out["squared_residuals"]["ljung_box_pvalue"],
        acorr_ljungbox(z**2, lags=[8])["lb_pvalue"],
    )
    with pytest.raises(ValueError):
        v.volatility_diagnostics([np.nan, 1, 2])
    with pytest.raises(ValueError, match="insufficient"):
        v.volatility_diagnostics(z[:10], lags=8)
