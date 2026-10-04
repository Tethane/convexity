import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from convexity.math import forecasting as f
from convexity.math.forecasting.builtins.regression import lag_features


@pytest.mark.parametrize('strategy', ['recursive', 'direct', 'multioutput'])
def test_regression_exact_linear_dynamics(strategy):
    y = np.column_stack((np.arange(60.), 2 * np.arange(60.) + 3))
    spec = f.RegressionForecaster(LinearRegression(), lags=(1, 2), strategy=strategy, max_horizon=4)
    fitted = spec.fit(y)
    assert_allclose(fitted.predict(4), [[60, 123], [61, 125], [62, 127], [63, 129]], atol=1e-10)
    assert_allclose(fitted.predict([1, 3]), [[60, 123], [62, 127]], atol=1e-10)
    if strategy != 'recursive':
        with pytest.raises(ValueError):
            fitted.forecast(5)


def test_lags_rolling_and_known_covariates_alignment():
    y = np.arange(10.)[:, None]
    assert_allclose(lag_features(y, 5, (1, 3), (3,)), [4., 2., 3., float(np.std([2, 3, 4]))])
    rng = np.random.default_rng(10)
    x = rng.normal(size=(100, 1))
    y = 4 * x[:, 0]
    for strategy in ('recursive', 'direct', 'multioutput'):
        spec = f.RegressionForecaster(LinearRegression(), lags=(1,), strategy=strategy, max_horizon=3)
        fitted = spec.fit(y, covariates=f.Covariates(known_future=x))
        assert_allclose(fitted.predict(3, covariates=[[1], [2], [3]])[:, 0], [4, 8, 12], atol=1e-10)
    tree = f.RegressionForecaster(RandomForestRegressor(n_estimators=5, random_state=1)).fit(y)
    assert tree.forecast(2).shape == (2, 1)


class SpyForecaster:
    def __init__(self):
        self.seen = []

    def fit(self, data, *, covariates=None):
        self.seen.append(data)
        class Adapter:
            def forecast(self, steps, covariates, *, rng=None):
                return f.ForecastResult(covariates, mean_values=covariates)
        return f.FittedForecaster(Adapter(), data, model_identity='spy')


def test_backtest_never_exposes_future_targets_or_past_covariates():
    n = 30
    y = np.arange(float(n))
    cov = f.Covariates(known_future=y, observed_past=100 + y, static=[2])
    data = f.TimeSeries(y, covariates=cov)
    model = SpyForecaster()
    result = f.backtest(model, data, horizon=[1, 3], origins=[10, 15, 20], window=7)
    assert result.origins == (10, 15, 20)
    assert result.score('rmse') == 0
    for history, record in zip(model.seen, result.records):
        assert len(history) == 7
        assert history.values[-1, 0] == record.origin - 1
        assert history.covariates.observed_past[-1, 0] == 100 + record.origin - 1
        assert history.covariates.known_future[-1, 0] == record.origin - 1
        assert record.forecast.horizon is not None
        assert record.forecast.horizon.timestamps is not None
        assert record.forecast.cutoff < record.forecast.horizon.timestamps[0]


def test_holdout_changes_cannot_change_forecasts():
    y = np.sin(np.arange(50.) / 3)
    spec = f.RegressionForecaster(LinearRegression(), strategy='direct', max_horizon=4)
    before = f.backtest(spec, y, horizon=4, origins=[30])
    y[30:] = 1e9
    after = f.backtest(spec, y, horizon=4, origins=[30])
    assert_allclose(before.records[0].forecast.point, after.records[0].forecast.point)


def test_metrics_against_hand_computed_formulas():
    m = f.metrics
    assert m.mae([0, 2], [1, 0]) == 1.5
    assert m.rmse([0, 2], [1, 0]) == np.sqrt(2.5)
    assert m.smape([0, 0], [0, 1]) == 100
    assert m.pinball_loss([0, 2], [1, 0], .25) == .625
    assert m.mase([[2], [4]], [[1], [2]], [[0], [1], [2]]) == 1.5
    paths = np.array([[[0.], [1.]], [[2.], [3.]], [[4.], [5.]]])
    out = f.ForecastResult.from_paths(paths)
    actual = np.array([[1.], [2.]])
    expected = abs(paths - actual).mean(axis=0) - .5 * abs(paths[:, None] - paths[None]).mean(axis=(0, 1))
    assert_allclose(m.crps(out, actual, axis=()), expected)
    flat = paths.reshape(3, -1)
    energy = np.linalg.norm(flat - actual.ravel(), axis=1).mean() - .5 * np.linalg.norm(flat[:, None] - flat[None], axis=-1).mean()
    assert_allclose(m.energy_score(out, actual), energy)
    gaussian = f.ForecastResult([[0]], mean_values=[[0]], variance_values=[[1]], gaussian=True)
    assert_allclose(m.crps(gaussian, [[0]]), (np.sqrt(2) - 1) / np.sqrt(np.pi))
    assert_allclose(m.log_score(gaussian, [[0]]), .5 * np.log(2 * np.pi))
    assert m.interval_coverage(gaussian, [[0]]) == 1
    assert m.sharpness(gaussian) > 3


def test_compare_baselines_selection_and_horizon_aggregation():
    y = np.arange(40.)
    results = f.compare({'drift': f.DriftForecaster()}, y, horizon=3, initial_window=20, step=5, seasonality=4)
    assert len(results) == 6
    assert results['drift'].score('mae') == 0
    naive = results['baseline:last']
    assert_allclose(naive.score('mae', by_horizon=True)[:, 0], [1, 2, 3])
    assert_allclose(naive.score('rmse'), np.sqrt(14 / 3))
    selection = f.select_model({'naive': f.NaiveForecaster(), 'drift': f.DriftForecaster()}, y,
                               horizon=3, initial_window=20, step=5)
    assert selection.best_name == 'drift'
    assert selection.fitted is not None
    assert selection.fitted.nobs == len(y)
    assert np.all(selection.best_by_horizon() == 'drift')
    grid = f.grid_search(f.MovingMeanForecaster, {'window': [2, 5]}, y, horizon=2, origins=[20, 30])
    assert isinstance(grid.best_model, f.MovingMeanForecaster)
    assert grid.best_model.window == 2
    regimes = naive.conditional_errors([0, 0, 1, 1])
    assert set(regimes) == {0, 1}
    with pytest.raises(ValueError):
        f.backtest(f.NaiveForecaster(), y, origins=[39], horizon=3)


def test_diagnostics_and_covariate_sensitivity():
    rng = np.random.default_rng(1)
    residuals = rng.normal(size=(100, 2))
    diagnostics = f.residual_diagnostics(residuals, lags=5)
    assert diagnostics['acf'].shape == (6, 2)
    assert_allclose(diagnostics['acf'][0], 1)
    x = rng.normal(size=(60, 1))
    model = f.RegressionForecaster(LinearRegression(), lags=(1,)).fit(3 * x, covariates=f.Covariates(known_future=x))
    derivative = f.covariate_sensitivity(model, 2, [[1], [2]])
    assert_allclose(derivative[:, 0, :, 0], np.eye(2) * 3, atol=1e-8)


def test_convexity_mean_learner_and_rejection_of_nonmean_learners():
    from convexity.math import learning
    from sklearn.linear_model import LogisticRegression
    y = np.arange(40.)
    fitted = f.RegressionForecaster(learning.Ridge(alpha=.01)).fit(y)
    assert_allclose(fitted.predict(2)[:, 0], [40, 41], atol=.01)
    with pytest.raises(ValueError, match='mean learner'):
        f.RegressionForecaster(learning.QuantileRegression()).fit(y)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(ValueError, match='classifier'):
        f.RegressionForecaster(LogisticRegression()).fit(y % 2)


def test_residual_diagnostics_accepts_statsmodels_acf_result_record(monkeypatch):
    from types import SimpleNamespace
    from statsmodels.tsa import stattools
    values = np.random.default_rng(19).normal(size=(40,2))
    expected = f.residual_diagnostics(values,lags=5)
    acf = stattools.acf
    monkeypatch.setattr(stattools, 'acf', lambda *args, **kwargs: SimpleNamespace(acf=acf(*args, **kwargs)))
    actual = f.residual_diagnostics(values,lags=5)
    for name in expected:
        assert_allclose(actual[name],expected[name])
