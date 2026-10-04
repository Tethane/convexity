from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from chronos import ChronosPipeline, ChronosBoltPipeline
    from timesfm3.torch import TimesFM3Forecaster

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import forecasting as f


def conditional_sampler(history, steps, future, *, size, rng):
    innovations = rng.normal(size=(size, steps, history.dimension))
    return history.values[-1] + innovations.cumsum(axis=1)


def foundation_predictor(history, steps, future, *, rng):
    return f.ForecastResult(np.repeat(history.values[-1:], steps, axis=0))


def test_generic_foundation_and_generative_paths():
    data = np.column_stack((np.arange(20.), -np.arange(20.)))
    foundation = f.FoundationForecaster(foundation_predictor).fit(data)
    assert_allclose(foundation.predict(2), [[19, -19]] * 2)
    model = f.GenerativeForecaster(conditional_sampler, samples=10000).fit(data)
    forecast = model.forecast([1, 3], rng=np.random.default_rng(1))
    assert forecast.paths is not None
    assert forecast.paths.shape == (10000, 2, 2)
    assert_allclose(model.predict([1, 3], rng=np.random.default_rng(1)), forecast.point)
    assert_allclose(forecast.variance(), [[1, 1], [3, 3]], atol=.1)
    assert_allclose(forecast.covariance()[0, 2], 1, atol=.06)
    assert .45 < forecast.probability(lambda path: path[-1, 0] > 19) < .55
    with pytest.raises(TypeError):
        model.forecast(3)


def test_point_pooling_and_mixture_path_dependence():
    y = np.arange(15.)
    point = f.ensemble([f.NaiveForecaster(), f.DriftForecaster()], weights=[.25, .75]).fit(y).forecast(2)
    assert_allclose(point.point[:, 0], [14.75, 15.5])
    assert 'sample' not in point.capabilities
    median = f.ensemble([f.NaiveForecaster(), f.DriftForecaster()], method='median').fit(y).forecast(2)
    assert_allclose(median.point[:, 0], [14.5, 15])
    generative = f.GenerativeForecaster(conditional_sampler, samples=3000)
    mixture = f.ensemble([generative, generative], method='mixture', samples=8000).fit(y)
    result = mixture.forecast(2, rng=np.random.default_rng(2))
    assert result.paths is not None
    assert result.paths.shape == (8000, 2, 1)
    assert_allclose(result.covariance(), [[1, 1], [1, 2]], atol=.15)
    with pytest.raises(NotImplementedError):
        f.ensemble([f.NaiveForecaster()], method='mixture').fit(y).forecast(2, rng=np.random.default_rng(1))


class FakeChronos:
    """Tests the documented tensor boundary without fetching model weights."""
    forecast_type = 'samples'

    def predict_quantiles(self, context, prediction_length, quantile_levels, limit_prediction_length, **kwargs):
        import torch
        assert context.ndim == 2 and context.shape[0] == 1
        mean = torch.ones((1, prediction_length))
        q = mean[..., None] + torch.tensor(quantile_levels)[None, None]
        return q, mean

    def predict(self, context, prediction_length, num_samples, limit_prediction_length):
        import torch
        return torch.randn((1, num_samples, prediction_length)).cumsum(-1)


def test_chronos_quantile_and_sample_api_boundaries():
    torch = pytest.importorskip('torch', exc_type=ModuleNotFoundError)
    before = torch.random.get_rng_state().clone()
    forecast = f.ChronosForecaster(cast("ChronosPipeline", FakeChronos())).fit(np.arange(20.)).forecast(3)
    assert_allclose(forecast.quantile(.5), 1.5)
    assert 'sample' not in forecast.capabilities
    model = f.ChronosForecaster(cast("ChronosPipeline", FakeChronos()), output='samples', samples=20).fit(np.arange(20.))
    a, b = model.forecast(3), model.forecast(3)
    assert a.paths is not None and b.paths is not None
    assert_allclose(a.paths, b.paths)
    assert torch.equal(before, torch.random.get_rng_state())
    assert a.paths is not None
    assert a.paths.shape == (20, 3, 1)


class FakeTimesFM:
    def predict_batch(self, contexts, horizon, return_quantiles, use_symmetric_averaging, **kwargs):
        context = contexts[0]
        assert kwargs['past_future_covariates'][0].shape == (1, context.shape[-1] + horizon)
        assert kwargs['past_only_covariates'][0].shape == (2, context.shape[-1])
        point = np.repeat(context[:, -1:], horizon, axis=1)
        return [SimpleNamespace(forecast=point, quantiles=point[..., None] + np.arange(9))]


def test_timesfm3_shapes_and_covariate_availability():
    y = np.column_stack((np.arange(20.), -np.arange(20.)))
    cov = f.Covariates(known_future=np.ones((20, 1)), observed_past=np.ones((20, 2)))
    result = f.TimesFMForecaster(cast("TimesFM3Forecaster", FakeTimesFM())).fit(y, covariates=cov).forecast(3, covariates=np.ones((3, 1)))
    assert_allclose(result.point, [[19, -19]] * 3)
    assert_allclose(result.quantile(.1), result.point)
    assert 'mean' not in result.capabilities
    assert 'sample' not in result.capabilities


def test_sktime_real_adapter():
    pytest.importorskip('sktime')
    from sktime.forecasting.naive import NaiveForecaster
    y = np.arange(25.)
    fitted = f.SktimeForecaster(NaiveForecaster(strategy='drift'), max_horizon=4).fit(y)
    assert_allclose(fitted.predict([1, 3])[:, 0], [25, 27])


def choose_drift(history):
    return [0, 1] if history.values[-1, 0] > history.values[0, 0] else [1, 0]


def test_dynamic_weights_and_temporal_stacking():
    y = np.arange(30.)
    models = (f.NaiveForecaster(), f.DriftForecaster())
    dynamic = f.EnsembleForecaster(models, weights=choose_drift).fit(y)
    assert_allclose(dynamic.predict(2)[:, 0], [30, 31])
    stacked = f.StackingForecaster(models, horizon=2, initial_window=15, step=3).fit(y)
    assert_allclose(np.asarray(stacked.parameters['weights'], dtype=float), [0, 1], atol=1e-8)
    assert_allclose(stacked.predict(2)[:, 0], [30, 31], atol=1e-8)


class FakeBolt(FakeChronos):
    forecast_type = 'quantiles'
    quantiles = [.1, .5, .9]


def test_chronos_bolt_median_is_not_a_mean_or_sample_law():
    pytest.importorskip('torch', exc_type=ModuleNotFoundError)
    result = f.ChronosForecaster(cast("ChronosBoltPipeline", FakeBolt())).fit(np.arange(20.)).forecast(3)
    assert 'mean' not in result.capabilities
    assert 'sample' not in result.capabilities
    with pytest.raises(ValueError, match='forecast_type'):
        f.ChronosForecaster(cast("ChronosBoltPipeline", FakeBolt()), output='samples', samples=3)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(ValueError, match='trained'):
        f.ChronosForecaster(cast("ChronosBoltPipeline", FakeBolt()), quantiles=(.05, .5, .95))  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
