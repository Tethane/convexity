from dataclasses import dataclass
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import forecasting as f


@pytest.mark.parametrize("bad", [[], True, 0, -1, [0], [2, 1], [1, 1], [1.5], [True]])
def test_invalid_horizons(bad):
    with pytest.raises(ValueError):
        f.ForecastHorizon(bad)


@pytest.mark.parametrize(
    "bad", [[], ["1"], [True], [np.inf], [[1j]], np.empty((2, 0)), np.zeros((2, 2, 2))]
)
def test_invalid_targets(bad):
    with pytest.raises(ValueError):
        f.TimeSeries(bad)


def test_time_alignment_sparse_irregular_and_datetime():
    data = f.TimeSeries(
        np.arange(5.0), np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    )
    fitted = f.DriftForecaster().fit(data)
    forecast = fitted.forecast([1, 3])
    assert_allclose(forecast.point[:, 0], [5, 7])
    assert forecast.horizon is not None
    assert_array_equal(
        forecast.horizon.timestamps,
        np.array(["2024-01-06", "2024-01-08"], dtype="datetime64[D]"),
    )
    with pytest.raises(ValueError):
        fitted.forecast(f.ForecastHorizon([1], [np.datetime64("2024-01-07")]))
    irregular = f.NaiveForecaster().fit(f.TimeSeries([1, 2, 3], [0, 2, 5]))
    with pytest.raises(ValueError):
        irregular.forecast(2)
    out = irregular.forecast(f.ForecastHorizon([1, 3], [6, 10]))
    assert out.horizon is not None
    assert_array_equal(out.horizon.timestamps, [6, 10])
    assert out.cutoff == 5


def test_covariate_availability_and_shape():
    y = np.arange(20.0)
    cov = f.Covariates(
        known_future=y[:, None], observed_past=(y * 2)[:, None], static=[1]
    )
    model = f.RegressionForecaster(strategy="direct", max_horizon=3).fit(
        y, covariates=cov
    )
    future = np.arange(20, 23.0)[:, None]
    assert model.forecast([1, 3], covariates=future).shape == (2, 1)
    for bad in (
        None,
        future[:2],
        np.column_stack([future, future]),
        f.Covariates(known_future=future, observed_past=future),
    ):
        with pytest.raises(ValueError):
            model.forecast([1, 3], covariates=bad)
    with pytest.raises(ValueError):
        f.NaiveForecaster().fit(y, covariates=cov)
    with pytest.raises(ValueError):
        f.TimeSeries(y, covariates=f.Covariates(known_future=[1]))
    with pytest.raises(ValueError):
        f.NaiveForecaster().fit(y).forecast(2, covariates=[1, 2])


def test_missing_policy_is_causal_and_explicit():
    y = [1.0, np.nan, 3.0, np.nan]
    with pytest.raises(ValueError):
        f.NaiveForecaster().fit(y)
    fitted = f.NaiveForecaster(missing="ffill").fit(y)
    assert_allclose(fitted.history.values[:, 0], [1, 1, 3, 3])
    with pytest.raises(ValueError):
        f.NaiveForecaster(missing="ffill").fit([np.nan, 2])
    assert_allclose(fitted.update([np.nan, 8]).history.values[:, 0], [1, 1, 3, 3, 3, 8])


def test_ownership_update_and_artifact(tmp_path):
    y = np.arange(10.0)
    fitted = f.DriftForecaster().fit(y)
    y[:] = -100
    assert_allclose(fitted.predict(2)[:, 0], [10, 11])
    updated = fitted.update([10, 11])
    assert fitted.nobs == 10 and updated.nobs == 12
    assert updated.artifact_identity != fitted.artifact_identity
    assert fitted.training_digest != updated.training_digest
    assert not fitted.history.values.flags.writeable
    assert not fitted.forecast(2).point.flags.writeable
    path = tmp_path / "model.pkl"
    updated.save(path)
    loaded = f.load_forecaster(path)
    assert not loaded.history.values.flags.writeable
    assert loaded.artifact_identity == updated.artifact_identity
    assert loaded.training_digest == updated.training_digest
    assert_allclose(loaded.predict(3), updated.predict(3))
    with pytest.raises(ValueError):
        fitted.update(f.TimeSeries([8, 9], [8, 9]))


@dataclass
class UserAdapter:
    def forecast(self, steps, covariates, *, rng=None):
        return f.ForecastResult(np.ones((steps, 1)))


class UserForecaster:
    def fit(self, data, *, covariates=None):
        return f.FittedForecaster(UserAdapter(), data, self)


def test_extension_and_bad_adapter():
    fitted = f.fit(np.arange(10), UserForecaster())
    assert_allclose(fitted.predict([1, 3]), 1)

    class Bad:
        def forecast(self, steps, covariates, *, rng=None):
            return f.ForecastResult([[1, 2]])

    with pytest.raises(ValueError):
        f.FittedForecaster(Bad(), f.TimeSeries([1, 2])).forecast(3)


def test_import_isolation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from convexity.math import forecasting as f
for name in ('torch', 'scipy', 'sklearn', 'statsmodels', 'pandas', 'sktime', 'convexity.math.distributions'):
    assert name not in sys.modules, name
for name in f.__all__:
    assert hasattr(f, name), name
""",
        ],
        check=True,
    )
