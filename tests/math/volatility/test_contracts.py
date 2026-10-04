import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import forecasting as f, volatility as v


@pytest.mark.parametrize('name,parameters', [('normal', ()), ('student_t', (7.,)),
    ('skew_student_t', (8., -.3)), ('ged', (1.5,))])
def test_innovation_law_contract_and_normalization(name, parameters):
    from scipy.integrate import quad
    law = v.innovation_distribution(name, parameters)
    assert_allclose(law.mean(), 0)
    assert_allclose(law.variance(), 1)
    assert_allclose(quad(lambda z: float(law.density(z)), -np.inf, np.inf)[0], 1, atol=1e-7)
    assert_allclose(quad(lambda z: float(z*z*law.density(z)), -np.inf, np.inf)[0], 1, atol=1e-7)
    p = np.array([[.01, .1], [.8, .99]])
    assert_allclose(law.cdf(law.quantile(p)), p, atol=1e-8)
    draws = law.sample((100000,), rng=np.random.default_rng(6))
    assert abs(draws.mean()) < .015
    assert abs(draws.var()-1) < .03
    assert law.sample((), rng=np.random.default_rng(1)).shape == ()
    assert law.sample(0, rng=np.random.default_rng(1)).shape == (0,)


def test_student_t_standardization():
    from scipy.stats import t
    law = v.innovation_distribution('student_t', [5])
    assert_allclose(law.quantile([.01, .99]), t.ppf([.01, .99], 5)*np.sqrt(3/5))
    for name, params in [('student_t', [1]), ('normal', [1]), ('ged', [0]),
                         ('skew_student_t', [8, 1.1]), ('skew_student_t', [8, -1]),
                         ('skew_student_t', [8, 1])]:
        with pytest.raises(ValueError):
            v.innovation_distribution(name, params)


def test_horizon_alignment_paths_and_artifact_roundtrip(returns, tmp_path):
    dates = np.datetime64('2020-01-01') + np.arange(len(returns))*np.timedelta64(1, 'D')
    data = v.TimeSeries(returns, dates, names=('return',))
    fitted = f.fit(data, v.GARCHForecaster(method='simulation', simulations=40))
    full = fitted.forecast(6, rng=np.random.default_rng(0))
    sparse = fitted.forecast([1, 3, 6], rng=np.random.default_rng(0))
    assert isinstance(full, v.VolatilityForecastResult)
    assert isinstance(sparse, v.VolatilityForecastResult)
    assert sparse.horizon is not None
    assert full.paths is not None and sparse.paths is not None
    assert_array_equal(sparse.paths, full.paths[:, [0, 2, 5]])
    assert full.conditional_variance_paths is not None and sparse.conditional_variance_paths is not None
    assert_array_equal(sparse.conditional_variance_paths, full.conditional_variance_paths[:, [0, 2, 5]])
    assert_array_equal(sparse.conditional_variance, full.conditional_variance[[0, 2, 5]])
    assert_array_equal(sparse.horizon.timestamps, dates[-1] + np.array([1, 3, 6])*np.timedelta64(1, 'D'))
    assert sparse.metadata['target_names'] == ('return',)
    assert sparse.cutoff == dates[-1]
    assert not sparse.conditional_variance.flags.writeable
    assert sparse.conditional_variance_paths is not None
    assert not sparse.conditional_variance_paths.flags.writeable
    path = tmp_path/'arch.pkl'
    fitted.save(path)
    restored = f.load_forecaster(path)
    assert restored.artifact_identity == fitted.artifact_identity
    assert restored.training_digest == fitted.training_digest
    assert_array_equal(restored.forecast(6, rng=np.random.default_rng(0)).paths, full.paths)
    assert fitted.nobs == len(returns)
    with pytest.raises(NotImplementedError):
        fitted.update([.1], mode='incremental')


def test_irregular_labels_and_har_artifacts(realized, tmp_path):
    timestamps = np.cumsum(np.random.default_rng(0).uniform(.5, 2, len(realized)))
    fitted = v.HARForecaster().fit(v.TimeSeries(realized, timestamps))
    with pytest.raises(ValueError, match='irregular'):
        fitted.forecast(3)
    horizon = v.ForecastHorizon([1, 3], timestamps[-1]+np.array([1., 6.]))
    path = tmp_path/'har.pkl'
    fitted.save(path)
    restored = v.load_forecaster(path)
    assert_allclose(restored.forecast(horizon).point, fitted.forecast(horizon).point)


def test_user_adapter_uses_existing_contract():
    class Adapter:
        def forecast(self, steps, covariates, *, rng=None):
            return v.VolatilityForecastResult(np.zeros((steps, 1)),
                mean_values=np.zeros((steps, 1)), variance_values=np.full((steps, 1), 2.),
                conditional_variance=np.full((steps, 1), 2.))
    class Model:
        def fit(self, data, *, covariates=None):
            return v.FittedForecaster(Adapter(), data)
    fitted = f.fit([1., 2., 3.], Model())
    out = fitted.forecast([1, 4])
    assert isinstance(out, v.VolatilityForecastResult)
    assert_allclose(out.volatility(), np.sqrt(2))
    assert out.shape == (2, 1)
    assert f.backtest(Model(), np.arange(20.), origins=[10, 15]).score('rmse') > 0


def test_result_shape_nonnegativity_and_copy_validation():
    original = np.array([[1.], [2.]])
    out = v.VolatilityForecastResult(original, conditional_variance=original)
    original[:] = -1
    assert_allclose(out.conditional_variance[:, 0], [1, 2])
    cases: tuple[dict[str, Any], ...] = ({'conditional_variance': [[-1]]}, {'conditional_variance': [[1], [2]]},
                   {'conditional_variance': [[np.nan]]}, {'conditional_variance': [[1]], 'target_kind': 'price'},
                   {'conditional_variance': [[1]], 'conditional_variance_paths': [[[1]]]})
    for kwargs in cases:
        with pytest.raises(ValueError):
            v.VolatilityForecastResult([[0]], **kwargs)
    with pytest.raises(ValueError):
        out.volatility(periods_per_year=0)


def test_invalid_model_specifications_and_inputs(returns):
    cases: tuple[dict[str, Any], ...] = ({'p': True}, {'o': 1}, {'model': 'unknown'}, {'model': 'arch'},
                   {'mean': 'zero', 'lags': 1}, {'mean': 'ar'}, {'decay': 1},
                   {'innovation': 'bad'}, {'simulations': 0}, {'fit_options': {'last_obs': 100}})
    for kwargs in cases:
        with pytest.raises(ValueError):
            v.ARCHModelForecaster(**kwargs)
    for y in ([1.], np.ones(100), [1, np.nan, 2], np.column_stack((returns, returns))):
        with pytest.raises(ValueError):
            v.GARCHForecaster().fit(y)
    for cov in (v.Covariates(observed_past=returns), v.Covariates(static=[1])):
        with pytest.raises(ValueError, match='covariates'):
            v.GARCHForecaster().fit(returns, covariates=cov)
    with pytest.raises(ValueError, match='known_future'):
        v.GARCHForecaster(mean='zero').fit(returns, covariates=v.Covariates(known_future=returns))
    options = {'options': {'maxiter': 100}}
    spec = v.GARCHForecaster(fit_options=options)
    options['options']['maxiter'] = 0
    retained = spec.fit_options['options']
    assert isinstance(retained, dict)
    assert retained['maxiter'] == 100


def test_lazy_imports_and_exports():
    script = """
import sys
from convexity.math import volatility as v
for name in ('arch', 'scipy', 'statsmodels', 'sklearn', 'torch'):
    assert name not in sys.modules, name
for name in v.__all__:
    assert getattr(v, name) is not None, name
"""
    subprocess.run([sys.executable, '-c', script], check=True)
