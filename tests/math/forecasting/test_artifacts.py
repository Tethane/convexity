"""Portable numerical state reproduces predictions without parameter refits."""
import json
from typing import Any
import zipfile

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import forecasting as f


def roundtrip(fitted, tmp_path, *, horizon=3):
    path = tmp_path / 'model.cvf'
    f.save_portable(fitted, path)
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {'manifest.json', 'arrays.npz'}
        assert json.loads(archive.read('manifest.json'))['version'] == 1
    restored = f.load_portable(path)
    a = fitted.forecast(horizon, rng=np.random.default_rng(99))
    b = restored.forecast(horizon, rng=np.random.default_rng(99))
    assert restored.artifact_identity == fitted.artifact_identity
    assert restored.training_digest == fitted.training_digest
    assert_allclose(a.point, b.point, rtol=1e-6, atol=1e-6)
    if 'variance' in a.capabilities:
        assert_allclose(a.variance(), b.variance(), rtol=1e-6, atol=1e-6)
    return restored


@pytest.mark.parametrize('spec', [f.NaiveForecaster(), f.OUForecaster(), f.ARForecaster(1),
                                  f.RegressionForecaster(strategy='direct', max_horizon=3)])
def test_portable_numeric_engines(spec, tmp_path):
    data = np.sin(np.arange(100.) / 6) + np.random.default_rng(1).normal(0, .1, 100)
    roundtrip(spec.fit(data), tmp_path)


def test_portable_regression_preserves_refit_configuration(tmp_path):
    from sklearn.linear_model import LinearRegression
    fitted = f.RegressionForecaster(LinearRegression(fit_intercept=False)).fit(np.arange(30.))
    restored = roundtrip(fitted, tmp_path)
    assert isinstance(restored.specification, f.RegressionForecaster)
    assert isinstance(restored.specification.estimator, LinearRegression)
    assert restored.specification.estimator.get_params()['fit_intercept'] is False
    assert_allclose(restored.update([30., 31.]).predict(3), fitted.update([30., 31.]).predict(3))


@pytest.mark.parametrize('kind', ['torch', 'ddpm', 'flow', 'vae', 'autoregressive', 'score'])
def test_portable_neural_engines(kind, tmp_path):
    torch = pytest.importorskip('torch')
    kwargs: dict[str, Any] = dict(context_length=5, max_horizon=3, epochs=1, hidden_size=8)
    if kind == 'torch':
        spec = f.TorchForecaster(head='student_t', **kwargs)
    elif kind == 'ddpm':
        spec = f.DiffusionForecaster(diffusion_steps=5, samples=8, **kwargs)
    else:
        spec = f.ConditionalGenerativeForecaster(kind, samples=8, noise_levels=3, langevin_steps=2, **kwargs)
    fitted = spec.fit(np.sin(np.arange(32.) / 4))
    state = torch.random.get_rng_state().clone()
    restored = roundtrip(fitted, tmp_path)
    if kind == 'ddpm':
        from convexity.math.forecasting._backends.diffusion import DiffusionAdapter

        assert isinstance(restored.adapter, DiffusionAdapter)
        assert restored.adapter.prediction_type == 'v_prediction'
        with zipfile.ZipFile(tmp_path / 'model.cvf') as archive:
            # Readers predating velocity support reject this kind explicitly.
            assert json.loads(archive.read('manifest.json'))['kind'] == 'ddpm-velocity'
    assert torch.equal(state, torch.random.get_rng_state())


def test_portable_ddpm_retains_legacy_epsilon_interpretation(tmp_path):
    from dataclasses import replace

    pytest.importorskip('torch', exc_type=ModuleNotFoundError)
    from convexity.math.forecasting._backends.diffusion import DiffusionAdapter

    fitted = f.DiffusionForecaster(context_length=3, max_horizon=2, epochs=1,
                                   diffusion_steps=5, samples=8, hidden_size=8).fit(np.arange(12.))
    assert isinstance(fitted.adapter, DiffusionAdapter)
    legacy = replace(fitted, adapter=replace(fitted.adapter, prediction_type='epsilon'))
    path = tmp_path / 'legacy-ddpm.cvf'
    f.save_portable(legacy, path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        arrays = archive.read('arrays.npz')
    del manifest['state']['prediction_type']
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('arrays.npz', arrays)
    restored = f.load_portable(path)
    assert isinstance(restored.adapter, DiffusionAdapter)
    assert restored.adapter.prediction_type == 'epsilon'
    actual = restored.forecast(2, rng=np.random.default_rng(9))
    expected = legacy.forecast(2, rng=np.random.default_rng(9))
    assert actual.paths is not None and expected.paths is not None
    assert_allclose(actual.paths, expected.paths)


def test_portable_version_rejection(tmp_path):
    path = tmp_path / 'bad.cvf'
    f.save_portable(f.NaiveForecaster().fit([1., 2.]), path)
    with zipfile.ZipFile(path) as z:
        manifest, arrays = json.loads(z.read('manifest.json')), z.read('arrays.npz')
    manifest['version'] = 900
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('manifest.json', json.dumps(manifest))
        z.writestr('arrays.npz', arrays)
    with pytest.raises(ValueError, match='version'):
        f.load_portable(path)
