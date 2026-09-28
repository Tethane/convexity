"""Family isolation, ownership and validation at public boundaries."""

from dataclasses import FrozenInstanceError
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import gaussian_processes as gp


@pytest.mark.parametrize("bad", [[], [1, 2], [[True]], [["1"]], [[1j]], [[np.nan]],
                                 [[np.inf]], np.empty((2, 0)), np.empty((0, 2))])
def test_invalid_features(bad):
    with pytest.raises(ValueError):
        gp.RBF()(bad)
    with pytest.raises(ValueError):
        gp.GaussianProcessRegressor().fit(bad, [1.])


@pytest.mark.parametrize("bad", [[], [[1], [2]], [1], [1, 2, 3], [True, False],
                                 ["1", "2"], [1j, 0j], [1, np.nan]])
def test_invalid_response(bad):
    with pytest.raises(ValueError):
        gp.GaussianProcessRegressor().fit([[1], [2]], bad)


@pytest.mark.parametrize("factory,kwargs", [
    (gp.RBF, {"length_scale": 0}), (gp.RBF, {"length_scale": []}),
    (gp.RBF, {"length_scale": [[1]]}), (gp.RBF, {"length_scale": True}),
    (gp.RBF, {"bounds": (2, 1)}), (gp.RBF, {"bounds": (0, 1)}),
    (gp.RBF, {"bounds": (1, np.inf)}), (gp.RBF, {"bounds": "bad"}),
    (gp.RBF, {"bounds": (.1, .2)}), (gp.RBF, {"bounds": [[.1, 2], [.1, 2]]}),
    (gp.Matern, {"nu": -1}), (gp.Matern, {"nu": np.nan}), (gp.Matern, {"nu": True}),
    (gp.Constant, {"value": -1}), (gp.DotProduct, {"sigma_0": 0}),
    (gp.WhiteNoise, {"noise_level": -.1}), (gp.Periodic, {"periodicity": 0}),
    (gp.RationalQuadratic, {"alpha": 0}), (gp.RationalQuadratic, {"length_scale": [1, 2]}),
    (gp.GaussianProcessRegressor, {"kernel": 1}), (gp.GaussianProcessRegressor, {"alpha": -1}),
    (gp.GaussianProcessRegressor, {"alpha": [[1]]}), (gp.GaussianProcessRegressor, {"alpha": True}),
    (gp.GaussianProcessRegressor, {"normalize_y": 1}), (gp.GaussianProcessRegressor, {"optimize": 1}),
    (gp.GaussianProcessRegressor, {"restarts": -1}), (gp.GaussianProcessRegressor, {"seed": 2**32}),
    (gp.GaussianProcessRegressor, {"restarts": 1, "optimize": False}),
    (gp.GaussianProcessRegressor, {"tolerance": 0}), (gp.GaussianProcessRegressor, {"maxiter": True}),
])
def test_invalid_specifications(factory, kwargs):
    with pytest.raises((TypeError, ValueError)):
        factory(**kwargs)


def test_mismatched_dimensions_and_variance_vectors():
    x = [[0.], [1.]]
    with pytest.raises(ValueError):
        gp.GaussianProcessRegressor(alpha=[.1]).fit(x, [0, 1])
    with pytest.raises(ValueError):
        gp.GaussianProcessRegressor(gp.RBF([1, 2])).fit(x, [0, 1])
    model = gp.GaussianProcessRegressor(optimize=False).fit(x, [0, 1])
    for method in (model.predict, model.predict_std, model.posterior):
        with pytest.raises(ValueError):
            method([[1, 2]])
    for bad in ([-1, 1], [.1], [[.1], [.1]], True):
        with pytest.raises(ValueError):
            model.posterior(x, noise_variance=bad)
    with pytest.raises(ValueError):
        gp.RBF()(x, [[1, 2]])
    with pytest.raises(ValueError):
        gp.RBF()(x, x, eval_gradient=True)
    with pytest.raises(ValueError):
        gp.RBF().with_theta([])
    with pytest.raises(ValueError):
        gp.RBF().with_theta([1000])
    with pytest.raises(ValueError):
        _ = gp.RBF()**.5


def test_ownership_and_public_outputs():
    x = np.linspace(0, 1, 10)[:, None]
    query = x.copy()
    y = np.sin(x[:, 0])
    lengths = [1.]
    noise = np.full(len(x), .01)
    spec: Any = gp.GaussianProcessRegressor(gp.RBF(lengths), alpha=noise, optimize=False)
    lengths[0] = 100
    noise[:] = 100
    fitted: Any = spec.fit(x, y)
    before = fitted.predict(query)
    law = fitted.posterior(query)
    x[:] = 100
    y[:] = 100
    assert_allclose(fitted.predict(query), before, atol=0, rtol=0)
    assert_allclose(spec.alpha, .01)
    assert_allclose(fitted.kernel.theta, [0.])
    for a in (before, fitted.training_features, fitted.kernel.theta, fitted.kernel.bounds,
              law.mean, law.covariance, law.std, law.interval()):
        assert not a.flags.writeable
    with pytest.raises(FrozenInstanceError):
        fitted.diagnostics = None
    with pytest.raises(FrozenInstanceError):
        spec.alpha = 0
    mean, covariance = np.zeros(2), np.eye(2)
    prediction = gp.GaussianPrediction(mean, covariance)
    mean[:] = 5
    covariance[:] = 5
    assert_allclose(prediction.mean, [0, 0])
    assert_allclose(prediction.covariance, np.eye(2))


def test_lazy_imports_and_public_exports():
    code = """
import sys
from convexity.math import gaussian_processes as gp
for name in ('scipy', 'sklearn', 'statsmodels', 'torch', 'pandas',
             'convexity.math.learning', 'convexity.math.distributions'):
    assert name not in sys.modules, name
for name in gp.__all__:
    assert hasattr(gp, name), name
assert 'gaussian_processes' in dir(__import__('convexity.math', fromlist=['']))
"""
    subprocess.run([sys.executable, "-c", code], check=True)
