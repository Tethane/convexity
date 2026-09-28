"""Ownership, validation, extensibility and import isolation for family 4."""

import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import transforms as tx


@pytest.mark.parametrize("samples", [[], [1., 2.], [[True]], [["1"]], [[1j]], [[np.nan]], [[np.inf]], np.empty((2, 0))])
@pytest.mark.parametrize("fitter", [tx.Standardize(), tx.PCA(), tx.FactorAnalysis(1)])
def test_training_matrix_contract(samples, fitter):
    with pytest.raises((TypeError, ValueError)):
        fitter.fit(samples)


@pytest.mark.parametrize("constructor,kwargs", [
    (tx.Standardize, {"ddof": True}), (tx.Standardize, {"ddof": 2}),
    (tx.Standardize, {"center": 1}), (tx.RobustScale, {"quantile_range": (90, 10)}),
    (tx.RobustScale, {"quantile_range": (0, 100)}),
    (tx.RobustScale, {"scale": False, "unit_variance": True}),
    (tx.MinMaxScale, {"lower": 2, "upper": 1}), (tx.MinMaxScale, {"upper": np.inf}),
    (tx.PCA, {"n_components": 0}), (tx.PCA, {"n_components": 1.2}),
    (tx.PCA, {"whiten": 1}), (tx.PCA, {"ddof": -1}),
    (tx.Whitening, {"method": "other"}), (tx.Whitening, {"n_components": 2}),
    (tx.FactorAnalysis, {"n_components": True}),
    (tx.FactorAnalysis, {"n_components": 2, "rotation": "oblique"}),
    (tx.FactorAnalysis, {"n_components": 1, "maxiter": 0}),
    (tx.FactorAnalysis, {"n_components": 1, "tolerance": 0}),
    (tx.Difference, {"lag": 0}), (tx.Difference, {"order": True}),
    (tx.LagFeatures, {"lags": ()}), (tx.LagFeatures, {"lags": (1, 1)}),
    (tx.LagFeatures, {"lags": (-1,)}), (tx.LagFeatures, {"lags": (True,)}),
    (tx.FourierFeatures, {"period": 12, "order": 6}),
    (tx.FourierFeatures, {"period": np.nan}),
    (tx.SeasonalFeatures, {"period": 1}), (tx.SeasonalFeatures, {"period": 3, "drop_first": 1}),
])
def test_invalid_specifications(constructor, kwargs):
    with pytest.raises((TypeError, ValueError)):
        constructor(**kwargs)


def test_owned_read_only_arrays_and_fixed_transform_dimensions():
    x = np.array([[1., 3.], [4., 8.], [6., 5.]])
    original = x.copy()
    fitted = tx.standardize(x)
    delta = tx.difference(x)
    x[:] = 99
    assert_allclose(fitted.location, original.mean(axis=0))
    assert_allclose(delta.inverse_transform(), original)
    for a in [fitted.location, fitted.scale, fitted.encoder, fitted.decoder, delta.values,
              *delta.initial, *delta.trailing, delta.rows, fitted.transform(original)]:
        assert not a.flags.writeable
    with pytest.raises(ValueError):
        fitted.transform([1.])
    with pytest.raises(ValueError):
        fitted.inverse_transform([1., 2., 3.])


def test_custom_fitters_and_pipeline_contract_checks():
    class Custom:
        def fit(self, samples):
            assert not samples.flags.writeable
            return tx.AffineTransform(samples.mean(axis=0), np.eye(2) * 2, np.eye(2) / 2)

    x = np.arange(12.).reshape(6, 2)
    model = tx.fit(x, Custom())
    assert_allclose(model.transform(x), 2 * (x - x.mean(axis=0)))
    pipeline = tx.Pipeline((Custom(), tx.PCA())).fit(x)
    assert_allclose(pipeline.inverse_transform(pipeline.transform(x)), x, atol=1e-14)

    class Bad:
        input_dimension = 2
        output_dimension = 2

        def fit(self, samples):
            return self

        def transform(self, values):
            return np.ones((1, 2))  # Drops training rows.

    with pytest.raises(ValueError, match="leading dimensions"):
        tx.fit(x, Bad())
    with pytest.raises(ValueError, match="leading dimensions"):
        tx.Pipeline((Bad(),)).fit(x)


def test_invalid_result_objects_and_capabilities():
    with pytest.raises(ValueError):
        tx.AffineTransform([1., 2.], np.eye(3), np.eye(3))
    singular = tx.AffineTransform([0., 0.], np.zeros((2, 2)), np.eye(2))
    assert not singular.is_invertible
    with pytest.raises(ValueError):
        singular.inverse_transform([1., 2.])
    with pytest.raises(ValueError):
        tx.Subspace([[1., 1.], [0., 0.]])
    with pytest.raises(ValueError):
        tx.GaussianFactorModel([0., 0.], [[1.], [1.]], [0., 1.])
    with pytest.raises(ValueError):
        tx.FeatureMatrix([[1], [2]], [1, 0], ("x",))
    with pytest.raises(ValueError):
        tx.FeatureMatrix([[1, 2]], [0], ("x", "x"))
    with pytest.raises(ValueError):
        tx.DifferenceResult([1, 2], ([0],), ([0, 1],), lag=1)
    with pytest.raises(ValueError):
        tx.PCAResult([0, 0], np.eye(2), [2, 2], 1, nobs=3)
    with pytest.raises(ValueError):
        tx.FittedPipeline((tx.standardize([[1., 2.]]), tx.standardize([[1.]])))


def test_short_sequences_and_invalid_feature_positions():
    with pytest.raises(ValueError):
        tx.difference([1, 2, 3], lag=2, order=2)
    with pytest.raises(ValueError):
        tx.lag_features([1, 2], (2,))
    with pytest.raises(ValueError):
        tx.fourier_features(2, 12, start=-1)
    with pytest.raises(ValueError):
        tx.seasonal_features(0, 4)
    with pytest.raises(ValueError):
        tx.difference([[1, 2], [3, 4]]).inverse_transform([1, 2])


def test_import_is_independent_and_public_exports_are_complete():
    code = """
import sys
from convexity.math import transforms
for name in ('convexity.math.numerical', 'convexity.math.estimation',
             'convexity.math.distributions', 'torch', 'sklearn', 'statsmodels', 'scipy', 'pandas'):
    assert name not in sys.modules, name
assert 'transforms' in dir(__import__('convexity.math', fromlist=['']))
for name in transforms.__all__:
    assert hasattr(transforms, name), name
"""
    subprocess.run([sys.executable, "-c", code], check=True)
