"""Learning input, result, ownership, extension and lazy import boundaries."""

from dataclasses import FrozenInstanceError
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import learning as learn


@pytest.mark.parametrize("bad", [[], [1, 2], [[True]], [["1"]], [[1j]], [[np.nan]], [[np.inf]], np.empty((2, 0))])
@pytest.mark.parametrize("tool", [learn.LinearRegression(), learn.Ridge(), learn.QuantileRegression(),
                                   learn.GeneralizedLinearModel(), learn.DecisionTree(), learn.NeuralRegressor()])
def test_invalid_training_features(bad, tool):
    with pytest.raises((TypeError, ValueError)):
        tool.fit(bad, [1.])


@pytest.mark.parametrize("bad", [[], [True, False], ["1", "2"], [1j, 0j], [np.nan, 1], [np.inf, 1], [[1]], [1, 2, 3]])
def test_invalid_regression_response(bad):
    with pytest.raises((TypeError, ValueError)):
        learn.LinearRegression().fit([[1], [2]], bad)


@pytest.mark.parametrize("bad", [[0, 0], [-1, 2], [1], [np.nan, 1], [True, True], [[1], [2]]])
def test_invalid_weights(bad):
    with pytest.raises((TypeError, ValueError)):
        learn.ridge([[1], [2]], [1, 2], sample_weight=bad)


@pytest.mark.parametrize("constructor,kwargs", [
    (learn.Ridge, {"alpha": 0}), (learn.Lasso, {"alpha": -1}),
    (learn.ElasticNet, {"l1_ratio": 1.1}), (learn.LinearRegression, {"penalty": "other"}),
    (learn.Ridge, {"fit_intercept": 1}), (learn.Lasso, {"maxiter": 0}),
    (learn.LogisticRegression, {"C": 0}), (learn.LogisticRegression, {"tolerance": 0}),
    (learn.QuantileRegression, {"quantiles": [.5, .5]}), (learn.QuantileRegression, {"quantiles": [0, 1]}),
    (learn.QuantileRegression, {"quantiles": [.9, .1]}), (learn.QuantileRegression, {"quantiles": []}),
    (learn.GeneralizedLinearModel, {"family": "other"}), (learn.GeneralizedLinearModel, {"maxiter": True}),
    (learn.DecisionTree, {"max_depth": 0}), (learn.RandomForest, {"n_estimators": 0}),
    (learn.ExtraTrees, {"task": "other"}), (learn.GradientBoosting, {"learning_rate": 0}),
    (learn.RandomForest, {"seed": 2**32}), (learn.NeuralRegressor, {"hidden_layers": [0]}),
    (learn.NeuralRegressor, {"activation": "other"}), (learn.NeuralClassifier, {"check_convergence": 1}),
])
def test_invalid_specifications(constructor, kwargs):
    with pytest.raises((TypeError, ValueError)):
        constructor(**kwargs)


@pytest.mark.parametrize("family,y,kwargs", [
    ("poisson", [-1, 0, 1, 2], {}), ("poisson", [.1, 0, 1, 2], {}),
    ("gamma", [0, 1, 2, 3], {}), ("binomial", [0, 2, 1, 0], {}),
    ("binomial", [0, .5, 1, 0], {}), ("binomial", [0, 1, 1, 0], {"trials": [0, 1, 1, 1]}),
    ("poisson", [0, 1, 1, 0], {"trials": [1, 1, 1, 1]}),
    ("gaussian", [0, 1, 1, 0], {"frequency_weights": [.5, 1, 1, 1]}),
    ("gaussian", [0, 1, 1, 0], {"offset": [0]}),
])
def test_invalid_glm_domains(family, y, kwargs):
    with pytest.raises((TypeError, ValueError)):
        learn.GeneralizedLinearModel(family).fit(np.arange(4.)[:, None], y, **kwargs)


def test_rank_deficient_glm_and_scalar_only_tools():
    with pytest.raises(ValueError, match="full-rank"):
        learn.GeneralizedLinearModel().fit(np.ones((10, 1)), np.arange(10.))
    for tool in [learn.GeneralizedLinearModel(), learn.QuantileRegression(), learn.GradientBoosting()]:
        with pytest.raises(ValueError):
            tool.fit(np.arange(10.)[:, None], np.ones((10, 1)))


@pytest.mark.parametrize("bad", [[1, 1, 1], [0., .1, 1.], [True, False, True], [0, np.nan, 1], ["", "a", "b"]])
def test_invalid_class_labels(bad):
    with pytest.raises(ValueError):
        learn.LogisticRegression().fit([[1], [2], [3]], bad)


def test_copied_state_and_readonly_predictions():
    x = np.arange(12.).reshape(6, 2)
    y = np.arange(6.)
    alpha: Any = np.array(.2)
    spec: Any = learn.Ridge(alpha=alpha)
    alpha[...] = 99
    assert spec.alpha == .2
    fitted: Any = spec.fit(x, y)
    before = fitted.predict(x)
    x[:] = 500
    y[:] = 500
    for value in (fitted.coefficients, fitted.intercept, before):
        assert not value.flags.writeable
    with pytest.raises(FrozenInstanceError):
        fitted.nobs = 2
    levels: Any = [.1, .5, .9]
    q = learn.QuantileRegression(levels)
    levels[0] = 0
    assert q.quantiles == (.1, .5, .9)


def test_user_learner_and_adapter_contracts():
    class Constant:
        input_dimension = 2
        output_shape = ()
        capabilities = frozenset({"mean"})

        def evaluate(self, operation, features, *, offset=None):
            assert not features.flags.writeable
            return np.full(len(features), 3.)

    class Custom:
        def fit(self, features, response):
            assert not features.flags.writeable
            return learn.FittedModel(Constant())

    fitted = learn.fit([[1, 2], [3, 4]], [1, 2], Custom())
    assert_allclose(fitted.predict_mean([3, 4]), 3)
    with pytest.raises(NotImplementedError):
        fitted.predict_proba([3, 4])
    with pytest.raises(ValueError):
        fitted.predict([3])
    with pytest.raises(ValueError):
        fitted.predict([3, 4], offset=1.)
    with pytest.raises(ValueError):
        learn.fit([[1], [2]], [1, 2], Custom())

    class Bad(Constant):
        def evaluate(self, operation, features, *, offset=None):
            return [np.nan] * len(features)

    with pytest.raises(ValueError, match="finite"):
        learn.FittedModel(Bad()).predict([[1, 2]])

    class BadShape(Constant):
        def evaluate(self, operation, features, *, offset=None):
            return np.ones((len(features), 1))

    with pytest.raises(ValueError, match="output shape"):
        learn.FittedModel(BadShape()).predict([[1, 2]])


@pytest.mark.parametrize("values", [[[-.1, 1.1]], [[.2, .2]], [[1.]], [[np.nan, .5]]])
def test_invalid_custom_probabilities(values):
    class Adapter:
        input_dimension = 1
        output_shape = ()
        capabilities = frozenset({"probability"})

        def evaluate(self, operation, features, *, offset=None):
            return values

    with pytest.raises(ValueError):
        learn.FittedModel(Adapter(), classes=["a", "b"]).predict_proba([[1.]])


def test_quantile_crossings_are_retained():
    class Crossing:
        input_dimension = 1
        output_shape = ()
        capabilities = frozenset({"quantile"})

        def evaluate(self, operation, features, *, offset=None):
            return np.tile([2., 1.], (len(features), 1))

    model = learn.FittedModel(Crossing(), quantiles=[.1, .9])
    assert_allclose(model.predict_quantiles([0.]), [2., 1.])


def test_lazy_imports_and_public_exports():
    code = """
import sys
from convexity.math import learning
for name in ('scipy', 'sklearn', 'statsmodels', 'torch', 'pandas',
             'convexity.math.estimation', 'convexity.math.transforms'):
    assert name not in sys.modules, name
for name in learning.__all__:
    assert hasattr(learning, name), name
assert 'learning' in dir(__import__('convexity.math', fromlist=['']))
"""
    subprocess.run([sys.executable, "-c", code], check=True)
