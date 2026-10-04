"""Nonlinear recovery, ensemble determinism and frozen feature preprocessing."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from typing import Any

from convexity.math import learning as learn, transforms as tx


@pytest.mark.parametrize("tool", [learn.DecisionTree(max_depth=5), learn.RandomForest(n_estimators=35, max_depth=6),
                                   learn.ExtraTrees(n_estimators=35, max_depth=6), learn.GradientBoosting(n_estimators=70)])
def test_tree_conditional_mean_recovery_and_snapshot(tool):
    rng = np.random.default_rng(16)
    x = rng.uniform(-2, 2, (700, 2))
    mean = x[:, 0] ** 2 + (x[:, 1] > 0)
    y = mean + .1 * rng.normal(size=len(x))
    a = tool.fit(x[:500], y[:500])
    b = tool.fit(x[:500], y[:500])
    predictions = a.predict_mean(x[500:])
    assert np.mean((predictions - mean[500:]) ** 2) < .15
    assert_allclose(predictions, b.predict_mean(x[500:]), atol=0, rtol=0)
    tool.fit(x[:500], -y[:500])
    assert_allclose(predictions, a.predict_mean(x[500:]), atol=0, rtol=0)
    x[:500] = 1e9
    y[:] = 1e9
    assert_allclose(predictions, a.predict_mean(x[500:]))
    assert a.predict_mean(x[500:506].reshape(2, 3, 2)).shape == (2, 3)
    with pytest.raises(NotImplementedError):
        a.predict_quantiles(x[500:])


@pytest.mark.parametrize("kind", [learn.DecisionTree, learn.RandomForest, learn.ExtraTrees, learn.GradientBoosting])
def test_tree_classification_probabilities_and_class_order(kind):
    rng = np.random.default_rng(17)
    x = rng.uniform(-2, 2, (300, 2))
    y = np.where(x[:, 0] ** 2 + x[:, 1] ** 2 > 2, "outside", "inside")
    result = kind(task="classification", n_estimators=30, max_depth=5).fit(x[:240], y[:240])
    assert np.array_equal(result.classes, ["inside", "outside"])
    probabilities = result.predict_proba(x[240:])
    assert_allclose(probabilities.sum(axis=-1), 1)
    assert np.mean(result.predict(x[240:]) == y[240:]) > .8


@pytest.mark.parametrize("shape", [(1,), (2,)])
@pytest.mark.parametrize("kind", [learn.DecisionTree, learn.RandomForest, learn.ExtraTrees])
def test_tree_multioutput_and_weighted_leaf_mean(kind, shape):
    x = np.zeros((10, 2))
    y = np.arange(10 * shape[0]).reshape((10,) + shape)
    w = np.arange(1, 11)
    result = kind(n_estimators=1).fit(x, y, sample_weight=w)
    assert result.predict_mean(x[0]).shape == shape
    if kind != learn.RandomForest:  # forests bootstrap observations before weighting
        assert_allclose(result.predict_mean(x[0]), np.average(y, weights=w, axis=0))


def test_pipeline_matches_manual_fit_and_never_refits_test_rows():
    rng = np.random.default_rng(18)
    training = rng.normal(size=(100, 4)) * [1., 2., 3., 4.]
    response = training @ [1., 0., 2., -1.]
    test = rng.normal(size=(20, 4)) + 30
    spec = learn.Pipeline((tx.Standardize(), tx.PCA(3)), learn.Ridge(alpha=.05))
    model: Any = spec.fit(training, response)
    preprocessing = tx.Pipeline(spec.transforms).fit(training)
    manual = spec.learner.fit(preprocessing.transform(training), response)
    assert_allclose(model.predict_mean(test), manual.predict_mean(preprocessing.transform(test)))
    old = model.predict_mean(test)
    spec.fit(training + 100, response)
    assert_allclose(model.predict_mean(test), old)
    assert model.input_dimension == 4
    assert model.coefficients is None  # coefficients live in transformed coordinates
    assert manual.coefficients is not None and model.adapter.model.coefficients is not None
    assert_allclose(model.adapter.model.coefficients, manual.coefficients)


@pytest.mark.parametrize("tool", [learn.LogisticRegression(), learn.QuantileRegression((.2, .8)),
                                   learn.GeneralizedLinearModel("poisson")])
def test_pipeline_propagates_capabilities_and_offsets(tool):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(100, 2))
    y = rng.poisson(2, 100)
    kwargs = {"offset": np.zeros(100)} if isinstance(tool, learn.GeneralizedLinearModel) else {}
    result: Any = learn.fit(x, y, learn.Pipeline((tx.Standardize(),), tool), **kwargs)
    assert result.capabilities == result.adapter.model.capabilities
    for operation in result.capabilities:
        output = result._evaluate(operation, x[:4], np.zeros(4) if kwargs else None)
        assert len(output) == 4
