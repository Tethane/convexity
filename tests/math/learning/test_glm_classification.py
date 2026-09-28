"""GLM link/count semantics and logistic class-probability contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, softmax

from convexity.math import learning as learn


def test_gaussian_glm_matches_ols_and_frequency_replication():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 2))
    y = 1 + x @ [2, 3] + rng.normal(size=100)
    weights = rng.integers(0, 5, 100)
    result = learn.GeneralizedLinearModel().fit(x, y, frequency_weights=weights)
    ols = learn.linear_regression(np.repeat(x, weights, axis=0), np.repeat(y, weights))
    assert_allclose(result.predict(x), ols.predict(x))


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_log_link_generated_recovery_and_required_offsets(family):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(5000, 2)) * .4
    offset = rng.uniform(-.5, .5, len(x))
    mean = np.exp(.8 + x @ [.6, -.5] + offset)
    y = rng.poisson(mean) if family == "poisson" else rng.gamma(4, mean / 4)
    result = learn.GeneralizedLinearModel(family).fit(x, y, offset=offset)
    assert_allclose(result.coefficients, [.6, -.5], atol=.08)
    assert_allclose(result.intercept, .8, atol=.03)
    assert_allclose(result.predict_mean(x, offset=offset),
                    np.exp(x @ result.coefficients + result.intercept + offset))
    assert result.predict_mean(x[:6].reshape(2, 3, 2), offset=offset[:6].reshape(2, 3)).shape == (2, 3)
    with pytest.raises(ValueError, match="explicit prediction offset"):
        result.predict(x)
    with pytest.raises(ValueError, match="leading axes"):
        result.predict(x, offset=0.)


def test_grouped_binomial_matches_expanded_bernoulli():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 2))
    trials = rng.integers(10, 30, len(x))
    successes = rng.binomial(trials, expit(-.3 + x @ [.8, -.5]))
    model = learn.GeneralizedLinearModel("binomial")
    grouped = model.fit(x, successes, trials=trials)
    expanded_x = np.repeat(x, trials, axis=0)
    expanded_y = np.concatenate([np.r_[np.ones(k), np.zeros(n-k)] for k, n in zip(successes, trials)])
    expanded = model.fit(expanded_x, expanded_y)
    assert_allclose(grouped.coefficients, expanded.coefficients, atol=1e-8)
    probability = expit(x @ grouped.coefficients + grouped.intercept)
    assert_allclose(grouped.predict(x), probability)
    assert_allclose(grouped.predict_proba(x), np.column_stack([1-probability, probability]))
    assert_allclose(grouped.predict_class(x), probability > .5)
    assert_allclose(grouped.coefficients, [.8, -.5], atol=.12)


def test_binomial_offset_and_frequency_match_statsmodels():
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.families import Binomial
    rng = np.random.default_rng(14)
    x = rng.normal(size=(100, 2))
    y = rng.binomial(1, .4, 100)
    offset = rng.normal(size=100) * .1
    w = rng.integers(1, 5, 100)
    result = learn.GeneralizedLinearModel("binomial").fit(x, y, offset=offset, frequency_weights=w)
    design = np.column_stack([np.ones(len(x)), x])
    reference = GLM(y, design, family=Binomial(), offset=offset, freq_weights=w).fit(tol=1e-8)
    assert_allclose(result.predict(x[:10], offset=offset[:10]), reference.predict(design[:10], offset=offset[:10]))


@pytest.mark.parametrize("classes", [np.array([-3, 7]), np.array(["bear", "bull", "flat"])])
def test_logistic_class_order_and_links(classes):
    rng = np.random.default_rng(30)
    x = rng.normal(size=(600, 3))
    beta = rng.normal(size=(3, len(classes)))
    probabilities = softmax(x @ beta, axis=-1)
    labels = np.array([classes[rng.choice(len(classes), p=p)] for p in probabilities])
    fitted = learn.logistic_regression(x, labels, C=2.)
    assert_allclose(fitted.predict_proba(x).sum(axis=-1), 1)
    assert np.array_equal(fitted.classes, np.sort(classes))
    logits = x @ fitted.coefficients + fitted.intercept
    expected = np.column_stack([1-expit(logits[:, 0]), expit(logits[:, 0])]) if len(classes) == 2 else softmax(logits, axis=-1)
    assert_allclose(fitted.predict_proba(x), expected)
    assert np.array_equal(fitted.predict(x), fitted.classes[expected.argmax(axis=1)])
    assert fitted.predict_proba(np.empty((0, 3))).shape == (0, len(classes))
    with pytest.raises(NotImplementedError):
        fitted.predict_mean(x)


def test_logistic_weight_replication_and_zero_class_removal():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(100, 2))
    y = rng.integers(0, 2, 100)
    w = rng.integers(1, 4, 100)
    w[-1], y[-1] = 0, 9
    tool = learn.LogisticRegression(tolerance=1e-10)
    a = tool.fit(x, y, sample_weight=w)
    b = tool.fit(np.repeat(x, w, axis=0), np.repeat(y, w))
    assert_allclose(a.predict_proba(x), b.predict_proba(x), atol=1e-9)
    assert np.array_equal(a.classes, [0, 1])


def test_glm_and_logistic_failed_convergence():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 3))
    y = rng.poisson(np.exp(x[:, 0]))
    with pytest.raises(learn.ConvergenceError):
        learn.GeneralizedLinearModel("poisson", maxiter=1).fit(x, y)
    with pytest.raises(learn.ConvergenceError):
        learn.LogisticRegression(maxiter=1, tolerance=1e-15).fit(x, (y > 1).astype(int))
    separated = np.arange(-20., 20.)[:, None]
    with pytest.raises(learn.ConvergenceError):
        learn.GeneralizedLinearModel("binomial").fit(separated, (separated[:, 0] > 0).astype(int))
