"""Analytic objective identities, weighted fits, quantiles and convergence."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import estimation as est, learning as learn


@pytest.mark.parametrize("shape", [(), (1,), (3,)])
@pytest.mark.parametrize(
    "tool",
    [
        learn.LinearRegression(),
        learn.Ridge(alpha=0.1),
        learn.Lasso(alpha=0.01),
        learn.ElasticNet(alpha=0.01),
    ],
)
def test_linear_shapes_and_independent_targets(shape, tool):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(120, 4))
    beta = rng.normal(size=(4,) + shape)
    y = x @ beta + 3.0
    fitted = tool.fit(x, y)
    assert fitted.output_shape == shape
    assert fitted.coefficients.shape == beta.shape
    assert fitted.intercept.shape == shape
    assert_allclose(fitted.predict_mean(x), x @ fitted.coefficients + fitted.intercept)
    assert fitted.predict_mean(x[:6].reshape(2, 3, 4)).shape == (2, 3) + shape
    assert fitted.predict_mean(x[0]).shape == shape
    assert fitted.predict_mean(np.empty((2, 0, 4))).shape == (2, 0) + shape
    if shape:
        for j in range(shape[0]):
            separate = tool.fit(x, y[:, j])
            assert_allclose(
                fitted.predict_mean(x)[:, j], separate.predict_mean(x), atol=1e-10
            )


def test_ols_weighted_normal_equations_and_rank_deficiency():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(50, 3))
    x[:, 2] = x[:, 0]
    y = 2 + x @ [1.0, 2.0, 1.0] + rng.normal(size=50)
    w = np.linspace(0.2, 2.0, 50)
    result = learn.linear_regression(x, y, sample_weight=w)
    design = np.column_stack([np.ones(len(x)), x])
    expected = np.linalg.lstsq(
        design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None
    )[0]
    assert_allclose(result.predict_mean(x), design @ expected, atol=1e-12)
    assert_allclose(design.T @ (w * (y - result.predict_mean(x))), 0, atol=1e-11)


def test_ridge_normalized_penalty_unpenalized_intercept_and_estimation_bridge():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 4))
    y = 5 + x @ [1.0, 0.0, 2.0, -3.0] + rng.normal(size=80)
    weights = rng.uniform(0.1, 2, 80)
    alpha = 0.2
    result = learn.ridge(x, y, alpha=alpha, sample_weight=weights)
    w = weights / weights.sum()
    xm, ym = w @ x, w @ y
    z = x - xm
    expected = np.linalg.solve(
        z.T @ (w[:, None] * z) + alpha * np.eye(4), z.T @ (w * (y - ym))
    )
    assert result.coefficients is not None
    assert_allclose(result.coefficients, expected)
    assert result.intercept is not None
    assert_allclose(result.intercept, ym - xm @ expected)
    for scale in (1e-310, 20.0):
        scaled = learn.ridge(x, y, alpha=alpha, sample_weight=weights * scale)
        assert scaled.coefficients is not None
        assert_allclose(scaled.coefficients, expected)
    for fitter, estimator in [
        (learn.ridge, est.ridge),
        (learn.lasso, est.lasso),
        (learn.elastic_net, est.elastic_net),
    ]:
        assert_allclose(
            fitter(x, y, alpha=alpha, fit_intercept=False).coefficients,
            estimator(x, y, alpha=alpha).parameters,
            atol=1e-8,
        )


@pytest.mark.parametrize("ratio", [0.0, 0.4, 1.0])
def test_elastic_net_closed_form_orthogonal_design(ratio):
    x = np.vstack([np.eye(3), -np.eye(3)]) * np.sqrt(3)
    beta = np.array([2.0, -0.1, -3.0])
    alpha = 0.5
    model = learn.elastic_net(x, 7 + x @ beta, alpha=alpha, l1_ratio=ratio)
    expected = (
        np.sign(beta)
        * np.maximum(np.abs(beta) - alpha * ratio, 0)
        / (1 + alpha * (1 - ratio))
    )
    assert model.coefficients is not None
    assert_allclose(model.coefficients, expected)
    assert model.intercept is not None
    assert_allclose(model.intercept, 7.0)


def test_zero_weights_omit_rows_including_penalty_scale():
    x = np.arange(12.0).reshape(6, 2)
    y = np.array([1.0, 3.0, 2.0, 4.0, 6.0, 1e9])
    w = [1, 1, 1, 1, 1, 0]
    for tool in [
        learn.LinearRegression(),
        learn.Ridge(),
        learn.Lasso(),
        learn.ElasticNet(),
    ]:
        assert_allclose(
            tool.fit(x, y, sample_weight=w).predict_mean(x),
            tool.fit(x[:5], y[:5]).predict_mean(x),
        )


def test_quantile_empirical_intercept_and_weighted_replication():
    x = np.zeros((9, 1))
    y = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 12.0])
    q = (0.2, 0.5, 0.8)
    fitted = learn.quantile_regression(x, y, quantiles=q)
    assert_allclose(fitted.predict_quantiles([0.0]), [-1.0, 2.0, 7.0])
    assert fitted.predict_quantiles(x[:2]).shape == (2, 3)
    weights = np.arange(1, 10)
    weighted = learn.QuantileRegression(q).fit(x, y, sample_weight=weights)
    repeated = learn.QuantileRegression(q).fit(
        np.repeat(x, weights, axis=0), np.repeat(y, weights)
    )
    assert_allclose(weighted.predict_quantiles(x), repeated.predict_quantiles(x))
    with pytest.raises(NotImplementedError):
        fitted.predict_mean(x)


def test_quantile_generated_conditional_recovery_and_pinball_optimum():
    rng = np.random.default_rng(9)
    x = rng.uniform(-1, 1, (1500, 1))
    y = 2 + 3 * x[:, 0] + rng.uniform(-1, 1, len(x))
    result = learn.QuantileRegression((0.1, 0.5, 0.9)).fit(x, y)
    assert result.coefficients is not None
    assert_allclose(result.coefficients, [[3, 3, 3]], atol=0.09)
    assert result.intercept is not None
    assert_allclose(result.intercept, [1.2, 2, 2.8], atol=0.05)
    residual = y[:, None] - result.predict_quantiles(x)
    assert result.quantiles is not None
    levels = result.quantiles
    objective = np.maximum(levels * residual, (levels - 1) * residual).mean(axis=0)
    for shift in [-0.2, 0.2]:
        alternative = residual + shift
        assert np.all(
            np.maximum(levels * alternative, (levels - 1) * alternative).mean(axis=0)
            > objective
        )


def test_penalized_convergence_error_retains_predictor():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(40, 20))
    y = rng.normal(size=40)
    with pytest.raises(learn.ConvergenceError) as error:
        learn.Lasso(alpha=0.001, tolerance=1e-15, maxiter=1).fit(x, y)
    assert not error.value.result.diagnostics.converged
    assert error.value.result.predict_mean(x).shape == y.shape
    inspected = learn.Lasso(
        alpha=0.001, tolerance=1e-15, maxiter=1, check_convergence=False
    ).fit(x, y)
    assert (
        error.value.result.coefficients is not None
        and inspected.coefficients is not None
    )
    assert_allclose(inspected.coefficients, error.value.result.coefficients)
