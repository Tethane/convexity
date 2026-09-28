import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from convexity.math import estimation as est


def design_sample():
    rng = np.random.default_rng(312)
    x = np.linspace(-2, 2, 100)
    design = np.column_stack((np.ones(len(x)), x))
    y = design @ [1., 2.] + rng.normal(0, .2, len(x))
    return design, y


@pytest.mark.parametrize("method", ["trf", "dogbox", "lm"])
def test_nonlinear_least_squares_recovers_exponential(method):
    x = np.linspace(0, 3, 60)
    y = 2.5 * np.exp(-.8 * x)
    result = est.least_squares(lambda t: t[0] * np.exp(-t[1] * x) - y, [1., 1.],
                               jacobian=lambda t: np.column_stack((np.exp(-t[1] * x), -t[0] * x * np.exp(-t[1] * x))),
                               estimator=est.LeastSquares(method=method, absolute_sigma=True))
    assert_allclose(result.parameters, [2.5, .8], atol=1e-7)
    assert result.covariance is not None
    assert result.diagnostics.rank == 2


def test_linear_and_nonlinear_covariance_scaling_and_t_interval():
    x, y = design_sample()
    linear = est.linear_least_squares(x, y, names=("intercept", "slope"))
    nonlinear = est.least_squares(lambda t: x @ t - y, [0., 0.], jacobian=lambda t: x)
    assert nonlinear.covariance is not None and linear.covariance is not None
    assert_allclose(nonlinear.parameters, linear.parameters, atol=1e-8)
    assert_allclose(nonlinear.covariance, linear.covariance, atol=1e-16)
    absolute = est.least_squares(lambda t: (x @ t - y) / .2, [0., 0.],
                                 jacobian=lambda t: x / .2, absolute_sigma=True)
    assert absolute.covariance is not None
    assert_allclose(absolute.covariance, .2**2 * np.linalg.inv(x.T @ x), atol=1e-16)
    assert absolute.df is None
    assert linear.df == len(y) - 2
    interval = est.wald_interval(linear, level=.9)
    assert_allclose(interval.upper, linear.parameters + stats.t.ppf(.95, len(y) - 2) * linear.standard_errors)


def test_wls_gls_whitening_equivalence():
    x, y = design_sample()
    weights = np.linspace(.5, 2, len(y))
    weighted = est.linear_least_squares(x, y, weights=weights)
    diagonal = est.linear_least_squares(x, y, sigma=np.diag(1 / weights))
    whitened = est.linear_least_squares(x * np.sqrt(weights)[:, None], y * np.sqrt(weights))
    assert weighted.covariance is not None and whitened.covariance is not None
    assert_allclose(weighted.parameters, diagonal.parameters)
    assert_allclose(weighted.covariance, whitened.covariance)
    sigma = .5 ** np.abs(np.subtract.outer(np.arange(len(y)), np.arange(len(y))))
    full = est.linear_least_squares(x, y, sigma=sigma)
    chol = np.linalg.cholesky(sigma)
    reference = est.linear_least_squares(np.linalg.solve(chol, x), np.linalg.solve(chol, y))
    assert full.covariance is not None and reference.covariance is not None
    assert_allclose(full.parameters, reference.parameters)
    assert_allclose(full.covariance, reference.covariance, atol=1e-16)


@pytest.mark.parametrize("kind", ["HC0", "HC1", "HC2", "HC3"])
def test_heteroskedastic_covariance_sandwich_identity(kind):
    x, y = design_sample()
    result = est.linear_least_squares(x, y, covariance_type=kind)
    assert result.residuals is not None and result.covariance is not None
    bread = np.linalg.inv(x.T @ x)
    hat_diagonal = np.sum((x @ bread) * x, axis=1)
    scale = result.residuals**2
    if kind == "HC1":
        scale *= len(y) / (len(y) - 2)
    elif kind == "HC2":
        scale /= 1 - hat_diagonal
    elif kind == "HC3":
        scale /= (1 - hat_diagonal)**2
    assert_allclose(result.covariance, bread @ (x.T @ (scale[:, None] * x)) @ bread)
    assert result.df is None


def test_rank_deficient_and_saturated_designs_have_no_standard_errors():
    x = np.ones((6, 2))
    result = est.linear_least_squares(x, np.arange(6))
    assert result.diagnostics.rank == 1
    assert result.covariance is None
    saturated = est.linear_least_squares(np.eye(2), [1., 2.])
    assert saturated.covariance is None
    nonlinear = est.least_squares(lambda t: np.ones(6) * t.sum() - np.arange(6), [0., 0.],
                                   jacobian=lambda t: x)
    assert nonlinear.covariance is None


@pytest.mark.parametrize("loss", ["huber", "tukey"])
@pytest.mark.parametrize("covariance_type", ["H1", "H2", "H3"])
def test_robust_linear_resists_outliers(loss, covariance_type):
    x, y = design_sample()
    y[-10:] += 30
    ordinary = est.linear_least_squares(x, y)
    robust = est.robust_linear(x, y, loss=loss, covariance_type=covariance_type)
    assert np.linalg.norm(robust.parameters - [1., 2.]) < .15
    assert np.linalg.norm(ordinary.parameters - [1., 2.]) > 3
    assert robust.covariance is not None
    assert robust.diagnostics.converged
    assert robust.objective is None


@pytest.mark.parametrize("loss", ["soft_l1", "huber", "cauchy", "arctan"])
@pytest.mark.parametrize("analytic_jacobian", [False, True])
def test_robust_nonlinear_estimation_does_not_attach_ols_covariance(loss, analytic_jacobian):
    y = np.r_[np.linspace(-.1, .1, 40), 30.]
    matrix = np.ones((len(y), 1))
    result = est.least_squares(lambda t: t[0] - y, [0.], loss=loss, f_scale=.2,
                               jacobian=(lambda t: matrix) if analytic_jacobian else None)
    assert abs(result.parameters[0]) < .02
    assert result.covariance is None
    assert result.residuals is not None
    assert_allclose(result.residuals, result.parameters[0] - y)
    assert_allclose(matrix, 1.)


def test_active_bounds_no_covariance_and_budget_failure():
    result = est.least_squares(lambda t: t - [2.], est.ParameterSpace([0.], upper=1.))
    assert_allclose(result.parameters, [1.])
    assert result.covariance is None
    with pytest.raises(est.ConvergenceError):
        est.least_squares(lambda t: [t[0]**2 - 4], [1.], estimator=est.LeastSquares(max_evaluations=1))


@pytest.mark.parametrize("ratio", [0., .3, 1.])
def test_regularization_has_common_objective_and_orthogonal_closed_form(ratio):
    n = 4
    x = np.eye(n) * np.sqrt(n)
    z = np.array([3., -.2, -1., 0.])
    y = x @ z
    alpha = .4
    result = est.regularized_linear(x, y, alpha=alpha, l1_ratio=ratio)
    expected = np.sign(z) * np.maximum(abs(z) - alpha * ratio, 0) / (1 + alpha * (1 - ratio))
    assert_allclose(result.parameters, expected, atol=1e-10)
    assert result.objective is not None
    assert_allclose(result.objective, np.sum((y - x @ expected)**2) / (2 * n)
                    + est.elastic_net_penalty(alpha, ratio)(expected))
    assert result.covariance is None


def test_named_regularized_constructors_and_failure():
    x, y = design_sample()
    for wrapper, ratio in [(est.ridge, 0), (est.lasso, 1), (est.elastic_net, .5)]:
        assert_allclose(wrapper(x, y, alpha=.2).parameters,
                        est.regularized_linear(x, y, alpha=.2, l1_ratio=ratio).parameters)
    rng = np.random.default_rng(42)
    correlated = rng.normal(size=(60, 10))
    correlated[:, 1] += correlated[:, 0] * 10
    with pytest.raises(est.ConvergenceError):
        est.regularized_linear(correlated, rng.normal(size=60), alpha=.0001, l1_ratio=.5, maxiter=1)
