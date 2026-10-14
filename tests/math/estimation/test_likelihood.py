import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import distributions as dist, estimation as est


@pytest.mark.parametrize("method", ["auto", "BFGS", "Powell", "SLSQP", "trust-constr"])
def test_gaussian_mean_analytic_solution_information_and_interval(method):
    observations = np.array([-2.0, 0.0, 1.0, 3.0, 4.0])
    sigma = 2.0
    model = est.likelihood(lambda theta: dist.normal(theta[0], sigma), observations)
    space = est.ParameterSpace([0.0], names=("mean",))
    result = est.maximum_likelihood(
        model,
        space,
        hessian=lambda t: [[len(observations) / sigma**2]],
        estimator=est.ScipyMinimizer(method=method),
    )
    assert_allclose(result.parameters, [observations.mean()], atol=2e-6)
    assert result.covariance is not None
    assert_allclose(result.covariance, [[sigma**2 / len(observations)]])
    assert result.named_parameters["mean"] == result.parameters[0]
    assert result.nobs == len(observations)
    assert result.covariance_kind == "observed_information"
    assert result.objective is not None
    assert_allclose(result.objective, -model.log_likelihood(result.parameters))
    interval = est.wald_interval(result)
    assert interval.kind == "confidence"
    assert_allclose(
        interval.upper - result.parameters,
        1.959963984540054 * sigma / np.sqrt(len(observations)),
    )


def test_gaussian_location_and_scale_with_numerical_information():
    y = dist.normal(2.0, 1.3).sample(400, rng=np.random.default_rng(52))
    model = est.likelihood(lambda t: dist.normal(t[0], np.exp(t[1])), y)
    result = est.maximum_likelihood(
        model, est.ParameterSpace([0.0, 0.0], names=("mu", "log_sigma"))
    )
    assert result.covariance is not None
    assert_allclose(result.parameters, [y.mean(), np.log(y.std())], atol=2e-5)
    assert_allclose(
        result.covariance, np.diag([y.var() / len(y), 1 / (2 * len(y))]), atol=2e-7
    )
    scale = est.delta_method(
        result,
        lambda t: np.exp(t[1]),
        jacobian=lambda t: [[0, np.exp(t[1])]],
        names=("sigma",),
    )
    assert scale.covariance is not None
    assert_allclose(scale.covariance, [[y.var() / (2 * len(y))]], rtol=2e-5)


def test_vector_law_and_user_conditional_likelihood():
    data = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 3.0]])
    model = est.likelihood(lambda t: dist.multivariate_normal(t, np.eye(2)), data)
    result = est.maximum_likelihood(
        model, [0.0, 0.0], hessian=lambda t: np.eye(2) * len(data)
    )
    assert_allclose(result.parameters, data.mean(axis=0), atol=2e-6)
    conditional = est.Likelihood(
        data, lambda t, xy: dist.normal(0, 1).log_density(xy[:, 1] - t[0] * xy[:, 0])
    )
    result = est.maximum_likelihood(conditional, [0.0])
    assert_allclose(result.parameters, [19 / 14], atol=2e-6)


@pytest.mark.parametrize(
    "factory,data,initial,expected",
    [
        (lambda t: dist.bernoulli(t[0]), [0, 1, 1, 0, 1], 0.5, 0.6),
        (lambda t: dist.poisson(t[0]), [0, 1, 2, 3, 4], 1.0, 2.0),
    ],
)
def test_discrete_mle(factory, data, initial, expected):
    upper = 1 if expected < 1 else np.inf
    result = est.maximum_likelihood(
        est.likelihood(factory, data),
        est.ParameterSpace([initial], lower=1e-6, upper=upper),
    )
    assert_allclose(result.parameters, [expected], atol=2e-6)
    assert result.covariance is not None


def test_boundary_estimate_has_no_interior_uncertainty():
    result = est.maximum_likelihood(
        est.likelihood(lambda t: dist.poisson(t[0]), [0.0, 0.0, 0.0]),
        est.ParameterSpace([1.0], lower=0.0),
    )
    assert_allclose(result.parameters, [0.0], atol=1e-12)
    assert result.covariance is None
    assert "bound" in result.diagnostics.uncertainty_note
    with pytest.raises(ValueError, match="bound"):
        est.wald_interval(result)


def test_frequency_weights_are_replication_and_zero_rows_are_removed():
    weighted = est.likelihood(
        lambda t: dist.normal(t[0], 2), [1.0, 3.0, 900.0], frequency_weights=[2, 3, 0]
    )
    repeated = est.likelihood(lambda t: dist.normal(t[0], 2), [1.0, 1.0, 3.0, 3.0, 3.0])
    a = est.maximum_likelihood(weighted, [0.0])
    b = est.maximum_likelihood(repeated, [0.0])
    assert a.covariance is not None and b.covariance is not None
    assert_allclose(a.parameters, b.parameters, atol=1e-6)
    assert_allclose(a.covariance, b.covariance, rtol=1e-5)
    assert a.objective is not None and b.objective is not None
    assert_allclose(a.objective, b.objective)
    assert weighted.nobs == 5
    assert_allclose(np.asarray(weighted.data), [1.0, 3.0])


def test_gaussian_map_matches_conjugate_posterior():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = est.likelihood(lambda t: dist.normal(t[0], 2), y)
    prior = est.gaussian_prior(loc=-1.0, scale=3.0)
    precision = len(y) / 4 + 1 / 9
    result = est.maximum_a_posteriori(model, prior, [0.0])
    assert result.covariance is not None
    assert_allclose(result.parameters, [(y.sum() / 4 - 1 / 9) / precision], atol=2e-6)
    assert_allclose(result.covariance, [[1 / precision]], rtol=1e-6)
    assert result.covariance_kind == "laplace"
    assert est.laplace_interval(result).kind == "credible"
    with pytest.raises(ValueError, match="laplace_interval"):
        est.wald_interval(result)


def test_penalty_and_prior_conventions():
    theta = np.array([-2.0, 3.0])
    assert est.l1_penalty(2)(theta) == 10
    assert est.l2_penalty(2)(theta) == 13
    assert est.elastic_net_penalty(2, 0.5)(theta) == 11.5
    assert est.l2_penalty(weights=[0.0, 1.0], center=[1.0, 2.0])(theta) == 0.5
    prior = est.laplace_prior([0.0, 1.0], [2.0, 3.0])
    assert_allclose(prior(theta), -np.log(4) - 1 - np.log(6) - 2 / 3)
    # A user objective can compose a likelihood and selectively weighted penalty.
    objective = lambda t: (t[0] - 3) ** 2 / 2 + est.l2_penalty(2)(t)
    result = est.minimize_estimate(objective, [0.0])
    assert_allclose(result.parameters, [1.0], atol=1e-6)
    assert result.covariance is None


def test_singular_information_does_not_invent_precision():
    model = est.Likelihood([1.0, 2.0], lambda t, y: -((y - t.sum()) ** 2) / 2)
    result = est.maximum_likelihood(
        model, [0.0, 0.0], hessian=lambda t: np.full((2, 2), 2.0)
    )
    assert_allclose(result.parameters.sum(), 1.5, atol=1e-6)
    assert result.covariance is None
    assert "rank" in result.diagnostics.uncertainty_note


def test_nonsmooth_map_matches_soft_thresholding_without_curvature():
    data = np.array([0.1, 0.2, 0.3])
    model = est.likelihood(lambda t: dist.normal(t[0], 1), data)
    prior = est.laplace_prior()
    with pytest.raises(ValueError, match="nonsmooth"):
        est.maximum_a_posteriori(model, prior, [0.0])
    result = est.maximum_a_posteriori(
        model, prior, [0.0], uncertainty=False, estimator=est.ScipyMinimizer("Powell")
    )
    assert_allclose(result.parameters, [0.0], atol=1e-7)
    assert result.covariance is None
