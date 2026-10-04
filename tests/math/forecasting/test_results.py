import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import forecasting as f
from convexity.math import distributions as d


def test_point_and_quantiles_do_not_invent_laws():
    point = f.ForecastResult([[1], [2]])
    for method in (point.mean, point.variance, point.covariance, point.distribution):
        with pytest.raises(NotImplementedError):
            method()
    q = f.ForecastResult([[1], [2]], quantile_levels=[.1, .5, .9],
                         quantile_values=[[[0, 1, 2]], [[1, 2, 3]]])
    assert_allclose(q.interval(.8), [[[0, 2]], [[1, 3]]])
    with pytest.raises(NotImplementedError):
        q.quantile(.2)
    with pytest.raises(NotImplementedError):
        q.sample(5, rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        f.ForecastResult([[1]], quantile_levels=[.1, .9], quantile_values=[[[2, 1]]])


def test_joint_gaussian_formula_samples_and_selection():
    mean = np.array([[1., 2.], [3., 4.]])
    cov = np.array([[1, .3, .6, 0], [.3, 2, 0, .4], [.6, 0, 3, 0], [0, .4, 0, 4.]])
    law = f.ForecastResult(mean, mean_values=mean, joint_covariance=cov, gaussian=True)
    draws = law.sample(40000, rng=np.random.default_rng(1))
    assert_allclose(draws.mean(axis=0), mean, atol=.03)
    assert_allclose(np.cov(draws.reshape(-1, 4), rowvar=False), cov, atol=.05)
    assert_allclose(law.variance(), np.diag(cov).reshape(2, 2))
    assert_allclose(law.marginal(1, 0).mean(), 3)
    assert_allclose(law.select([1]).covariance(), cov[2:, 2:])
    assert law.log_density(mean).shape == ()
    assert law.sample(0, rng=np.random.default_rng(0)).shape == (0, 2, 2)
    marginal = f.ForecastResult(mean, mean_values=mean, variance_values=np.ones_like(mean), gaussian=True)
    assert 'sample' not in marginal.capabilities
    with pytest.raises(NotImplementedError):
        marginal.covariance()
    with pytest.raises(ValueError):
        f.ForecastResult([[0], [0]], mean_values=[[0], [0]], joint_covariance=[[1, 2], [2, 1]], gaussian=True)


def test_singular_gaussian_and_empirical_paths():
    law = f.ForecastResult([[1], [2]], mean_values=[[1], [2]], joint_covariance=np.zeros((2, 2)), gaussian=True)
    assert_allclose(law.sample(3, rng=np.random.default_rng(1)), np.array([[[1], [2]]] * 3))
    assert law.log_density([[1], [2]]) == 0
    assert law.log_density([[1], [3]]) == -np.inf
    assert_allclose(law.marginal(0).sample(2, rng=np.random.default_rng(1)), 1)
    paths = np.array([[[1., 2.], [3., 4.]], [[3., 4.], [5., 6.]]])
    result = f.ForecastResult.from_paths(paths)
    paths[:] = 999
    assert_allclose(result.mean(), [[2, 3], [4, 5]])
    assert_allclose(result.variance(), 1)
    assert_allclose(result.covariance(), np.ones((4, 4)))
    assert result.paths is not None
    assert_allclose(result.quantile(.5), result.paths[0])
    assert result.probability(lambda path: path[0, 0] > 2) == .5
    assert_allclose(result.marginal(0, 0).quantile(.5), 1)
    assert result.select([1]).sample(4, rng=np.random.default_rng(0)).shape == (4, 1, 2)
    with pytest.raises(ValueError):
        result.probability(lambda path: path > 2)  # pyright: ignore[reportArgumentType] # Exercise runtime rejection.


def test_arbitrary_convexity_law():
    law = d.multivariate_student_t(5, [1, 2], [[1, .2], [.2, 1]])
    forecast = f.ForecastResult([[1], [2]], law=law)
    assert_allclose(forecast.mean()[:, 0], [1, 2])
    assert_allclose(forecast.covariance(), np.array([[1, .2], [.2, 1]]) * 5 / 3)
    assert forecast.sample(3, rng=np.random.default_rng(0)).shape == (3, 2, 1)
    assert forecast.select([1]).shape == (1, 1)


def test_exact_scalar_marginals_preserve_supplied_joint_dependence():
    mean = np.array([1., 2., 3.])
    covariance = np.array([[1., .3, .4], [.3, 2., .5], [.4, .5, 3.]])
    joint = d.multivariate_normal(mean, covariance)
    marginals = tuple(d.normal(mu, np.sqrt(var)) for mu, var in zip(mean, np.diag(covariance)))
    result = f.ForecastResult(mean[:, None], law=joint, marginal_laws=marginals)
    assert_allclose(result.covariance(), covariance)
    assert_allclose(result.quantile(.5)[:, 0], mean)
    assert_allclose(result.log_density(mean[:, None]), joint.log_density(mean))
    draws = result.sample(40000, rng=np.random.default_rng(19))[:, :, 0]
    assert_allclose(np.cov(draws, rowvar=False), covariance, atol=.05)
    selected = result.select([0, 2])
    assert_allclose(selected.covariance(), covariance[np.ix_([0, 2], [0, 2])])
    assert_allclose(selected.interval(.9), result.interval(.9)[[0, 2]])
    independent = f.ForecastResult.from_independent(marginals, (3, 1))
    assert_allclose(independent.covariance(), np.diag(np.diag(covariance)))
