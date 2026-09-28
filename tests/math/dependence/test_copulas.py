"""Copula margins, Sklar identities, rank recovery and boundary behavior."""

import numpy as np
import pytest
from typing import cast
from numpy.testing import assert_allclose

from convexity.math import dependence as dep, distributions as dist
from convexity.math.dependence._backends.copulas import Adapter


def copulas():
    corr = [[1., .6], [.6, 1.]]
    return [dep.IndependenceCopula(), dep.GaussianCopula(corr), dep.StudentTCopula(corr, 4),
            dep.ClaytonCopula(2), dep.GumbelCopula(2), dep.GumbelCopula(1)]


@pytest.mark.parametrize("copula", copulas())
def test_uniform_margins_sampling_shapes_and_reproducibility(copula):
    x = copula.sample(15000, rng=np.random.default_rng(12))
    assert_allclose(x.mean(axis=0), .5, atol=.008)
    assert_allclose(x.var(axis=0), 1/12, atol=.003)
    assert_allclose(np.mean(x < .1, axis=0), .1, atol=.008)
    a = copula.sample((2, 3), rng=np.random.default_rng(1))
    b = copula.sample((2, 3), rng=np.random.default_rng(1))
    assert_allclose(a, b)
    assert a.shape == (2, 3, 2)
    assert copula.sample(rng=np.random.default_rng(1)).shape == (2,)
    assert copula.sample((2, 0), rng=np.random.default_rng(1)).shape == (2, 0, 2)
    assert copula.log_density(a).shape == (2, 3)
    assert copula.log_density(np.empty((0, 2))).shape == (0,)


@pytest.mark.parametrize("copula", copulas())
def test_cdf_boundaries_and_leading_shapes(copula):
    u = np.array([[0., 1.], [1., 1.], [1., .3], [.7, 1.], [0., .4]])
    assert_allclose(copula.cdf(u, rng=np.random.default_rng(2)), [0, 1, .3, .7, 0])
    u = np.array([[[.3, .6]], [[.5, .5]]])
    first = copula.cdf(u, rng=np.random.default_rng(3))
    assert first.shape == (2, 1)
    assert_allclose(first, copula.cdf(u, rng=np.random.default_rng(3)))
    assert copula.cdf(np.empty((0, 2)), rng=np.random.default_rng(3)).shape == (0,)


@pytest.mark.parametrize("kind", ["gaussian", "student_t"])
def test_elliptical_cdf_identity_and_higher_dimensional_faces(kind):
    rho = .6
    matrix = np.full((3, 3), rho)
    np.fill_diagonal(matrix, 1)
    copula = dep.GaussianCopula(matrix) if kind == "gaussian" else dep.StudentTCopula(matrix, 5)
    expected = .25 + np.arcsin(rho) / (2*np.pi)
    assert_allclose(copula.cdf([.5, .5, 1], rng=np.random.default_rng(7)), expected, atol=3e-4)
    assert_allclose(copula.cdf([1, .37, 1], rng=np.random.default_rng(7)), .37)
    with pytest.raises(TypeError, match="Generator"):
        copula.cdf([.5, .5, .5])


def test_independence_and_tail_coefficients():
    from scipy.stats import t
    u = np.array([[.2, .6], [.9, .9]])
    independent = dep.IndependenceCopula()
    assert_allclose(independent.cdf(u), np.prod(u, axis=1))
    assert_allclose(independent.density(u), 1)
    assert_allclose(dep.GumbelCopula(1).density(u), 1)
    assert independent.tail_dependence() == (0, 0)
    assert_allclose(dep.ClaytonCopula(2).tail_dependence(), [2**(-.5), 0])
    assert_allclose(dep.GumbelCopula(2).tail_dependence(), [0, 2-np.sqrt(2)])
    copula = dep.StudentTCopula([[1, -.3], [-.3, 1]], 4)
    coefficient = 2*t.cdf(-np.sqrt(5*1.3/.7), 5)
    assert_allclose(copula.tail_dependence(), [coefficient, coefficient])
    # Zero shape correlation does not make t coordinates independent.
    assert dep.StudentTCopula(np.eye(2), 4).tail_dependence()[0] > 0


@pytest.mark.parametrize("kind", ["gaussian", "student_t"])
def test_copula_marginal_composition_recovers_elliptical_law(kind):
    corr = np.array([[1., .6], [.6, 1.]])
    loc, scale = np.array([1., -2.]), np.array([2., .5])
    shape = corr * scale[:, None] * scale[None, :]
    if kind == "gaussian":
        copula = dep.GaussianCopula(corr)
        marginals = [dist.normal(mu, sd) for mu, sd in zip(loc, scale)]
        reference = dist.multivariate_normal(loc, shape)
    else:
        copula = dep.StudentTCopula(corr, 5)
        marginals = [dist.student_t(5, mu, sd) for mu, sd in zip(loc, scale)]
        reference = dist.multivariate_student_t(5, loc, shape)
    joint = dep.CopulaDistribution(copula, marginals)
    points = np.array([[[1., -2.], [2., 0.]], [[0., -1.], [3., -3.]]])
    assert_allclose(joint.log_density(points), reference.log_density(points), atol=1e-12)
    assert_allclose(joint.cdf(points, rng=np.random.default_rng(8)),
                    reference.cdf(points, rng=np.random.default_rng(8)), atol=4e-4)
    assert joint.sample((2, 3), rng=np.random.default_rng(9)).shape == (2, 3, 2)
    assert joint.sample(0, rng=np.random.default_rng(9)).shape == (0, 2)
    assert joint.log_density(np.empty((0, 2))).shape == (0,)


def test_joint_support_and_discrete_marginal_rejection():
    joint = dep.CopulaDistribution(dep.IndependenceCopula(), [dist.lognormal(), dist.normal()])
    assert np.isneginf(joint.log_density([-1, 0]))
    assert_allclose(joint.cdf([-1, 0]), 0)
    with pytest.raises(ValueError, match="continuous"):
        dep.CopulaDistribution(dep.IndependenceCopula(), [dist.poisson(3), dist.normal()])


@pytest.mark.parametrize("family", ["gaussian", "student_t", "clayton", "gumbel"])
def test_kendall_parameter_recovery_and_rank_invariance(family):
    models = dict(gaussian=dep.GaussianCopula([[1., .65], [.65, 1.]]),
                  student_t=dep.StudentTCopula([[1., .65], [.65, 1.]], 5),
                  clayton=dep.ClaytonCopula(2), gumbel=dep.GumbelCopula(2))
    uniforms = models[family].sample(6000, rng=np.random.default_rng(16))
    fit = dep.fit_copula(uniforms, family=family, df=5 if family == "student_t" else None)
    transformed = dep.fit_copula(np.exp(uniforms), family=family, df=5 if family == "student_t" else None)
    if family in ("gaussian", "student_t"):
        assert_allclose(cast(Adapter, fit.copula.adapter).correlation, [[1., .65], [.65, 1.]], atol=.035)
        assert_allclose(cast(Adapter, fit.copula.adapter).correlation, cast(Adapter, transformed.copula.adapter).correlation)
    else:
        theta = cast(Adapter, fit.copula.adapter).theta
        assert theta is not None
        assert_allclose(theta, 2, atol=.16)
    assert np.isfinite(fit.log_likelihood)
    assert fit.nobs == 6000


def test_rank_fits_multivariate_repair_and_ties():
    x = np.array([[1, 3], [1, 2], [4, 5], [2, 1]])
    assert_allclose(dep.pseudo_observations(x)[:, 0], [.3, .3, .8, .6])
    data = np.arange(20.).reshape(-1, 1) * np.ones(3)
    with pytest.raises(ValueError):
        dep.fit_copula(data)
    fit = dep.fit_copula(data, repair_correlation=True)
    assert fit.repair_distance > 0
    assert fit.copula.dimension == 3
    assert np.isfinite(fit.log_likelihood)


def test_elliptical_laws_reject_numerically_singular_shapes_at_construction():
    almost_singular = [[1., 1-1e-12], [1-1e-12, 1.]]
    with pytest.raises(ValueError, match="numerically positive definite"):
        dep.GaussianCopula(almost_singular)
    with pytest.raises(ValueError, match="numerically positive definite"):
        dep.StudentTCopula(almost_singular, 5)


@pytest.mark.parametrize("copula", copulas())
def test_density_rejects_boundary_and_invalid_points(copula):
    for u in ([0, .5], [1, .5], [-.1, .5], [.5, np.nan], [.5], [True, False]):
        with pytest.raises((TypeError, ValueError)):
            copula.log_density(u)
    with pytest.raises(ValueError):
        copula.tail_dependence(0, 0)
    with pytest.raises(TypeError):
        copula.sample(2, rng=1)
