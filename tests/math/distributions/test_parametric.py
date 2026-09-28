"""Library agreement, probability identities, shape and ownership contracts."""

import numpy as np
import pytest
from typing import Any, cast
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from convexity.math import distributions as d


@pytest.mark.parametrize("law,reference", [
    (d.normal(2, 3), stats.norm(2, 3)),
    (d.student_t(7, 2, 3), stats.t(7, 2, 3)),
    (d.lognormal(.2, .7), stats.lognorm(.7, scale=np.exp(.2))),
    (d.bernoulli(.4), stats.bernoulli(.4)),
    (d.binomial(8, .4), stats.binom(8, .4)),
    (d.poisson(3), stats.poisson(3)),
])
def test_named_laws_match_scipy_and_preserve_shapes(law, reference):
    x = np.array([[-1., 0., 1.], [2., 3., 4.]])
    log = reference.logpmf if law.measure == "discrete" else reference.logpdf
    assert_allclose(law.log_density(x), log(x))
    assert_allclose(law.density(x), np.exp(log(x)))
    assert_allclose(law.cdf(x), reference.cdf(x))
    assert_allclose(law.survival(x), reference.sf(x))
    assert_allclose(law.mean(), reference.mean())
    assert_allclose(law.variance(), reference.var())
    assert_allclose(law.moment(2), reference.moment(2))
    assert_allclose(law.expect(lambda x: x*x), law.moment(2), rtol=1e-7)
    p = np.array([[.1, .4], [.7, .9]])
    assert_allclose(law.quantile(p), reference.ppf(p))
    for size in [(), 1, (2, 3), (0, 2)]:
        shape = (size,) if isinstance(size, int) else size
        a = law.sample(size, rng=np.random.default_rng(10))
        b = law.sample(size, rng=np.random.default_rng(10))
        assert a.shape == shape
        assert_array_equal(a, b)
        assert not a.flags.writeable


def test_scalar_boundaries_tail_accuracy_and_divergent_moments():
    assert d.normal().survival(10) > 0
    assert d.normal().log_density(100) < -1000
    assert_array_equal(d.lognormal().density([-np.inf, -1, 0]), [0, 0, 0])
    assert_array_equal(d.binomial(8, .2).quantile([0, 1]), [0, 8])
    assert np.isnan(d.student_t(.5).mean()) or np.isinf(d.student_t(.5).mean())
    assert np.isinf(d.student_t(1.5).variance())
    assert d.normal().cdf([]).shape == (0,)


@pytest.mark.parametrize("factory", [
    lambda: d.normal(scale=0), lambda: d.normal(loc=np.inf),
    lambda: d.normal(loc=cast(Any, "1")), lambda: d.normal(scale=True),
    lambda: d.student_t(0), lambda: d.student_t(np.inf),
    lambda: d.lognormal(log_loc=1000), lambda: d.lognormal(log_scale=-1),
    lambda: d.bernoulli(1.1), lambda: d.binomial(2.5, .3),
    lambda: d.binomial(True, .3), lambda: d.poisson(-1),
    lambda: d.multivariate_normal([0, 0], [[1, 2], [0, 1]]),
    lambda: d.multivariate_normal([0, 0], [[1, 1], [1, 1]]),
    lambda: d.multivariate_student_t(0, [0], [[1]]),
])
def test_invalid_parameters(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize("value", [np.nan, -0.1, 1.1, np.inf, True, "0.5"])
def test_invalid_quantile(value):
    with pytest.raises(ValueError):
        d.normal().quantile(value)


def test_invalid_queries_and_random_state():
    law = d.normal()
    with pytest.raises(ValueError):
        law.density(np.nan)
    with pytest.raises(TypeError):
        law.sample(2, rng=1)
    for size in [-1, True, [2], (1, -1), (1.5,)]:
        with pytest.raises(ValueError):
            law.sample(size, rng=np.random.default_rng())
    with pytest.raises(ValueError):
        law.expect(lambda x: x, conditional=True)
    with pytest.raises(ValueError):
        law.expect(lambda x: [x, x])


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("student", [False, True])
def test_vector_shapes_and_scipy_agreement(dimension, student):
    mean = np.arange(dimension, dtype=float)
    matrix = np.eye(dimension) + .2
    law = d.multivariate_student_t(6, mean, matrix) if student else d.multivariate_normal(mean, matrix)
    generator: Any = stats.multivariate_t if student else stats.multivariate_normal
    reference = generator(mean, matrix, 6) if student else generator(mean, matrix)
    x = np.zeros((2, 3, dimension))
    assert_allclose(law.log_density(x), np.asarray(reference.logpdf(x)).reshape(2, 3))
    assert_allclose(law.mean(), mean)
    assert_allclose(law.covariance(), matrix * (1.5 if student else 1))
    assert_allclose(law.moment(2), law.variance() + law.mean()**2)
    for size in [(), 1, (2, 1), (0, 2)]:
        shape = (size,) if isinstance(size, int) else size
        samples = law.sample(size, rng=np.random.default_rng(30))
        assert samples.shape == shape + (dimension,)
        assert_array_equal(samples, law.sample(size, rng=np.random.default_rng(30)))
    assert law.cdf(x, rng=np.random.default_rng(10)).shape == (2, 3)
    assert law.cdf(mean, rng=np.random.default_rng(10)).shape == ()
    assert law.log_density(np.zeros((0, dimension))).shape == (0,)
    with pytest.raises(ValueError):
        law.density(0)
    with pytest.raises(NotImplementedError):
        law.quantile(.5)
    with pytest.raises(TypeError):
        law.cdf(mean)


def test_vector_ownership_and_regular_conditional_identity():
    mean = np.array([1., 2., 3.])
    matrix = np.array([[4., .3, 1.], [.3, 2., .5], [1., .5, 3.]])
    for law in [d.multivariate_normal(mean, matrix), d.multivariate_student_t(7, mean, matrix)]:
        observed, values = [2, 0], [4., 2.]
        conditional = law.condition(observed, values)
        assert conditional.event_shape == (1,)
        marginal = law.marginal(observed)
        x = np.array([2., 1.5, 4.])
        assert_allclose(conditional.log_density([x[1]]) + marginal.log_density(values), law.log_density(x), atol=1e-12)
        assert_allclose(marginal.mean(), mean[observed])
        with pytest.raises(ValueError):
            law.condition([0, 1, 2], [0, 0, 0])
        with pytest.raises(ValueError):
            law.condition([0, 0], [0, 0])
        with pytest.raises(ValueError):
            law.marginal([-1])
    law = d.multivariate_normal(mean, matrix)
    mean[:] = 99
    matrix[:] = 0
    assert_allclose(law.mean(), [1, 2, 3])
    assert law.covariance()[0, 0] == 4


def test_gaussian_conditioning_known_answer_and_student_scale():
    normal = d.multivariate_normal([0, 0], [[1, .5], [.5, 1]]).condition([1], [2])
    assert_allclose(normal.mean(), [1])
    assert_allclose(normal.covariance(), [[.75]])
    student = d.multivariate_student_t(5, [0, 0], [[1, .5], [.5, 1]]).condition([1], [2])
    # df becomes 6; shape becomes .75*(5+4)/6; covariance adds 6/(6-2).
    assert_allclose(student.mean(), [1])
    assert_allclose(student.covariance(), [[1.6875]])


def test_sampling_moments_and_vector_cdf_identity():
    law = d.multivariate_normal([1, 2], [[2, .5], [.5, 1]])
    samples = law.sample(60000, rng=np.random.default_rng(892))
    assert_allclose(samples.mean(0), law.mean(), atol=.02)
    assert_allclose(np.cov(samples.T, ddof=0), law.covariance(), atol=.025)
    standard = d.multivariate_normal([0, 0], [[1, .5], [.5, 1]])
    assert_allclose(standard.cdf([0, 0], rng=np.random.default_rng(1)), 1/3, atol=2e-5)


@pytest.mark.parametrize("law,point", [(d.bernoulli(0), 0), (d.bernoulli(1), 1),
                                       (d.binomial(5, 0), 0), (d.binomial(5, 1), 5),
                                       (d.binomial(0, .4), 0), (d.poisson(0), 0)])
def test_degenerate_discrete_support(law, point):
    assert law.support() == (point, point)
    assert_array_equal(law.quantile([0, .25, 1]), [point]*3)
    assert law.variance() == 0


@pytest.mark.parametrize("student", [False, True])
def test_vector_cdf_infinite_bounds_reduce_to_marginals(student):
    law = d.multivariate_student_t(4, [0, 0], np.eye(2)) if student else d.multivariate_normal([0, 0], np.eye(2))
    assert_allclose(law.cdf([[-np.inf, 1], [np.inf, np.inf], [np.inf, 0]], rng=np.random.default_rng(2)), [0, 1, .5])
