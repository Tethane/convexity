"""Finite mass, mixture identities, transformations and explicit extension."""

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import distributions as d


def test_empirical_ties_weights_zero_mass_and_exact_expectation():
    values = np.array([0, 10, 10, 999.])
    weights = np.array([1., 1, 2, 0])
    law = d.empirical(values, weights)
    values[:] = 7
    weights[:] = 0
    assert_allclose(law.mean(), 7.5)
    assert_allclose(law.variance(), 18.75)
    assert_allclose(law.density([0, 10, 999]), [.25, .75, 0])
    assert_array_equal(law.quantile([0, .25, .26, 1]), [0, 0, 10, 10])
    assert_allclose(law.expect(lambda x: [x, x*x]), [7.5, 75])
    assert law.probability(lambda x: x > 0) == .75
    assert law.given(lambda x: x > 0).mean() == 10
    assert_allclose(law.map(lambda x: [x, -x]).mean(), [7.5, -7.5])
    assert d.empirical(np.arange(1134)).quantile(.5) == 566
    assert d.empirical(np.arange(1134)).cdf(566) == .5
    assert_allclose(d.empirical([0, 10], [1e308, 1e308]).mean(), 5)
    with pytest.raises(ValueError):
        law.given(lambda x: x < 0)
    with pytest.raises(ValueError):
        law.probability(lambda x: 1)


def test_empirical_vector_condition_and_pushforward():
    law = d.empirical([[0, 1], [0, 3], [1, 4]], [1, 3, 4])
    assert_allclose(law.mean(), [.5, 3.25])
    assert_allclose(law.cdf([[0, 2], [0, 3], [1, 9]]), [.125, .5, 1])
    assert_allclose(law.condition([0], [0]).mean(), [2.5])
    assert_allclose(law.marginal([1, 0]).mean(), [3.25, .5])
    assert law.map(lambda x: x.sum()).event_shape == ()
    assert law.sample((2, 3), rng=np.random.default_rng(1)).shape == (2, 3, 2)
    assert law.sample((), rng=np.random.default_rng(1)).shape == (2,)
    assert law.sample(0, rng=np.random.default_rng(1)).shape == (0, 2)
    assert_allclose(law.covariance(), [[.25, .375], [.375, .9375]])
    with pytest.raises(ValueError):
        law.condition([0], [.1])


@pytest.mark.parametrize("values,weights", [([], None), ([np.nan], None), ([np.inf], None),
                                            ([1, 2], [1]), ([1, 2], [0, 0]),
                                            ([1, 2], [-1, 2]), ([1, 2], [np.inf, 1]),
                                            ([[1], [2]], [[1], [1]]), ([[[]]], None)])
def test_invalid_empirical(values, weights):
    with pytest.raises(ValueError):
        d.empirical(values, weights)


def test_independent_law_density_cdf_covariance_and_condition():
    law = d.independent([d.normal(1, 2), d.student_t(5)])
    assert_allclose(law.log_density([0, 1]), d.normal(1, 2).log_density(0) + d.student_t(5).log_density(1))
    assert_allclose(law.cdf([1, 0]), .25)
    assert_allclose(law.covariance(), [[4, 0], [0, 5/3]])
    assert_allclose(law.condition([0], [4]).mean(), [0])
    assert law.marginal([1]).event_shape == (1,)
    assert law.sample((3, 2), rng=np.random.default_rng(1)).shape == (3, 2, 2)
    with pytest.raises(ValueError):
        d.independent([d.bernoulli(.4), d.poisson(3)]).condition([0], [2])


def test_mixture_stability_moments_quantile_and_sampling():
    law = d.mixture([d.normal(-2, 1), d.normal(4, 2)], [1, 3])
    assert_allclose(law.mean(), 2.5)
    assert_allclose(law.variance(), 10)
    assert_allclose(law.moment(2), 16.25)
    assert_allclose(law.expect(lambda x: x*x), 16.25)
    assert np.isfinite(law.log_density(1000))
    probabilities = np.array([.001, .1, .5, .9, .999])
    assert_allclose(law.cdf(law.quantile(probabilities)), probabilities, atol=2e-15)
    assert_array_equal(law.quantile([0, 1]), [-np.inf, np.inf])
    a = law.sample((100, 100), rng=np.random.default_rng(314))
    assert_array_equal(a, law.sample((100, 100), rng=np.random.default_rng(314)))
    assert abs(a.mean()-2.5) < .09
    assert abs(a.var()-10) < .3
    assert d.mixture([d.normal(), d.student_t(.5)], [1, 0]).mean() == 0
    with pytest.raises(NotImplementedError):
        d.mixture([d.poisson(1), d.poisson(2)]).quantile(.5)


def test_conditional_mixture_reweights_component_likelihoods():
    components = [d.multivariate_normal([0, 0], np.eye(2)),
                  d.multivariate_normal([2, 4], np.eye(2))]
    law = d.mixture(components, [1, 3])
    conditional = law.condition([0], [0])
    posterior_second = 3*np.exp(-2)/(1+3*np.exp(-2))
    assert_allclose(conditional.mean(), [4*posterior_second])
    assert_allclose(conditional.log_density([1]) + law.marginal([0]).log_density([0]), law.log_density([0, 1]))
    atoms = d.mixture([d.empirical([[0, 1]]), d.empirical([[1, 2]])])
    assert_allclose(atoms.condition([0], [0]).mean(), [1])
    with pytest.raises(ValueError):
        atoms.condition([0], [3])


def test_incompatible_compositions_are_rejected():
    for components in [[], [d.normal(), d.poisson(1)], [d.normal(), object()]]:
        with pytest.raises((TypeError, ValueError)):
            d.mixture(components)
    with pytest.raises(ValueError):
        d.mixture([d.normal(), d.multivariate_normal([0], [[1]])])
    with pytest.raises(ValueError):
        d.independent([d.multivariate_normal([0], [[1]])])


def test_exponential_change_of_variables_and_decreasing_affine():
    law = d.exponential(d.normal(.2, .7))
    reference = d.lognormal(.2, .7)
    x = np.array([-1., 0., .1, 1., 2., np.inf])
    assert_allclose(law.density(x), reference.density(x))
    assert_allclose(law.cdf(x), reference.cdf(x))
    assert_allclose(law.survival(x), reference.survival(x))
    assert_allclose(law.quantile([0, .25, .75, 1]), reference.quantile([0, .25, .75, 1]))
    assert_allclose(law.expect(lambda x: x), reference.mean(), rtol=1e-8)
    negative = d.affine(d.normal(2, 3), 4, -2)
    reference = d.normal(0, 6)
    assert_allclose(negative.density([-4, 0, 4]), reference.density([-4, 0, 4]))
    assert_allclose(negative.cdf([-4, 0, 4]), reference.cdf([-4, 0, 4]))
    assert_allclose(negative.quantile([0, .2, .8, 1]), reference.quantile([0, .2, .8, 1]))
    assert negative.support() == (-np.inf, np.inf)
    assert_allclose(negative.mean(), 0)
    assert_allclose(negative.variance(), 36)
    assert_allclose(negative.moment(2), 36)
    with pytest.raises(ValueError):
        d.affine(d.normal(), scale=0)
    with pytest.raises(ValueError):
        d.exponential(d.poisson(1))


def test_custom_adapter_uses_public_contract_and_rejects_bad_outputs():
    @dataclass(frozen=True)
    class PointMass:
        event_shape = ()
        measure: d.Measure = "discrete"
        capabilities = frozenset({"log_density", "sample"})

        def evaluate(self, operation, *args, **kwargs):
            if operation == "sample":
                return np.full(args[0], 3.)
            return np.where(args[0] == 3., 0., -np.inf)

    law = d.Distribution(PointMass())
    mixed = d.mixture([law, d.empirical([0, 1])])
    assert_allclose(mixed.density([0, 1, 3, 4]), [.25, .25, .5, 0])
    with pytest.raises(NotImplementedError):
        law.cdf(1)
    with pytest.raises(NotImplementedError):
        mixed.mean()

    class WrongShape(PointMass):
        def evaluate(self, operation, *args, **kwargs):
            return np.ones(3)

    with pytest.raises(ValueError, match="shape"):
        d.Distribution(WrongShape()).log_density(0)
