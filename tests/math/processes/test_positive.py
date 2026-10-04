import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import processes as p


def test_gbm_law_moments_and_generator():
    model = p.GeometricBrownianMotion(.07, .3)
    kernel = model.transition(100., 1.2)
    mean = 100*np.exp(.07*1.2)
    assert_allclose(kernel.mean(), mean)
    assert_allclose(kernel.covariance(), mean**2*np.expm1(.3**2*1.2))
    assert_allclose(model.generator(0, 100., 1., 0.), 7.)
    assert_allclose(model.generator(0, 100., .01, -.0001), .07-.5*.3**2)
    samples = kernel.sample(80000, rng=np.random.default_rng(13))
    assert np.all(samples > 0)
    assert_allclose(np.log(samples/100).mean(), (.07-.045)*1.2, atol=.004)
    assert_allclose(np.log(samples/100).var(), .09*1.2, atol=.002)


@pytest.mark.parametrize('model,state', [(p.GeometricBrownianMotion(.1, 0), 3.),
                                        (p.GeometricBrownianMotion(.1, .5), 0.),
                                        (p.CIR(2, .04, 0), .1)])
def test_deterministic_positive_models(model, state):
    law = model.transition(state, 1.).distribution()
    assert law.measure == 'discrete'
    assert law.variance() == 0


@pytest.mark.parametrize('sigma', [.15, .7])
def test_cir_exact_moments_boundary_and_feller(sigma):
    model = p.CIR(2., .04, sigma)
    assert model.feller_satisfied == (sigma**2 <= .16)
    for x in (0., .03):
        dt = .7
        decay = np.exp(-2*dt)
        mean = .04+(x-.04)*decay
        variance = x*sigma**2*decay*(1-decay)/2+.04*sigma**2*(1-decay)**2/4
        kernel = model.transition(x, dt)
        assert_allclose(kernel.mean(), mean)
        assert_allclose(kernel.covariance(), variance)
        samples = kernel.sample(100000, rng=np.random.default_rng(41))
        assert np.all(samples >= 0)
        assert_allclose(samples.mean(), mean, atol=5*np.sqrt(variance/len(samples)))
        assert_allclose(kernel.distribution().cdf(kernel.distribution().quantile([.1, .5, .9])), [.1, .5, .9])
    assert_allclose(model.stationary().mean(), .04)
    assert_allclose(model.stationary().variance(), .04*sigma**2/4)
    assert_allclose(model.generator(0, .03, 1., 0.), .02)
    with pytest.raises(ValueError):
        model.transition(.04, 1., method='euler')


@pytest.mark.parametrize('model', [p.CIR(1,.04,.2), p.GeometricBrownianMotion()])
def test_invalid_positive_state(model):
    with pytest.raises(ValueError):
        model.transition(-1., .1)
