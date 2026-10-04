import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import processes as p, distributions as d


def test_compound_poisson_additive_moments_and_generator():
    model = p.AdditiveJumpDiffusion(.3, [.2], 1.7, d.normal(-.2, .4))
    k = model.transition(2., .8)
    assert k.exact and 'distribution' not in k.capabilities
    assert_allclose(k.mean(), 2.+(.3-1.7*.2)*.8)
    assert_allclose(k.covariance(), (.04+1.7*(.16+.04))*.8)
    samples = k.sample(100000, rng=np.random.default_rng(11))
    assert_allclose(samples.mean(), k.mean(), atol=.008)
    assert_allclose(samples.var(), k.covariance(), atol=.009)
    assert_allclose(model.generator(0, 2., 4., 2., function=lambda x:x*x),
                    4*.3+.04+1.7*(4*(-.2)+.16+.04))
    with pytest.raises(NotImplementedError):
        k.distribution()


def test_vector_jumps_and_zero_intensity():
    marks = d.empirical([[1, -1], [2, -2]])
    model = p.AdditiveJumpDiffusion([0, 0], np.zeros((2, 1)), 2., marks)
    kernel = model.transition([0, 0], 1.)
    assert_allclose(kernel.mean(), [3, -3])
    assert_allclose(kernel.covariance(), [[5, -5], [-5, 5]])
    samples = kernel.sample((4, 5), rng=np.random.default_rng(0))
    assert_allclose(samples[..., 0]+samples[..., 1], 0)
    assert_allclose(model.generator(0, [0, 0], [0, 0], 2*np.eye(2), function=lambda x:x@x), 10)
    kernel = p.AdditiveJumpDiffusion(0., [0.], 0., d.student_t(1)).transition(2., 1.)
    assert kernel.distribution().measure == 'discrete'
    assert kernel.mean() == 2.


def test_jump_moments_are_not_invented_for_heavy_tails():
    k = p.AdditiveJumpDiffusion(0., [1.], 1., d.student_t(1)).transition(0., 1.)
    assert k.capabilities == {'sample'}
    with pytest.raises(NotImplementedError):
        k.mean()


def test_merton_exact_compensation_moments_and_log_count_mixture():
    model = p.MertonJumpDiffusion(.06, .2, 1.3, -.1, .25)
    k = model.transition(100., .7)
    samples = k.sample(180000, rng=np.random.default_rng(123))
    assert_allclose(k.mean(), 100*np.exp(.06*.7))
    assert_allclose(samples.mean(), k.mean(), atol=.4)
    assert_allclose(samples.var(), k.covariance(), rtol=.025)
    expected_log = (.06-1.3*model.compensator-.5*.2**2+1.3*(-.1))*.7
    assert_allclose(np.log(samples/100).mean(), expected_log, atol=.003)
    assert_allclose(model.generator(0, 100., 1., 0., function=lambda x:x,
                                   jump_expectation=100*model.compensator), 6.)
    assert_allclose(model.generator(0, 100., 1/100, -1/100**2, function=np.log),
                    .06-1.3*model.compensator-.02+1.3*(-.1))


def test_merton_reductions_and_atoms():
    no_jumps = p.MertonJumpDiffusion(.03,.2,0.,0.,0.).transition(1.,.7)
    gbm = p.GeometricBrownianMotion(.03,.2).transition(1.,.7)
    assert_allclose(no_jumps.mean(), gbm.mean())
    model = p.MertonJumpDiffusion(0.,0.,2.,-.3,0.)
    k = model.transition(1.,.5)
    samples = k.sample(50000,rng=np.random.default_rng(8))
    no_jump = np.exp(-2*model.compensator*.5)
    assert_allclose(np.mean(samples == no_jump), np.exp(-1), atol=.006)
    assert model.transition(0.,1).mean() == 0
