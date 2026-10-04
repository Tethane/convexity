import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import processes as p


def test_custom_sri2_and_euler_ode_limit_convergence():
    pytest.importorskip('sdeint')
    model = p.ItoDiffusion(lambda t,x:-2*x, lambda t,x:[0.])
    exact = np.exp(-2)
    errors = []
    for steps in (8, 16, 32):
        k = model.transition(1., 1., method='euler', substeps=steps)
        assert k.backend == 'sdeint' and not k.exact
        errors.append(abs(float(k.sample(rng=np.random.default_rng(1)))-exact))
    assert errors[1] < .55*errors[0]
    assert errors[2] < .55*errors[1]
    k = model.transition(1.,1.,method='sri2',substeps=32)
    assert abs(float(k.sample(rng=np.random.default_rng(1)))-exact) < errors[-1]/10
    with pytest.raises(NotImplementedError):
        k.distribution()


def test_custom_backend_rectangular_diffusion_and_reproducibility():
    pytest.importorskip('sdeint')
    model = p.ItoDiffusion(lambda t,x:[t,2*t], lambda t,x:[[1.],[2.]], (2,), 1)
    k = model.transition([0.,0.], .5, time=2., method='sri2', substeps=8)
    a = k.sample(30,rng=np.random.default_rng(1))
    b = k.sample(30,rng=np.random.default_rng(1))
    assert_allclose(a,b)
    assert_allclose(a[:,1],2*a[:,0])
    deterministic = p.ItoDiffusion(lambda t,x:t,lambda t,x:[0.])
    assert_allclose(deterministic.transition(0.,.5,time=2.,method='sri2',substeps=8)
                    .sample(rng=np.random.default_rng(1)), 1.125)


def test_euler_stochastic_convergence_to_exact_gbm_on_same_driver():
    pytest.importorskip('sdeint')
    # Each resolution compares Euler with the analytic solution using precisely
    # the same generated Brownian increments; this checks strong, not weak error.
    model = p.ItoDiffusion(lambda t,x:.1*x, lambda t,x:[.4*x])
    errors = []
    for n in (8, 64):
        count = 1200
        seed = 81
        increments = np.random.default_rng(seed).normal(0, np.sqrt(1/n), (count,n))
        exact = np.exp(.1-.5*.4**2+.4*increments.sum(axis=1))
        sampled = model.transition(1.,1.,method='euler',substeps=n).sample(count,rng=np.random.default_rng(seed))
        errors.append(np.sqrt(np.mean((sampled-exact)**2)))
    assert errors[1] < .5*errors[0]


def test_heston_coefficients_include_instantaneous_correlation():
    model = p.Heston(.05, 2., .04, .5, -.7)
    state = [100., .09]
    assert_allclose(model.drift(0,state), [5., -.1])
    b = model.diffusion(0,state)
    assert_allclose(b @ b.T, [[900.,-.7*.5*100*.09],[-.7*.5*100*.09,.25*.09]])
    assert_allclose(model.generator(0,state,[state[1],state[0]],[[0,1],[1,0]]),
                    .05*100*.09+100*2*(.04-.09)-.7*.5*100*.09)
    with pytest.raises(ValueError):
        model.transition(state,.1,method='exact')


def test_quantlib_heston_qe_moments_positivity_and_global_state():
    ql = pytest.importorskip('QuantLib')
    original = ql.Settings.instance().evaluationDate
    model = p.Heston(.03, 2., .04, .5, -.7)
    kernel = model.transition([100.,.03],.4,substeps=8)
    assert not kernel.exact and kernel.backend == 'QuantLib'
    assert kernel.capabilities == {'sample'}
    samples = kernel.sample(24000,rng=np.random.default_rng(14))
    assert np.all(samples[:,0]>0) and np.all(samples[:,1]>=0)
    expected_variance = .04+(.03-.04)*np.exp(-.8)
    assert_allclose(samples[:,1].mean(),expected_variance,atol=.001)
    assert_allclose(samples[:,0].mean(),100*np.exp(.03*.4),atol=.35)
    assert ql.Settings.instance().evaluationDate == original
    a = kernel.sample(3,rng=np.random.default_rng(1))
    b = kernel.sample(3,rng=np.random.default_rng(1))
    assert_allclose(a,b)
    paths = model.sample([100.,0.], [0,.1,.3],draws=10,rng=np.random.default_rng(1))
    assert paths.values.shape == (10,3,2)
    assert not paths.exact
