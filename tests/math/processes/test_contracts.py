import subprocess
import sys
from typing import Any
import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import processes as p, distributions as d


def test_lazy_import():
    subprocess.run([sys.executable, '-c', "import sys; from convexity.math import processes; assert not {'scipy', 'QuantLib', 'sdeint', 'torch'} & sys.modules.keys()"], check=True)


def test_owned_arrays_shapes_and_reproducibility():
    matrix = np.array([[-1., 0.], [0., -2.]])
    model = p.LinearDiffusion(matrix, np.eye(2))
    matrix[:] = 0
    assert model.matrix[0, 0] == -1
    with pytest.raises(ValueError):
        model.matrix[0, 0] = 1
    kernel = model.transition([1, 2], .3)
    a = kernel.sample((2, 3), rng=np.random.default_rng(1))
    b = kernel.sample((2, 3), rng=np.random.default_rng(1))
    assert a.shape == (2, 3, 2)
    assert_allclose(a, b)
    assert not a.flags.writeable
    assert kernel.sample((0, 3), rng=np.random.default_rng()).shape == (0, 3, 2)
    assert kernel.sample(rng=np.random.default_rng()).shape == (2,)
    with pytest.raises(TypeError, match='Generator'):
        kernel.sample(rng=1)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    for size in (True, -1, [2], (1.5,)):
        with pytest.raises(ValueError):
            kernel.sample(size, rng=np.random.default_rng())  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.


@pytest.mark.parametrize('model,state,method', [
    (p.BrownianMotion(), 2., 'exact'), (p.GeometricBrownianMotion(), 0., 'exact'),
    (p.CIR(1,.04,.3), 0., 'exact'), (p.Heston(.03,2,.04,.3,-.5), [100,0], 'qe'),
    (p.ItoDiffusion(lambda t,x: 0., lambda t,x: [1.]), 0., 'sri2'),
    (p.AdditiveJumpDiffusion(0.,[1.],2.,d.normal()), 0., 'exact'),
    (p.MertonJumpDiffusion(.03,.2,1.,-.1,.2), 100., 'exact')])
def test_zero_duration_preserves_state_and_rng(model, state, method):
    rng = np.random.default_rng(13)
    before = repr(rng.bit_generator.state)
    kernel = model.transition(state, 0, method=method)
    assert kernel.exact
    assert_allclose(kernel.sample(rng=rng), state)
    # Existing empirical atom sampling can consume a random number. Identity
    # kernels should not consume entropy merely to select their only point.
    assert repr(rng.bit_generator.state) == before


@pytest.mark.parametrize('dt', [-1., np.inf, np.nan, True])
def test_invalid_duration(dt):
    with pytest.raises(ValueError):
        p.BrownianMotion().transition(0., dt)


@pytest.mark.parametrize('times', [[], [1, 1], [1, 0], [0, np.nan], [0, np.inf]])
def test_invalid_times(times):
    with pytest.raises(ValueError):
        p.BrownianMotion().sample(0, times, rng=np.random.default_rng())


def test_checked_callback_contracts_and_readonly_input():
    seen = []
    def drift(t, x):
        seen.append(x.flags.writeable)
        return -x
    model = p.ItoDiffusion(drift, lambda t,x: [[1.]], (1,))
    k = model.transition([2.], .1, method='euler')
    assert_allclose(k.mean(), [1.8])
    assert seen == [False]
    for model in (p.ItoDiffusion(lambda t,x: [0], lambda t,x:[1]),
                  p.ItoDiffusion(lambda t,x: 0, lambda t,x:1),
                  p.ItoDiffusion(lambda t,x: np.nan, lambda t,x:[1])):
        with pytest.raises(ValueError):
            model.transition(0., .1, method='euler')
    with pytest.raises(ValueError):
        p.ItoDiffusion(lambda t,x: 0., lambda t,x:[1.]).transition(0., .1)


def test_user_process_needs_no_inheritance():
    class Custom:
        event_shape = ()
        def transition(self, state, dt, *, time=0., method='exact', substeps=1):
            return p.Transition.from_distribution(d.empirical([float(state)+dt*time]))
    paths = p.sample_paths(Custom(), 1., [0., .4, 1.], draws=2, rng=np.random.default_rng(1))
    assert_allclose(paths.values, [[1, 1, 1.24]]*2)
    assert paths.exact and not paths.values.flags.writeable
    assert paths.marginal(2).mean() == 1.24
    class Broken:
        event_shape = ()
        def transition(self, *args, **kwargs):
            return d.normal()
    broken: Any = Broken()
    with pytest.raises(TypeError):
        p.transition(broken, 0., 1.)
    with pytest.raises(ValueError):
        p.Transition((), 'bad', 'test', False, lambda size,rng: [0]).sample(rng=np.random.default_rng())


def test_irregular_paths_joint_covariance_and_initial():
    times = [3., 3.2, 3.9]
    paths = p.BrownianMotion().sample(2., times, draws=15000, rng=np.random.default_rng(12))
    assert paths.values.shape == (15000, 3)
    assert_allclose(paths.values[:,0], 2.)
    assert_allclose(np.cov(paths.values[:,1:].T), [[.2,.2],[.2,.9]], atol=.018)
    assert_allclose(paths.times, times)
    with pytest.raises(ValueError):
        p.BrownianMotion().sample(0., [0.], rng=np.random.default_rng(), method='unknown')
