"""Adversarial regressions for elapsed time, limiting cases and declared laws."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import distributions as d, processes as p


@pytest.mark.parametrize('model', [p.OrnsteinUhlenbeck(1., 1., 0.), p.CIR(1., 1., 0.)])
def test_mean_reversion_does_not_cancel_the_long_run_level(model):
    assert_allclose(model.transition(1e20, 100.).mean(), 1., rtol=1e-14)
    # exp(-40)*x still matters even when 1-exp(-40) rounds to one.
    assert_allclose(model.transition(1e20, 40.).mean(), 1.+1e20*np.exp(-40), rtol=1e-14)


def test_ou_brownian_limit_ignores_unused_long_run_level():
    kernel = p.OrnsteinUhlenbeck(0., -1e308, 1.).transition(1e308, .5)
    assert kernel.mean() == 1e308
    assert_allclose(kernel.covariance(), .5, rtol=1e-14)


def test_merton_representable_level_does_not_overflow_before_scaling():
    kernel = p.MertonJumpDiffusion(710., 0., 1., 0., 0.).transition(1e-300, 1.)
    expected = np.exp(np.log(1e-300)+710.)
    assert_allclose(kernel.mean(), expected, rtol=1e-14)
    assert_allclose(kernel.sample(4,rng=np.random.default_rng(13)), expected, rtol=1e-14)
    assert kernel.covariance() == 0.


def test_gbm_absorbing_zero_and_representable_exponential_product():
    assert p.GeometricBrownianMotion(1000., 2.).transition(0., 1.).mean() == 0
    kernel = p.GeometricBrownianMotion(710., 0.).transition(1e-300, 1.)
    assert_allclose(kernel.mean(), np.exp(np.log(1e-300)+710.), rtol=1e-14)


@pytest.mark.parametrize('method', ['euler', 'sri2'])
def test_custom_sde_integrates_elapsed_duration_at_large_absolute_origin(method):
    model = p.ItoDiffusion(lambda t,x: 1., lambda t,x: [0.])
    for substeps in (2, 8):
        result = model.transition(0., .3, time=1e15, method=method, substeps=substeps)
        assert_allclose(result.sample(rng=np.random.default_rng(3)), .3, rtol=1e-14)


def test_sde_callback_keeps_absolute_time_after_relative_integration():
    origin = 7.
    model = p.ItoDiffusion(lambda t,x: t-origin, lambda t,x: [0.])
    result = model.transition(0., .5, time=origin, method='sri2', substeps=16)
    assert_allclose(result.sample(rng=np.random.default_rng(3)), .125, atol=1e-14)


def test_compound_jump_finite_mean_survives_infinite_variance():
    kernel = p.AdditiveJumpDiffusion(1., [0.], 2., d.student_t(1.5,loc=3.)).transition(4.,.5)
    assert 'mean' in kernel.capabilities
    assert 'covariance' not in kernel.capabilities
    assert_allclose(kernel.mean(), 7.5)
    with pytest.raises(NotImplementedError):
        kernel.covariance()


def test_transition_rejects_conflicting_explicit_and_distribution_moments():
    law = d.normal(0.,1.)
    for moments in ({'_mean': 2.}, {'_covariance': 2.}):
        with pytest.raises(ValueError, match='agree'):
            p.Transition((), 'exact', 'test', True, lambda size,rng: law.sample(size,rng=rng),
                         _law=law, **moments)


@pytest.mark.parametrize('covariance', [-1e-16, -1.])
def test_transition_rejects_negative_scalar_variance_at_any_scale(covariance):
    with pytest.raises(ValueError, match='covariance'):
        p.Transition((), 'exact', 'test', True, lambda size,rng: np.zeros(size), _covariance=covariance)


def test_vector_covariance_validation_is_relative_to_its_scale():
    for matrix in ([[1e-20, 0], [0, -1e-22]], [[1e-20, 1e-24], [0, 1e-20]]):
        with pytest.raises(ValueError, match='covariance'):
            p.Transition((2,), 'exact', 'test', True, lambda size,rng: np.zeros(size+(2,)), _covariance=matrix)


def test_path_metadata_retains_every_used_kernel_not_only_the_last():
    class SwitchingSolver:
        event_shape = ()
        def transition(self,state,dt,*,time=0.,method='exact',substeps=1):
            if dt == 0:
                return p.Transition.identity(state)
            early = time < 1
            return p.Transition((), 'euler' if early else 'exact', 'sdeint' if early else 'scipy',
                                not early, lambda size,rng: np.full(size,state+dt))
    paths = p.sample_paths(SwitchingSolver(),0.,[0.,1.,2.],draws=2,rng=np.random.default_rng(2))
    assert not paths.exact
    assert paths.method == paths.backend == 'mixed'
    assert set(paths.methods) == {'euler','exact'}
    assert set(paths.backends) == {'sdeint','scipy'}


@pytest.mark.parametrize('name', ['method','backend'])
def test_process_paths_rejects_invalid_metadata(name):
    kwargs = {'method': 'exact', 'backend': 'numpy', 'exact': True}
    kwargs[name] = ''
    with pytest.raises(ValueError,match=name):
        p.ProcessPaths([0.], [[1.]], **kwargs)


def test_sri2_preserves_ito_log_drift_with_multiple_drivers(monkeypatch):
    # Capture the real backend's drivers, then compare every sample with the
    # analytic Ito solution on those drivers. No Monte Carlo moment tolerance.
    import sdeint.integrate as backend
    original = backend.deltaW
    increments = []

    def recorded(*args, **kwargs):
        value = original(*args, **kwargs)
        increments.append(value)
        return value

    monkeypatch.setattr(backend, 'deltaW', recorded)
    model = p.ItoDiffusion(lambda t,x: .2*x, lambda t,x: np.array([.2,.3])*x,
                          noise_dimension=2)
    errors = []
    for steps in (8, 64):
        increments.clear()
        kernel = model.transition(1., 1., method='sri2', substeps=steps)
        samples = kernel.sample(200,rng=np.random.default_rng(137))
        exact_logs = .2-.5*.13+np.stack(increments).sum(axis=1) @ np.array([.2,.3])
        errors.append(np.sqrt(np.mean((np.log(samples)-exact_logs)**2)))
    assert errors[1] < .003
    assert errors[1] < .3*errors[0]
