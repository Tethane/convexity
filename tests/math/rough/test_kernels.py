import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import gamma, gammainc
from convexity.math import rough as r


@pytest.mark.parametrize('h', [.05, .1, .5, .8])
def test_fractional_integrals_covariance_and_scaling(h):
    k = r.FractionalKernel(h)
    assert_allclose(k.covariance(.7, .7), .7**(2*h))
    assert_allclose(k.covariance(.7, .7, until=.2), .7**(2*h)-.5**(2*h))
    assert_allclose(k.integral(0, .7), np.sqrt(2*h)*.7**(h+.5)/(h+.5))
    reference = quad(lambda u: 2*h*(.7-u)**(h-.5)*(.3-u)**(h-.5), 0, .3,
                     epsabs=1e-10)[0]
    assert_allclose(k.covariance(.7, .3), reference, rtol=1e-8)
    assert_allclose(k.covariance(7, 3), 10**(2*h)*reference, rtol=1e-8)
    assert k.covariance(0, 3) == 0
    assert k.integral(.3, .3) == 0
    assert_allclose(k.integral(1, 1+1e-10), np.sqrt(2*h)*1e-10, rtol=1e-6)


def test_exponential_kernel_signed_repeated_and_zero_rates():
    k = r.ExponentialKernel([1., -.3, .2], [0., 2., 2.])
    assert_allclose(k([0, 1]), [0.9, 1-.1*np.exp(-2)])
    assert_allclose(k.integral(.2, .8), quad(lambda t: 1-.1*np.exp(-2*t), .2, .8)[0])
    assert_allclose(k.covariance(.9, .7, until=.4),
                    quad(lambda u: (1-.1*np.exp(-2*(.9-u)))*(1-.1*np.exp(-2*(.7-u))), 0, .4)[0])
    transition = k.factor_process().transition([0, 0, 0], .4)
    assert_allclose(transition.covariance()[0, 0], .4)
    assert_allclose(k.weights @ transition.covariance() @ k.weights, k.covariance(.4, .4))


def test_custom_kernel_and_structural_extension():
    custom = r.CallableKernel(lambda u: np.exp(-2*u))
    analytic = r.ExponentialKernel([1.], [2.])
    assert_allclose(custom.integral(.1, 1), analytic.integral(.1, 1))
    assert_allclose(custom.covariance(.7, .4), analytic.covariance(.7, .4))
    class UserKernel:
        def __call__(self, lag):
            return np.ones_like(lag)
        def integral(self, lower, upper):
            return upper-lower
        def covariance(self, t, s, *, until=None):
            return min(t, s) if until is None else until
    driver = r.VolterraGaussian(UserKernel())
    assert_allclose(driver.covariance([0, .2, .8]), np.minimum.outer([0, .2, .8], [0, .2, .8]))
    paths = r.sample_driver(driver, [0, .2], draws=2, rng=np.random.default_rng(14))
    assert paths.values.shape == (2, 2)


def test_laplace_approximation_refinement_and_independent_error_identity():
    errors = []
    for n in [4, 8, 16, 32]:
        k, error = r.approximate_fractional(.1, factors=n)
        errors.append(error.relative_l2_error)
        assert np.all(k.weights > 0) and np.all(k.rates > 0)
        p = .6
        cross = np.sqrt(.2)*np.sum(k.weights*gamma(p)*gammainc(p, k.rates)/k.rates**p)
        exact_error = 1+k.covariance(1, 1)-2*cross
        assert_allclose(error.squared_error, exact_error, rtol=1e-7, atol=1e-10)
        assert error.quadrature_error < 1e-7
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < .2
    with pytest.raises(ValueError, match='exceeds'):
        r.approximate_fractional(.1, factors=4, tolerance=.01)
    k, error = r.approximate_fractional(.5, tolerance=0)
    assert error.l2_error == 0
    assert_allclose(k([0, 1, 100]), 1)


def test_error_captures_near_zero_tail_and_time_units():
    k = r.ExponentialKernel([0.], [0.])
    for h in [.01, .1, .45]:
        error = r.kernel_error(r.FractionalKernel(h), k, 2.)
        assert_allclose(error.squared_error, 2**(2*h))
    _, a = r.approximate_fractional(.2, factors=24, horizon=1)
    _, b = r.approximate_fractional(.2, factors=24, horizon=10)
    assert_allclose(a.relative_l2_error, b.relative_l2_error)
    assert_allclose(b.squared_error, a.squared_error*10**.4)
