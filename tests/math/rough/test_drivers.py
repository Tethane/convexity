import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import rough as r


@pytest.mark.parametrize('h', [.15, .5, .8])
def test_fbm_stationary_increments_and_empirical_covariance(h):
    grid = np.linspace(0, 1, 9)
    model = r.FractionalBrownianMotion(h)
    covariance = model.covariance(grid)
    difference = np.diff(np.eye(len(grid)), axis=0)
    increments = difference @ covariance @ difference.T
    assert_allclose(np.diag(increments), (1/8)**(2*h))
    assert_allclose(increments[0, 1], .5*(2**(2*h)-2)*(1/8)**(2*h))
    paths = model.sample(grid, draws=30000, rng=np.random.default_rng(15))
    assert_allclose(paths.values.T @ paths.values/30000, covariance, atol=.025)
    assert paths.exact
    if h == .5:
        assert_allclose(covariance, np.minimum.outer(grid, grid))


def test_riemann_liouville_is_not_stationary_increment_fbm():
    grid = [0., .3, .7, 1.]
    volterra = r.VolterraGaussian(r.FractionalKernel(.1)).covariance(grid)
    fbm = r.FractionalBrownianMotion(.1).covariance(grid)
    assert_allclose(np.diag(volterra), np.diag(fbm))
    assert abs(volterra[1, 3]-fbm[1, 3]) > .1


def test_joint_volterra_driver_brownian_cross_covariance_and_causality():
    grid = np.array([0, .03, .2, .6, 1.])
    model = r.VolterraGaussian(r.FractionalKernel(.1))
    result = model.sample_joint(grid, draws=60000, rng=np.random.default_rng(14))
    samples = np.column_stack((result.values, result.brownian_increments))
    cross = model.brownian_covariance(grid)
    expected = np.block([[model.covariance(grid), cross], [cross.T, np.diag(np.diff(grid))]])
    assert_allclose(samples.T @ samples/len(samples), expected, atol=.013)
    assert_allclose(cross[1, 0], np.sqrt(.2)*.03**.6/.6)
    assert_allclose(cross[np.triu_indices(5, 0, 4)], 0)


def test_brownian_limit_and_zero_kernel():
    for method in ['dense', 'factors']:
        k = r.ExponentialKernel([1], [0])
        result = r.VolterraGaussian(k).sample_joint([0, .2, .3, 1], draws=20,
                                                   rng=np.random.default_rng(5), method=method)
        assert_allclose(result.values[:, 1:], np.cumsum(result.brownian_increments, axis=1), atol=1e-7)
    zero = r.VolterraGaussian(r.ExponentialKernel([0], [1]))
    assert_allclose(zero.sample([0, 1], draws=3, rng=np.random.default_rng(5)).values, 0)


def test_factor_scheme_matches_dense_covariance_and_shares_noise():
    kernel = r.ExponentialKernel([.7, .3, .1], [0, 2, 2])
    model = r.VolterraGaussian(kernel)
    grid = np.array([0, .1, .5, 1.])
    result = model.sample_joint(grid, draws=50000, rng=np.random.default_rng(15), method='factors')
    assert result.exact and result.factors is not None
    assert_allclose(result.values.T @ result.values/50000, model.covariance(grid), atol=.012)
    assert_allclose(result.values.T @ result.brownian_increments/50000,
                    model.brownian_covariance(grid), atol=.012)
    assert_allclose(result.factors[:, :, 0], np.column_stack((np.zeros(50000),
                    np.cumsum(result.brownian_increments, axis=1))), atol=1e-7)
    assert_allclose(result.factors[:, :, 1], result.factors[:, :, 2], atol=1e-7)


def test_origin_no_entropy_reproducibility_and_ownership():
    for driver in [r.FractionalBrownianMotion(), r.VolterraGaussian(r.FractionalKernel())]:
        rng = np.random.default_rng(14)
        before = str(rng.bit_generator.state)
        assert_allclose(driver.sample([0], draws=2, rng=rng).values, 0)
        assert str(rng.bit_generator.state) == before
        a = driver.sample([0, .1, .4], draws=4, rng=np.random.default_rng(14))
        b = driver.sample([0, .1, .4], draws=4, rng=np.random.default_rng(14))
        assert_allclose(a.values, b.values, rtol=0, atol=0)
        assert not a.values.flags.writeable and not a.times.flags.writeable


@pytest.mark.parametrize('amplitude', [1e-12, 1e12])
def test_joint_sampling_preserves_disparate_driver_and_brownian_units(amplitude):
    driver = r.VolterraGaussian(r.ExponentialKernel([amplitude], [1.]))
    paths = driver.sample_joint([0, .3, 1], draws=30000, rng=np.random.default_rng(140))
    scaled = paths.values/amplitude
    reference = r.VolterraGaussian(r.ExponentialKernel([1.], [1.]))
    assert_allclose(scaled.T @ scaled/len(scaled), reference.covariance(paths.times), atol=.012)
    assert_allclose(scaled.T @ paths.brownian_increments/len(scaled),
                    reference.brownian_covariance(paths.times), atol=.012)


def test_invalid_custom_joint_covariance_is_rejected():
    kernel = r.CallableKernel(lambda u: 1., integral_function=lambda a, b: b-a,
                              covariance_function=lambda t, s, end: 0.)
    # Claimed zero driver variance conflicts with nonzero Brownian covariance.
    with pytest.raises(ValueError, match='zero-variance'):
        r.VolterraGaussian(kernel).sample_joint([0, 1], rng=np.random.default_rng(14))
