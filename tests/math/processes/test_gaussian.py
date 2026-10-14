import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import processes as p


def test_brownian_and_arithmetic_moments_and_generator():
    brownian = p.BrownianMotion()
    kernel = brownian.transition(3.0, 2.0)
    assert_allclose(kernel.mean(), 3.0)
    assert_allclose(kernel.covariance(), 2.0)
    assert brownian.generator(0, 3.0, 6.0, 2.0) == 1.0
    model = p.ArithmeticBrownianMotion(0.3, 0.7)
    kernel = model.transition(2.0, 0.5)
    assert_allclose(kernel.mean(), 2.15)
    assert_allclose(kernel.covariance(), 0.245)
    assert_allclose(model.generator(1.0, 2.0, lambda x: 2 * x, lambda x: 2.0), 1.69)
    samples = kernel.sample(80000, rng=np.random.default_rng(13))
    assert_allclose(samples.mean(), 2.15, atol=0.006)
    assert_allclose(samples.var(), 0.245, atol=0.004)


def test_scalar_ou_semigroup_stationary_and_zero_reversion():
    model = p.OrnsteinUhlenbeck(1.7, -2.0, 0.6)
    a, b = 0.4, 0.9
    first = model.transition(3.0, a)
    sequential = model.transition(first.mean(), b)
    joint = model.transition(3.0, a + b)
    assert_allclose(sequential.mean(), joint.mean())
    assert_allclose(
        np.exp(-2 * model.kappa * b) * first.covariance() + sequential.covariance(),
        joint.covariance(),
    )
    assert_allclose(model.stationary().variance(), 0.36 / 3.4)
    assert_allclose(model.transition(3.0, 1000).mean(), -2.0)
    assert_allclose(
        model.transition(3.0, 1000).covariance(), model.stationary().variance()
    )
    zero = p.OrnsteinUhlenbeck(0, -500, 0.2).transition(1.0, 0.7)
    assert_allclose(zero.mean(), 1.0)
    assert_allclose(zero.covariance(), 0.028)
    with pytest.raises(ValueError, match="stationary"):
        p.OrnsteinUhlenbeck(0).stationary()


def test_integrated_brownian_exact_linear_transition():
    model = p.LinearDiffusion([[0, 1], [0, 0]], [[0], [2]], [1, 0])
    dt = 0.6
    matrix, offset, covariance = model.coefficients(dt)
    assert_allclose(matrix, [[1, dt], [0, 1]])
    assert_allclose(offset, [dt, 0])
    assert_allclose(covariance, 4 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]))
    assert_allclose(model.transition([2, 3], dt).mean(), [2 + 4 * dt, 3])
    assert model.noise_dimension == 1
    with pytest.raises(ValueError, match="Hurwitz"):
        model.stationary()


def test_vector_ou_stationarity_semigroup_and_long_gap():
    model = p.VectorOrnsteinUhlenbeck(
        [[2, -0.4], [0.3, 1]], [1, -1], [[0.5, 0.1], [0.1, 0.6]]
    )
    stationary = model.stationary()
    cov = stationary.covariance()
    assert_allclose(
        model.matrix @ cov + cov @ model.matrix.T + model.loading @ model.loading.T,
        0,
        atol=1e-14,
    )
    f, b, q = model.coefficients(0.7)
    g, c, r = model.coefficients(1.3)
    h, e, s = model.coefficients(2.0)
    assert_allclose(h, g @ f)
    assert_allclose(e, g @ b + c)
    assert_allclose(s, g @ q @ g.T + r)
    assert_allclose(model.transition([40, 20], 1e4).mean(), [1, -1], atol=1e-12)
    assert_allclose(model.transition([40, 20], 1e4).covariance(), cov, atol=1e-12)
    assert_allclose(
        model.generator(0, [1, 2], [2, 4], 2 * np.eye(2)),
        2 * np.array([1, 2]) @ model.drift(0, [1, 2]) + np.sum(model.loading**2),
    )


def test_singular_noise_and_deterministic_limits():
    kernel = p.LinearDiffusion(np.zeros((2, 2)), [[1], [1]]).transition([1, 3], 0.4)
    samples = kernel.sample(100, rng=np.random.default_rng(12))
    assert_allclose(samples[:, 1] - samples[:, 0], 2.0)
    assert_allclose(kernel.covariance(), np.full((2, 2), 0.4))
    for model, state in [
        (p.ArithmeticBrownianMotion(0.2, 0), 2.0),
        (p.OrnsteinUhlenbeck(2, 1, 0), 3.0),
        (p.LinearDiffusion([[0]], [[0]], [2]), [3.0]),
    ]:
        law = model.transition(state, 0.5).distribution()
        assert law.measure == "discrete"
        samples = law.sample(3, rng=np.random.default_rng(10))
        assert_allclose(samples, np.broadcast_to(law.mean(), samples.shape))
