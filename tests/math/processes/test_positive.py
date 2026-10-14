import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import processes as p


def test_gbm_law_moments_and_generator():
    model = p.GeometricBrownianMotion(0.07, 0.3)
    kernel = model.transition(100.0, 1.2)
    mean = 100 * np.exp(0.07 * 1.2)
    assert_allclose(kernel.mean(), mean)
    assert_allclose(kernel.covariance(), mean**2 * np.expm1(0.3**2 * 1.2))
    assert_allclose(model.generator(0, 100.0, 1.0, 0.0), 7.0)
    assert_allclose(model.generator(0, 100.0, 0.01, -0.0001), 0.07 - 0.5 * 0.3**2)
    samples = kernel.sample(80000, rng=np.random.default_rng(13))
    assert np.all(samples > 0)
    assert_allclose(np.log(samples / 100).mean(), (0.07 - 0.045) * 1.2, atol=0.004)
    assert_allclose(np.log(samples / 100).var(), 0.09 * 1.2, atol=0.002)


@pytest.mark.parametrize(
    "model,state",
    [
        (p.GeometricBrownianMotion(0.1, 0), 3.0),
        (p.GeometricBrownianMotion(0.1, 0.5), 0.0),
        (p.CIR(2, 0.04, 0), 0.1),
    ],
)
def test_deterministic_positive_models(model, state):
    law = model.transition(state, 1.0).distribution()
    assert law.measure == "discrete"
    assert law.variance() == 0


@pytest.mark.parametrize("sigma", [0.15, 0.7])
def test_cir_exact_moments_boundary_and_feller(sigma):
    model = p.CIR(2.0, 0.04, sigma)
    assert model.feller_satisfied == (sigma**2 <= 0.16)
    for x in (0.0, 0.03):
        dt = 0.7
        decay = np.exp(-2 * dt)
        mean = 0.04 + (x - 0.04) * decay
        variance = (
            x * sigma**2 * decay * (1 - decay) / 2
            + 0.04 * sigma**2 * (1 - decay) ** 2 / 4
        )
        kernel = model.transition(x, dt)
        assert_allclose(kernel.mean(), mean)
        assert_allclose(kernel.covariance(), variance)
        samples = kernel.sample(100000, rng=np.random.default_rng(41))
        assert np.all(samples >= 0)
        assert_allclose(samples.mean(), mean, atol=5 * np.sqrt(variance / len(samples)))
        assert_allclose(
            kernel.distribution().cdf(kernel.distribution().quantile([0.1, 0.5, 0.9])),
            [0.1, 0.5, 0.9],
        )
    assert_allclose(model.stationary().mean(), 0.04)
    assert_allclose(model.stationary().variance(), 0.04 * sigma**2 / 4)
    assert_allclose(model.generator(0, 0.03, 1.0, 0.0), 0.02)
    with pytest.raises(ValueError):
        model.transition(0.04, 1.0, method="euler")


@pytest.mark.parametrize("model", [p.CIR(1, 0.04, 0.2), p.GeometricBrownianMotion()])
def test_invalid_positive_state(model):
    with pytest.raises(ValueError):
        model.transition(-1.0, 0.1)
