import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import processes as p, distributions as d


def test_compound_poisson_additive_moments_and_generator():
    model = p.AdditiveJumpDiffusion(0.3, [0.2], 1.7, d.normal(-0.2, 0.4))
    k = model.transition(2.0, 0.8)
    assert k.exact and "distribution" not in k.capabilities
    assert_allclose(k.mean(), 2.0 + (0.3 - 1.7 * 0.2) * 0.8)
    assert_allclose(k.covariance(), (0.04 + 1.7 * (0.16 + 0.04)) * 0.8)
    samples = k.sample(100000, rng=np.random.default_rng(11))
    assert_allclose(samples.mean(), k.mean(), atol=0.008)
    assert_allclose(samples.var(), k.covariance(), atol=0.009)
    assert_allclose(
        model.generator(0, 2.0, 4.0, 2.0, function=lambda x: x * x),
        4 * 0.3 + 0.04 + 1.7 * (4 * (-0.2) + 0.16 + 0.04),
    )
    with pytest.raises(NotImplementedError):
        k.distribution()


def test_vector_jumps_and_zero_intensity():
    marks = d.empirical([[1, -1], [2, -2]])
    model = p.AdditiveJumpDiffusion([0, 0], np.zeros((2, 1)), 2.0, marks)
    kernel = model.transition([0, 0], 1.0)
    assert_allclose(kernel.mean(), [3, -3])
    assert_allclose(kernel.covariance(), [[5, -5], [-5, 5]])
    samples = kernel.sample((4, 5), rng=np.random.default_rng(0))
    assert_allclose(samples[..., 0] + samples[..., 1], 0)
    assert_allclose(
        model.generator(0, [0, 0], [0, 0], 2 * np.eye(2), function=lambda x: x @ x), 10
    )
    kernel = p.AdditiveJumpDiffusion(0.0, [0.0], 0.0, d.student_t(1)).transition(
        2.0, 1.0
    )
    assert kernel.distribution().measure == "discrete"
    assert kernel.mean() == 2.0


def test_jump_moments_are_not_invented_for_heavy_tails():
    k = p.AdditiveJumpDiffusion(0.0, [1.0], 1.0, d.student_t(1)).transition(0.0, 1.0)
    assert k.capabilities == {"sample"}
    with pytest.raises(NotImplementedError):
        k.mean()


def test_merton_exact_compensation_moments_and_log_count_mixture():
    model = p.MertonJumpDiffusion(0.06, 0.2, 1.3, -0.1, 0.25)
    k = model.transition(100.0, 0.7)
    samples = k.sample(180000, rng=np.random.default_rng(123))
    assert_allclose(k.mean(), 100 * np.exp(0.06 * 0.7))
    assert_allclose(samples.mean(), k.mean(), atol=0.4)
    assert_allclose(samples.var(), k.covariance(), rtol=0.025)
    expected_log = (0.06 - 1.3 * model.compensator - 0.5 * 0.2**2 + 1.3 * (-0.1)) * 0.7
    assert_allclose(np.log(samples / 100).mean(), expected_log, atol=0.003)
    assert_allclose(
        model.generator(
            0,
            100.0,
            1.0,
            0.0,
            function=lambda x: x,
            jump_expectation=100 * model.compensator,
        ),
        6.0,
    )
    assert_allclose(
        model.generator(0, 100.0, 1 / 100, -1 / 100**2, function=np.log),
        0.06 - 1.3 * model.compensator - 0.02 + 1.3 * (-0.1),
    )


def test_merton_reductions_and_atoms():
    no_jumps = p.MertonJumpDiffusion(0.03, 0.2, 0.0, 0.0, 0.0).transition(1.0, 0.7)
    gbm = p.GeometricBrownianMotion(0.03, 0.2).transition(1.0, 0.7)
    assert_allclose(no_jumps.mean(), gbm.mean())
    model = p.MertonJumpDiffusion(0.0, 0.0, 2.0, -0.3, 0.0)
    k = model.transition(1.0, 0.5)
    samples = k.sample(50000, rng=np.random.default_rng(8))
    no_jump = np.exp(-2 * model.compensator * 0.5)
    assert_allclose(np.mean(samples == no_jump), np.exp(-1), atol=0.006)
    assert model.transition(0.0, 1).mean() == 0
