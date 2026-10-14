import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import rough as r


def test_eta_zero_recovers_gbm_on_identical_increments():
    model = r.RoughBergomi(eta=0, xi0=0.04, mu=0.03)
    result = model.sample(
        100, [0, 0.1, 0.4, 1], draws=20000, rng=np.random.default_rng(14)
    )
    expected = 100 * np.exp(
        0.01 * result.times[1:] + 0.2 * np.cumsum(result.asset_increments, axis=1)
    )
    assert_allclose(result.spot[:, 1:], expected)
    assert_allclose(result.variance, 0.04)
    assert_allclose(result.spot[:, -1].mean(), 100 * np.exp(0.03), atol=0.5)
    assert not result.paths.exact and result.driver.exact


@pytest.mark.parametrize("rho", [-1.0, -0.7, 0.0, 1.0])
def test_leverage_and_martingale_grid(rho):
    result = r.RoughBergomi(eta=0.5, rho=rho).sample(
        100, np.linspace(0, 1, 17), draws=25000, rng=np.random.default_rng(14)
    )
    w = result.driver.brownian_increments.ravel()
    b = result.asset_increments.ravel()
    assert_allclose(np.corrcoef(w, b)[0, 1], rho, atol=0.006)
    assert_allclose(result.spot[:, -1].mean(), 100, atol=0.7)
    assert_allclose(result.variance.mean(axis=0), 0.04, atol=0.0007)
    # The first return uses deterministic initial variance, not future variance.
    assert_allclose(
        np.log(result.spot[:, 1] / 100),
        -0.5 * 0.04 / 16 + 0.2 * result.asset_increments[:, 0],
    )


def test_finite_factor_bergomi_and_deterministic_curve_refinement():
    kernel, error = r.approximate_fractional(0.2, factors=8)
    result = r.VolterraBergomi(kernel, eta=0.6).sample(
        100, [0, 0.2, 1], draws=100, rng=np.random.default_rng(14), method="factors"
    )
    assert result.driver.factors is not None
    assert result.driver.factors.shape == (100, 3, 8)
    assert result.driver.exact and not result.paths.exact
    assert error.squared_error > 0
    # eta=0 and xi0(t)=.04+.02t: exact integrated variance is .05 at T=1.
    # The sampled log-Euler variance is the left Riemann sum; bias halves.
    errors = []
    for n in [8, 16, 32]:
        grid = np.linspace(0, 1, n + 1)
        paths = r.RoughBergomi(eta=0, xi0=lambda t: 0.04 + 0.02 * t).sample(
            1, grid, draws=2, rng=np.random.default_rng(14)
        )
        errors.append(0.05 - np.sum(paths.variance[0, :-1] * np.diff(grid)))
    assert_allclose(np.array(errors[:-1]) / errors[1:], 2)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: r.FractionalKernel(0),
        lambda: r.FractionalBrownianMotion(1),
        lambda: r.RoughBergomi(hurst=0.8),
        lambda: r.RoughBergomi(rho=1.1),
        lambda: r.RoughBergomi(eta=-1),
        lambda: r.RoughBergomi(xi0=0),
        lambda: r.ExponentialKernel([1], [-1]),
        lambda: r.ExponentialKernel([1, 2], [1]),
        lambda: r.ForwardVarianceCurve([0, 1], [0.04, -0.1]),
        lambda: r.approximate_fractional(0.1, factors=True),
        lambda: r.approximate_fractional(0.1, rate_min=2, rate_max=1),
    ],
)
def test_invalid_parameters(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


@pytest.mark.parametrize(
    "grid", [[0.1, 0.2], [0, 0], [0, -0.1], [0, np.inf], [], [0, True]]
)
def test_invalid_history_grid(grid):
    with pytest.raises(ValueError):
        r.VolterraGaussian(r.FractionalKernel()).sample(
            grid, rng=np.random.default_rng(14)
        )


def test_methods_rng_shapes_and_boundary_validation():
    model = r.RoughBergomi()
    with pytest.raises(TypeError, match="Generator"):
        model.sample(100, [0, 1], rng=14)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(ValueError):
        model.sample(100, [0], draws=0, rng=np.random.default_rng(14))
    with pytest.raises(ValueError, match="ExponentialKernel"):
        model.sample(100, [0], method="factors", rng=np.random.default_rng(14))
    with pytest.raises(ValueError, match="method"):
        model.sample(100, [0], method="euler", rng=np.random.default_rng(14))
    with pytest.raises(ValueError, match="maturities"):
        model.variance_model.sample_forward(1, [0.5], rng=np.random.default_rng(14))
    with pytest.raises(ValueError, match="increments"):
        model.variance_model.forward_from_brownian([0, 0.1], [0.1, 0.2], [0.1])
    rng = np.random.default_rng(14)
    before = str(rng.bit_generator.state)
    result = model.sample(100, [0], rng=rng)
    assert str(rng.bit_generator.state) == before
    assert result.spot[0, 0] == 100
    assert_allclose(result.variance, 0.04)
    assert not result.asset_increments.flags.writeable
