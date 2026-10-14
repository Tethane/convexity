import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from convexity.math import simulation as s, processes as p


@pytest.mark.parametrize("method", ["mc", "sobol", "halton"])
def test_innovations_covariance_reproducibility_and_generator_advancement(method):
    cov = np.array([[1.0, -0.6], [-0.6, 2.0]])
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    x = s.gaussian_innovations(8192, cov, rng=rng, method=method)
    # IID Gaussian sample covariance has a known Wishart sampling error.
    # Fixed absolute thresholds made this depend on the chosen matrix root.
    if method == "mc":
        se = np.sqrt((cov**2 + np.outer(np.diag(cov), np.diag(cov))) / (len(x) - 1))
        assert np.all(abs(np.cov(x.T) - cov) < 5 * se)
    else:
        assert_allclose(np.cov(x.T), cov, atol=0.04)
    assert not np.array_equal(
        x, s.gaussian_innovations(8192, cov, rng=rng, method=method)
    )
    rng.bit_generator.state = state
    assert_array_equal(x, s.gaussian_innovations(8192, cov, rng=rng, method=method))
    singular = s.gaussian_innovations(8, [[1, 1], [1, 1]], rng=rng, method=method)
    assert_allclose(singular[:, 0], singular[:, 1])
    with pytest.raises(ValueError):
        x[0, 0] = 1


def test_brownian_irregular_grid_antithetics_and_coarsening():
    fine = s.brownian_increments(
        [2, 2.1, 2.5, 3],
        draws=12000,
        dimension=2,
        rng=np.random.default_rng(0),
        antithetic=True,
    )
    assert_array_equal(fine.values[:6000], -fine.values[6000:])
    assert_allclose(
        fine.values.var(axis=0), [[0.1, 0.1], [0.4, 0.4], [0.5, 0.5]], atol=0.015
    )
    coarse = fine.coarsen([0, 2, 3])
    assert_allclose(coarse.values[:, 0], fine.values[:, :2].sum(axis=1))
    assert_allclose(coarse.values.sum(axis=1), fine.values.sum(axis=1), atol=1e-15)
    assert coarse.sampling == "antithetic"
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    assert s.brownian_increments(
        [3], draws=8, rng=rng, method="sobol"
    ).values.shape == (8, 0, 1)
    assert rng.bit_generator.state == state


def test_vector_gbm_exact_cross_time_and_asset_log_covariance():
    times = [0, 0.3, 1.0]
    loading = np.array([[0.2, 0], [-0.15, 0.3]])
    drift = np.array([0.05, -0.02])
    noise = s.brownian_increments(
        times, draws=16384, dimension=2, rng=np.random.default_rng(1), method="sobol"
    )
    result = s.geometric_brownian_paths([10, 20], drift, loading, noise)
    assert result.exact and result.sampling == "sobol"
    logs = np.log(result.values[:, 1:] / [10, 20]).reshape(16384, 4)
    expected = np.kron(np.minimum.outer(times[1:], times[1:]), loading @ loading.T)
    assert_allclose(np.cov(logs.T), expected, atol=0.0003)
    assert_allclose(result.mean()[-1], np.array([10, 20]) * np.exp(drift), rtol=0.0003)
    absorbing = s.geometric_brownian_paths([0, 1], drift, loading, noise)
    assert np.all(absorbing.values[:, :, 0] == 0)


def test_euler_and_milstein_strong_convergence_on_same_brownian_path():
    model = p.ItoDiffusion(lambda t, x: 0.1 * x, lambda t, x: [0.4 * x])
    noise = s.brownian_increments(
        np.linspace(0, 1, 65), draws=500, rng=np.random.default_rng(18)
    )
    exact = np.exp(0.1 - 0.5 * 0.4**2 + 0.4 * noise.values[:, :, 0].sum(axis=1))
    errors = {method: [] for method in ("euler", "milstein")}
    for stride in (4, 2, 1):
        coarse = noise.coarsen(np.arange(0, 65, stride))
        for method in errors:
            options = (
                {"diffusion_derivative": lambda t, x: 0.4}
                if method == "milstein"
                else {}
            )
            paths = s.integrate(model, 1.0, coarse, method=method, **options)
            errors[method].append(np.sqrt(np.mean((paths.values[:, -1] - exact) ** 2)))
            assert not paths.exact
    assert errors["euler"][-1] < 0.65 * errors["euler"][0]
    assert errors["milstein"][-1] < 0.35 * errors["milstein"][0]
    assert errors["milstein"][-1] < 0.15 * errors["euler"][-1]


def test_euler_vector_rectangular_loading_and_nonzero_time_origin():
    model = p.ItoDiffusion(lambda t, x: [t, 2 * t], lambda t, x: [[1], [2]], (2,), 1)
    w = s.brownian_increments([2, 2.5, 3], draws=4, rng=np.random.default_rng(1))
    paths = s.integrate(model, [0, 0], w)
    assert_allclose(paths.values[:, :, 1], 2 * paths.values[:, :, 0])
    assert_allclose(paths.values[:, -1, 0] - w.values[:, :, 0].sum(axis=1), 2.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"draws": 3, "method": "sobol"},
        {"draws": 3, "antithetic": True},
        {"draws": 4, "method": "sobol", "antithetic": True},
        {"dimension": True},
        {"draws": 0},
        {"method": "bad"},
    ],
)
def test_driver_invalid_designs(kwargs):
    with pytest.raises(ValueError):
        s.brownian_increments([0, 1], rng=np.random.default_rng(1), **kwargs)


def test_invalid_covariance_and_milstein_scope():
    rng = np.random.default_rng(1)
    for covariance in ([[1, 2], [2, 1]], [[1e-30, 0], [0, -1e-30]], [[1, 1], [0, 1]]):
        with pytest.raises(ValueError):
            s.gaussian_innovations(4, covariance, rng=rng)
    w = s.brownian_increments([0, 1], draws=4, rng=rng)
    with pytest.raises(ValueError):
        s.integrate(p.BrownianMotion(), 0, w, method="milstein")
    with pytest.raises(ValueError):
        s.integrate(
            p.LinearDiffusion([[0]], [[1]]),
            [0],
            w,
            method="milstein",
            diffusion_derivative=lambda t, x: 0,
        )
    for indices in ([1], [0, 0, 1], [0, 2], [0.0, 1.0]):
        with pytest.raises(ValueError):
            w.coarsen(indices)
