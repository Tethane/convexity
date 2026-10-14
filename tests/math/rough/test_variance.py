import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import rough as r


def test_curve_interpolation_ownership_and_no_extrapolation():
    values = np.array([0.04, 0.09])
    curve = r.ForwardVarianceCurve([0, 1], values)
    values[:] = 3
    assert_allclose(curve([0, 0.5, 1]), [0.04, 0.06, 0.09])
    assert_allclose(curve(0.5), 0.06)
    with pytest.raises(ValueError, match="extrapolate"):
        curve([1.1])


def test_lognormal_variance_mean_and_log_covariance():
    grid = [0, 0.1, 0.5, 1]
    kernel = r.FractionalKernel(0.15)
    driver = r.VolterraGaussian(kernel)
    values = driver.sample(grid, draws=80000, rng=np.random.default_rng(14)).values
    curve = r.ForwardVarianceCurve([0, 1], [0.04, 0.09])
    model = r.LognormalVolterraVariance(kernel, curve, 0.7)
    variance = model.evaluate(grid, values)
    covariance = driver.covariance(grid)
    assert_allclose(variance.mean(axis=0), curve(grid), rtol=0.012)
    assert_allclose(
        np.cov(np.log(variance), rowvar=False), 0.49 * covariance, atol=0.007
    )


def test_forward_full_state_mean_and_covariance_compensation():
    k = r.FractionalKernel(0.1)
    model = r.LognormalVolterraVariance(k, 0.04, 0.8)
    result = model.sample_forward(
        0.3, [0.3, 0.7, 1.0], draws=70000, rng=np.random.default_rng(14)
    )
    assert_allclose(result.values.mean(axis=0), 0.04, atol=0.0006)
    covariance = np.array(
        [
            [k.covariance(t, s, until=0.3) for t in result.maturities]
            for s in result.maturities
        ]
    )
    assert_allclose(
        np.cov(np.log(result.values), rowvar=False), 0.64 * covariance, atol=0.008
    )
    assert result.method == "full-brownian-filtration"
    initial = model.sample_forward(0, [0, 1], draws=3, rng=np.random.default_rng(14))
    assert_allclose(initial.values, 0.04)


def test_brownian_history_conditions_out_unobserved_bridge_variance():
    k = r.FractionalKernel(0.1)
    model = r.LognormalVolterraVariance(k, 0.04, 0.9)
    result = model.forward_from_brownian([0, 0.5], [0.2], [0.5, 1])
    weights = np.array([k.integral(0, 0.5), k.integral(0.5, 1)]) / 0.5
    explained = weights**2 * 0.5
    assert_allclose(
        result.values[0], 0.04 * np.exp(0.9 * weights * 0.2 - 0.5 * 0.9**2 * explained)
    )
    # Independently integrate the conditional residual Gaussian at maturity.
    total = np.array([k.covariance(t, t) for t in [0.5, 1]])
    rng = np.random.default_rng(14)
    residual = rng.standard_normal((150000, 2)) * np.sqrt(total - explained)
    conditional_samples = 0.04 * np.exp(
        0.9 * (weights * 0.2 + residual) - 0.5 * 0.9**2 * total
    )
    assert_allclose(conditional_samples.mean(axis=0), result.values[0], rtol=0.008)
    assert result.method == "observed-brownian-increments"
    assert_allclose(model.forward_from_brownian([0], [], [0, 1]).values, 0.04)


def test_forward_tower_property_under_history_refinement():
    model = r.LognormalVolterraVariance(r.FractionalKernel(0.15), 0.04, 0.6)
    # Fine halves conditional on a fixed W(1): dW0 ~ N(W(1)/2, 1/4).
    rng = np.random.default_rng(14)
    first = 0.1 + 0.5 * rng.standard_normal(100000)
    fine = model.forward_from_brownian(
        [0, 0.5, 1], np.column_stack((first, 0.2 - first)), [1, 2]
    )
    coarse = model.forward_from_brownian([0, 1], [0.2], [1, 2])
    assert_allclose(fine.values.mean(axis=0), coarse.values[0], rtol=0.003)


def test_forward_factors_recover_current_variance_and_future_means():
    kernel = r.ExponentialKernel([0.7, 0.3], [0.2, 3.0])
    model = r.LognormalVolterraVariance(kernel, eta=0.7)
    paths = r.VolterraGaussian(kernel).sample_joint(
        [0, 0.3, 0.8], draws=40000, rng=np.random.default_rng(14), method="factors"
    )
    assert paths.factors is not None
    result = model.forward_from_factors(0.8, paths.factors[:, -1], [0.8, 1, 2])
    variance = model.evaluate(paths.times, paths.values)
    assert_allclose(result.values[:, 0], variance[:, -1])
    assert_allclose(result.values.mean(axis=0), 0.04, atol=0.0006)
    # Continue a supplied factor state with the existing exact OU kernel.
    assert paths.factors is not None
    state = paths.factors[0, -1]
    continuation = (
        kernel.factor_process()
        .transition(state, 0.2)
        .sample(40000, rng=np.random.default_rng(3))
    )
    terminal = model.forward_from_factors(1, continuation, [1])
    assert_allclose(terminal.values.mean(), result.values[0, 1], rtol=0.008)
