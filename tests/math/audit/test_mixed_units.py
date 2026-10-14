"""Changes of coordinate units must preserve validity and probability laws."""

from datetime import date
from typing import Any, cast
import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm, multivariate_t
from convexity.math import (
    distributions as d,
    forecasting as f,
    gaussian_processes as gp,
    state_space as ss,
    optimization as opt,
    curves,
    estimation,
    bayesian,
)


def forecast(cov):
    n = len(cov)
    return f.ForecastResult(
        np.zeros((n, 1)),
        mean_values=np.zeros((n, 1)),
        gaussian=True,
        joint_covariance=cov,
    ).distribution()


@pytest.mark.parametrize(
    "factory",
    [
        lambda c: gp.GaussianPrediction(np.zeros(len(c)), c),
        lambda c: ss.GaussianState(np.zeros(len(c)), c),
        lambda c: opt.QuadraticProgram(np.zeros(len(c)), quadratic=c),
        forecast,
        lambda c: estimation.Estimate(
            np.zeros(len(c)), covariance=c, covariance_kind="sampling"
        ),
        lambda c: bayesian.GaussianLinearRegression(np.zeros(len(c)), c),
        lambda c: d.multivariate_normal(np.zeros(len(c)), c),
    ],
)
@pytest.mark.parametrize("small", [1.0, 1e-20])
def test_invalid_small_block_cannot_hide_under_large_valid_coordinate(factory, small):
    bad = np.array([[1e20, 0, 0], [0, small, 2 * small], [0, 2 * small, small]])
    with pytest.raises(ValueError, match="positive"):
        factory(bad)


@pytest.mark.parametrize(
    "factory",
    [
        lambda c: gp.GaussianPrediction(np.zeros(len(c)), c),
        lambda c: ss.GaussianState(np.zeros(len(c)), c).distribution(),
        forecast,
        lambda c: d.multivariate_normal(np.zeros(len(c)), c),
    ],
)
@pytest.mark.parametrize("variances", [[1e20, 1.0], [1.0, 1e-20], [1e-100, 1e100]])
def test_gaussian_density_keeps_every_positive_coordinate(factory, variances):
    scale = np.sqrt(variances)
    law = factory(np.diag(variances))
    points = np.array([[0, 0], [1, -2], [-0.5, 0.25]]) * scale
    expected = norm.logpdf(points / scale).sum(axis=1) - np.log(scale).sum()
    assert_allclose(law.log_density(points), expected, rtol=1e-13, atol=1e-13)
    draws = law.sample(20000, rng=np.random.default_rng(171)) / scale
    assert_allclose(draws.mean(axis=0), 0, atol=0.025)
    assert_allclose(np.cov(draws.T), np.eye(2), atol=0.025)
    assert_allclose(
        law.sample(7, rng=np.random.default_rng(7)),
        law.sample(7, rng=np.random.default_rng(7)),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda c: gp.GaussianPrediction(np.zeros(len(c)), c),
        lambda c: ss.GaussianState(np.zeros(len(c)), c).distribution(),
        forecast,
    ],
)
def test_singular_density_uses_original_support_volume(factory):
    loading = np.array([2.0, 3.0, 0.0])
    law = factory(np.outer(loading, loading))
    z = np.array([0.0, 1.0, -2.0])
    assert_allclose(
        law.log_density(z[:, None] * loading),
        norm.logpdf(z) - np.log(np.linalg.norm(loading)),
        atol=1e-12,
    )
    assert np.isneginf(law.log_density([2.0, 3.0, 1e-30]))
    assert np.isneginf(law.log_density([2.0, 3.1, 0.0]))


def test_student_t_density_changes_units_with_full_dimensional_jacobian():
    scale = np.array([1e-10, 1e10])
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    law = d.multivariate_student_t(5.0, [0.0, 0.0], scale[:, None] * corr * scale)
    z = np.array([[1.0, 2.0], [-1.0, 0.5]])
    assert_allclose(
        law.log_density(z * scale),
        cast(Any, multivariate_t).logpdf(z, shape=corr, df=5) - np.log(scale).sum(),
    )
    assert_allclose(
        law.cdf([0.0, 0.0], rng=np.random.default_rng(13)),
        d.multivariate_student_t(5.0, [0.0, 0.0], corr).cdf(
            [0.0, 0.0], rng=np.random.default_rng(13)
        ),
    )


def test_forecast_rejects_marginals_inconsistent_with_joint_law():
    with pytest.raises(ValueError, match="marginal laws"):
        f.ForecastResult(
            [[0.0]],
            law=d.multivariate_normal([0.0], [[1.0]]),
            marginal_laws=[d.normal(20.0, 2.0)],
        )


def test_curve_fit_returns_owned_dated_diagnostics_and_recovers_generated_rate():
    t = np.array([1.0, 2.0, 3.0])
    fit = curves.fit_discount_curve(
        "2026-01-01",
        t,
        np.eye(3),
        np.exp(-0.031 * t),
        [0.05],
        model=lambda t, p: np.exp(-p[0] * t),
    )
    assert isinstance(fit, curves.CurveFitResult)
    assert fit.reference_date == date(2026, 1, 1)
    assert fit.success
    assert_allclose(fit.parameters, [0.031], atol=1e-10)
    assert not fit.parameters.flags.writeable
    restored = pickle.loads(pickle.dumps(fit))
    assert_allclose(restored.parameters, fit.parameters)
    assert restored.reference_date == fit.reference_date
    with pytest.raises((ValueError, TypeError)):
        curves.fit_discount_curve(
            "invalid",
            t,
            np.eye(3),
            np.exp(-0.031 * t),
            [0.05],
            model=lambda t, p: np.exp(-p[0] * t),
        )


def test_correlated_innovations_and_chance_cone_preserve_mixed_units():
    from convexity.math import simulation, robust

    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    units = np.array([1e10, 1e-10])
    covariance = units[:, None] * corr * units
    draws = simulation.gaussian_innovations(
        40000, covariance, rng=np.random.default_rng(61)
    )
    assert_allclose(np.cov((draws / units).T), corr, atol=0.02)
    constraint = robust.gaussian_chance_constraint(
        [0.0, 0.0], covariance, 1.0, probability=0.95
    )
    # Inverse-unit portfolio is exposed equally to both standardized factors.
    portfolio = 1 / units
    assert_allclose(
        np.linalg.norm(constraint.matrix @ portfolio),
        norm.ppf(0.95) * np.sqrt(np.ones(2) @ corr @ np.ones(2)),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "estimator", [ss.ExtendedKalmanFilter(), ss.UnscentedKalmanFilter()]
)
def test_filter_and_smoother_commute_with_coordinate_scaling(estimator):
    pytest.importorskip("filterpy")

    def model(units):
        return ss.StateSpaceModel(
            ss.GaussianState([0.0, 0.0], np.diag(units**2)),
            ss.LinearGaussianTransition(np.eye(2), np.diag(0.1 * units**2)),
            ss.LinearGaussianObservation(np.diag(1 / units), 0.2 * np.eye(2)),
        )

    units = np.array([1e10, 1e-10])
    observations = [[1.0, -1.0], [0.5, -2.0], [0.2, 1.0]]
    expected = estimator.smooth(model(np.ones(2)), observations)
    actual = estimator.smooth(model(units), observations)
    for name in ("filtered", "predicted", "smoothed"):
        first, second = getattr(actual, name), getattr(expected, name)
        assert_allclose(first.mean / units, second.mean, atol=1e-12)
        assert_allclose(
            first.covariance / units[:, None] / units[None, :],
            second.covariance,
            atol=1e-12,
        )
    assert_allclose(actual.log_likelihoods, expected.log_likelihoods, atol=1e-12)


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
@pytest.mark.parametrize("df", [1e6, 1e20, 1e100, 1e308])
def test_multivariate_t_normal_limit_remains_finite(dimension, df):
    point = np.ones(dimension)
    law = d.multivariate_student_t(df, np.zeros(dimension), np.eye(dimension))
    assert_allclose(law.log_density(point), norm.logpdf(point).sum(), atol=1e-5)
