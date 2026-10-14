"""Generated-data recovery, shape conventions, tail calibration and failures."""

import numpy as np
import pytest
from typing import Any
from numpy.testing import assert_allclose
from scipy import stats

from convexity.math import dependence as dep


@pytest.mark.parametrize("estimate_df", [False, True])
def test_student_t_likelihood_recovers_shape_and_degrees_of_freedom(estimate_df):
    location = np.array([1.0, -2.0])
    shape = np.array([[2.0, 0.7], [0.7, 1.0]])
    x = dep.multivariate_student_t(6, location, shape).sample(
        5000, rng=np.random.default_rng(20)
    )
    result = dep.fit_student_t(
        x, df=6 if not estimate_df else 8, estimate_df=estimate_df
    )
    assert result.diagnostics.converged
    assert_allclose(result.location, location, atol=0.06)
    assert_allclose(result.shape, shape, rtol=0.07, atol=0.04)
    assert_allclose(result.df, 6, atol=0.9)
    assert_allclose(result.covariance, result.shape * result.df / (result.df - 2))
    assert_allclose(result.log_likelihood, result.distribution.log_density(x).sum())
    generator: Any = stats.multivariate_t
    initial_score = generator.logpdf(
        x, loc=x.mean(axis=0), shape=np.cov(x, rowvar=False) * (4 / 6), df=6
    ).sum()
    assert result.log_likelihood >= initial_score - 1e-5
    assert result.distribution.condition([0], [1.0]).event_shape == (1,)


def test_student_t_small_df_and_univariate_shape():
    x = dep.multivariate_student_t(1.5, [1.0], [[2.0]]).sample(
        700, rng=np.random.default_rng(21)
    )
    result = dep.StudentT(df=1.5).fit(x)
    assert result.shape.shape == (1, 1)
    assert_allclose(result.location, [1.0], atol=0.2)
    with pytest.raises(ValueError, match="finite covariance"):
        _ = result.covariance
    assert np.isinf(result.distribution.variance()).all()


def test_student_t_convergence_and_identifiability_failures():
    x = np.random.default_rng(22).standard_t(4, size=(200, 2))
    with pytest.raises(dep.ConvergenceError) as caught:
        dep.StudentT(maxiter=1).fit(x)
    assert not caught.value.result.diagnostics.converged
    result = dep.StudentT(maxiter=1, check_convergence=False).fit(x)
    assert not result.diagnostics.converged
    for data in (np.ones((10, 2)), np.eye(3), np.arange(20.0).reshape(10, 2)):
        with pytest.raises(ValueError):
            dep.fit_student_t(data)


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2])
def test_extreme_value_laws_match_scipy_and_exponential_gumbel_limits(xi):
    probabilities = np.array([0.0, 0.01, 0.5, 0.95, 1.0])
    for construct, scipy_law in [
        (
            dep.generalized_pareto(xi, scale=2, threshold=3),
            stats.genpareto(xi, loc=3, scale=2),
        ),
        (
            dep.generalized_extreme_value(xi, loc=3, scale=2),
            stats.genextreme(-xi, loc=3, scale=2),
        ),
    ]:
        scipy_law: Any = scipy_law
        assert_allclose(construct.quantile(probabilities), scipy_law.ppf(probabilities))
        values = construct.quantile(probabilities[1:-1])
        assert_allclose(construct.cdf(values), probabilities[1:-1])
        assert_allclose(construct.survival(values), 1 - probabilities[1:-1])
        assert_allclose(construct.log_density(values), scipy_law.logpdf(values))
        assert_allclose(construct.mean(), scipy_law.mean())
        assert_allclose(construct.variance(), scipy_law.var())
        assert_allclose(construct.moment(1), scipy_law.moment(1))
        assert_allclose(construct.support(), scipy_law.support())
        assert construct.sample((2, 0), rng=np.random.default_rng(23)).shape == (2, 0)
    if xi == 0:
        assert_allclose(
            dep.generalized_pareto(0).survival([0, 1, 2]), np.exp(-np.array([0, 1, 2]))
        )
        assert_allclose(
            dep.generalized_extreme_value(0).cdf([-1, 0, 1]),
            np.exp(-np.exp(-np.array([-1, 0, 1]))),
        )


@pytest.mark.parametrize(
    "family,xi", [("gpd", 0.2), ("gpd", -0.2), ("gev", 0.15), ("gev", -0.15)]
)
def test_extreme_value_parameter_recovery(family, xi):
    law = (
        dep.generalized_pareto(xi, scale=2)
        if family == "gpd"
        else dep.generalized_extreme_value(xi, loc=3, scale=2)
    )
    x = law.sample(6000, rng=np.random.default_rng(24))
    result = dep.fit_gpd(x) if family == "gpd" else dep.fit_gev(x)
    assert result.diagnostics.converged
    # Sampling tolerances span roughly four asymptotic standard errors.
    assert_allclose(result.xi, xi, atol=0.06)
    assert_allclose(result.scale, 2, atol=0.12)
    assert_allclose(result.location, 0 if family == "gpd" else 3, atol=0.08)
    assert_allclose(result.log_likelihood, result.distribution.log_density(x).sum())
    periods = np.array([2.0, 10.0, 100.0])
    assert_allclose(
        result.distribution.survival(result.return_level(periods)),
        1 / periods,
        atol=2e-14,
    )


def test_pot_uses_strict_exceedances_and_original_observation_rate():
    rng = np.random.default_rng(25)
    excesses = dep.generalized_pareto(0.2, scale=1.5).sample(2500, rng=rng)
    x = np.r_[np.zeros(7500), np.full(100, 3.0), 3 + excesses]
    fit = dep.peaks_over_threshold(x, 3.0)
    assert fit.nobs == 10100
    assert fit.tail.nobs == 2500
    assert_allclose(fit.exceedance_rate, 2500 / 10100)
    assert_allclose(fit.survival(3.0), fit.exceedance_rate)
    assert_allclose(fit.quantile(1 - fit.exceedance_rate), 3.0)
    assert_allclose(fit.return_level(1 / fit.exceedance_rate), 3.0)
    periods = np.array([10.0, 100.0, 1e5])
    assert_allclose(fit.survival(fit.return_level(periods)), 1 / periods, rtol=1e-13)
    assert_allclose(
        fit.quantile(1 - 1 / periods), fit.return_level(periods), rtol=1e-11
    )
    with pytest.raises(ValueError, match="below"):
        fit.quantile(0.2)
    with pytest.raises(ValueError, match="upper tail"):
        fit.survival(2.0)
    with pytest.raises(ValueError):
        fit.return_level(2.0)
    assert fit.diagnostics == fit.tail.diagnostics


def test_bounded_tail_and_stable_large_return_period():
    tail = dep.TailFit("gpd", -0.25, 0.0, 2.0, 100, -100.0)
    pot = dep.PeaksOverThresholdResult(4.0, tail, 1000)
    assert_allclose(pot.quantile(1.0), 12.0)
    assert_allclose(pot.survival([12.0, 13.0, np.inf]), 0.0)
    exponential = dep.TailFit("gpd", 0.0, 0.0, 2.0, 100, -100.0)
    result = dep.PeaksOverThresholdResult(4.0, exponential, 1000)
    assert_allclose(result.return_level(1e20), 4 + 2 * np.log(1e19))


def test_block_alignment_and_incomplete_blocks():
    x = np.array([1, 4, 3, 2, 8, 5, 10])
    with pytest.raises(ValueError, match="complete"):
        dep.block_maxima(x, 3)
    assert_allclose(dep.block_maxima(x, 3, drop_incomplete=True), [4, 8])
    assert_allclose(dep.block_maxima(x[:6], 3), [4, 8])
    with pytest.raises(ValueError):
        dep.block_maxima(x, 8, drop_incomplete=True)


@pytest.mark.parametrize("family", ["gpd", "gev"])
def test_tail_fit_reports_optimizer_failure(family):
    x = np.random.default_rng(26).exponential(size=100)
    fitter = dep.GeneralizedPareto if family == "gpd" else dep.GeneralizedExtremeValue
    with pytest.raises(dep.ConvergenceError) as caught:
        fitter(maxiter=1).fit(x)
    assert not caught.value.result.diagnostics.converged
    assert not fitter(maxiter=1, check_convergence=False).fit(x).diagnostics.converged


@pytest.mark.parametrize(
    "bad", [[1, 1, 1, 1], [1, 2], [[1, 2], [3, 4]], [1.0, np.nan, 3, 4]]
)
def test_invalid_tail_data(bad):
    for fitter in (dep.GeneralizedPareto(), dep.GeneralizedExtremeValue()):
        with pytest.raises(ValueError):
            fitter.fit(bad)


def test_negative_excesses_and_insufficient_exceedances():
    with pytest.raises(ValueError):
        dep.fit_gpd([-1.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        dep.peaks_over_threshold([1.0, 2.0, 3.0, 4.0], 3.0)
