"""Public-API counterexamples discovered in the 36-family adversarial audit."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import (
    control,
    credit,
    diagnostics,
    distributions,
    execution,
    forecasting,
    measures,
    numerical,
    optimization,
    risk,
)


def forecast_covariance(cov):
    return forecasting.ForecastResult(
        np.zeros((len(cov), 1)),
        mean_values=np.zeros((len(cov), 1)),
        gaussian=True,
        joint_covariance=cov,
    )


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize(
    "factory",
    [
        lambda c: optimization.QuadraticProgram([0.0, 0.0], quadratic=c),
        lambda c: risk.volatility_contributions([1.0, 0.0], c),
        lambda c: execution.QuadraticCost([0.0, 0.0], c),
        forecast_covariance,
    ],
)
@pytest.mark.parametrize(
    "matrix", [np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([[2.0, 0.6], [0.2, 1.0]])]
)
def test_invalid_matrices_stay_invalid_when_units_change(scale, factory, matrix):
    with pytest.raises(ValueError):
        factory(scale * matrix)


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_hermitian_solver_rejects_asymmetric_input_in_any_units(scale):
    with pytest.raises(ValueError, match="Hermitian"):
        numerical.eigh(scale * np.array([[1.0, 9.0], [0.0, 2.0]]))


@pytest.mark.parametrize("diagonal", [[1e20, -1.0], [1.0, -1e-20]])
def test_large_coordinate_cannot_hide_negative_quadratic_cost(diagonal):
    with pytest.raises(ValueError):
        optimization.QuadraticProgram([0.0, 0.0], quadratic=np.diag(diagonal))


@pytest.mark.parametrize("offset", [0.0, 1e10, 1e16, -1e16])
def test_crps_translation_invariance_against_pairwise_definition(offset):
    samples = np.arange(0.0, 14.0, 2.0) + offset
    actual = offset + 6.0
    w = np.ones(7) / 7
    reference = (
        w @ abs(samples - actual) - 0.5 * w @ abs(samples[:, None] - samples) @ w
    )
    assert_allclose(
        diagnostics.crps(samples, actual), reference, rtol=1e-13, atol=1e-13
    )
    assert diagnostics.crps([offset] * 7, offset) == 0


def test_forecast_cannot_claim_quantiles_for_laws_without_inversion():
    law = distributions.mixture(
        [distributions.bernoulli(0.2), distributions.bernoulli(0.8)]
    )
    assert "quantile" not in law.capabilities
    result = forecasting.ForecastResult([[0.5]], marginal_laws=[law])
    assert "quantile" not in result.capabilities
    assert "interval" not in result.capabilities
    with pytest.raises(NotImplementedError):
        result.quantile(0.5)


def test_explicit_quantiles_can_supplement_noninvertible_marginal_laws():
    law = distributions.mixture(
        [distributions.bernoulli(0.2), distributions.bernoulli(0.8)]
    )
    result = forecasting.ForecastResult(
        [[0.5]], marginal_laws=[law], quantile_levels=[0.5], quantile_values=[[[0.0]]]
    )
    assert "quantile" in result.capabilities
    assert_allclose(result.quantile(0.5), [[0.0]])


def test_small_inconsistent_forecast_variances_are_rejected():
    with pytest.raises(ValueError, match="diagonal"):
        forecasting.ForecastResult(
            [[0.0]],
            mean_values=[[0.0]],
            variance_values=[[1e-20]],
            gaussian=True,
            joint_covariance=[[1e-30]],
        )


@pytest.mark.parametrize("drift", [1e-20, 1e-13, 1.0])
def test_deterministic_drift_cannot_change_under_equivalent_measure(drift):
    pair = measures.gbm_pair(drift, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="span"):
        pair.market_price_of_risk(0.0, 1.0)


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_migration_generator_conservation_is_independent_of_time_units(scale):
    with pytest.raises(ValueError):
        credit.MigrationModel(scale * np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_hjb_zero_generator_has_analytic_solution_even_at_small_discount():
    result = control.generator_hjb(
        np.zeros((1, 2, 1)), [[1e-14, 2e-14]], 1e-14, maxiter=10
    )
    assert_allclose(result.values, [2.0])
    assert_allclose(result.policy, [1])


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_hjb_rejects_nonconservative_generator_in_any_time_units(scale):
    with pytest.raises(ValueError):
        control.generator_hjb(np.array([[[scale]]]), [[1.0]], 1.0, maxiter=2)


@pytest.mark.parametrize("offset", [0.0, 1e10, 1e16])
def test_forecasting_crps_agrees_with_diagnostics_under_translation(offset):
    x = offset + np.arange(0.0, 14.0, 2.0)
    f = forecasting.ForecastResult.from_paths(x[:, None, None])
    reference = diagnostics.crps(x, offset + 6.0)
    assert_allclose(
        forecasting.metrics.crps(f, [[offset + 6.0]]), reference, rtol=1e-13, atol=1e-13
    )


def test_ensemble_weights_are_invariant_to_large_common_scale():
    models = (forecasting.NaiveForecaster(), forecasting.DriftForecaster())
    expected = (
        forecasting.ensemble(models, weights=[1.0, 1.0])
        .fit([1.0, 2.0, 3.0])
        .forecast(2)
    )
    actual = (
        forecasting.ensemble(models, weights=[1e308, 1e308])
        .fit([1.0, 2.0, 3.0])
        .forecast(2)
    )
    assert_allclose(actual.point, expected.point)
