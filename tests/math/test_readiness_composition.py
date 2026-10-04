"""Public composition contracts consumed by the planned risk/control families."""

import importlib
import numpy as np
import pytest
import convexity.math as math
from convexity.math import simulation as sim, measures as me, pricing as pr
from convexity.math import contracts as ct, sensitivities as se


@pytest.mark.parametrize("family", math.__all__)
def test_every_family_public_export_resolves(family):
    module = importlib.import_module("convexity.math." + family)
    assert len(module.__all__) == len(set(module.__all__))
    assert all(hasattr(module, name) for name in module.__all__)


def test_process_measure_scenarios_forecast_and_pricing_share_one_path_law():
    pair = me.gbm_pair(0.08, 0.04, 0.01, 0.2)
    increments = sim.brownian_increments(
        [0.0, 0.5, 1.0], dimension=1, draws=10000, rng=np.random.default_rng(73)
    )
    paths = sim.geometric_brownian_paths([100.0], [0.03], [[0.2]], increments)
    forecast = paths.to_forecast()
    np.testing.assert_allclose(forecast.covariance(), paths.covariance(), atol=1e-11)
    np.testing.assert_allclose(forecast.mean(), paths.mean())
    # Risk mappings keep the same whole-path weights and temporal dependence.
    payoff = lambda path: max(path[-1, 0] - 100, 0)
    payoff_law = paths.map(payoff)
    result = pr.scenario_price(paths, payoff, pair.pricing)
    assert result.value == pytest.approx(np.exp(-0.04) * payoff_law.mean(), abs=1e-12)
    model = pr.BlackScholesModel.from_measure_pair(pair, 100, "2026-01-01")
    contract = ct.VanillaContract(100.0, ct.Exercise("european", ("2027-01-01",)))
    analytic = pr.price(
        pr.PricingProblem(contract, model, "2026-01-01"), pr.VanillaEngine()
    )
    assert result.error_estimate is not None
    assert abs(result.value - analytic.value) < 5 * result.error_estimate
    # Reweighting does not manufacture an IID Monte Carlo error estimate.
    weighted = me.reweight(paths, np.linspace(-2.0, 2.0, len(paths.values)))
    assert pr.scenario_price(weighted, payoff, pair.pricing).error_estimate is None
    sparse = forecast.select([0, 2])
    np.testing.assert_allclose(
        sparse.covariance(), forecast.covariance()[np.ix_([0, 2], [0, 2])]
    )


def test_pricing_greeks_feed_hedging_without_vendor_adapters():
    from convexity.math import curves as cv

    curve = cv.flat_curve("2026-01-01", 0.04)
    model = pr.BlackScholesModel(100.0, curve, cv.flat_curve("2026-01-01", 0.01), 0.2)

    def option(strike):
        claim = ct.VanillaContract(strike, ct.Exercise("european", ("2027-01-01",)))
        return pr.price(
            pr.PricingProblem(claim, model, "2026-01-01"), pr.VanillaEngine()
        )

    held, hedge_option = option(100), option(110)
    exposures = np.array(
        [[1.0, hedge_option.greeks["delta"]], [0.0, hedge_option.greeks["vega"]]]
    )
    target = np.array([held.greeks["delta"], held.greeks["vega"]])
    solution = se.hedge(exposures, target)
    np.testing.assert_allclose(exposures @ solution.holdings + target, 0, atol=1e-8)
