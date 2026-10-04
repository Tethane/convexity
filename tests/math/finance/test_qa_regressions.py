"""Adversarial financial contracts, independent of the adapter implementation."""

import numpy as np
import pytest
import QuantLib as ql
from convexity.math import contracts as ct, curves as cv, pricing as pr
from convexity.math import fixed_income as fi, credit as cr, implied_volatility as iv
from convexity.math import measures as me, stopping as st, rates as ra
from convexity.math.calibration import CalibrationProblem, CalibrationResult, calibrate

REF = "2026-01-01"


def market():
    curve = cv.flat_curve(REF, 0.04)
    return curve, pr.BlackScholesModel(100, curve, cv.flat_curve(REF, 0.01), 0.2)


def test_callable_bond_requires_exercise_engine():
    curve, _ = market()
    bond = fi.callable_bond(
        ct.schedule(REF, "2031-01-01"), 0.08, [("2028-01-01", 100, "call")]
    )
    with pytest.raises(ValueError, match="engine"):
        fi.bond_analytics(bond, curve)
    with pytest.raises(ValueError, match="engine"):
        fi.key_rate_duration(bond, curve, [REF, "2028-01-01", "2031-01-01"])
    with pytest.raises(ValueError, match="callable|exercise"):
        fi.carry_roll_down(bond, curve, "2027-01-01")


def test_cds_accrual_rebate_in_value_and_spread_risk():
    curve, _ = market()
    hazard = cr.flat_hazard(REF, 0.02)
    dates = ct.schedule("2025-12-20", "2030-12-20", months=3)
    result = cr.cds_price(dates, 0.012, curve, hazard)
    assert result["value"] == pytest.approx(
        result["premium_leg"] + result["protection_leg"], abs=1e-12
    )
    up = cr.cds_price(dates, 0.0121, curve, hazard)["value"]
    down = cr.cds_price(dates, 0.0119, curve, hazard)["value"]
    assert result["spread_sensitivity"] == pytest.approx(
        (up - down) / 0.0002, abs=1e-10
    )


def test_smile_calendar_checks_include_every_input_expiry():
    low = iv.SVI(0.01, 0.0, 0.0, 0.0, 0.1)
    high = iv.SVI(0.05, 0.0, 0.0, 0.0, 0.1)
    surface = iv.SmileSurface([1.0, 1.001, 1.002, 2.0], (low, high, low, high))
    assert not surface.diagnose([-1.0, 0.0, 1.0], time_points=3).passed
    with pytest.raises(ValueError):
        surface.diagnose([0.0], time_points=1)


def test_normal_density_includes_negative_strikes():
    from scipy.stats import norm

    k = np.array([-0.03, -0.01, 0.0, 0.01])
    sigma = 0.02
    density = iv.risk_neutral_density(
        lambda strike: pr.forward_option(-0.01, strike, sigma, 1, model="bachelier"),
        k,
        1.0,
        step=1e-5,
    )
    np.testing.assert_allclose(density, norm.pdf((k + 0.01) / sigma) / sigma, rtol=1e-6)


def test_measure_valuation_rejects_invalid_numeraire():
    class Bad:
        def value(self, t):
            return 1.0 if t == 0 else -1.0

    with pytest.raises(ValueError, match="numeraire"):
        me.MeasureModel(None, "Q", Bad()).value(  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
            np.array([10.0]), 1.0, expectation=np.mean
        )


def test_stopping_probability_tolerance():
    with pytest.raises(ValueError, match="stochastic"):
        st.StoppingProblem([[1.0], [2.0]], [[[1.000001]]], 1.0)


def test_lsmc_validation_rejects_negative_rewards():
    x = np.ones((10, 3))
    rewards = np.ones_like(x)
    policy = st.least_squares_policy(x, rewards, 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        policy.evaluate(x, -rewards, 1.0)


def test_seed_zero_does_not_claim_reproducibility():
    with pytest.raises(ValueError, match="seed"):
        pr.VanillaEngine("mc", seed=0)


def test_asian_accumulator_matches_past_fixings():
    _, model = market()
    with pytest.raises(ValueError, match="accumulator"):
        pr.asian_price(
            model, REF, 100, ["2027-01-01"], past_fixings=0, running_accumulator=100
        )
    with pytest.raises(ValueError, match="accumulator"):
        pr.asian_price(
            model,
            REF,
            100,
            ["2027-01-01"],
            average="geometric",
            past_fixings=1,
            running_accumulator=-1,
        )


def test_rate_engine_curve_consistency():
    curve, _ = market()
    other = cv.flat_curve(REF, 0.08)
    hw = ra.ShortRateModel("hull_white", dict(kappa=0.1, sigma=0.01), other)
    with ct.evaluation_date(REF):
        dates = ct.schedule("2027-01-01", "2030-01-01")
        cap = ql.Cap(ql.IborLeg([100.0], dates, ql.Euribor6M(curve.handle)), [0.04])
        with pytest.raises(ValueError, match="curve"):
            ra.rate_option_price(
                cap, curve, kind="cap_floor", method="analytic", model=hw
            )


def test_callable_curve_risk_reprices_option():
    curve, _ = market()
    dates = ct.schedule(REF, "2031-01-01")
    bond = fi.callable_bond(dates, 0.08, [("2028-01-01", 100, "call")])

    def engine(c):
        return ql.TreeCallableFixedRateBondEngine(
            ra.ShortRateModel(
                "hull_white", dict(kappa=0.1, sigma=0.01), c
            ).to_quantlib(),
            150,
        )

    risk = fi.key_rate_duration(bond, curve, [REF, "2031-01-01"], engine_factory=engine)
    h = 1e-4
    down = fi.bond_analytics(
        bond, curve.shifted(-h), engine=engine(curve.shifted(-h)), compute_yield=False
    ).dirty
    up = fi.bond_analytics(
        bond, curve.shifted(h), engine=engine(curve.shifted(h)), compute_yield=False
    ).dirty
    base = fi.bond_analytics(
        bond, curve, engine=engine(curve), compute_yield=False
    ).dirty
    assert risk.sum() == pytest.approx((down - up) / (2 * h * base), rel=3e-4)


def test_cir_negative_state_is_rejected():
    model = ra.ShortRateModel("cir", dict(r0=0.04, kappa=0.2, theta=0.04, sigma=0.1))
    with pytest.raises(ValueError, match="nonnegative"):
        model.discount_bond(0.0, 1.0, -0.1)


def test_curve_time_conventions_match_process_clock():
    curve, _ = market()
    with pytest.raises(ValueError, match="day|convention"):
        pr.BlackScholesModel(
            100, curve, cv.flat_curve(REF, 0.01, convention="actual360"), 0.2
        )


def test_calibration_validates_custom_solver_against_problem():
    class WrongSolver:
        def fit(self, problem):
            return CalibrationResult(
                [1, 2], [0, 0], np.eye(2), [1, 1], 2, None, True, "ok", 1, [0, 0]
            )

    problem = CalibrationProblem(lambda p: np.array([p[0]]), [1], [0.5])
    with pytest.raises(ValueError, match="shape|axes|parameter"):
        calibrate(problem, solver=WrongSolver())


def test_constant_coefficient_engines_reject_shaped_curves():
    curve = cv.zero_curve([REF, "2026-07-01", "2027-01-01"], [0.01, 0.1, 0.04])
    model = pr.BlackScholesModel(100, curve, cv.flat_curve(REF, 0), 0.2)
    contract = ct.VanillaContract(
        100, ct.Exercise("american", (REF, "2027-01-01")), "put"
    )
    with pytest.raises(ValueError, match="constant|flat"):
        pr.price(pr.PricingProblem(contract, model, REF), pr.VanillaEngine("tree"))
    assert (
        pr.price(pr.PricingProblem(contract, model, REF), pr.VanillaEngine("fd")).value
        > 0
    )
    with pytest.raises(ValueError, match="constant|flat"):
        pr.barrier_price(model, REF, "2027-01-01", 100, 80)


def test_zero_volatility_does_not_report_kink_greeks_as_zero():
    curve = cv.flat_curve(REF, 0)
    model = pr.BlackScholesModel(100, curve, curve, 0)
    contract = ct.VanillaContract(100, ct.Exercise("european", ("2027-01-01",)))
    result = pr.price(pr.PricingProblem(contract, model, REF), pr.VanillaEngine())
    assert result.value == 0 and not result.greeks


def test_basket_cannot_mix_numeraire_curves_that_only_agree_at_expiry():
    one = cv.zero_curve([REF, "2026-07-01", "2027-01-01"], [0.04, 0.02, 0.04])
    two = cv.zero_curve([REF, "2026-07-01", "2027-01-01"], [0.04, 0.10, 0.04])
    dividend = cv.flat_curve(REF, 0)
    models = [pr.BlackScholesModel(100, c, dividend, 0.2) for c in (one, two)]
    with pytest.raises(ValueError, match="numeraire"):
        pr.basket_price(models, [[1, 0], [0, 1]], [0.5, 0.5], REF, "2027-01-01", 100)


def test_lsmc_engine_repeats_calibration_and_valuation_streams():
    _, model = market()
    problem = pr.PricingProblem(
        ct.VanillaContract(100, ct.Exercise("american", (REF, "2027-01-01")), "put"),
        model,
        REF,
    )
    engine = pr.VanillaEngine("lsmc", steps=20, samples=2000, seed=73)
    assert pr.price(problem, engine).value == pr.price(problem, engine).value


@pytest.mark.parametrize("spread", [0.0, 0.012])
def test_current_trade_cds_reconciles_rebate_and_affine_spread_risk(spread):
    curve, _ = market()
    hazard = cr.flat_hazard(REF, 0.02)
    dates = ct.schedule("2025-12-20", "2030-12-20", months=3)

    def value(s):
        return cr.cds_price(dates, s, curve, hazard, trade_date=REF)

    result = value(spread)
    assert isinstance(result, cr.CDSResult)
    if spread:
        assert result.accrual_rebate > 0
    assert result.value == pytest.approx(result.premium_leg + result.protection_leg)
    assert result.spread_sensitivity == pytest.approx(
        (value(spread + 0.001).value - result.value) / 0.001, abs=1e-10
    )
    assert value(result.fair_spread).value == pytest.approx(0, abs=1e-10)


def test_floating_yield_uses_explicit_valuation_not_global_date():
    curve, _ = market()
    index = ql.Euribor6M(curve.handle)
    bond = fi.floating_bond(
        ct.schedule("2026-07-01", "2028-07-01"), index, fixing_days=2
    )
    prices = []
    for ambient in ("2025-01-01", "2030-01-01"):
        with ct.evaluation_date(ambient):
            prices.append(
                fi.price_from_yield(bond, 0.04, "2026-03-01", valuation_date=REF)
            )
            y = fi.yield_from_price(bond, prices[-1], "2026-03-01", valuation_date=REF)
            assert y == pytest.approx(0.04, abs=1e-9)
            assert ql.Settings.instance().evaluationDate == ct.to_quantlib_date(ambient)
    assert prices[0] == pytest.approx(prices[1], abs=1e-12)


def test_heston_rejects_mixed_process_clocks():
    curve, _ = market()
    with pytest.raises(ValueError, match="day-count"):
        pr.HestonModel(
            100,
            curve,
            cv.flat_curve(REF, 0.01, convention="actual360"),
            0.04,
            1,
            0.04,
            0.3,
            -0.5,
        )


def test_surface_finite_difference_recovers_constant_vol_price():
    curve, _ = market()
    vol = iv.black_variance_surface(
        REF,
        ["2026-07-01", "2027-01-01", "2028-01-01"],
        [30, 100, 300],
        np.full((3, 3), 0.2),
    )
    model = pr.BlackScholesModel(100, curve, cv.flat_curve(REF, 0.01), vol)
    problem = pr.PricingProblem(
        ct.VanillaContract(100, ct.Exercise("european", ("2027-01-01",))), model, REF
    )
    exact = pr.price(problem, pr.VanillaEngine()).value
    numerical = pr.price(problem, pr.VanillaEngine("fd")).value
    assert numerical == pytest.approx(exact, abs=0.005)
    with pytest.raises(ValueError, match="constant"):
        pr.basket_price([model], [[1]], [1], REF, "2027-01-01", 100)


def test_empty_arbitrage_report_cannot_certify_a_surface():
    with pytest.raises(ValueError):
        iv.ArbitrageReport([], [], 0.04)


def test_typed_path_result_owns_inputs_and_enforces_measure():
    values = np.ones((3, 2, 1))
    result = ra.RatePaths(values, [1, 1, 1], [1, 2])
    values[:] = 7
    assert np.all(result.values == 1)
    assert not result.values.flags.writeable
    assert result["measure"] == "terminal_forward"
    with pytest.raises(ValueError, match="measure"):
        ra.RatePaths(values, [1, 1, 1], [1, 2], "Q")


def test_custom_pricing_protocol_does_not_require_native_inheritance():
    class Model:
        measure = "Q"

    class Engine:
        def price(self, problem):
            return pr.PricingResult(12.0, engine="custom")

    assert pr.price(pr.PricingProblem(object(), Model(), REF), Engine()).value == 12


def test_policy_constructor_rejects_misaligned_time_axes():
    with pytest.raises(ValueError, match="aligned"):
        st.ExercisePolicy((None,), (True, True, True), 1)


def test_geometric_asian_does_not_flatten_term_structures():
    curve = cv.zero_curve([REF, "2026-07-01", "2027-01-01"], [0.01, 0.1, 0.04])
    model = pr.BlackScholesModel(100, curve, cv.flat_curve(REF, 0), 0.2)
    with pytest.raises(ValueError, match="constant|flat"):
        pr.asian_price(
            model, REF, 100, ["2026-07-01", "2027-01-01"], average="geometric"
        )


def test_native_zero_volatility_preserves_price_without_kink_greeks():
    curve = cv.flat_curve(REF, 0)
    vol = ql.BlackConstantVol(
        ct.to_quantlib_date(REF), ql.NullCalendar(), 0, ql.Actual365Fixed()
    )
    model = pr.BlackScholesModel(100, curve, curve, vol)
    contract = ct.VanillaContract(100, ct.Exercise("european", ("2027-01-01",)))
    result = pr.price(pr.PricingProblem(contract, model, REF), pr.VanillaEngine())
    assert result.value == 0 and not result.greeks
