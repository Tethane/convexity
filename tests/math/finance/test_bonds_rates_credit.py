import numpy as np
import pytest
import QuantLib as ql
from convexity.math._finance import evaluation_date, qdate
from convexity.math.contracts import schedule
from convexity.math.curves import flat_curve, zero_curve
from convexity.math.fixed_income import *
from convexity.math.rates import *
from convexity.math.credit import *


def test_bond_yield_duration_convexity(discount):
    dates = schedule("2025-01-01", "2031-01-01")
    bond = fixed_bond(dates, 0.06)
    result = bond_analytics(bond, discount, settlement_date="2026-03-01")
    assert result.dirty == pytest.approx(result.clean + result.accrued)
    assert result.accrued > 0
    assert result.yield_to_maturity is not None
    assert price_from_yield(
        bond, result.yield_to_maturity, result.settlement_date
    ) == pytest.approx(result.clean, abs=1e-9)
    h = 1e-4
    plus = price_from_yield(bond, result.yield_to_maturity + h, result.settlement_date)
    minus = price_from_yield(bond, result.yield_to_maturity - h, result.settlement_date)
    assert (minus - plus) / (2 * h * result.dirty) == pytest.approx(
        result.duration, rel=1e-6
    )
    assert (minus - 2 * result.clean + plus) / (h * h * result.dirty) == pytest.approx(
        result.convexity, rel=1e-6
    )
    zero = zero_coupon("2031-01-01")
    assert bond_analytics(zero, discount).dirty == pytest.approx(
        100 * discount.discount("2031-01-01")
    )


def test_amortization_and_key_rate_risk(discount):
    ds = schedule("2026-01-01", "2029-01-01", months=12)
    bond = amortizing_bond(ds, [100, 70, 40], 0.05)
    flows = cashflows(bond, "2026-01-01")
    principal = sum(cf.amount for cf in flows) - sum(
        n * 0.05 * ql.Actual365Fixed().yearFraction(ds[i], ds[i + 1])
        for i, n in enumerate([100, 70, 40])
    )
    assert principal == pytest.approx(100)
    ordinary = fixed_bond(ds, 0.05)
    krd = key_rate_duration(
        ordinary, discount, ["2026-01-01", "2027-01-01", "2029-01-01"]
    )
    base = bond_analytics(ordinary, discount).dirty
    down = bond_analytics(ordinary, discount.shifted(-1e-4)).dirty
    up = bond_analytics(ordinary, discount.shifted(1e-4)).dirty
    assert krd.sum() == pytest.approx((down - up) / (2e-4 * base), rel=1e-7)
    carry = carry_roll_down(ordinary, discount, "2027-01-01")
    assert carry["horizon_value"] + carry["reinvested_coupons"] - carry[
        "initial_value"
    ] == pytest.approx(carry["carry"] + carry["roll_down"])


def test_floating_bond_forecast(discount):
    ds = schedule("2026-07-01", "2028-07-01")
    index = ql.Euribor6M(discount.handle)
    bond = floating_bond(ds, index, fixing_days=2)
    result = bond_analytics(bond, discount, compute_yield=False)
    assert 90 < result.dirty < 105
    assert all(
        np.isfinite(cf.amount) for cf in cashflows(bond, discount.reference_date)
    )


def test_callable_bond(discount):
    ds = schedule("2026-01-01", "2031-01-01")
    plain = fixed_bond(ds, 0.07)
    callable = callable_bond(ds, 0.07, [("2028-01-01", 100, "call")])
    putable = callable_bond(ds, 0.07, [("2028-01-01", 110, "put")])
    model = ShortRateModel("hull_white", {"kappa": 0.1, "sigma": 0.01}, discount)
    engine = ql.TreeCallableFixedRateBondEngine(model.to_quantlib(), 200)
    cp = bond_analytics(callable, discount, engine=engine, compute_yield=False).dirty
    pp = bond_analytics(plain, discount).dirty
    put = bond_analytics(putable, discount, engine=engine, compute_yield=False).dirty
    assert cp < pp and put >= pp - 0.05


@pytest.mark.parametrize(
    "kind,params",
    [
        ("vasicek", dict(r0=0.05, kappa=0.2, theta=0.05, sigma=0.01)),
        ("cir", dict(r0=0.05, kappa=0.2, theta=0.05, sigma=0.1)),
        ("hull_white", dict(kappa=0.2, sigma=0.01)),
        ("g2", dict(a=0.2, sigma=0.01, b=0.3, eta=0.02, rho=-0.5)),
    ],
)
def test_rate_models(kind, params, discount):
    model = ShortRateModel(kind, params, discount)
    state = [0.0, 0.0] if kind == "g2" else 0.05
    assert model.discount_bond(0, 0, state) == pytest.approx(1)
    assert 0 < model.discount_bond(0, 5, state) < 1
    if kind in ("hull_white", "g2"):
        assert model.discount_bond(0, 5, state) == pytest.approx(
            discount.discount(5), rel=1e-10
        )
    if kind in ("vasicek", "cir"):
        physical = ShortRateModel(kind, params, measure="P")
        process = physical.process()
        assert not isinstance(process, ql.HullWhiteProcess)
        assert process.transition(0.05, 1).mean() > 0
        with pytest.raises(ValueError):
            physical.discount_bond(0, 1, 0.05)


def test_cap_and_swaption_engines(discount):
    with evaluation_date(discount.reference_date):
        ds = schedule("2027-01-01", "2030-01-01")
        index = ql.Euribor6M(discount.handle)
        leg = ql.IborLeg([100.0], ds, index)
        cap = ql.Cap(leg, [0.04])
        hw = ShortRateModel("hull_white", {"kappa": 0.1, "sigma": 0.01}, discount)
        for method, vol in [
            ("black", 0.2),
            ("normal", 0.01),
            ("analytic", None),
            ("tree", None),
        ]:
            result = rate_option_price(
                cap, discount, kind="cap_floor", method=method, volatility=vol, model=hw
            )
            assert result.value > 0
        swap = ql.VanillaSwap(
            ql.VanillaSwap.Payer,
            100.0,
            ds,
            0.04,
            ql.Actual365Fixed(),
            ds,
            index,
            0.0,
            ql.Actual360(),
        )
        option = ql.Swaption(swap, ql.EuropeanExercise(qdate("2027-01-01")))
        for method, vol in [
            ("black", 0.2),
            ("normal", 0.01),
            ("analytic", None),
            ("tree", None),
        ]:
            result = rate_option_price(
                option,
                discount,
                kind="swaption",
                method=method,
                volatility=vol,
                model=hw,
            )
            assert result.value > 0
        g2 = ShortRateModel(
            "g2", dict(a=0.1, sigma=0.01, b=0.2, eta=0.015, rho=-0.5), discount
        )
        assert (
            rate_option_price(
                option, discount, kind="swaption", method="g2", model=g2
            ).value
            > 0
        )


def test_hjm_market_model(discount):
    gsr = gaussian_hjm(discount, ["2027-01-01"], [0.01, 0.015], [0.1])
    assert gsr.zerobond(2.0) == pytest.approx(discount.discount(2), rel=1e-9)
    rate_times = [0.5, 1.0, 1.5, 2.0]
    evolution = ql.EvolutionDescription(rate_times)
    corr = ql.ExponentialForwardCorrelation(rate_times)
    native = ql.AbcdVol(
        0.0,
        0.0,
        1.0,
        0.2,
        [1.0, 1.0, 1.0],
        corr,
        evolution,
        2,
        [0.04, 0.04, 0.04],
        [0.0, 0.0, 0.0],
    )
    paths = market_model_paths(native, draws=1000, seed=41)
    assert paths["values"].shape == (1000, 3, 3)
    assert np.all(paths["values"] > 0)
    assert paths["measure"] == "terminal_forward"
    # The last forward is a martingale under the terminal bond measure.
    assert paths["values"][:, -1, -1].mean() == pytest.approx(0.04, abs=0.001)


def test_inflation_bond_and_swap(discount):
    reference = discount.reference_date
    curve = inflation_curve(
        reference, ["2025-10-01", "2026-10-01", "2030-10-01"], [0.02, 0.02, 0.02]
    )
    index = inflation_index(curve)
    try:
        index.addFixing(qdate("2025-10-01"), 100.0, True)
        ds = schedule("2026-01-01", "2029-01-01", months=12)
        bond = inflation_bond(ds, index, 100.0, 0.01)
        flows = cashflows(bond, reference)
        assert flows[-1].amount > 100
        assert bond_analytics(bond, discount, compute_yield=False).dirty > 0
        swap = ql.ZeroCouponInflationSwap(
            ql.Swap.Payer,
            100.0,
            qdate("2026-01-01"),
            qdate("2029-01-01"),
            ql.NullCalendar(),
            ql.Unadjusted,
            ql.Actual365Fixed(),
            0.02,
            index,
            ql.Period(3, ql.Months),
            ql.CPI.Flat,
        )
        assert abs(inflation_swap_price(swap, discount).value) < 0.01
        yoy = inflation_curve(
            reference,
            ["2025-10-01", "2026-10-01", "2030-10-01"],
            [0.02, 0.02, 0.02],
            year_on_year=True,
        )
        assert inflation_index(yoy, year_on_year=True) is not None
    finally:
        index.clearFixings()


def test_default_distribution_and_risky_identity(discount, valuation):
    hazard = flat_hazard(valuation, 0.02)
    assert hazard.survival(3) == pytest.approx(np.exp(-0.06))
    assert hazard.default_probability(1, 3) == pytest.approx(
        np.exp(-0.02) - np.exp(-0.06)
    )
    assert hazard.density(2) == pytest.approx(0.02 * np.exp(-0.04))
    draws = hazard.sample(10000, horizon=5, rng=np.random.default_rng(3))
    probability = 1 - np.exp(-0.1)
    assert np.isfinite(draws).mean() == pytest.approx(
        probability, abs=5 * np.sqrt(probability * (1 - probability) / len(draws))
    )
    actual = risky_cashflows(
        [3.0], [100.0], discount, hazard, recovery=0.4, recovery_notional=100
    )
    expected = 100 * np.exp(-0.07 * 3) + 40 * 0.02 / 0.07 * (1 - np.exp(-0.07 * 3))
    assert actual == pytest.approx(expected, abs=1e-8)
    with pytest.raises(ValueError):
        risky_cashflows(
            [3.0], [100.0], discount, flat_hazard(valuation, 0.02, measure="P")
        )
    curve = survival_curve([valuation, "2027-01-01", "2028-01-01"], [1, 0.98, 0.96])
    assert curve.survival(1) == pytest.approx(0.98)
    with pytest.raises(ValueError):
        survival_curve([valuation, "2027-01-01"], [1, 1.1])


def test_cds_calibration_reprice(discount, valuation):
    with evaluation_date(valuation):
        helpers = [
            ql.SpreadCdsHelper(
                spread,
                ql.Period(years, ql.Years),
                0,
                ql.NullCalendar(),
                ql.Quarterly,
                ql.Unadjusted,
                ql.DateGeneration.Forward,
                ql.Actual360(),
                0.4,
                discount.handle,
            )
            for years, spread in [(1, 0.01), (3, 0.012), (5, 0.014)]
        ]
        curve = bootstrap_cds(valuation, helpers)
        for helper in helpers:
            assert helper.impliedQuote() == pytest.approx(
                helper.quote().value(), abs=1e-8
            )
    ds = schedule(valuation, "2031-01-01", months=3)
    first = cds_price(ds, 0.01, discount, curve)
    par = cds_price(ds, first["fair_spread"], discount, curve)
    assert par["value"] == pytest.approx(0, abs=1e-10)
    assert first["value"] == pytest.approx(
        first["premium_leg"] + first["protection_leg"]
    )
    up = cds_price(ds, 0.0101, discount, curve)["value"]
    down = cds_price(ds, 0.0099, discount, curve)["value"]
    assert (up - down) / 0.0002 == pytest.approx(first["spread_sensitivity"])


def test_physical_hazard_migration_structural():
    estimate = estimate_hazard([2, 3, 5], [True, False, True])
    assert estimate["hazard"] == 0.2 and estimate["measure"] == "P"
    migration = MigrationModel([[-0.1, 0.08, 0.02], [0.02, -0.22, 0.2], [0, 0, 0]])
    a = migration.transition(2).transition
    b = migration.transition(1).transition
    np.testing.assert_allclose(a, b @ b, atol=1e-12)
    np.testing.assert_allclose(a[-1], [0, 0, 1])
    fitted = MigrationModel.fit([[0, 8, 2], [2, 0, 20], [0, 0, 0]], [100, 100, 0])
    np.testing.assert_allclose(fitted.generator, migration.generator)
    result = merton_structural(120, 100, 0.2, 1, 0.04, physical_drift=0.1)
    assert result["equity"] + result["debt"] == pytest.approx(120)
    assert (
        result["credit_spread"] > 0
        and result["p_default_probability"] < result["q_default_probability"]
    )


def test_stochastic_inflation_and_p_curve_forecast(discount, valuation):
    from convexity.math.forecasting import TimeSeries, NaiveForecaster

    real = flat_curve(valuation, 0.02)
    q = InflationDynamics(0.1, "Q", nominal=discount, real=real)
    assert q.transition(100, 0, 2).mean() == pytest.approx(100 * np.exp(0.06))
    paths = q.sample(100, [0, 1, 2], draws=50000, rng=np.random.default_rng(91))
    assert paths.mean()[-1] == pytest.approx(100 * np.exp(0.06), abs=0.25)
    assert q.option(100, 100, 1) > 0
    p = InflationDynamics(0.1, "P", drift=0.04)
    with pytest.raises(ValueError):
        p.option(100, 100, 1)
    f = EmpiricalCurveForecaster(NaiveForecaster(), [1, 2, 5])
    fit = f.fit(TimeSeries([[0.02, 0.03, 0.04], [0.021, 0.031, 0.041]]))
    np.testing.assert_allclose(
        fit.predict(2), [[0.021, 0.031, 0.041], [0.021, 0.031, 0.041]]
    )


def test_nonflat_carry_roll_down(valuation):
    curve = zero_curve(
        [valuation, "2027-01-01", "2029-01-01", "2031-01-01"], [0.02, 0.03, 0.05, 0.06]
    )
    bond = zero_coupon("2031-01-01")
    constant = carry_roll_down(bond, curve, "2027-01-01")
    implied = carry_roll_down(bond, curve, "2027-01-01", evolution="implied_forward")
    assert abs(constant["roll_down"]) > 0.1
    assert implied["roll_down"] == pytest.approx(0, abs=1e-10)


def test_cir_backend_domain_is_explicit():
    params = dict(r0=0.05, kappa=0.2, theta=0.05, sigma=0.4)
    model = ShortRateModel("cir", params)
    with pytest.raises(ValueError, match="QuantLib CIR"):
        model.discount_bond(0.0, 2.0, 0.05)
    process = ShortRateModel("cir", params, measure="P").process()
    assert not isinstance(process, ql.HullWhiteProcess)
    assert process.transition(0.05, 1.0).mean() > 0


def test_structural_debt_avoids_cancellation():
    result = merton_structural(1e20, 100.0, 0.2, 1.0, 0.04)
    assert result["debt"] == pytest.approx(100 * np.exp(-0.04))
    assert result["credit_spread"] == pytest.approx(0.0, abs=1e-14)
