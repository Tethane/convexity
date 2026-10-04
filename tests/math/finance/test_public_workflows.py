"""Acceptance workflows that need only Convexity and ordinary Python/NumPy."""

from functools import partial
import numpy as np
import pytest
from convexity.math import contracts as ct, curves as cv, fixed_income as fi
from convexity.math import rates as ra, credit as cr, processes as ps

REF = "2026-01-01"


def test_curve_quotes_reprice_without_vendor_objects_or_settings():
    quotes = [
        cv.deposit_quote(0.03, "6M"),
        cv.deposit_quote(0.032, "1Y"),
        cv.swap_quote(0.035, "2Y"),
        cv.swap_quote(0.04, "5Y"),
    ]
    curve = cv.bootstrap(REF, quotes)
    assert curve.calibration_errors is not None
    np.testing.assert_allclose(curve.calibration_errors, 0, atol=1e-10)
    assert curve.calibration_errors is not None
    assert not curve.calibration_errors.flags.writeable
    assert 0 < curve.discount("2030-01-01") < 1
    assert 0.03 < curve.forward_rate("2027-01-01", "2028-01-01") < 0.05


def test_credit_quotes_reprice_without_vendor_helpers():
    discount = cv.flat_curve(REF, 0.04)
    quotes = [
        cr.cds_quote(s, t, discount)
        for t, s in [("1Y", 0.01), ("3Y", 0.012), ("5Y", 0.014)]
    ]
    default = cr.bootstrap_cds(REF, quotes)
    assert default.calibration_errors is not None
    np.testing.assert_allclose(default.calibration_errors, 0, atol=1e-10)
    dates = ct.schedule(REF, "2031-01-01", months=3)
    result = cr.cds_price(dates, 0.014, discount, default, trade_date=REF)
    assert abs(result.value) < 1e-8


def test_human_readable_conventions_and_bond_yield_round_trip():
    dates = ct.schedule(
        "2026-01-31", "2029-01-31", calendar="weekends", convention="modified_following"
    )
    bond = fi.fixed_bond(dates, 0.05)
    price = fi.price_from_yield(
        bond, 0.043, "2026-02-02", compounding="periodic", frequency="semiannual"
    )
    assert fi.yield_from_price(
        bond, price, "2026-02-02", compounding="periodic", frequency="semiannual"
    ) == pytest.approx(0.043, abs=1e-9)
    with pytest.raises(ValueError, match="calendar"):
        ct.schedule(REF, "2027-01-01", calendar="unknown")


def test_floating_bonds_use_owned_histories_without_cross_talk():
    curve = cv.flat_curve("2026-03-01", 0.04)
    dates = ct.schedule(REF, "2028-01-01")
    low_history = {REF: 0.01}
    low = ra.rate_index(curve, fixings=low_history)
    high = ra.rate_index(curve, fixings={REF: 0.09})
    low_history[REF] = 0.99
    a, b = fi.floating_bond(dates, low), fi.floating_bond(dates, high)

    def value(bond):
        return fi.bond_analytics(bond, curve, compute_yield=False).dirty

    low_value = value(a)
    assert value(b) > low_value + 3
    assert value(a) == pytest.approx(low_value, abs=1e-12)
    # The same histories work in standalone yield calculation and cash-flow extraction.
    price = fi.price_from_yield(a, 0.04, "2026-03-01")
    assert fi.yield_from_price(a, price, "2026-03-01") == pytest.approx(0.04, abs=1e-9)
    assert (
        fi.cashflows(a, "2026-03-01")[0].amount
        < fi.cashflows(b, "2026-03-01")[0].amount
    )


def test_rate_instruments_and_cap_floor_parity():
    curve = cv.flat_curve(REF, 0.04)
    dates = ct.schedule("2027-01-01", "2030-01-01")
    index = ra.rate_index(curve)
    swap = ra.interest_rate_swap(
        dates, dates, index, 0.04, fixed_convention="actual360"
    )
    result = ra.swap_price(swap, curve)
    cap = ra.cap_floor(dates, index, 0.04)
    floor = ra.cap_floor(dates, index, 0.04, kind="floor")
    cap_price = ra.rate_option_price(cap, curve, volatility=0.2).value
    floor_price = ra.rate_option_price(floor, curve, volatility=0.2).value
    assert cap_price - floor_price == pytest.approx(result.value, abs=1e-10)
    par = ra.interest_rate_swap(
        dates, dates, index, result.fair_rate, fixed_convention="actual360"
    )
    assert ra.swap_price(par, curve).value == pytest.approx(0, abs=1e-10)
    option = ra.swaption(swap, ["2027-01-01"])
    assert ra.rate_option_price(option, curve, volatility=0.2).value > 0
    bermudan = ra.swaption(swap, ["2027-01-01", "2027-07-01"], style="bermudan")
    with pytest.raises(ValueError, match="European"):
        ra.rate_option_price(bermudan, curve, volatility=0.2)
    hw = ra.ShortRateModel("hull_white", dict(kappa=0.1, sigma=0.01), curve)
    assert ra.rate_option_price(bermudan, curve, method="tree", model=hw).value > 0


def test_callable_bond_and_effective_risk_use_public_engine_factory():
    curve = cv.flat_curve(REF, 0.04)
    dates = ct.schedule(REF, "2031-01-01")
    bond = fi.callable_bond(dates, 0.07, [("2028-01-01", 100.0, "call")])
    factory = partial(
        fi.callable_bond_engine, mean_reversion=0.1, volatility=0.01, steps=150
    )
    result = fi.bond_analytics(bond, curve, engine=factory(curve), compute_yield=False)
    assert result.dirty < fi.bond_analytics(fi.fixed_bond(dates, 0.07), curve).dirty
    risk = fi.key_rate_duration(
        bond, curve, [REF, "2028-01-01", "2031-01-01"], engine_factory=factory
    )
    assert np.all(np.isfinite(risk)) and risk.sum() > 0


def test_inflation_history_and_swap_need_no_vendor_fixing_api():
    discount = cv.flat_curve(REF, 0.04)
    curve = ra.inflation_curve(
        REF, ["2025-10-01", "2026-10-01", "2030-10-01"], [0.02] * 3, frequency="monthly"
    )
    index = ra.inflation_index(curve, fixings={"2025-10-01": 100.0})
    dates = ct.schedule(REF, "2029-01-01", months=12)
    bond = fi.inflation_bond(dates, index, 100, 0.01)
    assert fi.bond_analytics(bond, discount, compute_yield=False).dirty > 0
    swap = ra.inflation_swap(REF, "2029-01-01", index, 0.02)
    assert abs(ra.inflation_swap_price(swap, discount).value) < 0.01


def test_forward_model_and_gaussian_hjm_can_be_used_without_vendor_calls():
    model = ra.forward_market_model(
        [0.5, 1.0, 1.5, 2.0], [0.04] * 3, volatility=0.2, factors=2
    )
    paths = ra.market_model_paths(model, draws=2000, seed=42)
    assert paths.values.shape == (2000, 3, 3)
    assert paths.values[:, -1, -1].mean() == pytest.approx(0.04, abs=0.001)
    curve = cv.flat_curve(REF, 0.04)
    hjm = ra.gaussian_hjm(curve, ["2027-01-01"], [0.01, 0.015], [0.1])
    assert ra.gaussian_bond_price(hjm, 2.0) == pytest.approx(
        curve.discount(2.0), rel=1e-10
    )


def test_hull_white_process_composes_with_existing_transition_and_paths():
    curve = cv.flat_curve(REF, 0.04)
    model = ra.ShortRateModel("hull_white", dict(kappa=0.1, sigma=0.01), curve)
    process = model.process()
    kernel = ps.transition(process, 0.04, 0.5)
    assert kernel.event_shape == () and kernel.exact
    assert float(kernel.mean()) == pytest.approx(
        0.04 + 0.01**2 / (2 * 0.1**2) * (1 - np.exp(-0.1 * 0.5)) ** 2, abs=1e-9
    )
    paths = ps.sample_paths(
        process, 0.04, [0.0, 0.5, 1.0], draws=20, rng=np.random.default_rng(1)
    )
    assert paths.values.shape == (20, 3)


def test_g2_factor_process_uses_existing_multivariate_protocol():
    model = ra.ShortRateModel(
        "g2",
        dict(a=0.1, sigma=0.01, b=0.2, eta=0.015, rho=-0.5),
        cv.flat_curve(REF, 0.04),
    )
    kernel = ps.transition(model.process(), [0.0, 0.0], 1.0)
    assert kernel.event_shape == (2,)
    np.testing.assert_allclose(kernel.mean(), [0, 0], atol=1e-12)
    assert kernel.covariance()[0, 1] < 0


def test_ois_quotes_and_standard_python_dates():
    from datetime import date

    quotes = [cv.ois_quote(0.03, "1Y"), cv.ois_quote(0.035, "3Y")]
    curve = cv.bootstrap(REF, quotes)
    assert curve.calibration_errors is not None
    np.testing.assert_allclose(curve.calibration_errors, 0, atol=1e-10)
    dates = ct.schedule_dates(
        ct.schedule("2026-01-31", "2027-01-31", calendar="weekends")
    )
    assert all(isinstance(d, date) for d in dates)
    assert dates[0] == date(2026, 1, 30)
