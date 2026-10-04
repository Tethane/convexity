"""Boundary checks for downstream valuation, scenarios and risk composition."""

import numpy as np
import pytest
import QuantLib as ql
from convexity.math import curves as cv, contracts as ct, rates as ra
from convexity.math import fixed_income as fi, pricing as pr, credit as cr

REF = "2026-01-01"


def test_numpy_engine_configuration_is_accepted_at_vendor_boundary():
    curve = cv.flat_curve(REF, 0.04)
    model = pr.BlackScholesModel(100.0, curve, cv.flat_curve(REF, 0.0), 0.2)
    contract = ct.VanillaContract(100.0, ct.Exercise("european", ("2027-01-01",)))
    engine = pr.VanillaEngine("tree", steps=np.int64(100), grid=np.int64(100))
    assert pr.price(pr.PricingProblem(contract, model, REF), engine).value > 0


def test_custom_price_cannot_silently_change_contract_currency():
    class Model:
        measure = "Q"

    class Engine:
        def price(self, problem):
            return pr.PricingResult(12.0, currency="USD")

    contract = ct.VanillaContract(
        100.0, ct.Exercise("european", ("2027-01-01",)), currency="EUR"
    )
    with pytest.raises(ValueError, match="currency"):
        pr.price(pr.PricingProblem(contract, Model(), REF), Engine())  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.


def test_rate_prices_retain_index_currency():
    curve = cv.flat_curve(REF, 0.04)
    dates = ct.schedule("2027-01-01", "2030-01-01")
    index = ra.rate_index(curve, currency="EUR")
    cap = ra.cap_floor(dates, index, 0.04)
    assert ra.rate_option_price(cap, curve, volatility=0.2).currency == "EUR"
    swap = ra.interest_rate_swap(dates, dates, index, 0.04)
    assert ra.swap_price(swap, curve).currency == "EUR"


def test_floating_bond_inherits_index_fixing_lag():
    curve = cv.flat_curve("2026-03-01", 0.04)
    index = ra.rate_index(curve, fixing_days=2, fixings={"2025-12-30": 0.03})
    bond = fi.floating_bond(ct.schedule(REF, "2028-01-01"), index)
    assert fi.bond_analytics(bond, curve, compute_yield=False).dirty > 0


def test_owned_fixings_restore_external_history_after_success_and_error():
    curve = cv.flat_curve("2026-03-01", 0.04)
    index = ra.rate_index(curve, fixings={REF: 0.03})
    date = ct.to_quantlib_date(REF)
    index.addFixing(date, 0.08, True)
    try:
        bond = fi.floating_bond(ct.schedule(REF, "2028-01-01"), index)
        assert fi.cashflows(bond, "2026-03-01")[0].amount < 2
        with ct.evaluation_date("2026-03-01"):
            assert index.fixing(date) == 0.08
        with pytest.raises(ValueError, match="settlement"):
            fi.bond_analytics(bond, curve, settlement_date="2030-01-01")
        with ct.evaluation_date("2026-03-01"):
            assert index.fixing(date) == 0.08
    finally:
        index.clearFixings()


def test_curve_bootstrap_and_owned_fixings_restore_ambient_settings():
    ambient = ql.Settings.instance().evaluationDate
    with ct.evaluation_date("2040-01-01"):
        curve = cv.bootstrap(
            REF, [cv.deposit_quote(0.03, "6M"), cv.swap_quote(0.04, "2Y")]
        )
        assert curve.calibration_errors is not None
        np.testing.assert_allclose(curve.calibration_errors, 0, atol=1e-10)
        assert ql.Settings.instance().evaluationDate == ct.to_quantlib_date(
            "2040-01-01"
        )
    assert ql.Settings.instance().evaluationDate == ambient
    before = curve.discount("2027-01-01")
    with ct.evaluation_date("2029-01-01"):
        assert curve.discount("2027-01-01") == before


def test_negative_curve_time_and_generator_shape_fail_cleanly():
    curve = cv.flat_curve(REF, 0.04)
    for fn in (curve.time, curve.zero_rate, curve.discount):
        with pytest.raises(ValueError):
            fn("2025-01-01")
    with pytest.raises(ValueError, match="square"):
        cr.MigrationModel(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        cr.MigrationModel.fit([[0, 1], [0, 0]], [1, -1])


def test_forward_carry_accepts_empty_generator_and_checks_time():
    curve = cv.flat_curve(REF, 0.04)
    assert cv.forward_price(
        100.0, 1.0, curve, dividend=curve, cash_dividends=iter(())
    ) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        cv.forward_price(100.0, "2025-01-01", curve)


def test_rate_engine_rejects_ignored_normal_displacement():
    curve = cv.flat_curve(REF, 0.04)
    cap = ra.cap_floor(
        ct.schedule("2027-01-01", "2030-01-01"), ra.rate_index(curve), 0.04
    )
    with pytest.raises(ValueError, match="displacement"):
        ra.rate_option_price(
            cap, curve, method="normal", volatility=0.01, displacement=0.1
        )


@pytest.mark.parametrize("number", [np.float32, np.float64])
def test_numpy_scalar_models_and_contracts_normalize_at_boundary(number):
    curve = cv.flat_curve(REF, 0.04)
    model = pr.BlackScholesModel(number(100), curve, curve, number(0.2))
    claim = ct.VanillaContract(number(100), ct.Exercise("european", ("2027-01-01",)))
    result = pr.price(pr.PricingProblem(claim, model, REF), pr.VanillaEngine())
    assert result.value > 0
    heston = pr.HestonModel(
        number(100),
        curve,
        curve,
        number(0.04),
        number(1.0),
        number(0.04),
        number(0.3),
        number(-0.5),
    )
    assert (
        pr.price(
            pr.PricingProblem(claim, heston, REF), pr.VanillaEngine("heston")
        ).value
        > 0
    )
    hw = ra.ShortRateModel(
        "hull_white", dict(kappa=number(0.1), sigma=number(0.01)), curve
    )
    assert hw.discount_bond(0, 1, 0.04) == pytest.approx(curve.discount(1), rel=1e-8)
    assert ct.CashFlow("2027-01-01", number(2)).to_quantlib().amount() == 2
