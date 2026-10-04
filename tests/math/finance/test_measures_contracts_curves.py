import numpy as np
import pytest
import QuantLib as ql
from convexity.math._finance import evaluation_date, qdate
from convexity.math.measures import *
from convexity.math.processes import Heston, GeometricBrownianMotion
from convexity.math.simulation import PathScenarios
from convexity.math.contracts import *
from convexity.math.curves import *


def test_settings_nested_exception():
    original = ql.Settings.instance().evaluationDate
    original_flag = ql.Settings.instance().includeReferenceDateEvents
    with pytest.raises(RuntimeError):
        with evaluation_date("2026-01-01"):
            ql.Settings.instance().includeReferenceDateEvents = not original_flag
            with evaluation_date("2027-01-01"):
                assert ql.Settings.instance().evaluationDate == qdate("2027-01-01")
            assert ql.Settings.instance().evaluationDate == qdate("2026-01-01")
            raise RuntimeError("test")
    assert ql.Settings.instance().evaluationDate == original
    assert ql.Settings.instance().includeReferenceDateEvents == original_flag


def test_gbm_change_of_measure():
    pair = gbm_pair(0.12, 0.04, 0.01, 0.2)
    assert pair.market_price_of_risk(0, 100) == pytest.approx([0.45])
    rng = np.random.default_rng(23)
    dw = rng.normal(size=(200000, 1, 1))
    ll = girsanov_log_density(dw, [0.45], [1.0])
    terminal = 100 * np.exp(0.12 - 0.5 * 0.2**2 + 0.2 * dw[:, 0, 0])
    assert np.mean(np.exp(ll)) == pytest.approx(1, abs=0.004)
    assert np.mean(terminal * np.exp(ll)) == pytest.approx(100 * np.exp(0.03), abs=0.25)
    paths = PathScenarios(
        [0, 1], np.c_[np.full(len(dw), 100), terminal], sampling="iid"
    )
    changed = reweight(paths, ll - 10000)
    assert changed.mean()[-1] == pytest.approx(100 * np.exp(0.03), abs=0.25)
    assert changed.sampling == "unspecified"
    with pytest.raises(ValueError):
        reweight(paths, np.full(len(dw), -np.inf))  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.


def test_pair_diffusion_boundaries():
    with pytest.raises(ValueError):
        DiffusionPair(
            MeasureModel(GeometricBrownianMotion(0.1, 0.2), "P"),
            MeasureModel(
                GeometricBrownianMotion(0.04, 0.3), "Q", MoneyMarketNumeraire(0.04)
            ),
        )
    with pytest.raises(ValueError):
        gbm_pair(0.1, 0.04, 0.0, 0.0).market_price_of_risk(0, 100)
    pair = DiffusionPair(
        MeasureModel(Heston(0.1, 1.0, 0.04, 0.3, -0.5), "P"),
        MeasureModel(
            Heston(0.03, 2.0, 0.05, 0.3, -0.5), "Q", MoneyMarketNumeraire(0.03)
        ),
    )
    risk = pair.market_price_of_risk(0, [100, 0.04])
    np.testing.assert_allclose(
        pair.physical.model.diffusion(0, [100, 0.04]) @ risk,
        pair.physical.model.drift(0, [100, 0.04])
        - pair.pricing.model.drift(0, [100, 0.04]),
    )
    with pytest.raises(ValueError):
        MeasureModel(None, "Q")


def test_numeraire_identity(discount):
    model = MeasureModel(None, "Q", CurveNumeraire(discount))
    assert model.value(
        np.array([10.0, 20.0]), 1.0, expectation=np.mean
    ) == pytest.approx(15 * np.exp(-0.05))
    assert change_numeraire_weights([1, 2], [2, 8], 1, 2) == pytest.approx([1, 2])
    with pytest.raises(ValueError):
        MeasureModel(None, "P").value(1, 1, expectation=np.mean)


@pytest.mark.parametrize(
    "kind,spot,units,cash",
    [("call", 120, 2, -200), ("put", 80, -2, 200), ("call", 80, 0, 0)],
)
def test_settlement(kind, spot, units, cash):
    exercise = Exercise("european", ("2027-01-01",))
    physical = VanillaContract(100, exercise, kind, 2, "physical")
    settled = physical.delivery(spot)
    assert settled == (units, cash)
    monetary = VanillaContract(100, exercise, kind, 2, "cash").delivery(spot)[1]
    assert settled[0] * spot + settled[1] == monetary


def test_conventions_cashflows_and_actions(discount):
    ds = schedule("2026-01-01", "2027-01-01", months=6)
    leg = fixed_cashflows(ds, 0.05)
    expected = 2.5 * discount.discount(ds[1]) + 102.5 * discount.discount(ds[-1])
    # Actual/365 allocates the coupon amounts to the actual semiannual lengths.
    expected = sum(c.amount * discount.discount(c.date) for c in leg)
    assert present_value(leg, discount) == pytest.approx(expected)
    assert present_value(
        [CashFlow("2027-01-01", 10, "EUR")], discount, fx={"EUR": 1.2}
    ) == pytest.approx(12 * np.exp(-0.05))
    assert year_fraction("2026-01-01", "2027-01-01") == 1
    assert interest_factor(
        0.04, 2, compounding="periodic", frequency=2
    ) == pytest.approx(1.02**4)
    assert interest_factor(0.04, 2, compounding="simple") == pytest.approx(1.08)
    assert interest_factor(0.04, 2) == pytest.approx(np.exp(0.08))
    shares, price = split_holdings(10, 100, 4)
    assert shares * price == 1000
    shares, price, cash = dividend_adjustment(10, 100, 2)
    assert shares * price + cash == 1000
    with pytest.raises(ValueError):
        Exercise("american", ("2027-01-01",))


def test_curves_identity_and_carry(discount, dividend):
    assert discount.discount(2) == pytest.approx(np.exp(-0.1))
    assert discount.zero_rate(2) == pytest.approx(0.05)
    assert discount.forward_rate(0.5, 2) == pytest.approx(0.05)
    assert discount.shifted(0.01).discount(2) == pytest.approx(np.exp(-0.12))
    assert discount.rolled("2027-01-01").discount(1) == pytest.approx(np.exp(-0.05))
    assert forward_price(100, 1, discount, dividend=dividend) == pytest.approx(
        100 * np.exp(0.03)
    )
    assert forward_price(
        100, 1, discount, storage_rate=0.03, convenience_yield=0.01
    ) == pytest.approx(100 * np.exp(0.07))
    cf = CashFlow("2026-07-01", 2)
    assert forward_price(100, 1, discount, cash_dividends=[cf]) == pytest.approx(
        (100 - 2 * discount.discount(cf.date)) / discount.discount(1)
    )
    assert futures_basis(104, 103) == 1
    with pytest.raises(ValueError):
        forward_price(100, 1, discount, dividend=dividend, cash_dividends=[cf])
    with pytest.raises(ValueError):
        discount.forward_rate(1, 0.5)


def test_interpolation_bootstrap_and_fit(valuation):
    dates = [valuation, "2027-01-01", "2028-01-01"]
    dfs = np.exp(-0.05 * np.arange(3))
    curve = discount_curve(dates, dfs)
    assert curve.discount(0.5) == pytest.approx(np.exp(-0.025))
    zero = zero_curve(dates, [0.05, 0.05, 0.05])
    assert zero.discount(0.5) == pytest.approx(curve.discount(0.5))
    with pytest.raises(RuntimeError):
        curve.discount(3)
    helpers = [
        ql.DepositRateHelper(
            r,
            ql.Period(n, ql.Months),
            0,
            ql.NullCalendar(),
            ql.Unadjusted,
            False,
            ql.Actual365Fixed(),
        )
        for n, r in [(6, 0.04), (12, 0.05)]
    ]
    boot = bootstrap(valuation, helpers)
    assert boot.discount(1) == pytest.approx(1 / 1.05, abs=1e-10)
    with evaluation_date(valuation):
        for h in helpers:
            assert h.impliedQuote() == pytest.approx(h.quote().value(), abs=1e-10)
    with evaluation_date("2035-01-01"):
        assert boot.discount(1) == pytest.approx(1 / 1.05, abs=1e-10)
    fit = fit_discount_curve(
        valuation,
        [1, 2, 3],
        np.eye(3),
        np.exp(-0.04 * np.arange(1, 4)),
        [0.1],
        model=lambda t, p: np.exp(-p[0] * t),
    )
    assert fit.x[0] == pytest.approx(0.04, abs=1e-8)


def test_composition_with_pricing(valuation):
    from convexity.math.pricing import BlackScholesModel

    model = BlackScholesModel.from_measure_pair(
        gbm_pair(0.1, 0.05, 0.02, 0.2), 100, valuation
    )
    assert model.discount.zero_rate(1) == pytest.approx(0.05)
    assert model.dividend.zero_rate(1) == pytest.approx(0.02)


@pytest.mark.parametrize(
    "interpolation", ["log_discount", "linear_zero", "flat_forward"]
)
def test_bootstrap_interpolators(valuation, interpolation):
    helpers = [
        ql.DepositRateHelper(
            0.05,
            ql.Period(1, ql.Years),
            0,
            ql.NullCalendar(),
            ql.Unadjusted,
            False,
            ql.Actual365Fixed(),
        )
    ]
    curve = bootstrap(valuation, helpers, interpolation=interpolation)
    assert curve.discount(1.0) == pytest.approx(1 / 1.05, abs=1e-10)


def test_ou_cir_measure_pairing():
    from convexity.math.processes import OrnsteinUhlenbeck, CIR

    for p, q in [
        (OrnsteinUhlenbeck(0.2, 0.04, 0.01), OrnsteinUhlenbeck(0.3, 0.05, 0.01)),
        (CIR(0.2, 0.04, 0.1), CIR(0.3, 0.05, 0.1)),
    ]:
        pair = DiffusionPair(
            MeasureModel(p, "P"), MeasureModel(q, "Q", MoneyMarketNumeraire(0.04))
        )
        risk = pair.market_price_of_risk(0.0, 0.04)
        np.testing.assert_allclose(
            p.diffusion(0.0, 0.04) @ risk, p.drift(0.0, 0.04) - q.drift(0.0, 0.04)
        )
