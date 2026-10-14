from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from convexity.backtest import Backtest, HistoricalData, SimulatedExchange
from convexity.runtime import Bar, Instrument, Position, Runtime, Strategy, SubmitOrder
from convexity.runtime.corporate_actions import (
    CorporateAction,
    UnsupportedCorporateAction,
    apply_corporate_action,
    settle_dividend,
)

T = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
RAW = "unadjusted_no_corporate_actions"


def runtime():
    instruments = {s: Instrument(s) for s in ("A", "B")}
    r = Runtime(
        {"owner": Strategy()}, SimulatedExchange(instruments), instruments, 1000
    )
    r.start(T)
    r.positions[("owner", "A")] = Position(10, D(80))
    r.marks.update(A=D(100), B=D(25))
    r.mark_times.update(A=T, B=T)
    return r


def action(kind, **kwargs):
    return CorporateAction(kind, "A", kind, T, T, "verified test fixture", **kwargs)


def test_split_preserves_value_and_cost_reverse_split():
    r = runtime()
    before = r.equity
    apply_corporate_action(r, action("split", ratio=2), RAW)
    assert r.positions[("owner", "A")] == Position(20, D(40))
    assert r.marks["A"] == 50 and r.equity == before
    reverse = CorporateAction("reverse", "A", "split", T, T, "fixture", ratio=D(".5"))
    apply_corporate_action(r, reverse, RAW)
    assert r.positions[("owner", "A")] == Position(10, D(80))


def test_fractional_entitlements_and_adjusted_prices_fail_atomically():
    r = runtime()
    with pytest.raises(UnsupportedCorporateAction, match="Fractional"):
        apply_corporate_action(r, action("split", ratio=D(".03")), RAW)
    assert r.positions[("owner", "A")].quantity == 10
    with pytest.raises(UnsupportedCorporateAction, match="raw"):
        apply_corporate_action(
            r, action("split", ratio=2), "split_adjusted_price_return"
        )


def test_dividend_entitlement_survives_sale_until_payment_and_preserves_equity():
    r = runtime()
    dividend = action(
        "dividend",
        cash_per_share=2,
        pay_at=T + timedelta(days=3),
        valuation_price=98,
        valuation_at=T,
    )
    before = r.equity
    apply_corporate_action(r, dividend, RAW)
    assert r.cash == 1000 and r.marks["A"] == 98
    assert r.equity == before
    r.positions[("owner", "A")] = Position()
    r.time = dividend.pay_at
    settle_dividend(r, dividend)
    assert r.cash == 1020 and not getattr(r, "_corporate_receivables")
    with pytest.raises(ValueError, match="No recorded"):
        settle_dividend(r, dividend)


def test_cash_and_share_merger_records_explicit_consideration():
    r = runtime()
    apply_corporate_action(
        r, action("merger", cash_per_share=20, target_symbol="B", ratio=3), RAW
    )
    assert r.cash == 1200
    assert r.positions[("owner", "A")].quantity == 0
    assert r.positions[("owner", "A")].realized_pnl == 150
    assert r.positions[("owner", "B")] == Position(30, D(25))
    assert r.equity == 1950


@pytest.mark.parametrize("kind", ["delisting", "bankruptcy", "spinoff"])
def test_unknown_settlement_is_not_zeroed_or_fabricated(kind):
    r = runtime()
    with pytest.raises(UnsupportedCorporateAction, match="no price is invented"):
        apply_corporate_action(r, action(kind), RAW)
    assert r.positions[("owner", "A")].quantity == 10


def test_late_knowledge_rejected():
    with pytest.raises(ValueError, match="Late-discovered"):
        CorporateAction("late", "A", "split", T, T + timedelta(days=1), "fixture")


def test_backtest_split_between_bars_changes_existing_accounting():
    class Buy(Strategy):
        def on_start(self, context):
            return SubmitOrder.market("A", "BUY", 10, time_in_force="GTC")

    bars = HistoricalData(
        tuple(
            Bar(
                "A",
                T + timedelta(hours=i),
                T + timedelta(hours=i + 1),
                D(p),
                D(p),
                D(p),
                D(p),
                10000,
            )
            for i, p in enumerate((100, 50, 55))
        )
    )
    split = CorporateAction(
        "split", "A", "split", T + timedelta(hours=1), T, "fixture", ratio=D(2)
    )
    # Preload a prior observed price for the existing order risk gate.
    prior = Bar("A", T - timedelta(hours=1), T, D(100), D(100), D(100), D(100), 10000)
    data = HistoricalData((prior, *bars.bars))
    result = Backtest(
        data=data,
        strategies={"owner": Buy},
        initial_cash=10000,
        start=T,
        preload_marks=True,
        corporate_actions=(split,),
    ).run()
    assert result.status == "completed", result.diagnostics
    assert result.positions["owner"]["A"].quantity == 20
    assert result.metrics["final_equity"] == 10100
    assert len(result.fills) == 1
    assert result.cash_flows[0]["action_id"] == "split"


def test_dividend_never_infers_payment_date_or_ex_market_price():
    with pytest.raises(ValueError, match="payment timestamp"):
        action("dividend", cash_per_share=2)
    r = runtime()
    with pytest.raises(UnsupportedCorporateAction, match="sourced ex-date"):
        apply_corporate_action(r, action("dividend", cash_per_share=2, pay_at=T), RAW)
    assert r.marks["A"] == 100 and r.cash == 1000


def test_simultaneous_entitlements_cannot_depend_on_action_id_order():
    bars = HistoricalData(
        (Bar("A", T, T + timedelta(hours=1), D(100), D(100), D(100), D(100), 100),)
    )
    with pytest.raises(ValueError, match="Simultaneous"):
        Backtest(
            data=bars,
            strategies={"owner": Strategy},
            corporate_actions=(
                action("split", ratio=2),
                action("dividend", cash_per_share=1, pay_at=T),
            ),
        )


def test_unsupported_action_terms_are_not_silently_ignored():
    with pytest.raises(UnsupportedCorporateAction, match="cash-in-lieu"):
        action("split", ratio=2, cash_per_share=1)
    with pytest.raises(UnsupportedCorporateAction, match="Share dividends"):
        action("dividend", cash_per_share=1, ratio=2, pay_at=T)
