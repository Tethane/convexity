"""Observed prices, account capacity, native cash and dated FX accounting."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

import convexity as cx
from convexity.runtime import Context, Position, Runtime
from convexity.runtime.account import broker_account
from convexity.backtest import SimulatedExchange
from convexity.brokers.ibkr import AccountValue, PaperAccountSnapshot

T = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)


def context():
    account = cx.AccountState(
        T,
        "USD",
        {"USD": D(1000)},
        {},
        D(1000),
        D(1000),
        "test",
        buying_power=D(1000),
        sizing_power=D(1000),
    )
    return Context(
        T,
        "s",
        D(1000),
        D(1000),
        {},
        {},
        {"NVDA": D(100), "AAPL": D(200)},
        {"NVDA": T, "AAPL": T},
        (),
        {},
        account=account,
    )


def test_price_uses_latest_mark_not_oldest_history_and_checks_time():
    ctx = context()
    assert ctx.price("NVDA") == 100
    assert ctx.price_info("NVDA").time == T
    with pytest.raises(ValueError, match="No observed"):
        ctx.price("MISSING")
    with pytest.raises(ValueError, match="Future"):
        replace(ctx, mark_times={"NVDA": T + timedelta(seconds=1)}).price("NVDA")
    with pytest.raises(ValueError, match="Stale"):
        replace(ctx, time=T + timedelta(minutes=5)).price(
            "NVDA", max_age=timedelta(minutes=1)
        )


def test_buy_price_uses_observed_ask():
    ctx = replace(
        context(), ask_prices={"NVDA": cx.PriceObservation(D(101), T, "USD", "ask")}
    )
    assert ctx.price("NVDA", side="BUY") == 101


def test_account_conversion_requires_observed_currency_rate():
    state = context().account
    assert state is not None
    account = replace(state, fx_rates={"CAD": cx.FXRate("CAD", "USD", D(".75"), T)})
    assert account.equity_in("CAD") == D(1000) / D(".75")
    with pytest.raises(ValueError, match="No observed FX"):
        account.equity_in("GBP")


def test_fx_rejects_future_and_stale_and_keeps_cash_immutable():
    state = context().account
    assert state is not None
    with pytest.raises(ValueError, match="future"):
        replace(
            state,
            fx_rates={
                "CAD": cx.FXRate("CAD", "USD", D(".75"), T + timedelta(seconds=1))
            },
        )
    state = replace(
        state,
        fx_rates={"CAD": cx.FXRate("CAD", "USD", D(".75"), T - timedelta(days=1))},
    )
    with pytest.raises(ValueError, match="Stale FX"):
        state.equity_in("CAD", max_age=timedelta(hours=1))
    with pytest.raises(TypeError):
        state.cash_balances["USD"] = D(0)  # type: ignore[index]


def runtime(limits=None):
    instruments = {s: cx.Instrument(s) for s in ("NVDA", "AAPL")}
    rt = Runtime(
        {"s": cx.Strategy()}, SimulatedExchange(instruments), instruments, 1000, limits
    )
    rt.marks.update(NVDA=D(100), AAPL=D(200))
    rt.mark_times.update(NVDA=T, AAPL=T)
    rt.start(T)
    return rt


def test_pending_orders_reduce_capacity_and_cancellation_releases_it():
    rt = runtime()
    rt.command(
        "s",
        cx.SubmitOrder.market(
            "NVDA", "BUY", 4, client_order_id="entry", time_in_force="GTC"
        ),
    )
    state = rt.context("s").account
    assert state is not None
    assert state.reserved == 400
    assert state.sizing_power == 600
    rt.command("s", cx.CancelOrder("s:entry"))
    assert rt.account_state().sizing_power == 1000


def test_margin_fields_have_explicit_simulated_model():
    rt = runtime(cx.RiskLimits(gross_leverage=D(2), maintenance_margin_rate=D(".25")))
    rt.positions[("s", "NVDA")] = Position(4, D(100))
    rt.cash_balances["USD"] = D(600)
    state = rt.context("s").account
    assert state is not None
    assert (state.equity, state.initial_margin, state.maintenance_margin) == (
        1000,
        200,
        100,
    )
    assert (state.available_funds, state.excess_liquidity, state.buying_power) == (
        800,
        900,
        1600,
    )
    assert state.source == "simulated_gross_leverage"


def test_oco_reserves_only_largest_peer():
    rt = runtime()
    rt.command(
        "s",
        cx.OCOOrders(
            (
                cx.SubmitOrder.limit("NVDA", "BUY", 4, 100, time_in_force="GTC"),
                cx.SubmitOrder.limit("NVDA", "BUY", 3, 100, time_in_force="GTC"),
            ),
            "entry",
        ),
    )
    assert rt.account_state().reserved == 400


@pytest.mark.parametrize(
    "stock_currency,expected", [("USD", D(799)), ("CAD", D("799.2"))]
)
def test_backtest_native_cash_fx_revaluation_and_no_future_rate(
    stock_currency, expected, tmp_path
):
    observed = []

    class Buy(cx.Strategy):
        def on_start(self, context):
            observed.append(context.account)

        def on_bar(self, event, context):
            observed.append(context.account)
            if len(observed) == 2:
                return cx.SubmitOrder.market("NVDA", "BUY", 2, time_in_force="GTC")

    bars = tuple(
        cx.Bar(
            "NVDA",
            T + timedelta(hours=i),
            T + timedelta(hours=i + 1),
            D(100),
            D(100),
            D(100),
            D(100),
            100,
        )
        for i in range(2)
    )
    result = (
        cx.Backtest(
            data=cx.HistoricalData(bars),
            strategies={"s": Buy},
            initial_cash={"CAD": D(1000)},
            instruments=(cx.Instrument("NVDA", currency=stock_currency),),
            fx_rates=(
                cx.FXRate("CAD", "USD", D(".75"), T),
                cx.FXRate("CAD", "USD", D(".8"), T + timedelta(hours=2)),
            ),
            execution=cx.ExecutionModel(minimum_commission=D(1)),
        )
        .run()
        .raise_for_failure()
    )
    assert observed[0].equity == 750
    assert observed[1].fx_rates["CAD"].rate == D(".75")
    assert observed[2].fx_rates["CAD"].rate == D(".8")
    assert result.metrics["final_equity"] == expected
    assert result.fills[0].currency == stock_currency
    assert result.positions["s"]["NVDA"].currency == stock_currency
    balances = result.equity[-1]["cash_balances"]
    assert balances == (
        {"CAD": D(1000), "USD": D(-201)} if stock_currency == "USD" else {"CAD": D(799)}
    )
    assert result.metrics["total_return"] == expected / D(750) - 1
    result.export(tmp_path)
    assert '"base_currency": "USD"' in (tmp_path / "result.json").read_text()


def test_missing_initial_fx_fails_instead_of_using_future_rate():
    with pytest.raises(ValueError, match="No observed FX"):
        cx.Backtest(
            data=None,
            strategies={"s": cx.Strategy},
            start=T,
            end=T + timedelta(hours=1),
            initial_cash={"CAD": D(100)},
            fx_rates=(cx.FXRate("CAD", "USD", D(".75"), T + timedelta(minutes=1)),),
        ).run()


def test_broker_account_keeps_native_cash_and_distinct_margin_values():
    rows = [
        AccountValue("NetLiquidation", "1500", "CAD"),
        AccountValue("TotalCashValue", "1000", "CAD"),
        AccountValue("CashBalance", "100", "USD"),
        AccountValue("CashBalance", "860", "CAD"),
        AccountValue("CashBalance", "1000", "BASE"),
        AccountValue("ExchangeRate", "1.4", "USD"),
        AccountValue("BuyingPower", "4000", "CAD"),
        AccountValue("AvailableFunds", "700", "CAD"),
        AccountValue("InitMarginReq", "800", "CAD"),
        AccountValue("MaintMarginReq", "600", "CAD"),
        AccountValue("ExcessLiquidity", "900", "CAD"),
    ]
    state = broker_account(PaperAccountSnapshot("DU123", T, T, tuple(rows), (), ()))
    assert state.cash_balances == {"USD": 100, "CAD": 860}
    assert state.base_currency == "CAD"
    assert state.convert(700, "CAD", "USD") == 500
    assert (
        state.initial_margin,
        state.maintenance_margin,
        state.available_funds,
        state.buying_power,
    ) == (800, 600, 700, 4000)
    assert state.cash_balances_complete


def test_broker_missing_ledger_or_margin_does_not_fabricate_values():
    rows = (
        AccountValue("NetLiquidation", "1000", "USD"),
        AccountValue("TotalCashValue", "900", "USD"),
    )
    state = broker_account(PaperAccountSnapshot("DU123", T, T, rows, (), ()))
    assert not state.cash_balances_complete and not state.cash_balances
    assert state.buying_power is None and state.initial_margin is None


def test_callback_prices_follow_observation_time_not_next_bar():
    seen = []

    class Observe(cx.Strategy):
        def on_bar(self, event, context):
            seen.append(
                (context.time, context.price("NVDA"), context.price_info("NVDA").time)
            )

    bars = tuple(
        cx.Bar(
            "NVDA",
            T + timedelta(hours=i),
            T + timedelta(hours=i + 1),
            D(p),
            D(p),
            D(p),
            D(p),
            100,
        )
        for i, p in enumerate((100, 200))
    )
    cx.Backtest(
        data=cx.HistoricalData(bars), strategies={"s": Observe}
    ).run().raise_for_failure()
    assert seen == [
        (T + timedelta(hours=1), 100, T + timedelta(hours=1)),
        (T + timedelta(hours=2), 200, T + timedelta(hours=2)),
    ]


def test_foreign_instrument_notional_limit_converts_before_comparison():
    instruments = {"CA": cx.Instrument("CA", currency="CAD")}
    rt = Runtime(
        {"s": cx.Strategy()},
        SimulatedExchange(instruments),
        instruments,
        1000,
        cx.RiskLimits(max_order_notional=D(100)),
        fx_rates=(cx.FXRate("CAD", "USD", D(".75"), T),),
    )
    rt.marks["CA"], rt.mark_times["CA"] = D(100), T
    rt.start(T)
    rt.command("s", cx.SubmitOrder("CA", "BUY", 1, client_order_id="one"))
    rt.command("s", cx.SubmitOrder("CA", "BUY", 2, client_order_id="two"))
    assert rt.orders["s:one"].status == "working"
    assert rt.orders["s:two"].status == "rejected"


def test_price_filter_does_not_expose_undeclared_symbol():
    from convexity.runtime.contracts import filter_context

    class Declared(cx.Strategy):
        contract = cx.StrategyContract(
            instruments=("NVDA",),
            subscriptions=(cx.BarSubscription(("NVDA",), bar_size="1 day"),),
        )

    ctx = replace(
        context(), prices={"AAPL": cx.PriceObservation(D(200), T, "USD", "bar_close")}
    )
    ctx = filter_context(Declared(), ctx)
    with pytest.raises(ValueError, match="No observed"):
        ctx.price("AAPL")


def test_opening_fill_cannot_use_fx_rate_only_observed_at_bar_close():
    class Buy(cx.Strategy):
        def on_bar(self, event, context):
            if not context.position("CA"):
                return cx.SubmitOrder("CA", "BUY", 9)

    bars = tuple(
        cx.Bar(
            "CA",
            T + timedelta(hours=i),
            T + timedelta(hours=i + 1),
            D(100),
            D(100),
            D(100),
            D(100),
            100,
        )
        for i in range(2)
    )
    result = (
        cx.Backtest(
            data=cx.HistoricalData(bars),
            strategies={"s": Buy},
            initial_cash=1000,
            instruments=(cx.Instrument("CA", currency="CAD"),),
            fx_rates=(
                cx.FXRate("CAD", "USD", D(1), T),
                cx.FXRate("CAD", "USD", D(2), T + timedelta(hours=2)),
            ),
        )
        .run()
        .raise_for_failure()
    )
    # At the opening fill time nine shares cost USD 900, not the later USD 1800.
    assert len(result.fills) == 1 and result.fills[0].quantity == 9
