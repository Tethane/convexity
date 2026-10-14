"""Exact entry bounds, durable trade limits, cutoff exits and bounded recovery."""

from typing import Literal
from dataclasses import replace
from datetime import datetime, timedelta, timezone, date
from decimal import Decimal as D

import pytest

from convexity.brokers.ibkr import Contract, Quote
from convexity.runtime import CancelOrder, Context, OptionQuotes, Timer
from convexity.strategies.options import ScheduledVertical, VerticalRule, ZeroDTE
from convexity.trading import Execution, OptionIdentity, OrderRecord, VerticalOrder

T = datetime(2026, 10, 9, 14, tzinfo=timezone.utc)


def rule(**kwargs):
    return VerticalRule(
        OptionIdentity(101, "SPY", date(2026, 10, 16), D(777), "C", "SPY"),
        OptionIdentity(102, "SPY", date(2026, 10, 16), D(778), "C", "SPY"),
        T,
        T + timedelta(minutes=5),
        T + timedelta(minutes=10),
        **kwargs,
    )


def context(time=T, records=()):
    return Context(
        time, "spread", D(1000), D(1000), {}, {}, {}, {}, (), {}, tuple(records)
    )


def quotes(config, time=T):
    result = []
    for leg, bid, ask in (
        (config.long_leg, D("1.20"), D("1.25")),
        (config.short_leg, D(".50"), D(".55")),
    ):
        contract = Contract(
            leg.contract_id,
            leg.symbol,
            "OPT",
            "",
            "USD",
            "SMART",
            leg.expiry.strftime("%Y%m%d"),
            leg.strike,
            leg.right,
            "100",
            (),
            "",
            leg.trading_class,
        )
        result.append(
            Quote(
                contract,
                "live",
                "live",
                bid,
                ask,
                D(10),
                D(10),
                time,
                time,
                time,
                30,
                True,
                "received",
                (),
            )
        )
    return OptionQuotes(time, tuple(result))


def record(
    config,
    identity,
    side: Literal["BUY", "SELL"] = "BUY",
    status="Filled",
    created=T,
    legs=True,
    partial=False,
):
    request = VerticalOrder(
        identity, "spread", config.long_leg, config.short_leg, side, 1, D(".75")
    )
    executions = (
        tuple(
            Execution(
                identity + str(leg.contract_id),
                identity,
                D(1),
                D(1),
                request.leg_side(leg.contract_id),
                created.isoformat(),
                contract_id=leg.contract_id,
            )
            for leg in (config.long_leg, config.short_leg)[: 1 if partial else 2]
        )
        if legs
        else ()
    )
    return OrderRecord(
        request,
        "DU123",
        1,
        71,
        0,
        1,
        status,
        True,
        False,
        False,
        D(1) if status == "Filled" else D(0),
        executions,
        (),
        created,
    )


def test_entry_price_never_rounds_above_configured_debit_cap():
    config = rule(max_entry_debit=D(".755"))
    strategy = ScheduledVertical(config)
    strategy.on_event(quotes(config), context())
    command = strategy.on_event(Timer(T, "entry"), context())
    assert isinstance(command, VerticalOrder)
    assert command.side == "BUY" and command.limit_price == D(".75")
    assert isinstance(command, VerticalOrder)
    assert command.limit_price <= config.max_entry_debit


@pytest.mark.parametrize("kind", ["missing", "stale", "non_live", "wide"])
def test_entry_requires_complete_fresh_liquid_option_quotes(kind):
    config = rule()
    strategy = ScheduledVertical(config)
    event = quotes(config)
    if kind == "missing":
        event = replace(
            event, quotes=(replace(event.quotes[0], ask=None), event.quotes[1])
        )
    elif kind == "non_live":
        event = replace(
            event, quotes=(replace(event.quotes[0], status="non_live"), event.quotes[1])
        )
    elif kind == "wide":
        event = replace(
            event, quotes=(replace(event.quotes[0], ask=D("1.40")), event.quotes[1])
        )
    strategy.on_event(event, context())
    time = T + timedelta(seconds=31) if kind == "stale" else T
    assert strategy.on_event(Timer(time, "entry"), context(time)) is None
    assert strategy.last_decision.startswith("skipped:")


def test_restart_daily_limit_counts_this_session_not_every_trade_in_same_expiry():
    config = rule()
    for created, allowed in ((T, False), (T - timedelta(days=1), True)):
        prior = record(config, "prior", status="Cancelled", created=created, legs=False)
        restarted = ScheduledVertical(config)
        restarted.on_event(quotes(config), context(records=(prior,)))
        command = restarted.on_event(Timer(T, "entry"), context(records=(prior,)))
        assert (command is not None) is allowed


def test_restart_after_cutoff_exits_without_waiting_for_another_timer():
    config = rule()
    held = record(config, "open")
    restarted = ScheduledVertical(config)
    late = config.exit_at + timedelta(minutes=1)
    command = restarted.on_event(quotes(config, late), context(late, (held,)))
    assert isinstance(command, VerticalOrder)
    assert command.side == "SELL" and command.quantity == 1


def test_cutoff_cancels_working_entry_when_quotes_arrive_before_exit_timer():
    config = rule()
    pending = record(config, "pending", status="Submitted", legs=False)
    strategy = ScheduledVertical(config)
    commands = strategy.on_event(
        quotes(config, config.exit_at), context(config.exit_at, (pending,))
    )
    assert commands == (CancelOrder("pending"),)


def test_partial_leg_fault_never_legs_out_and_exit_retries_stop_at_cap():
    config = rule(max_exit_attempts=2)
    strategy = ScheduledVertical(config)
    partial = record(config, "partial", status="Cancelled", partial=True)
    assert (
        strategy.on_event(
            quotes(config, config.exit_at), context(config.exit_at, (partial,))
        )
        is None
    )
    assert "unbalanced" in strategy.last_decision
    records = (
        record(config, "open"),
        record(
            config, "exit1", "SELL", "Cancelled", T + timedelta(minutes=1), legs=False
        ),
        record(
            config, "exit2", "SELL", "Cancelled", T + timedelta(minutes=2), legs=False
        ),
    )
    restarted = ScheduledVertical(config)
    assert (
        restarted.on_event(
            quotes(config, config.exit_at), context(config.exit_at, records)
        )
        is None
    )
    assert "retries exhausted" in restarted.last_decision


def test_exit_attempts_and_ids_are_scoped_to_current_entry_cycle():
    config = rule(max_exit_attempts=1, max_trades_per_day=2)
    first = record(config, "first-open")
    previous = ScheduledVertical(config)
    first_exit = previous.on_event(
        quotes(config, config.exit_at), context(config.exit_at, (first,))
    )
    assert isinstance(first_exit, VerticalOrder)
    records = (
        first,
        record(
            config, first_exit.client_order_id, "SELL", created=T + timedelta(minutes=1)
        ),
        record(config, "second-open", created=T + timedelta(minutes=2)),
    )
    restarted = ScheduledVertical(config)
    second_exit = restarted.on_event(
        quotes(config, config.exit_at), context(config.exit_at, records)
    )
    assert isinstance(second_exit, VerticalOrder)
    assert second_exit.side == "SELL"
    assert isinstance(second_exit, VerticalOrder)
    assert second_exit.client_order_id != first_exit.client_order_id


def test_zero_dte_daily_trade_limit_is_per_underlying():
    base = rule()
    spy = replace(
        base,
        long_leg=replace(base.long_leg, expiry=T.date()),
        short_leg=replace(base.short_leg, expiry=T.date()),
        same_day=True,
    )
    qqq = replace(
        spy,
        long_leg=replace(
            spy.long_leg, contract_id=201, symbol="QQQ", trading_class="QQQ"
        ),
        short_leg=replace(
            spy.short_leg, contract_id=202, symbol="QQQ", trading_class="QQQ"
        ),
    )
    strategy = ZeroDTE((spy, qqq))
    prior = record(spy, "spy-attempt", status="Cancelled", legs=False)
    strategy.on_event(
        OptionQuotes(T, quotes(spy).quotes + quotes(qqq).quotes),
        context(records=(prior,)),
    )
    commands = strategy.on_event(Timer(T, "entry"), context(records=(prior,)))
    assert len(commands) == 1 and commands[0].symbol == "QQQ"


def test_exit_rejection_does_not_recursively_submit_again_in_same_event_drain():
    from convexity.runtime import Rejection

    config = rule()
    held = record(config, "open")
    strategy = ScheduledVertical(config)
    ctx = context(config.exit_at, (held,))
    command = strategy.on_event(quotes(config, config.exit_at), ctx)
    assert isinstance(command, VerticalOrder)
    assert command.side == "SELL"
    assert (
        strategy.on_event(
            Rejection(config.exit_at, "spread", "reconcile broker ownership", command),
            ctx,
        )
        is None
    )
    assert "fault: order rejected" in strategy.last_decision


def test_repeated_exit_timers_allow_new_sell_to_rest_until_retry_delay():
    config = rule(exit_retry_after=timedelta(seconds=20))
    entry = record(config, "entry")
    pending = record(config, "exit", "SELL", "Submitted", config.exit_at, legs=False)
    strategy = ScheduledVertical(config)
    for elapsed in (0, 2, 19):
        time = config.exit_at + timedelta(seconds=elapsed)
        assert (
            strategy.on_event(Timer(time, "exit"), context(time, (entry, pending)))
            is None
        )
    time = config.exit_at + timedelta(seconds=20)
    assert strategy.on_event(Timer(time, "exit"), context(time, (entry, pending))) == (
        CancelOrder("exit"),
    )


def test_explicit_early_exit_survives_restart_without_reusing_quotes():
    config = rule()
    held = record(config, "entry")
    first = ScheduledVertical(config)
    first.on_event(quotes(config), context(records=(held,)))
    first.on_event(Timer(T, "exit"), context(records=(held,)))
    restarted = ScheduledVertical(config)
    restarted.load_state(first.dump_state())
    # A restarted early exit cannot price itself from the old process's quotes.
    assert restarted.on_event(Timer(T, "unrelated"), context(records=(held,))) is None
    command = restarted.on_event(quotes(config), context(records=(held,)))
    assert isinstance(command, VerticalOrder) and command.side == "SELL"


def test_grouped_exit_state_survives_restart_and_rejects_different_underlyings():
    base = rule()
    spy = replace(
        base,
        long_leg=replace(base.long_leg, expiry=T.date()),
        short_leg=replace(base.short_leg, expiry=T.date()),
        same_day=True,
    )
    first = ZeroDTE((spy,))
    first.on_event(Timer(T, "exit"), context())
    restarted = ZeroDTE((spy,))
    restarted.load_state(first.dump_state())
    commands = restarted.on_event(quotes(spy), context(records=(record(spy, "entry"),)))
    assert len(commands) == 1 and commands[0].side == "SELL"
    corrupt = dict(first.dump_state(), underlyings=["QQQ"])
    with pytest.raises(ValueError, match="underlyings"):
        restarted.load_state(corrupt)
