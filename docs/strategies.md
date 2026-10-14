# Building strategies

A strategy declares its name, allowed instruments, observations, and warmup.
The same callback interface runs in historical research and IBKR paper trading.

```python
import convexity as cx

class MovingAverage(cx.Strategy):
    metadata = cx.StrategyMetadata("moving-average")
    contract = cx.StrategyContract(
        instruments=("SPY",),
        subscriptions=(cx.BarSubscription(("SPY",), bar_size="1 day"),),
        warmup_sessions=20,
    )

    def on_bar(self, event, context):
        closes = [bar.close for bar in context.history["SPY"][-20:]]
        if len(closes) < 20 or context.pending("SPY"):
            return None

        invested = context.position("SPY") > 0
        above_average = closes[-1] > sum(closes) / len(closes)

        if above_average and not invested:
            return cx.SubmitOrder.market("SPY", "BUY", 10)

        if invested and not above_average:
            return cx.SubmitOrder.market("SPY", "SELL", context.position("SPY"))
```

Trading permission and event subscription are separate. Declaring SPY permits
orders but does not request bar callbacks. Commands outside the contract fail
before routing. [TargetWeightStrategy](portfolio.md) handles portfolio sizing
and rebalance orders when a strategy supplies weights.

## Callbacks

| Subscription | Callback |
| --- | --- |
| `BarSubscription` | `on_bar(event, context)` |
| `EventSubscription(Timer, schedule=...)` | `on_timer(event, context)` |
| `EventSubscription(FillEvent)` | `on_fill(event, context)` |
| `EventSubscription(OrderUpdate)` | `on_partial` for partial fills, otherwise `on_order` |
| `EventSubscription(Rejection)` | `on_rejection(event, context)` |
| Subscription lifecycle, delivered automatically to its owner | `on_subscription(event, context)` |

Specific callbacks fall back to `on_event` and receive each event once.
`on_start` follows warmup; `on_stop` runs at shutdown. Warmup loads completed
history without trading callbacks. Return one command, an iterable, or `None`.
Callbacks run serially against an immutable context. Orders and fills belong to
the strategy that submitted them. [Subscription changes](order-lifecycle.md#dynamic-subscriptions)
take effect after the callback; a failed callback publishes none of its queued
requests.

## Current prices and buying quantities

`context.history[symbol][-1]` is the latest observed bar. `context.price(symbol)`
is a completed mark, or the relevant broker quote with `side="BUY"` / `"SELL"`.
`price_info` includes time, currency, and source. Use `max_age=timedelta(...)` to
reject stale prices. Neither a mark nor a quote promises the next fill price.

Size from cash and available funds, after pending reservations and a fee/price
buffer. Equity includes held assets and is not spendable cash. Divide a shared
budget before sizing multiple orders. A market gap can exceed any chosen buffer;
a limit bounds the price but may not fill. Whole shares are required.

`context.account` exposes native `cash_balances`, base-currency cash/equity,
margin requirements, `available_funds`, `buying_power`, `reserved`, `sizing_power`,
and dated FX rates. Negative cash can represent a currency loan. Missing broker
fields remain `None`; check `cash_balances_complete` before cash-funded sizing.
`account.convert(...)` and `equity_in(...)` value currencies without trading them.
See [FX accounting](backtesting.md#currency-ledgers-and-fx).

## Schedules and order IDs

```python
from datetime import time

entry = cx.EventSubscription(
    cx.Timer, schedule=cx.ClockSchedule("entry", at=time(15, 50)),
)
```

Put the subscription in the strategy contract. Clock schedules default to XNYS
and New York time, respecting holidays and DST; a 15:50 event is absent on a
13:00 close. `SessionSchedule` supports daily/monthly open/close-relative events
and offsets. Keep its start and namespace fixed across paper restarts.

`client_order_id="entry"` identifies one intent for the entire run/deployment,
including after a fill or cancellation. Identical retries deduplicate; conflicting
reuse rejects. For recurring decisions use `f"entry:{event.time.isoformat()}"`
or omit the ID. Public IDs are owner-prefixed, such as `my-strategy:entry`.
Subscribe to Rejection and inspect open orders when a strategy stops trading.
[OvernightHold](../examples/overnight_hold.py) is a complete timer example.

## One strategy, research or paper

`Research(...).run(MyStrategy)` constructs a fresh strategy. In notebooks use
`await research.run_async(...)`. PaperTrading owns subscriptions, warmup,
callbacks, schedules, checkpoints, and reconnects:

```python
from decimal import Decimal

paper = cx.PaperTrading(
    config=cx.PaperAccountConfig.from_env(),
    strategies=(MyStrategy,),
    path="out/my-paper",
    budgets={"my-strategy": Decimal("2000")},
    max_order_notional=Decimal("1000"),
    market_collar_bps=Decimal("25"),
    strategy_versions={"my-strategy": "v1"},
)
await paper.run()
```

Define `MyStrategy` with the matching name and one-minute bars before running
that snippet; invoking it can submit paper orders. Managed paper bars currently
support one-minute regular sessions, USD whole-share stocks, and standard USD
debit verticals. Use matching signal cadence when comparing with research.
Automatic dynamic-universe paper subscriptions are unsupported.

Paper market commands become bounded limit orders using the configured collar
and may remain unfilled. Broker AvailableFunds and fresh live bid/ask receipts
gate submissions. Manual holdings remain unassigned. Stock FillEvent waits for
commission reports; option leg executions use BrokerOrderUpdate.

Keep the deployment directory, account, names, versions, and risk settings stable
when resuming. Change versions when changing behavior or parameters, and handle
existing exposure deliberately. `paper.status()`, `paper.health`, and
`paper.schedule_status()` expose errors and delivery state. `await paper.start()`
returns a background task; retain it to observe failures, and stop with
`await paper.stop()`. Shutdown does not cancel orders or flatten positions.

## Interrupted callbacks

A failed strategy pauses while order tracking and healthy strategies continue.
`fail_fast=True` stops the session instead. Infrastructure faults propagate.
Paused strategies may cancel or reduce verified owned exposure through
`await paper.command(owner, command)`; submissions need an explicit client ID.
An operator command cannot resume the strategy or open new exposure silently.

Durable schedules claim before invoking callbacks. Failure leaves that owner's
delivery uncertain; it is not retried automatically. Use `policy="catch_up"`
for overdue-safe exits and `skip` for entries. A delivered exit timer does not
prove that an order filled.

Reconcile the journal with actual broker effects before resolving uncertainty.
`StrategyState(path_to_orders_sqlite, account).unresolved(owner)` lists callback
failures; `resolve(owner, identity, detail=...)` records reconciliation without
rerunning user code. Close the store afterwards. Interrupted timers also need
`DurableSchedule.resolve(...)`, using the durable namespace shown by
`paper.schedule_status()`. Preserve the journal and schedule database on restart.

## Scheduled option rules

`ScheduledVertical(VerticalRule(...))` manages one exact standard debit spread;
`ZeroDTE(rules)` composes same-day rules. Qualified leg identities are retained.
Entries require fresh live quotes within the configured window, spread/debit
limits, and trade-attempt budget. Missing quotes skip entry. These are execution
demonstrations, not predictive models.

Exits cancel working entries and reduce balanced owned legs. Retry limits and
attempt counts persist across restarts. Imbalanced or unreconciled legs fault;
the runtime never substitutes naked-leg orders. Exhausted/rejected exits require
inspection. [IBKR setup and recovery](ibkr.md) describes broker details.
