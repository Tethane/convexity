# Building strategies

For research setup, run options, cached history, and results, start with
[event-driven research](backtesting.md). This guide covers strategy callbacks,
target planning, and managed paper deployment. Amendments, contingent orders,
and subscription changes are detailed in the [order lifecycle guide](order-lifecycle.md).

For a complete guided lesson with executable cells and charts, open
[`examples/convexity_demo.ipynb`](../examples/convexity_demo.ipynb). It covers the
real-data research loop, experiments, runtime contracts and opt-in paper deployment.

## One strategy, research or paper

The public entry points are available directly after `import convexity as cx`.
Implement the strategy; `Research` acquires real stock history and owns replay,
while `PaperTrading` owns paper execution. `Backtest` accepts explicitly supplied
local data when acquisition is unnecessary.
For a complete runnable example, see
[`examples/strategy_workflow.py`](../examples/strategy_workflow.py). It computes
a target from observed bars and uses exactly the same strategy class in both modes:

```sh
# Real one-minute history; symbols and inclusive dates are required.
conda run -n quant python examples/strategy_workflow.py --symbols IWM \
  --start 2025-01-02 --end 2025-01-03 --warmup-sessions 1 \
  --output out/my-research

# Actual paper trading. Choose a symbol without unassigned account holdings.
conda run -n quant python examples/strategy_workflow.py --mode paper \
  --symbol IWM --output out/my-paper --seconds 300
```

The research command downloads actual IBKR prices, verifies session coverage and
saves input bars, provenance and simulated executions. Covered requests can
replay offline; retrieval failure never substitutes synthetic data. See the
[research API and price-return accounting limits](backtesting.md). The example strategy declares a
one-minute interval matching the paper feed. Bar subscriptions accept custom durations
such as `7 mins`, `90 secs`, or `2 weeks`; other cadences require equivalent signal
aggregation before paper deployment.

The paper command can submit orders. Stopping the runner disconnects; it does not
cancel resting orders or liquidate positions. Keep its output directory to resume
the same strategy ownership and checkpoint on another run.

For your own Python module, the composition is:

```python
from datetime import date
import asyncio
from decimal import Decimal
import convexity as cx

class MyStrategy(cx.Strategy):
    metadata = cx.StrategyMetadata("my-strategy")
    contract = cx.StrategyContract(
        instruments=("IWM",),
        subscriptions=(cx.BarSubscription(("IWM",), bar_size="1 min"),),
        warmup_sessions=1,
    )

    def on_bar(self, event, context):
        # Calculate a signal using observed history; return commands or None.
        return None

result = cx.Research(date(2025, 1, 2), date(2025, 1, 3)).run(
    MyStrategy, output="out/my-research",
)
paper = cx.PaperTrading(
    config=cx.PaperAccountConfig.from_env(),
    strategies=(MyStrategy,),
    path="out/my-paper",
    budgets={"my-strategy": Decimal("2000")},
    max_order_notional=Decimal("1000"),
    market_collar_bps=Decimal("25"),
    strategy_versions={"my-strategy": "v1"},
)
# Explicitly invoking this enables paper strategy execution:
asyncio.run(paper.run())
```

In a notebook, use `await paper.start()` and keep its returned task to observe
failures; `await paper.stop()` waits for cleanup. `paper.status()` provides a
detached stock/options status while running and after shutdown. `paper.health`
records connection transitions. A failed strategy is paused while the connection,
order tracking, operator actions, and healthy strategies continue. Inspect
`paper.status().errors` and `paper.schedule_status()` for failures. Account, data,
and journal faults still propagate from `run()` or the background task. Set
`fail_fast=True` to stop the managed session on a strategy failure as well.

For declared strategies, `PaperTrading` derives feeds and warmup from each
`StrategyContract`. Declaring a trading instrument alone does not subscribe to
bar callbacks. Managed bar feeds currently support one-minute regular-session
observations; the default calendar is XNYS. Install `.[ibkr,calendars]`.
The lower-level mapping-based runner also accepts `StockSubscription` overrides.
`option_requests` accepts
`OptionQuoteRequest` identities and uses the same verified execution connection.
Paper supports whole-share USD stocks and standard debit verticals; options
backtesting remains outside the stock simulator's capabilities.

The runner loads historical warmup without invoking trading callbacks, restores
checkpointed strategy state, and then enables startup and live delivery. Backfilled
old bars do not themselves cause orders. Missing/stale decision data blocks
trading; session calendars distinguish missing bars from overnight/holiday gaps.

The persistence directory identifies a deployment. Keep strategy names, account,
subscriptions, risk settings and `strategy_versions` consistent when resuming.
Change the strategy version when changing its parameters or behavior; changed
deployment specifications require deliberate handling of existing exposure and
a new directory, rather than silently adopting old ownership/checkpoints.

## Callback contract

`Strategy.on_start(context)` and `Strategy.on_event(event, context)` return
a supported command, an iterable of commands, or `None`.
See [order and subscription lifecycle](order-lifecycle.md) for replacements,
filtered cancellation, stops, bracket/OCO orders, and queued `ctx.subscribe()` /
`ctx.unsubscribe()` requests. The runtime
serializes callbacks and routes commands through the configured execution adapter. Context contains
observed history, marks, owned positions, pending orders and account totals.
Callbacks cannot mutate the context. Exceptions pause that strategy; execution
tracking continues. Shutdown does not liquidate positions.

## Current prices and buying quantities

Use `context.price("NVDA")` for the latest observed valuation price.
`context.history["NVDA"][-1]` is the latest history bar; index `0` is
the oldest. Neither history nor marks provide future bars. A backtest mark is
the most recently completed execution bar's close; it is not a promise of the
next fill price. At startup a prior completed bar can supply the mark.

`context.price_info(symbol)` returns a `PriceObservation` with `value`,
`time`, `currency`, and `source`. `context.price(symbol, side="BUY")`
uses the ask when broker quotes are available; `side="SELL"` uses the bid.
Backtests use observed marks for both. Missing, future-dated, and unusable
broker prices raise `ValueError`. Set `max_age=timedelta(...)` to reject
old observations explicitly; choose an age appropriate to the subscription.

Position sizing belongs in the strategy. Context exposes observed prices and
account data; it does not choose allocations or calculate share quantities.
For example, this single-entry strategy explicitly chooses cash funding, a fee
reserve and price headroom:

```python
from datetime import date
from datetime import timedelta
from decimal import Decimal
import convexity as cx

class BuyAndHold(cx.Strategy):
    metadata = cx.StrategyMetadata("buy-hold", version="2")
    contract = cx.StrategyContract(instruments=("SPY",))

    def on_start(self, context):
        account = context.account
        if account is None or account.base_currency != "USD":
            raise ValueError("This strategy requires a USD reporting account.")
        price = context.price("SPY", max_age=timedelta(days=4))
        position = context.positions.get("SPY")
        if (position is not None and position.quantity != 0) or any(
            not order.terminal for order in context.orders
        ):
            return None
        if not account.cash_balances_complete or account.available_funds is None:
            raise ValueError("Cash balances and available funds are required.")

        cash = account.cash_balances.get("USD", Decimal(0))
        budget = max(
            Decimal(0),
            min(cash, account.available_funds) - account.reserved - Decimal("10"),
        )
        shares = int(budget // (price * Decimal("1.01")))
        if shares == 0:
            return None
        return cx.SubmitOrder.market(
            "SPY", "BUY", shares,
            time_in_force="DAY", client_order_id="entry",
        )

result = await cx.Research(date(2025, 1, 2), date(2026, 10, 12)).run_async(
    BuyAndHold, initial_cash=100_000, base_currency="USD",
)
```

The four-calendar-day age limit accommodates ordinary weekends and the New Year
holiday; it is a strategy policy, not a guarantee that every startup has a recent
price. The startup price is the prior completed bar, not the coming opening
price. A market order may gap past the 1% headroom and be rejected by execution
risk checks. Use a limit order if you need a hard maximum purchase price, accepting
that it may not fill. A DAY order can also expire unfilled; this startup-only
strategy does not retry.

For multiple symbols, split a single budget explicitly before calculating each
integer quantity. Account equity includes held assets and is not the same as
spendable cash. For margin or multi-currency sizing, make the funding and FX policy
explicit in your strategy using the account fields below. Runtime strategy caps,
pending-order checks and broker risk checks continue to apply to submitted orders.

## Account state, margin, and currencies

`context.account` is an immutable `AccountState`. `context.cash` and
`context.equity` remain convenient totals, now explicitly in the account's
`base_currency`. Marks, fill prices, fees on individual simulated fills, and
position average costs/realized P&L are in the instrument currency. Instruments,
fills and positions carry their currency.

| Account field | Meaning |
| --- | --- |
| `time`, `source`, `base_currency` | Snapshot time, broker/model provenance, reporting currency |
| `cash_balances` | Native balances such as USD, CAD and GBP; includes currency loans as negative balances |
| `cash`, `equity` | Total cash valuation and net liquidation value in base currency |
| `initial_margin`, `maintenance_margin` | Initial and maintenance requirements |
| `available_funds`, `excess_liquidity` | Remaining initial-margin and maintenance-margin capacity |
| `buying_power` | Broker-reported or model buying power before local reservations |
| `reserved`, `sizing_power` | Conservative pending-order reservation and remaining sizing capacity |
| `fx_rates` | Dated rates: one unit of each currency expressed in base-currency units |

```python
account = context.account
usd_cash = account.cash_balances.get("USD", Decimal(0))
equity_cad = account.equity_in("CAD")
usd_value = account.convert(Decimal("1000"), "GBP", "USD")
```

Valuation never converts the actual cash holdings. Missing FX rates raise errors.
Pass `max_age` to `account.convert` or `equity_in` to bound rate age.
Check `context.time - account.time` explicitly when your strategy requires a
fresh account snapshot.

Paper accounts preserve IBKR's currency ledger and base-currency totals. Missing
optional margin fields remain `None`; absent currency-ledger data leaves
`cash_balances_complete=False`, rather than pretending all cash is held in the
base currency. Broker summary timestamps describe snapshot acquisition, not a
guarantee of tick-by-tick account or FX updates. The stock order gate uses
AvailableFunds, and sizing uses the lesser of AvailableFunds and BuyingPower
when both exist, less conservative local reservations. This can understate
capacity already reserved by the broker. Existing paper order budgets remain
USD-denominated and are converted explicitly against non-USD account totals.
Paper routing still supports USD stocks and the existing USD verticals; this
account view does not add foreign-stock routing.

IBKR distinguishes [account-summary margin totals and currency ledgers](https://interactivebrokers.github.io/tws-api/account_summary.html);
its [ExchangeRate field](https://www.interactivebrokers.com/docs/tws-api/doc/account-portfolio-data/account-updates/account-value-keys)
expresses a currency in account base units.

Backtests use the declared **gross-leverage model**, not IBKR portfolio margin.
Initial margin is gross position exposure divided by `RiskLimits.gross_leverage`.
Maintenance margin uses `maintenance_margin_rate` when supplied, otherwise the
same rate as initial margin. Remaining sizing notional is equity times leverage,
less gross exposure and conservative pending-order/fee reservations.
There is no automatic margin liquidation or currency-loan interest model.
See [backtest currency configuration](backtesting.md#currency-ledgers-and-fx).

For models that produce target share counts, use `TargetStrategy`:

```python
from convexity.strategies import TargetStrategy

def targets(context):
    if len(context.history["SPY"]) < 20:
        return None
    # Call convexity.math using context.history here.
    return {"SPY": 10}

strategy_factory = lambda: TargetStrategy(targets, timer="rebalance")
```

The function runs on the named timer. A mapping fixes desired absolute holdings;
subsequent bars pursue those targets. `None` discards outstanding desired targets,
but does not cancel orders already sent. Explicit cancellation remains a command.
`rebalance_orders(context, targets)` computes target minus current and pending
quantities. It waits for working orders on a symbol, closes reversals before
opening the other side, and completes reductions before additions. Omitted
symbols remain unchanged. Repeating the planner after routing commands does not
duplicate pending orders. Risk rejections remain observable runtime events.

See [the complete math example](../examples/backtest_math.py): it estimates a
covariance matrix from observed returns and calls
`convexity.math.optimization.solve(QuadraticProgram(...))` inside the target
function. Solver failures pause the strategy instead of manufacturing weights.
The example fetches real history for explicit dates/stocks; its model and
parameters demonstrate composition rather than establish profitable performance.

## Monthly demonstration rule

`MonthlyLongShort(StockRule(...))` is a public configurable example. These are
demonstration rules, with explicit caller-supplied paper budgets and timing. They make no profitability claim.

| Setting | Rule |
| --- | --- |
| Universe | Explicit distinct stock symbols supplied to the research/paper configuration |
| Ranking | Last observed close / close `lookback` observations ago minus one; default 3 closes |
| Positions | Long the winner and short the loser; each uses 30% of current account equity by default |
| Ties | Equal top/bottom scores skip; symbol order resolves intermediate rank ties |
| Rounding | Whole shares, toward zero |
| Rebalance | Named `rebalance` timer; research example uses the first actual exchange session's close each month |
| Budget | Rule caps each leg at 50%; runtime account/order/strategy limits still apply |
| Short unavailable | Explicit `shortable` universe; skip the whole new rebalance when the loser is unavailable |
| Execution | Reduce before increasing exposure; reversal passes through flat |
| Exit | Optional distinct `exit_timer` sets all configured targets to zero; subsequent bars continue reducing owned positions |
| Exit retry | Opening orders cancel immediately; reducing orders rest for `exit_retry_seconds` (default 20) before a repeated exit timer cancels them |

Worked example: equity $1,000; A closes 8, 9, 10; B closes 25, 23, 20. A ranks
above B. At a 30% leg weight targets are A +30 and B -15 shares. If currently
A +10 and B -5, the eventual additions are A BUY 20 and B SELL 10. Pending
orders must finish before replacement orders are planned. This calculation and
repeat-invocation behavior are tested in `tests/strategies/test_stocks.py`.

Backtest exports automatically include a strategy's `config` attribute when
present, including `StockRule`. Arbitrary callback functions are not serialized;
retain their source and supply all model parameters through `Backtest(metadata=...)`.
`TargetStrategy` checkpoints desired targets and its last processed timer.
`PaperRuntime` restores these before enabling callbacks, so an unfinished rebalance
continues on subsequent fresh bars after restart. Broker orders and ownership are
reconciled independently through the durable execution journal.

For an existing deployment, newly reconciled executions and late commissions
deliver the still-unconsumed fill callbacks after checkpoint restoration. Completed
callback identities prevent a second delivery on the next restart. A first
deployment baselines pre-existing journal history without retroactively running
strategy callbacks for those old trades.

Stateful custom strategies implement `dump_state()` and `load_state(state)` using
JSON-serializable values. `load_state(None)` means no prior checkpoint. The default
hooks are suitable for stateless strategies that derive decisions from context.
Built-in stock and option strategies provide their own hooks. Do not put clients,
file handles, models, or other process resources in checkpoints; construct those
in `__init__`. Persist the model parameters/version as part of your deployment.

Startup is delivered once per durable account/strategy owner; reconnecting does
not repeat an unconditional startup order. Every callback is durably claimed
before user code, its checkpoint is saved before command routing, and completion
is recorded after commands are processed. A crash during that interval pauses the
owner for reconciliation; the runtime never assumes that a checkpoint proves
an order was sent, nor retries an unknown broker submission with a new ID.

## Durable timers

`SessionCalendar` uses an explicit session table for holidays, early closes and
DST. `DurableSchedule` persists delivery of the resulting daily/monthly timers:

```python
from datetime import datetime, timezone
from convexity.runtime import DurableSchedule, Timer

started = datetime.now(timezone.utc)
with DurableSchedule("out/schedules.sqlite", "monthly-v1",
                     calendar.monthly("rebalance"), started_at=started) as schedule:
    def deliver(run):
        # Runtime must already be started; poll before later market events.
        runtime.advance(run.delivered_at,
                        timers=(Timer(run.delivered_at, run.timer.name),))
        # For broker intents, derive one stable ID per symbol/action:
        # run.order_id("SPY-buy")

    schedule.poll(datetime.now(timezone.utc), deliver)
```

For a low-level integration, poll from the application's loop; `PaperTrading`
handles this itself. The journal skips entries/rebalances whose
scheduled time preceded startup. Use a separate `policy="catch_up"` schedule for
exits; overdue exits are delivered at the next poll with their original identity.
Namespaces have immutable schedules; use a new namespace when intentionally
changing the session table, and distinct namespaces for one-shot demonstrations.
Delivery time is separate from the original scheduled time.

A claim commits before callback invocation. A crash or exception leaves an
uncertain run and blocks subsequent deliveries in that namespace. Reconcile
broker and order-journal state, then call `resolve(timer, completed=..., detail=...)`.
The schedule does not claim exactly-once broker side effects or silently retry
an uncertain submission. A callback completing does not prove an exit filled.

## Inspecting state

`runtime_status(runtime).format()` displays strategy states, owned positions,
outstanding orders, recent fills and errors. `DurableSchedule.status()` exposes
scheduled times and persisted pending/completed/skipped/uncertain states.
The broker status command reports broker balances and manual/unassigned holdings
separately from execution-derived strategy ownership:

```sh
conda run -n quant python scripts/ibkr_status.py
```

`PaperRuntime` connects these same callbacks to `PaperBroker` or `VerticalBroker`.
Its context exposes broker-authoritative USD cash/equity and execution-derived
strategy holdings. Manual account holdings remain separate. It requires explicit
per-strategy dollar budgets, order caps, and stock limit collars. Stock market
commands become bounded limits with the requested DAY/GTC lifetime; quote receipts must remain fresh. Broker
AvailableFunds and pending commitments gate additions; the runtime does not
pretend its stock simulator models broker margin for bonds, futures, or options.
An empty stock instrument mapping supports options-only workflows.

The offline `Runtime` and `SimulatedExchange` support stocks only and explicitly
reject vertical commands. `PaperRuntime` delivers `OptionQuotes` to callbacks and
`BrokerOrderUpdate` containing durable option leg executions. Stock `FillEvent`
waits for a known commission; option fills are not fabricated as stock fills.

See [the paper runner](../examples/paper_mvp.py) for explicit construction,
completed historical bars, quote events, persistent plans, and restart handling.
Starting it can place orders. Disconnecting does not liquidate positions, and a
completed timer delivery does not prove the resulting order filled.


## Scheduled option rules

`ScheduledVertical(VerticalRule(...))` manages one explicitly identified standard
USD debit vertical as a SMART BAG. `ZeroDTE(rules)` composes same-day rules for
distinct underlyings through the same Strategy interface. Each leg retains its
qualified contract ID, expiry, strike, right, trading class, and multiplier 100.
There is no automatic legging or replacement with a different contract.

Entry requires the named timer within `[entry_start, entry_end)`, fresh live
bid/ask receipts on both legs, each leg spread within `max_leg_spread` (default
0.10), and positive long-ask minus short-bid no greater than `max_entry_debit`
(default 0.90). The limit adds `price_collar` (default 0.05), capped at the debit
limit and rounded down to cents. Missing or stale quotes skip entry. These are
transparent quote-based demonstration rules, not an alpha model.

`max_trades_per_day` defaults to one per owner and underlying, counted from
persisted BUY intent creation dates in New York time, including failed intents.
Restart does not reset the count. Unknown intent dates count conservatively.
`ZeroDTE` additionally requires expiration on the entry session's New York date.

The exit timer, or another callback at/after `exit_at`, starts reducing the
position. Working entries are canceled immediately. A working exit is not
canceled until a repeated exit timer observes its persisted age reaching
`exit_retry_after` (default 20 seconds). Balanced owned legs are mandatory;
unbalanced legs produce a fault, never a naked-leg order. Exit credit uses fresh
long bid minus short ask, subtracting another collar per attempt, bounded by
0.01 and spread width. `max_exit_attempts` defaults to two per entry cycle and
survives restart. Rejections remain visible and do not trigger an immediate
recursive retry. Operators must inspect unresolved or exhausted exits.

## Async delivery and calendar tables

`SessionCalendar.from_exchange("XNYS", start, end)` loads a version-labelled
session table from the optional `calendars` extra, including holidays, early
closes and DST. Supply complete months when generating monthly timers.
Calendars with midday breaks are rejected by this single-session model.

For paper delivery, use `await schedule.poll_async(now, deliver)` with an async
callback that awaits `paper_runtime.step(run.timer)`. Preserve the original
timer identity; the runtime context reflects current delivery time.
The claim is durable before awaiting the callback; failure or cancellation leaves
it uncertain. `poll` remains the synchronous simulator API. Use `catch_up` for
exits and reconcile uncertain delivery before explicit resolution. Persist the
same schedule namespace and order journal across restarts.

## Managed schedules and commands

`PaperTrading` polls `ScheduleSpec` schedules itself. Supply
`ScheduleSpec("monthly-v1", calendar.monthly("rebalance"))` through `schedules=`;
use a separate schedule with `policy="catch_up"` for overdue-safe exits. You do
not need an application polling loop. `DurableSchedule` remains available for
specialized integrations.

For continuous daily/monthly operation, declare a `SessionSchedule` instead of
precomputing a finite timer list:

```python
from datetime import date, timedelta
from convexity import SessionSchedule

monthly = SessionSchedule("monthly-v1", "rebalance", start=date(2026, 10, 9),
                          frequency="monthly", at="open", offset=timedelta(minutes=5))
# Pass schedules=[monthly] to PaperTrading. Keep start fixed on restart.
```

The runner resolves full exchange years, handles holidays/early closes/DST,
and adds future durable schedules across year boundaries. Offsets must be less
than one day in either direction. Reconnect skips missed `skip` entries and
delivers overdue `catch_up` exits; due events across schedules are chronological.

Working stock history retains `StockSubscription.max_bars` (default 10,000).
Recent events, errors and health retain `event_history_limit` (default 10,000).
Choose sufficient history for your model's lookback. Completed quote/account/
connection callback records retain the latest 100 per owner and kind. Uncertain
callbacks, executions, order records and trading-event identities remain durable;
their audit storage grows with activity. Back up the deployment directory.

`SubmitOrder(..., time_in_force="DAY" | "GTC")` preserves its declared lifetime
on paper and in simulation. Paper market commands become bounded limits using
the configured collar; they may remain unfilled. This execution policy differs
from the simulator's next-bar-open market-fill model. Paper broker records retain
the actual transmitted limit alongside the original command.

Explicit stock client IDs are scoped to the strategy: a command with
`client_order_id="entry-1"` from `my-strategy` exposes order ID
`my-strategy:entry-1` in context, order updates and fills. Cancel that public ID,
or read `order.order_id` from context. Broker-safe transport IDs remain internal.
Reusing an ID for a different side, quantity, symbol, type, price or lifetime
emits `Rejection`; an identical retry does not resubmit. IDs remain reserved
for the whole run/deployment after an order fills or cancels. A recurring entry
must omit the ID or use `f"entry:{event.time.isoformat()}"`; a constant `"entry"`
will not create a new daily order. Old journals without
original-command metadata remain readable, but ambiguous intent reuse is rejected.

Both execution modes emit `Rejection` for rejected commands. When a recorded
order becomes rejected, its `OrderUpdate` also exposes terminal state.
Stock `FillEvent` on paper waits for its commission report. Option leg executions
remain explicit `BrokerOrderUpdate` records.

For an operator action use `await paper.command(owner, command)` (or the lower
level `PaperRuntime.command`). Stock submissions through this API require an
explicit client ID. A paused strategy may cancel or reduce verified owned
exposure; an operator call cannot silently resume it or open a new position.

Callback events include `Bars`, `Timer`, `StockQuotes`, `OptionQuotes`,
`AccountUpdate`, `ConnectionHealth`, `OrderUpdate`, `FillEvent`, `BrokerOrderUpdate`
and `Rejection`. Stock/account observations retain their broker timestamps and
quality. A quote event is not proof that every quote is tradable; execution gates
still check live bid/ask receipts, identity, ownership and available capacity.

## Interrupted callbacks

Strategy exceptions and unresolved command outcomes pause their owner and remain
visible in status. The managed runner keeps observing working orders and fills,
permits explicit cancellation or position reduction, and continues healthy owners.
Startup callback failures are isolated the same way after account/data validation.
The runner never resumes a paused owner or retries its failed callback automatically.

Managed schedules persist a separate delivery claim for each owner. A failed
owner's timer remains uncertain and suspends that owner's schedule namespace;
healthy owners continue receiving later timers from the same declared schedule.
`paper.schedule_status()` exposes each row's logical `schedule`, `strategy_id`,
actual durable `namespace`, and delivery status, including after shutdown. Paused
owners' future timers remain unclaimed. Low-level `PaperRuntime.step(Timer(...))`
still raises `DeliveryError`, even if an adapter discards its result; an explicit
`owners=(...)` selects and reports only those recipients. `fail_fast=True` opts
into terminating the managed session on any failed strategy delivery. Shutdown
never implicitly cancels orders or liquidates positions.

Resolve actual unknown outcomes against the journal and broker before restarting.
`StrategyState(path_to_orders_sqlite, account).unresolved(owner)` reports the
callback identity and error; `resolve(owner, identity, detail=...)` records the
reconciliation without replaying user code. Close that store after use. A timer
whose delivery was interrupted also needs `DurableSchedule.resolve(...)` to record
its reconciled outcome. For a managed timer, use the durable namespace reported by
`paper.schedule_status()`, rather than the logical declaration's namespace. These are explicit recovery APIs for ambiguous effects,
not normal application event-loop responsibilities.
