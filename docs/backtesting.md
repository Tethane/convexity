# Event-driven research

A strategy owns its identity, trading permissions, subscriptions, and warmup.
`Research` defines an inclusive range of session dates and connects the runtime,
historical feeds, and simulated exchange. The strategy uses the same commands,
context, and event callbacks in research and paper execution.

Use this guide to run strategies against historical events. For exploratory
DataFrames, see [historical data](data.md); for models and estimators, see
[Convexity math](math/README.md). Order amendments, stops, groups, and dynamic
subscriptions are covered in the [order lifecycle guide](order-lifecycle.md).

## Setup and first run

From the repository root, install the research dependencies with Python 3.14+:

```sh
python -m pip install -e '.[ibkr,calendars]'
```

The default historical provider uses a read-only IBKR connection. TWS or IB
Gateway must be available for data not already cached, with the required data
permissions. With no host/port configured, discovery tries Gateway paper (4002),
TWS paper (7497), Gateway live (4001), then TWS live (7496), always read-only.
For each port it tries localhost and, under WSL, the Windows host detected from
the default route. Explicit `IBKR_HOST` and `IBKR_PORT` pin their respective
settings and disable fallback for that setting. Client ID defaults to `72`;
override it with `IBKR_CLIENT_ID`. `IBKR_TIMEOUT` (default 15 seconds) bounds
all connection attempts together. Errors list attempted endpoints.
The standalone `cx.data.IBKR` adapter uses the same host/port discovery.
Research does not require a paper account configuration or submit broker orders.

```python
import convexity as cx

class BuyAndHold(cx.Strategy):
    metadata = cx.StrategyMetadata("buy-hold", version="1")
    contract = cx.StrategyContract(instruments=("SPY",))

    def on_start(self, context):
        return cx.SubmitOrder.market("SPY", "BUY", 10, time_in_force="DAY",
                                     client_order_id="entry")

result = cx.Research("2025-01-02", "2025-02-07").run(BuyAndHold)
print(result.metrics, result.positions, result.open_orders)
```

There is no bar subscription in this strategy. Its startup order enters the
exchange queue; returning it does not mean it filled. The simulator acquires
execution prices itself. An observed price from the prior session supports
startup risk checks without requiring strategy warmup or exposing future prices.

Submitting QQQ from this strategy violates its SPY contract. The runtime validates
all commands from a callback before routing any of them. Research raises
`BacktestFailure` with the partial result and contract diagnostic; direct runtime
commands raise `ContractViolation`. Observing an instrument does not grant
permission to trade it.

## Run configuration

`Research(start, end)` accepts ISO date strings or Python `date` objects. Both
dates are inclusive exchange-session dates; weekends and holidays contribute no
sessions. A range containing no sessions fails explicitly.

Each run creates fresh instances from the supplied class or factory. For multiple
strategies, pass a sequence of factories. Names come from `StrategyMetadata` and
must be unique. Parameterized strategies may set metadata and contract in
`__init__`.

```python
research = cx.Research("2025-01-02", "2025-02-07")
result = research.run(
    BuyAndHold,
    execution="conservative",
    initial_cash=100_000,
    output="out/buy-hold",
)
```

Infrastructure options belong to `run` or `run_async`:

| Option | Default | Purpose |
| --- | --- | --- |
| `provider` | Read-only IBKR history | A `HistoricalProvider` implementation |
| `cache` | `.convexity/history.sqlite3` | Persistent validated history; `None` disables the outer cache |
| `execution` | `"standard"` | A named preset or `ExecutionModel` |
| `initial_cash` | `100_000` | Starting simulated account cash |
| `limits` | Default `RiskLimits` | Account and strategy risk constraints |
| `output` | `None` | Directory for result and replay exports |
| `refresh` | `False` | Reacquire requested history |
| `metadata` | `None` | Additional run metadata |

In a notebook or another running event loop, use the asynchronous entry point:

```python
result = await research.run_async(BuyAndHold, output="out/buy-hold")
```

`run()` cannot run inside an existing event loop. A single `Research` instance
also rejects concurrent runs; use separate instances for concurrent experiments.

### Connection refused (including WSL)

`HistoryError` containing `ConnectionRefusedError` means the configured IBKR
socket endpoint refused the connection. Start and log into TWS or IB Gateway,
enable API socket access, and match its configured host and port. Changing the
strategy or switching between `run` and `run_async` does not fix this connection.

When Python runs in WSL and Gateway runs on Windows, `127.0.0.1` may not reach
Windows. Discovery now includes the WSL default-route host automatically,
following [Microsoft's WSL networking guidance](https://learn.microsoft.com/windows/wsl/networking).
For custom networking or ports, check `ip route` and pass the connection
explicitly in a notebook:

```python
provider = cx.IBKRHistoricalData(cx.HistoricalConnectionConfig(
    host="172.31.96.1",  # Replace with your reachable Gateway host.
    port=4002,          # Match Gateway's configured API socket port.
    client_id=72,
))
result = await cx.Research("2025-01-02", "2025-02-07").run_async(
    BuyAndHold, provider=provider,
)
print(result.metrics, result.positions, result.open_orders)
```

Alternatively set `IBKR_HOST` and `IBKR_PORT` in the Python process environment
before constructing the default provider. Shell exports made after a notebook
kernel starts do not update that kernel's environment. Research downloads real
history and simulates orders locally; unavailable history has no synthetic fallback.

## Clock events

```python
class Overnight(cx.Strategy):
    metadata = cx.StrategyMetadata("overnight")
    contract = cx.StrategyContract(
        instruments=("SPY",),
        subscriptions=(
            cx.ClockSubscription("enter", at="15:50"),
            cx.ClockSubscription("exit", at="09:30"),
            cx.EventSubscription(cx.OrderUpdate),
            cx.EventSubscription(cx.FillEvent),
        ),
    )

    def on_timer(self, event, context):
        if event.name == "enter" and not context.position("SPY") and not context.pending("SPY"):
            return cx.SubmitOrder.market("SPY", "BUY", 10)
        if event.name == "exit" and context.position("SPY") > 0:
            return cx.SubmitOrder.market("SPY", "SELL", context.position("SPY"))

    def on_partial(self, event, context):
        # event.order describes the partially filled order. Context already
        # reflects executions. This strategy leaves the remainder working.
        pass

    def on_fill(self, event, context):
        # event.fill contains execution quantity, price, fee and timestamps.
        pass

result = cx.Research("2025-01-02", "2025-02-07").run(Overnight)
```

Clock subscriptions default to `America/New_York` and the `XNYS` exchange
calendar. They occur only within actual sessions: holidays are absent, DST is
respected, and 15:50 is skipped on a session closing at 13:00. The simulator
selects a sufficiently fine execution feed for these events. This strategy does
not receive bar callbacks. For exchange-relative daily/monthly schedules, include
`SessionSchedule(...)` in the strategy's subscriptions.

## Bars, warmup, and callbacks

To observe bars, declare `BarSubscription(("SPY",), bar_size="1 day")` in
`contract.subscriptions`. Set `contract.warmup_sessions=5` if the strategy needs
five prior sessions in `context.history` at startup. Warmup does not invoke
trading callbacks or create simulated executions. Only completed observations
are exposed. The runtime scopes market events and history to subscriptions;
trading permissions remain the `instruments` declaration.

| Subscription | Callback (fallback: `on_event`) |
| --- | --- |
| `BarSubscription` | `on_bar(event, context)` |
| `ClockSubscription` / `SessionSchedule` | `on_timer(event, context)` |
| `EventSubscription(FillEvent)` | `on_fill(event, context)` |
| `EventSubscription(OrderUpdate)` | `on_partial` for partial status, falling back to `on_order` |
| `EventSubscription(Rejection)` | `on_rejection(event, context)` |
| Subscription lifecycle (automatically delivered to the owner) | `on_subscription(event, context)` |

A specific callback takes precedence over `on_event`; each event is delivered
once. `on_start` runs after declared warmup and `on_stop` runs at shutdown.
Callbacks return one command, an iterable of commands, or `None`. Context is an
immutable snapshot of observed history, marks, attributed positions, pending
orders, cash, and equity. Exchange events are delivered to the owning strategy.

To activate or deactivate subscriptions during a run, declare the permitted pool
in `contract.available_subscriptions` and use `context.subscribe()` /
`context.unsubscribe()`. Changes take effect after the callback completes and
produce `SubscriptionUpdate` events. See the
[subscription lifecycle](order-lifecycle.md#dynamic-subscriptions) for identities,
history visibility, and reconnection behavior.

## Explicit order intent and execution

```python
cx.SubmitOrder.market("SPY", "BUY", 10, time_in_force="DAY")
cx.SubmitOrder.limit("SPY", "BUY", 10, limit_price="500.00", time_in_force="GTC")
cx.SubmitOrder.limit("SPY", "BUY", 10, limit_price="500.00", time_in_force="IOC")
cx.CancelOrder(order_id="buy-hold:entry")
```

The named constructors default to `DAY`. The lower-level
`SubmitOrder(symbol, side, quantity, ...)` remains market/GTC for compatibility;
prefer the named constructors so intent is visible.

| Time in force | Simulated behavior |
| --- | --- |
| DAY | Rest until filled, canceled, rejected, or the session close |
| GTC | Rest across sessions; remain inspectable when research ends |
| IOC | At the first eligible execution opportunity, fill available quantity and cancel the remainder |
| FOK | At that opportunity, fill the whole quantity or cancel without any fill |

Market, limit, stop, and stop-limit orders are supported. Stops and grouped
orders require DAY/GTC. Invalid combinations and unsupported order types fail
explicitly. Immediate limit orders must be executable at the
opening price proxy; a later intrabar crossing cannot rescue them. Paper execution
currently rejects IOC/FOK explicitly because its adapter cannot preserve these
semantics. Paper market commands use the adapter's documented protective limit
collar; inspect broker updates for the actual submitted intent.

Use `execution="standard"` (default), `"frictionless"`, or `"conservative"` in
`run`. The standard preset uses a 2 bp spread, 1 bp adverse slippage, $0.005/share
commission with $1/order minimum, 10% volume participation and 3% annual borrow.
Conservative uses 5/3 bp spread/slippage, 1% participation, one execution-interval
submission/cancel delays, strict limit penetration and 5% borrow. These are
model assumptions, not calibrated exchange guarantees. Advanced callers can
supply `ExecutionModel(...)` and `RiskLimits(...)`.

The current exchange backend uses completed stock OHLCV observations as its
liquidity/price model. Strategies are event-driven; the model does not reconstruct
L2 queue positions or an intrabar tape. FIFO order acceptance shares volume
across strategies, permitting partial fills and cancellation races. Orders
cannot fill from the bar that generated them. Fills at an eligible bar's open
become available at its end; intrabar limit crossings are timestamped at the end.
Fills update accounting before callbacks, and bars precede equal-time timers.
There is no automatic end-of-run liquidation.

For amendments, filtered cancellation, and bracket/OCO behavior in simulation
and paper execution, see [order and subscription lifecycle](order-lifecycle.md).

## Historical data and persistent cache

To override the default connection in Python, pass
`provider=cx.IBKRHistoricalData(cx.HistoricalConnectionConfig(...))` to `run`.
There is no synthetic fallback when acquisition fails.

Validated observations are stored in `.convexity/history.sqlite3`. Repeating a
covered request reads SQLite without downloading history. Expanding the date
range or instrument set fetches missing sessions and inserts them into the same
database. Provider identity, calendar, resolution, and price basis separate
incompatible datasets. Use `cache=...` to choose the database and `refresh=True`
to deliberately reacquire requested data. Custom `HistoricalProvider` adapters
are wrapped by the same cache; give them a stable `cache_identity`.
Covered requests can replay offline, including subsets of previously acquired
history. Execution feeds and required prior-session data must also be cached.

This cache belongs to `Research`. The standalone `convexity.data.history()` API
returns DataFrames and does not share it. Its `end` boundary is exclusive,
whereas `Research` includes the end session. A DataFrame provider is not a
`HistoricalProvider`; see [historical data](data.md) for that API's contract.

## Scope and interpretation

Prices and coverage are validated before replay. Missing intervals fail rather
than being forward-filled. IBKR datasets use split-adjusted price-return units;
dividend cash flows, corporate-action accounting, historical universe membership,
and L2 replay are outside this MVP. The simulator currently uses one exchange
calendar per run and one bar resolution per symbol within each strategy.
Different strategies may observe the same symbol at different resolutions.
Research supports bar, clock, fill, order-update, rejection, and subscription
lifecycle events; unsupported feed requests fail before acquisition. Inspect
`cx.RESEARCH_CAPABILITIES` and `cx.PAPER_CAPABILITIES` for supported commands,
events, and restrictions. The managed paper feed currently supports one-minute
bars. Use matching observation cadence when comparing research and paper runs.

## Currency ledgers and FX

Both `Research.run_async(...)` and `Backtest(...)` accept `base_currency`
(default `"USD"`), `initial_cash` as either a scalar in base currency or a
currency/balance mapping, and a tuple of dated `FXRate` observations.

For example, configuration with caller-supplied historical rates looks like:

```python
from decimal import Decimal

result = await cx.Research("2025-01-02", "2025-02-07").run_async(
    MyStrategy,
    base_currency="USD",
    initial_cash={"USD": Decimal("50000"), "CAD": Decimal("20000")},
    fx_rates=historical_fx_rates,  # tuple of cx.FXRate objects supplied by you
)
```

Each `cx.FXRate(currency, base_currency, rate, time)` means one unit of
`currency` is worth `rate` base units, **known at the timezone-aware time**.
Supply an observation at or before replay startup for every nonzero foreign
cash balance. Future rates cannot fill a missing initial conversion. Subsequent
observations update valuation at replay steps; callbacks only see rates already
available at their time. Opening-fill risk checks use FX known at the opening,
rather than the later bar close. FX observations do not independently trigger
strategy callbacks. These rates are explicit replay inputs; Research does not
automatically download or invent FX history.

Cash stays in its native ledger. Buying a USD stock debits USD; if funded by
foreign-currency collateral this can create a negative USD balance (a currency
loan). There is no implicit FX trade, FX spread/fee, deposit interest or currency
loan interest. Short-stock borrow fees remain supported. Currency conversion
changes reporting value, not currency ownership.

Research's historical acquisition currently remains US USD stocks.
For supplied data, `Backtest(instruments=(cx.Instrument("LABEL", currency="CAD"), ...))`
supports instrument currencies explicitly. Simulated execution commissions are
in each instrument's currency; fee and borrow-cost totals are translated into
base currency when recorded. Position average costs and realized P&L retain
instrument currency. Base-currency equity and returns include FX revaluation.

`result.account` exposes the final account, including native cash and FX rates.
`result.equity` records cash balances, margin, buying power, reservations and
remaining sizing capacity over time. Account totals and result metrics use
`result.config["base_currency"]`; exports retain currency metadata.
See [strategy prices, sizing and margin semantics](strategies.md#current-prices-and-buying-quantities).

## Inspect and save results

`output` saves orders, fills, equity, cash flows, events, resolved contracts,
execution assumptions, history provenance, and replay inputs. Failed strategy
runs retain their inspectable result on `BacktestFailure.result`. Returns and
maximum drawdown are fractions. A completed run can contain rejections or open
orders; inspect those alongside metrics.

```python
print(result.metrics)
print(result.positions)
print(result.open_orders)
result.export("out/review")  # Save the result after a run.
```

The result is a `BacktestResult` dataclass, not a pandas DataFrame or NumPy
array. `metrics` is a dictionary of `Decimal` values; `equity` is a tuple
of timestamped dictionaries; `orders` and `fills` are tuples of dataclasses.
`positions` maps strategy names to symbol/position dictionaries.

Convert the equity history to pandas for plotting or further analysis. Convert
`Decimal` values to floats for plotting while retaining the original result for
exact accounting:

```python
import pandas as pd
import matplotlib.pyplot as plt

equity = pd.DataFrame(result.equity).set_index("time")
equity.index = pd.to_datetime(equity.index, utc=True)
equity[["cash", "equity"]] = equity[["cash", "equity"]].astype(float)
equity["equity"].plot(title="Portfolio equity", ylabel=result.config["base_currency"], grid=True)
plt.tight_layout()
plt.show()

values = equity["equity"].to_numpy()  # NumPy array, if needed.
```

Use `output=...` on the research run to include acquired bars, sessions, and
the history manifest alongside the result export. `result.export(...)` saves
the result and its CSV tables, without copying the acquired history files.

For a worked experiment, open the [research notebook](../examples/convexity_demo.ipynb)
or [same-strategy script](../examples/strategy_workflow.py). The
[math strategy example](../examples/backtest_math.py) uses observed returns to
compute portfolio targets.

## Migrating older calls

Migrate old `Research(symbols=..., strategies={...}, bar_size=...,
warmup_sessions=..., config=...)` calls by moving identity and data requirements
into the strategy and passing factories to `run`. Infrastructure settings such
as provider, cache, capital, execution preset, and output belong to `run`.
`Backtest` remains the explicit local-data replay API.
