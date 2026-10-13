"""Build the comprehensive, real-data Convexity teaching notebook.

Run with the quant Python environment. This writes source cells only; execute
the notebook separately to verify research and render figures. Paper trading is
disabled in the generated notebook.
"""
import ast
import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text, *, tags=()):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip(), metadata={"tags": list(tags)}))


md('''
# Convexity: from an idea to a real-data backtest and a paper strategy

**A guided, executable demonstration.** Start at the top and run cells in order.
You will write a strategy, retrieve actual IBKR historical observations, inspect
the simulated execution, test assumptions, and learn the paper deployment API.

The default lesson uses **SPY and QQQ, January 2–3, 2025, one-minute bars**.
The small period keeps the demo quick; it is not enough to validate an investment
idea. Prices come from IBKR or its previously validated cache. There is no
synthetic-price fallback. Research simulates executions and never sends orders
to the broker. **The paper-order section is disabled by default.**

### How to present this

For a 15-minute demo, show sections **1–3, 5–9, 11, and 17**. For a full lesson,
allow 60–90 minutes and work through the experiments and exercises.

1. [What the system does](#map)
2. [Environment and configuration](#setup)
3. [The strategy contract](#strategy)
4. [Execution and risk assumptions](#assumptions)
5. [The first real-data run](#research)
6. [Prove which prices were used](#data)
7. [Read results and draw charts](#results)
8. [Trace an order without lookahead](#causality)
9. [Stress costs and compare a baseline](#costs)
10. [A chronological research experiment](#experiments)
11. [Offline replay and reproducibility](#replay)
12. [Commands, IDs, limits, and failures](#commands)
13. [Schedules and target strategies](#schedules)
14. [Use convexity.math in a strategy](#math)
15. [Checkpoint state and restart semantics](#state)
16. [Paper trading: what changes](#paper)
17. [An explicit paper demonstration](#paper-demo)
18. [Options, troubleshooting, and your next idea](#next)

**Presenter opening:** “I supply a rule, instruments, dates, and assumptions.
Convexity owns data acquisition, replay, events, order bookkeeping, and reports.”
''')

md('''
<a id="map"></a>
## 1. What is real, and what is simulated?

| Component | Responsibility |
|---|---|
| `Research` | Fetch real historical bars, validate sessions, cache, warm up, run, export |
| `Strategy` | Read the current `Context` and return commands |
| `Runtime` | Advance the clock, dispatch events, track orders, risk and accounting |
| `SimulatedExchange` | Model order acceptance, fills, cancellations and execution costs |
| `Backtest` | Replay an explicitly supplied dataset without downloading prices |
| `PaperTrading` | Own a running IBKR paper session, observations, timers and reconnects |
| `convexity.math` | Calculations and models you can call from your strategy |

The research path is **stocks + dates → broker observations → strategy callbacks
→ simulated fills → inspectable results**. Historical prices are inputs, not
generated outputs. In paper mode, IBKR supplies order/execution/account truth.

The implemented research scope is **US USD stocks/ETFs, regular sessions,
one-minute or daily bars, split-adjusted price returns**. Dividend cash flows,
full point-in-time corporate-action accounting, options backtests, historical
index membership and automatic liquidation are not implemented. We will keep
those boundaries visible when interpreting the results.
''')

md('''
<a id="setup"></a>
## 2. Set up the notebook once

Select the **quant / Python 3.14** kernel. From a terminal in the repository:

```bash
conda run -n quant python -m pip install -e '.[ibkr,calendars,notebooks]'
```

Historical connections use `IBKR_HOST`, `IBKR_PORT`, and `IBKR_CLIENT_ID` from
the environment, with localhost:7497 defaults. A first fetch needs TWS/IB Gateway running,
socket API access, an unused client ID, and the relevant historical-data access.
Exact cache hits do not need a connected broker. A failed download stops the
lesson with an error; it does not substitute prices.

Run from the repository or its `examples` directory. The next cell discovers the
root and prints package versions, but deliberately does not display your account
configuration. This notebook uses top-level **`await`**; do not wrap it in
`asyncio.run()` inside Jupyter.
''')
code('''
from pathlib import Path
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from collections import Counter
import asyncio
import hashlib
import importlib.metadata as package_metadata
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

import convexity as cx
from convexity.strategies import rebalance_orders, TargetStrategy
from convexity.runtime import Rejection, StrategyState

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "pyproject.toml").is_file() and (p / "convexity").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook from the Convexity repository or its examples directory.")

pd.set_option("display.max_columns", 14)
pd.set_option("display.max_rows", 12)
plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 110})
print("Python:", sys.version.split()[0])
print("Convexity:", package_metadata.version("convexity"))
print("IBKR client:", package_metadata.version("ib_async"))
print("Exchange calendars:", package_metadata.version("exchange-calendars"))
''')
md('''
### The experiment settings

This is the main cell you edit for your own idea. Dates are **inclusive exchange
session dates**, not UTC cutoffs. Holidays are omitted using the calendar. Future
or unfinished sessions fail validation. `WARMUP_SESSIONS=1` means one *prior
trading session*, not one calendar day.

We deliberately use one-minute observations to match the managed paper runner.
Research accepts custom durations such as `7 mins`, `90 secs`, `2.5 hours`,
`3 days`, `2 weeks`, or `1 month`. Intraday bars aggregate complete source
observations from each session open, with the final bar clipped at the close.
Multi-session signals use daily execution bars. Changing the interval changes
the signal's lookback horizon; second data availability depends on IBKR.
This lesson's paper section requires `1 min` data.

The demo creates a timestamped research output directory. Paper deployment later
uses a separate **stable** directory because its ownership/checkpoints must survive
restarts. Historical requests use a separate client ID from the paper runner.
''')
code('''
SYMBOLS = ("SPY", "QQQ")
START, END = "2025-01-02", "2025-01-03"
BAR_SIZE = "1 min"
WARMUP_SESSIONS = 1
LOOKBACK = 20
SHARES = 2
INITIAL_CASH = D("100000")
OWNER = "demo-trend"

CACHE_DIR = ROOT / ".convexity" / "history.sqlite3"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
RUN_DIR = ROOT / "out" / "convexity-demo" / RUN_ID

history_config = cx.HistoricalConnectionConfig.from_env()
# Paper account settings are loaded only inside the opt-in paper cell.
print({"symbols": SYMBOLS, "start": START, "end": END, "bars": BAR_SIZE,
       "warmup_sessions": WARMUP_SESSIONS, "lookback_bars": LOOKBACK,
       "shares_per_positive_signal": SHARES})
''')

md('''
<a id="strategy"></a>
## 3. The entire strategy contract

| Method / input | Meaning |
|---|---|
| `on_start(context)` | Startup decision after warmup; on paper it is durably delivered once per owner |
| `on_event(event, context)` | Respond to bars, timers, order/fill/rejection events or paper observations |
| `on_stop(context)` | Shutdown notification; it does **not** liquidate or route returned commands |
| Handler return | One submit/cancel command, an iterable of commands, or `None` |
| `context.history[symbol]` | Completed bars visible as of the current runtime time |
| `context.position(symbol)` | This strategy's owned whole-share position |
| `context.pending(symbol)` | Signed remaining shares in its working orders |
| `context.orders` | This owner's current and terminal stock order records |
| `context.cash`, `context.equity` | Account totals, not a separate per-strategy wallet |
| `context.marks`, `mark_times` | Latest observed prices and when they became available |
| `context.broker_orders` | Durable paper broker records, including option leg executions |

Context is a read-only snapshot. Return intent; do not modify its positions.
Filter event types: submitting on every acknowledgment can cause a feedback loop.
Keep callbacks short and deterministic. Download data before replay, not from
inside the signal calculation.

### A transparent rule

For each stock: if the latest close exceeds the close `LOOKBACK` observed bars
ago, target `SHARES`; otherwise target zero. This is a teaching rule, not a claim
of alpha. Its config makes the choices inspectable.

`rebalance_orders` accounts for owned **and pending** shares, waits for working
orders, prioritizes reductions, and closes reversals before adding opposite
exposure. A target is an absolute holding, not “buy this many more every minute.”
''')
code('''
class DemoTrend(cx.Strategy):
    metadata = cx.StrategyMetadata(OWNER, version="demo-trend-v1")
    def __init__(self, symbols=SYMBOLS, lookback=LOOKBACK, shares=SHARES):
        if type(lookback) is not int or lookback < 2:
            raise ValueError("lookback must be an integer >= 2")
        if type(shares) is not int or shares < 1:
            raise ValueError("shares must be a positive integer")
        self.config = {"symbols": tuple(symbols), "lookback": lookback, "shares": shares}
        self.contract = cx.StrategyContract(
            instruments=tuple(symbols),
            subscriptions=(cx.BarSubscription(tuple(symbols), bar_size=BAR_SIZE),),
            warmup_sessions=WARMUP_SESSIONS,
        )

    def on_event(self, event, context):
        if not isinstance(event, cx.Bars):
            return None
        targets = {}
        for symbol in self.config["symbols"]:
            history = context.history[symbol]
            if len(history) < self.config["lookback"]:
                return None
            rising = history[-1].close > history[-self.config["lookback"]].close
            targets[symbol] = self.config["shares"] if rising else 0
        return rebalance_orders(context, targets)


def factories(lookback=LOOKBACK, shares=SHARES):
    return {OWNER: lambda: DemoTrend(SYMBOLS, lookback, shares)}

assert factories()[OWNER]() is not factories()[OWNER]()
display(DemoTrend().config)
''')
md('''
Pass **factories**, such as `lambda: MyStrategy(parameters)`. Research takes a
factory or sequence; the lower-level replay API also accepts owner mappings. Every
backtest needs fresh mutable state. Do not pass `lambda: already_used_instance`.
The runtime rejects reused instances across repeat runs.

This rule derives decisions from context, so it needs no custom checkpoint.
Later we will distinguish such stateless rules from strategies with durable
decision state. Real monetary calculations use `Decimal`; conversion to floats
below is only for pandas summaries and plotting.
''')

md('''
<a id="assumptions"></a>
## 4. Make execution costs and risk explicit

An attractive frictionless curve is not enough. We declare spread, slippage,
commissions and a maximum share of each bar's volume. These are assumptions,
not a guarantee of your broker's pricing. Partial fills share a FIFO liquidity
budget across orders. Delay fields count eligible bars, not wall-clock seconds.

The demo is long-only. Research risk limits reserve capacity for pending orders;
opposite orders do not magically free funds. The per-owner gross cap is distinct
from initial account cash. Small fixed-share exposure will produce a small
account return even when the stock price moves substantially.
''')
code('''
execution = cx.ExecutionModel(
    spread_bps=D("2"), slippage_bps=D("1"),
    commission_per_share=D("0.005"), minimum_commission=D("1"),
    volume_participation=D("0.10"),
    submission_delay_bars=0, cancel_delay_bars=0,
    limit_rule="touch", allow_short=False, annual_borrow_rate=D("0.03"),
)

def risk_for(owner=OWNER, order_cap=D("2000")):
    return cx.RiskLimits(gross_leverage=D("1"), max_order_notional=order_cap,
                         strategy_gross_limits={owner: D("6000")})

limits = risk_for()
display(pd.DataFrame([
    {"assumption": "spread", "value": "2 bps full spread"},
    {"assumption": "slippage", "value": "1 bp adverse per fill"},
    {"assumption": "commission", "value": "$0.005/share, $1 minimum per order"},
    {"assumption": "participation", "value": "10% of broker-supplied bar volume"},
    {"assumption": "order / strategy cap", "value": "$2,000 / $6,000 gross"},
]))
''')

md('''
<a id="research"></a>
## 5. Run the complete real-data workflow

This is the core API you use in your own notebook. The first matching request
downloads real prices. Subsequent identical requests validate and reuse the cache.
`refresh=True` explicitly creates a fresh snapshot; it can reflect broker data
revisions. We leave refresh off so our experiments use one fixed dataset.

**This cell is read-only with respect to IBKR.** Orders returned by the strategy
are routed to the simulator. The optional paper execution cell is much later.
''')
code('''
research = cx.Research(START, END)
result = await research.run_async(
    tuple(factories().values()), provider=cx.IBKRHistoricalData(history_config),
    cache=CACHE_DIR, output=RUN_DIR / "baseline",
    initial_cash=INITIAL_CASH, execution=execution, limits=limits,
    metadata={"purpose": "Convexity teaching demo", "notebook": "convexity_demo.ipynb"},
)
assert result.status == "completed"
print(f"{result.status}: {len(result.orders)} orders; {len(result.fills)} fills; "
      f"{len(result.open_orders)} orders still open")
print("Source:", result.config["data_source"])
print("Price basis:", result.config["price_basis"])
print("Dataset SHA-256:", result.config["data_identity"])
''')
md('''
**What just happened?** The library resolved the contracts and exchange sessions,
fetched the requested bars plus prior warmup sessions, validated every interval,
and saved provenance. It then created fresh runtime/exchange/strategy objects.
Warmup populated history and marks without invoking strategy callbacks. The
simulation began at the first requested session open.

An error here is useful information: unavailable data, missing bars or a wrong
account/configuration must not turn into a successful-looking test. See the
troubleshooting table at the end; do not replace the failure with invented prices.
''')

md('''
<a id="data"></a>
## 6. Inspect the prices and their provenance

The exported bars include warmup. The manifest identifies the real broker
contracts, acquisition times, exact requests, raw returned rows, adjustment
convention and hashes. We display a compact summary rather than thousands of
raw rows. This also distinguishes a completed replay from evidence about which
dataset was actually used.
''')
code('''
baseline_dir = RUN_DIR / "baseline"
manifest = json.loads((baseline_dir / "history_manifest.json").read_text())
bars = pd.read_csv(baseline_dir / "bars.csv")
for column in ("start", "end"):
    bars[column] = pd.to_datetime(bars[column], utc=True)
sessions_table = pd.read_csv(baseline_dir / "sessions.csv")
for column in ("open", "close"):
    sessions_table[column] = pd.to_datetime(sessions_table[column], utc=True)

display(pd.DataFrame([{
    "provider": manifest["provider"], "source": manifest["source"],
    "bar_size": manifest["request"]["bar_size"],
    "signal_bar_size": result.config["metadata"]["research"]["observation_bar_size"],
    "acquired_at_UTC": manifest["acquired_end"],
    "price_basis": manifest["price_basis"],
    "bars_including_warmup": len(bars),
}]))
display(pd.DataFrame(manifest["contracts"].values()))
display(bars.head(6))
display(sessions_table)
assert manifest["data_identity"] == result.config["data_identity"]
assert manifest["provider"] == "IBKR"
''')
md('''
IBKR `TRADES` observations are **split-adjusted, not dividend-adjusted**.
The simulator works in those adjusted price/share units. It does not credit
dividend cash flows or reconstruct original share/tick/fee behavior around splits.
Volume is kept as returned by the broker. This is **price-return research**, not
complete point-in-time corporate-action accounting.

A user-selected list such as today's SPY/QQQ does not model changing index
membership or delisted stocks. A missing interval causes a failure rather than
forward filling. A historical snapshot can be revised on a later refresh.
These properties matter when you choose a larger, economically meaningful study.
''')
code('''
study_start = pd.Timestamp(result.config["start"])
study_end = pd.Timestamp(result.config["end"])
warmup_bars = bars[bars["end"] <= study_start]
study_bars = bars[(bars["start"] >= study_start) & (bars["end"] <= study_end)]
print("Warmup bars:", len(warmup_bars), "| study bars:", len(study_bars))
assert not warmup_bars.empty
assert (warmup_bars["end"] <= study_start).all()

close_panel = study_bars.pivot(index="end", columns="symbol", values="close")
ax = (100 * close_panel / close_panel.iloc[0]).plot(title="Actual broker observations: normalized closes")
ax.set_ylabel("First observed study close = 100")
ax.set_xlabel("UTC; overnight gaps remain visible")
plt.show()
''')

md('''
<a id="results"></a>
## 7. Read the result, not just the return

`completed` means the replay completed. It does not mean all orders filled,
positions are flat, or the strategy is profitable. Start with status, diagnostics,
rejections, open orders and positions; then inspect performance.

`total_return` is modeled account-equity change divided by initial cash. It is a
fraction, not a percentage, and it excludes omitted dividend cash flows.
`max_drawdown` is also a fraction. Fees and borrow costs are separate metrics.
We report dollar equity change as well because only a small part of the account
is invested by this fixed-share demonstration.
''')
code('''
def summary(label, value):
    return {
        "run": label, "status": value.status, "fills": len(value.fills),
        "rejected": sum(o.status == "rejected" for o in value.orders),
        "open_orders": len(value.open_orders),
        "equity_change_USD": float(value.metrics["final_equity"] - INITIAL_CASH),
        "return_pct": float(value.metrics["total_return"] * 100),
        "max_drawdown_pct": float(value.metrics["max_drawdown"] * 100),
        "fees_USD": float(value.metrics["fees"]),
        "borrow_USD": float(value.metrics["borrow_cost"]),
    }

def equity_frame(value):
    frame = pd.DataFrame(value.equity)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    for column in ("cash", "equity", "fees", "borrow_cost"):
        frame[column] = frame[column].astype(float)
    return frame.set_index("time")

display(pd.DataFrame([summary("baseline", result)]).set_index("run"))
print("Diagnostics:", result.diagnostics)
display(pd.DataFrame([{"owner": owner, "symbol": symbol, "quantity": position.quantity,
                      "average_cost": str(position.average_cost),
                      "realized_pnl_before_fees": str(position.realized_pnl)}
                     for owner, holdings in result.positions.items() for symbol, position in holdings.items()]))
''')
code('''
equity = equity_frame(result)
drawdown = equity["equity"] / equity["equity"].cummax() - 1
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
(equity["equity"] - float(INITIAL_CASH)).plot(ax=axes[0], color="#146b75")
axes[0].set(title="Strategy account: modeled outcomes on real observations", ylabel="Equity change ($)")
(100 * drawdown).plot(ax=axes[1], color="#b04a48")
axes[1].set_ylabel("Drawdown (%)")
equity[["fees", "borrow_cost"]].plot(ax=axes[2])
axes[2].set(ylabel="Cumulative costs ($)", xlabel="UTC")
plt.show()
''')
code('''
fills = pd.DataFrame(result.to_dict()["fills"])
if not fills.empty:
    for column in ("occurred_at", "available_at"):
        fills[column] = pd.to_datetime(fills[column], utc=True)
    display(fills.head(10))
    display(fills.groupby(["symbol", "side"]).agg(fill_count=("execution_id", "count"),
                                               shares=("quantity", "sum")))
else:
    print("No fills. Inspect signals, rejections, pending orders and the requested study window.")

order_rows = [{"id": o.order_id, "symbol": o.request.symbol, "side": o.request.side,
               "quantity": o.request.quantity, "filled": o.filled_quantity,
               "status": o.status, "reason": o.reason} for o in result.orders]
orders = pd.DataFrame(order_rows)
display(orders.head(10))
print("Terminal status counts:", dict(Counter(o.status for o in result.orders)))
print("Last observed holdings remain open unless your strategy explicitly closed them.")
''')

md('''
<a id="causality"></a>
## 8. Follow one order through time

The strategy sees a completed bar at its **end**. Its new market order can fill
at the next eligible bar's open, with modeled adverse costs. That execution is
published to the strategy at the bar's end. Intrabar limit fills are conservatively
timestamped at bar end because OHLC does not reveal the crossing time.

At a replay step the ordering is: borrow accrual → existing-order fills and
cancellations → DAY expiry → install the completed bars → order/fill/bar callbacks
→ equal-time named timers → returned commands → equity record.

Here we check actual artifacts from this run. No invented prices are needed to
demonstrate causality. An opening fill may occur at exactly the preceding bar's
end; that is the boundary of the *next* bar, not a fill using the signal bar's OHLC.
''')
code('''
if result.fills:
    fill = result.fills[0]
    order = next(o for o in result.orders if o.order_id == fill.order_id)
    eligible = bars[(bars["symbol"] == fill.symbol)
                    & (bars["start"] == pd.Timestamp(fill.occurred_at))]
    display(pd.DataFrame([
        {"stage": "order accepted from observed signal", "UTC": order.accepted_at},
        {"stage": "modeled execution occurrence", "UTC": fill.occurred_at},
        {"stage": "fill available to strategy", "UTC": fill.available_at},
    ]))
    display(eligible)
    assert order.accepted_at <= fill.occurred_at < fill.available_at
    assert fill.available_at <= result.config["end"]
    print("Modeled fill price:", fill.price, "| fee:", fill.fee)

print("Recorded event types:", dict(Counter(type(event).__name__ for event in result.events)))
''')

md('''
<a id="costs"></a>
## 9. Challenge the result: costs and a simple baseline

We now use **the saved real bars**, so these experiments need no new download.
`Backtest` is the lower-level offline API behind `Research`. Pass the recorded
price basis and session calendar; otherwise it is easy to mislabel adjusted data
or lose DAY-expiry/session semantics.

The frictionless run is a diagnostic comparison only. The stress run raises
slippage and minimum commission. Costs can change capacity and even which orders
execute, so the difference is not always a simple subtraction from one fixed
fill list. Compare the resulting fills and orders too.
''')
code('''
real_data = cx.HistoricalData.from_csv(baseline_dir / "bars.csv", price_basis=manifest["price_basis"])
real_calendar = cx.SessionCalendar.from_csv(baseline_dir / "sessions.csv")
assert real_data.identity == result.config["data_identity"]

def replay(*, strategy_factories=None, model=execution, lookback=LOOKBACK,
           start=None, end=None, owner=OWNER, timers=(), risk=None):
    first, last = start or result.config["start"], end or result.config["end"]
    observations = real_data.resample(BAR_SIZE, real_calendar, boundaries=(first, last))
    return cx.Backtest(
        data=real_data, calendar=real_calendar,
        observations=observations,
        strategies=factories(lookback) if strategy_factories is None else strategy_factories,
        initial_cash=INITIAL_CASH, execution=model,
        limits=risk if risk is not None else risk_for(owner),
        start=first, end=last,
        warmup=True, timers=timers,
        metadata={"input_data_sha256": real_data.identity, "purpose": "notebook experiment"},
    ).run().raise_for_failure()

frictionless = replay(model=replace(execution, spread_bps=D(0), slippage_bps=D(0),
                                   commission_per_share=D(0), minimum_commission=D(0)))
stressed = replay(model=replace(execution, slippage_bps=D(10), minimum_commission=D(2)))
display(pd.DataFrame([summary("frictionless diagnostic", frictionless), summary("baseline", result),
                      summary("cost stress", stressed)]).set_index("run"))
''')
md('''
A useful comparison is a **same-size buy-and-hold portfolio**. It holds the same
number of shares per symbol and uses the same model, cash and constraints. It
is not an equal-dollar market index benchmark, and its exposure time differs
from the timing strategy. Both remain marked to the last observed price without
an invented end-of-test sale.
''')
code('''
class BuyAndHold(cx.Strategy):
    def on_start(self, context):
        return tuple(cx.SubmitOrder(symbol, "BUY", SHARES, client_order_id=f"hold-{symbol}")
                     for symbol in SYMBOLS)

hold = replay(strategy_factories={OWNER: BuyAndHold})
display(pd.DataFrame([summary("trend rule", result), summary("same-size buy & hold", hold)]).set_index("run"))
fig, ax = plt.subplots()
for label, value in (("Trend", result), ("Same-size hold", hold), ("Cost stress", stressed)):
    (equity_frame(value)["equity"] - float(INITIAL_CASH)).plot(ax=ax, label=label)
ax.set(title="Compare account P&L on the same real dataset", ylabel="Equity change ($)", xlabel="UTC")
ax.legend()
plt.show()
''')

md('''
<a id="experiments"></a>
## 10. A chronological experiment, without selecting on the test set

The mechanics of an honest experiment:

1. State the rule and a small candidate parameter set before testing.
2. Split by **time**, keeping complete sessions together.
3. Choose a parameter using training results only.
4. Evaluate that single choice on the later holdout period.
5. Record unsuccessful results, costs, diagnostics and exposure as well as winners.

The two-day default has just one training session and one test session. This
demonstrates the API; it is not statistical evidence. For a substantive study,
use many sessions and suitable independent/walk-forward validation. Repeatedly
tuning after inspecting the holdout makes it part of the training process.

Each split below starts with fresh cash and positions. Prior observations can
warm up the later test, but trained positions do not carry over. That is a
deliberate independent-test convention, not a continuously held portfolio.
''')
code('''
study_sessions = [s for s in real_calendar.sessions
                  if result.config["start"] <= s.open and s.close <= result.config["end"]]
if len(study_sessions) < 2:
    raise ValueError("Use at least two study sessions for the chronological-split lesson.")
split_index = max(1, min(len(study_sessions) - 1, int(0.7 * len(study_sessions))))
train_sessions, test_sessions = study_sessions[:split_index], study_sessions[split_index:]
candidates = (10, 20, 60)
training_results = {}
for lookback in candidates:
    training_results[lookback] = replay(lookback=lookback, start=train_sessions[0].open,
                                        end=train_sessions[-1].close)
training_table = pd.DataFrame([dict(lookback=k, **summary("train", v))
                               for k, v in training_results.items()]).set_index("lookback")
display(training_table)
chosen = max(candidates, key=lambda k: (training_results[k].metrics["total_return"], -k))
holdout = replay(lookback=chosen, start=test_sessions[0].open, end=test_sessions[-1].close)
print("Chosen on training only:", chosen)
print("Holdout sessions:", test_sessions[0].label, "through", test_sessions[-1].label)
display(pd.DataFrame([summary("untuned holdout", holdout)]))
experiment = {"candidate_lookbacks": candidates, "chosen_lookback": chosen,
              "training_end": str(train_sessions[-1].label),
              "holdout_start": str(test_sessions[0].label),
              "dataset_sha256": real_data.identity,
              "selection_rule": "training modeled equity return, smaller lookback breaks ties"}
(RUN_DIR / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\\n")
holdout.export(RUN_DIR / "holdout")
''')

md('''
<a id="replay"></a>
## 11. Prove an exact cached replay works offline

This second `Research` run uses the same input snapshot. We give the provider a
client factory that **raises if any network client is created**. Success therefore
demonstrates a true cache hit, not a hidden second download.

We compare all seven artifacts: results, bars, sessions, provenance, fills, equity
and cash flows. Identical results require unchanged deterministic strategy code,
parameters, execution assumptions and observations. A changed source/version or
explicit refresh is a new experiment. Notebook-defined source hashes may be
unavailable; keep the notebook plus explicit strategy versions/configuration.
''')
code('''
def no_network_client():
    raise AssertionError("This demonstration requires an exact offline cache hit")

offline_provider = cx.IBKRHistoricalData(history_config, client_factory=no_network_client)
offline_result = await cx.Research(START, END).run_async(
    tuple(factories().values()), provider=offline_provider, cache=CACHE_DIR,
    output=RUN_DIR / "offline-replay", initial_cash=INITIAL_CASH,
    execution=execution, limits=limits,
    metadata={"purpose": "Convexity teaching demo", "notebook": "convexity_demo.ipynb"},
)
assert result.to_dict() == offline_result.to_dict()
artifact_names = ("result.json", "bars.csv", "sessions.csv", "history_manifest.json",
                  "fills.csv", "equity.csv", "cash_flows.csv")
artifact_rows = []
for name in artifact_names:
    first = (baseline_dir / name).read_bytes()
    second = (RUN_DIR / "offline-replay" / name).read_bytes()
    assert first == second, name
    artifact_rows.append({"artifact": name, "bytes": len(first),
                          "sha256": hashlib.sha256(first).hexdigest(), "identical": True})
display(pd.DataFrame(artifact_rows))
''')

md('''
<a id="commands"></a>
## 12. Orders, cancellation, rejection, and failure

These command objects describe intent. **Constructing them does not submit an
order.** The runtime routes commands returned by handlers; a paper operator can
also explicitly call `await paper.command(owner, command)`.

- Stocks support market/limit orders, BUY/SELL, whole positive shares and DAY/GTC.
- Explicit IDs are scoped to the owner: `entry-1` becomes `owner:entry-1`.
- An identical retry is deduplicated; conflicting intent with the same ID is rejected.
- Cancel the public ID from the order/context, not an invented IBKR transport ID.
- A canceled order may already have partial fills. Read the execution state.
- Operator submissions require explicit stable client IDs.
''')
code('''
example_market = cx.SubmitOrder(SYMBOLS[0], "BUY", 1, client_order_id="entry-1")
observed_price = real_data.bars[0].close
example_limit = cx.SubmitOrder(SYMBOLS[0], "BUY", 1, order_type="limit",
                               limit_price=observed_price, time_in_force="DAY",
                               client_order_id="limit-1")
example_cancel = cx.CancelOrder(f"{OWNER}:entry-1")
display(example_market, example_limit, example_cancel)
print("Constructed only: no broker action.")
''')
md('''
### A rejected command is visible

This intentionally impossible research order exceeds the configured cap. We
use the real observations already loaded, and inspect `Rejection` events. A
successful replay can include rejected orders; counting only return would hide
that the intended trade never happened.
''')
code('''
class OverBudget(cx.Strategy):
    def on_start(self, context):
        return cx.SubmitOrder(SYMBOLS[0], "BUY", 100_000, client_order_id="too-large")

rejected = replay(strategy_factories={OWNER: OverBudget})
reasons = [{"time": event.time, "owner": event.strategy_id, "reason": event.reason}
           for event in rejected.events if isinstance(event, Rejection)]
assert reasons and not rejected.fills
display(pd.DataFrame(reasons))
''')
md('''
### A broken callback is a failed strategy

The next example intentionally raises a programming error. Low-level `Backtest`
returns a failed result so you can inspect it. Calling `.raise_for_failure()`
raises `BacktestFailure` carrying that `.result`. High-level `Research` performs
that check automatically and saves partial output first when an output path was
configured. Catching this expected demonstration error is not a data fallback.
''')
code('''
class BrokenForTeaching(cx.Strategy):
    def on_start(self, context):
        raise ValueError("Intentional demo: my signal calculation is broken")

failed = cx.Backtest(data=real_data, calendar=real_calendar,
    strategies={OWNER: BrokenForTeaching}, execution=execution,
    initial_cash=INITIAL_CASH, start=result.config["start"], end=result.config["end"],
    warmup=True).run()
try:
    failed.raise_for_failure()
except cx.BacktestFailure as error:
    print("Expected failure:", error)
    assert error.result.status == "failed"
''')

md('''
<a id="schedules"></a>
## 13. Schedule decisions and use target functions

Not every strategy should decide on every bar. `TargetStrategy(function,
timer="rebalance")` calls a target function on a named timer, then pursues its
absolute share targets on subsequent bars. Returning `None` discards targets;
it does not cancel an already working order. Built-in `MonthlyLongShort` and
`StockRule` are another configurable starting point.

`SessionSchedule` works in both `Research` and `PaperTrading`. Calendars account
for holidays, early closes and DST. Monthly schedules resolve full months before
filtering dates, so starting mid-month does not fabricate a first-session timer.
The simulator acquires a sufficiently fine execution feed for intraday clocks,
independently of the bar observations requested by the strategy.
''')
code('''
scheduled_owner = "demo-scheduled"
daily_schedule = cx.SessionSchedule(
    "daily-signal-v1", "rebalance", start=date.fromisoformat(START),
    frequency="daily", at="open", offset=timedelta(minutes=5),
)

def morning_targets(context):
    return {symbol: (SHARES if context.history[symbol][-1].close >
                    context.history[symbol][-LOOKBACK].close else 0) for symbol in SYMBOLS}

class ScheduledTargets(TargetStrategy):
    metadata = cx.StrategyMetadata(scheduled_owner)
    contract = cx.StrategyContract(
        instruments=SYMBOLS, warmup_sessions=WARMUP_SESSIONS,
        subscriptions=(cx.BarSubscription(SYMBOLS, bar_size=BAR_SIZE), daily_schedule),
    )
    def __init__(self):
        super().__init__(morning_targets)

scheduled = await cx.Research(START, END).run_async(
    ScheduledTargets, provider=cx.IBKRHistoricalData(history_config), cache=CACHE_DIR,
    execution=execution, initial_cash=INITIAL_CASH,
    limits=risk_for(scheduled_owner), output=RUN_DIR / "scheduled",
)
display(pd.DataFrame([{"name": event.name, "UTC": event.time}
                      for event in scheduled.events if isinstance(event, cx.Timer)]))
display(pd.DataFrame([summary("daily timed targets", scheduled)]))

monthly_declaration = cx.SessionSchedule("monthly-v1", "rebalance",
    start=date.fromisoformat(START), frequency="monthly", at="close")
print("A monthly declaration is also available; it has not been deployed:", monthly_declaration)
''')
md('''
On paper, schedule claims are durable **per owner**. Failed delivery stays
uncertain; healthy owners can continue on the same logical schedule. `skip`
avoids replaying missed entries after downtime. Use `catch_up` only for decisions
that remain appropriate when late, such as an explicitly designed overdue exit.
A completed exit timer proves delivery, not a fill or a flat position.
''')

md('''
<a id="math"></a>
## 14. Call convexity.math from the strategy

The mathematical library and trading runtime compose through ordinary Python.
Here we estimate covariance from observed minute returns and solve a long-only
minimum-variance problem. The target function receives only current context;
the optimizer returns weights, which we translate into whole shares under a
small capital budget. A covariance ridge stabilizes this tiny demonstration.

This example uses 60 observed bars, not 60 trading days. Estimation horizon,
conditioning, turnover, integer sizing and out-of-sample behavior still need
research. The library cannot turn a convenient optimizer into a validated model.
''')
code('''
from convexity.math.optimization import QuadraticProgram, solve

def minimum_variance_targets(context):
    window = 60
    if any(len(context.history[s]) < window for s in SYMBOLS):
        return None
    closes = np.array([[float(b.close) for b in context.history[s][-window:]] for s in SYMBOLS])
    returns = closes[:, 1:] / closes[:, :-1] - 1
    covariance = np.atleast_2d(np.cov(returns)) + np.eye(len(SYMBOLS)) * 1e-6
    answer = solve(QuadraticProgram(
        linear=np.zeros(len(SYMBOLS)), quadratic=2 * covariance,
        equality=np.ones((1, len(SYMBOLS))), target=[1], lower_bounds=0, upper_bounds=1,
    ))
    if answer.solution is None:
        raise RuntimeError("Optimizer produced no solution")
    capital = min(context.equity, D("2000"))
    return {symbol: int(capital * D(str(max(0.0, float(weight)))) / context.marks[symbol])
            for symbol, weight in zip(SYMBOLS, answer.solution)}

optimizer_owner = "demo-optimizer"
class OptimizedTargets(TargetStrategy):
    metadata = cx.StrategyMetadata(optimizer_owner)
    contract = cx.StrategyContract(
        instruments=SYMBOLS, warmup_sessions=WARMUP_SESSIONS,
        subscriptions=(cx.BarSubscription(SYMBOLS, bar_size=BAR_SIZE), daily_schedule),
    )
    def __init__(self):
        super().__init__(minimum_variance_targets)

optimized = await cx.Research(START, END).run_async(
    OptimizedTargets, provider=cx.IBKRHistoricalData(history_config), cache=CACHE_DIR,
    execution=execution, initial_cash=INITIAL_CASH,
    limits=risk_for(optimizer_owner), output=RUN_DIR / "optimizer",
    metadata={"model": "long-only minimum variance", "window_bars": 60,
              "covariance_ridge": "1e-6", "capital_budget": "2000"},
)
display(pd.DataFrame([summary("math-driven targets", optimized)]))
''')

md('''
### Multiple strategies share one account

The keys in `strategies` are durable **owner identities**. Each owner receives
its own orders, fills and positions; account cash/equity remain shared. Give each
owner a budget rather than independently allocating all account equity twice.
The account-level capacity check includes both owners' working orders.

Here two variants run simultaneously against the same real observations. This
is different from the separate parameter experiments above. The reported equity
curve belongs to the combined account; the final attributed positions below do
not by themselves constitute per-strategy return series. The paper API uses the
same owner mapping, with matching keys in `budgets` and `strategy_versions`.
''')
code('''
multi = replay(
    strategy_factories={"fast": lambda: DemoTrend(SYMBOLS, 10, SHARES),
                        "slow": lambda: DemoTrend(SYMBOLS, 60, SHARES)},
    risk=cx.RiskLimits(gross_leverage=D("1"), max_order_notional=D("2000"),
                      strategy_gross_limits={"fast": D("3000"), "slow": D("3000")}),
)
display(pd.DataFrame([summary("combined fast + slow account", multi)]))
display(pd.DataFrame([{"owner": owner, "symbol": symbol, "shares": position.quantity}
                     for owner, positions in multi.positions.items() for symbol, position in positions.items()]))
print("Fills by owner:", dict(Counter(fill.strategy_id for fill in multi.fills)))
''')

md('''
<a id="state"></a>
## 15. Stateful strategies and restart semantics

For paper execution, persist **decision state** with JSON-compatible
`dump_state()` and `load_state(state)`. Reconstruct external models/connections in
`__init__`; do not checkpoint client handles or copy broker positions into your
own competing ledger. The built-in target/option strategies have their own hooks.

`load_state(None)` is a fresh deployment. Checkpoints are restored before live
delivery. Startup is durably delivered once per account/owner, so it is not the
place to rely on initializing resources anew after every reconnect.

Callbacks are claimed before user code; checkpoints are saved before routing
commands. A crash in that interval leaves uncertainty. The system does not
pretend a checkpoint proves an order reached the broker. Reconcile actual effects
before resolving a claim. This is durable recovery, not a guarantee that arbitrary
external Python side effects execute exactly once.
''')
code('''
class StatefulCounter(cx.Strategy):
    """Checkpoint teaching example: observes bars and never emits orders."""
    def __init__(self):
        self.completed_decisions = 0

    def on_event(self, event, context):
        if isinstance(event, cx.Bars):
            self.completed_decisions += 1

    def dump_state(self):
        return {"version": 1, "completed_decisions": self.completed_decisions}

    def load_state(self, state):
        if state is None:
            return
        if (not isinstance(state, dict) or state.get("version") != 1
                or type(state.get("completed_decisions")) is not int
                or state["completed_decisions"] < 0):
            raise ValueError("Unsupported checkpoint")
        self.completed_decisions = state["completed_decisions"]

first = StatefulCounter()
first.completed_decisions = 7
checkpoint = json.loads(json.dumps(first.dump_state()))
restored = StatefulCounter()
restored.load_state(checkpoint)
assert restored.completed_decisions == 7
display(checkpoint)
''')

md('''
<a id="paper"></a>
## 16. Research and paper share an API, not identical market mechanics

| Topic | Research | Managed IBKR paper |
|---|---|---|
| Clock / observations | Replay of supplied completed bars | Current broker data and wall clock |
| Market command | Next eligible open with configured costs | Bounded limit using configured price collar; may stay unfilled |
| Stock fills | Simulator and modeled commission | IBKR execution; `FillEvent` waits for known commission |
| Balances / capacity | Explicit stock accounting and risk assumptions | Broker-authoritative account plus ownership/budget gates |
| Observation cadence | Explicit daily or one-minute bars | Current managed stock subscription is one-minute |
| State | Fresh strategy instances per research run | Checkpoints, durable intents, reconciliation across reconnects |
| Failure | Failed results/errors with diagnostics | Failed owner pauses while tracking and healthy owners continue |
| Shutdown | No implied closing trade | No implied cancellation or liquidation |

Paper account totals can include other assets. Strategy budgets are explicit,
and existing manual holdings remain unassigned. Configured trading symbols must
not conflict with unassigned holdings. Do not change owner names or discard the
deployment directory to make an ownership fault disappear.

`PaperTrading` owns data polling, freshness checks, callback delivery, timers and
bounded transport reconnects. Normal application code does not need a refresh
loop. Uncertain callbacks/orders and account/data/journal faults still need
attention; invalid observations are not replaced with stale/fabricated signals.

### Event reference

Both stock paths use `Bars`, `Timer`, `OrderUpdate`, `FillEvent`, and `Rejection`.
Paper additionally exposes `StockQuotes`, `AccountUpdate`, `ConnectionHealth`,
`OptionQuotes`, and `BrokerOrderUpdate`. Option leg executions remain broker
records, not fabricated stock `FillEvent`s. A quote event can contain unavailable
or stale data; the execution gate still checks actual quote quality and receipt age.
''')

md('''
<a id="paper-demo"></a>
## 17. Explicit paper demonstration — off by default

**Running All Cells with the defaults below sends no broker orders.**
To deliberately demonstrate paper execution, review the selected symbols and
budgets, confirm the paper login and market-data access, then change the flag.
This uses the same `DemoTrend` class and one-minute signal cadence as research.

The default 30-second observation period may contain no new completed bar, so
zero orders/fills is a legitimate outcome. Even a signal does not guarantee a
collared-limit fill. This is current paper execution, not replay of the old dates.

Keep `PAPER_PATH`, owner, strategy version and configuration stable when resuming.
The runner refuses an incompatible deployment manifest. A new output directory
does not authorize adopting or ignoring existing exposure. History retention
must cover your lookback; this small demo's 30-minute warmup covers its 20 bars.
''')
code('''
ENABLE_PAPER_TRADING = False
PAPER_DEMO_SECONDS = 30
PAPER_PATH = ROOT / "out" / "notebook-paper-demo"
paper = None
paper_task = None

def make_paper_runner():
    if BAR_SIZE != "1 min":
        raise ValueError("Match research and paper cadence; this example requires one-minute bars.")
    if LOOKBACK > 28:
        raise ValueError("Increase the paper history window explicitly before using this longer lookback.")
    return cx.PaperTrading(
        config=cx.PaperAccountConfig.from_env(), strategies=tuple(factories().values()),
        instruments={symbol: cx.Instrument(symbol, shortable=False) for symbol in SYMBOLS},
        subscriptions={symbol: cx.StockSubscription(history_seconds=1800, max_bars=10_000)
                       for symbol in SYMBOLS},
        path=PAPER_PATH, budgets={OWNER: D("6000")},
        max_order_notional=D("2000"), market_collar_bps=D("25"), allow_short=False,
        strategy_versions={OWNER: f"demo-trend-v1:{','.join(SYMBOLS)}:{LOOKBACK}:{SHARES}"},
    )

if ENABLE_PAPER_TRADING:
    paper = make_paper_runner()
    try:
        paper_task = await paper.start()
        await asyncio.wait_for(asyncio.shield(paper_task), timeout=PAPER_DEMO_SECONDS)
    except asyncio.TimeoutError:
        pass  # Planned demo duration; cleanup runs below.
    finally:
        await paper.stop()
    print(paper.status().format())
    display(pd.DataFrame(paper.schedule_status()))
else:
    print("Paper execution disabled. All earlier executions were simulated against real history.")
''', tags=('paper-opt-in',))
md('''
For an interactive notebook session, call `paper = make_paper_runner()` and
`paper_task = await paper.start()`, inspect `paper.status()` in later cells, and
always finish with `await paper.stop()`. Retain the task to observe terminal
errors. The timed demo above wraps shutdown in `finally`; closing/killing the
kernel cannot perform orderly cleanup.

`paper.status()` returns a detached snapshot of positions, working orders, recent
fills, strategy states and errors. `paper.schedule_status()` includes logical
schedule, owner and durable namespace. Inspect these, not just whether a task
exists. `fail_fast=True` is an opt-in alternative to default owner isolation.

### Explicit operator actions

These are **examples to execute deliberately in an active session**, not actions
performed by this notebook. Read current owned position and pending orders first.

```python
# Cancel the exact public order ID shown in context/status.
events = await paper.command(OWNER, cx.CancelOrder(public_order_id))

# Explicitly close a verified long position after handling working orders.
events = await paper.command(OWNER, cx.SubmitOrder(
    symbol, "SELL", owned_quantity, client_order_id="operator-close-unique-1"))
```

Read returned events for rejections, and then verify fills and remaining exposure.
Never blindly rerun a close with a new ID because the previous outcome is unclear.
Paused owners may cancel or reduce verified exposure; operator commands do not
silently resume the strategy. `stop()` disconnects—it does not flatten positions.

### Recovery APIs

`StrategyState(path_to_orders_sqlite, account).unresolved(owner)` identifies
interrupted callbacks. `resolve(owner, identity, detail=...)` records a reconciled
outcome without replaying user code. Close the store when done. An uncertain timer
also needs `DurableSchedule.resolve(timer, completed=..., detail=...)` using its
actual per-owner namespace from `schedule_status()`. These resolution methods
change durable state; the notebook does not call them automatically.
''')

md('''
<a id="next"></a>
## 18. Options, limitations, and troubleshooting

### The options boundary

Paper execution supports standard USD **debit verticals**, using qualified
`OptionIdentity` legs, `VerticalOrder`, `VerticalRule`, `ScheduledVertical`, and
`ZeroDTE`. `OptionQuoteRequest` supplies explicit subscriptions to `PaperTrading`;
contract expiry, strike, right, trading class and multiplier matter. Chain discovery
does not prove every strike/expiry combination is a valid contract.

Defined entry/exit schedules, fresh leg quotes and actual leg executions determine
the outcome. Entry/exit attempts are bounded; an unfilled exit remains exposure.
There is no automatic conversion into naked-leg orders. Options backtesting is
not implemented, so this notebook does not pretend the stock simulator validates
an option strategy. See [strategy rules](../docs/strategies.md#scheduled-option-rules)
and [broker APIs](../docs/ibkr.md) for the exact supported composition.

### Troubleshooting the normal workflow

| Symptom | Inspect / act |
|---|---|
| Wrong Python or missing package | Select the quant kernel; install declared extras in that environment |
| Account/config validation error | Check your local TOML and verified paper login; do not print secrets into shared outputs |
| Connection/client-ID failure | Check Gateway/TWS, socket settings and another process using that ID |
| Missing bars or entitlement error | Confirm symbol/listing period/access; correct the request instead of filling gaps |
| Future/unfinished session | Choose fully completed historical session dates |
| Cache validation error | Preserve evidence; explicitly refresh to reacquire a validated snapshot |
| No fills | Inspect warmup, signals, rejections, pending orders, model assumptions and dataset end |
| Rejected order | Read `Rejection.reason`, risk caps, pending exposure and short permissions |
| Intrabar timer rejected | Use minute data for intraday scheduling; daily OHLC cannot support that timing |
| Strategy exception | Inspect `BacktestFailure.result` or paused-owner errors; fix the strategy before interpreting performance |
| Paper owner paused / unknown effect | Reconcile journal and broker; do not delete state or invent a new order ID to retry |
| Configuration differs on restart | Restore/migrate the deployment deliberately; keep exposure ownership intact |
| Stopping did not close positions | Shutdown never implies liquidation; explicit exit intent and verified fills are required |

### What the tool does not prove

The simulator does not reconstruct tick paths, queue position, exchange latency,
full broker margin/settlement, dividend cash flows, assignment/exercise or automatic
liquidation. Spread/slippage, volume participation and borrow are declared models.
Fixed-universe and adjusted-data biases remain relevant. The demo's small sample
and arbitrary rule establish software behavior, not expected profitability.
''')

md('''
## 19. Your own idea: a practical research brief

Before coding the next rule, write down:

1. **Hypothesis:** what observable relationship is being tested?
2. **Universe and dates:** what was actually available, and what is held out?
3. **Observation clock:** minute or daily, completed bars only, decision schedule.
4. **Signal and sizing:** inputs, exact lookback, integer targets, position limits.
5. **Exits:** signal exit, timer exit, what happens to outstanding orders and end-of-test holdings.
6. **Execution assumptions:** spread, slippage, commissions, borrow, participation, delays.
7. **Evaluation:** baseline, turnover/cost sensitivity, chronological validation, failure criteria.
8. **Reproducibility:** immutable data identity, source/version, parameters, outputs and notes.

### Exercises

- Change the signal while preserving the Strategy interface; explain each input's availability time.
- Increase the lookback, then check both research and paper warmup/retention.
- Compare small-share results to a deliberately budgeted target-sizing rule.
- Add a session-close exit timer, then inspect whether orders actually filled before the dataset ended.
- Try a limit-order rule and compare `touch` with `penetration` on the same real snapshot.
- Expand the dates before making performance claims; keep the holdout genuinely unused for selection.
- Move the class into your own `.py` file and import it into `Research` and `PaperTrading`.

**Presenter closing:** “We can trace every result back to real observations,
declared execution assumptions and a strategy implementation. The next iteration
is changing the idea and testing it against those same controls.”

### Where to go next

- [Research API and data semantics](../docs/backtesting.md)
- [Strategy, schedules and recovery contracts](../docs/strategies.md)
- [Runnable research/paper script](strategy_workflow.py)
- [Mathematical strategy example](backtest_math.py)
- [Connection settings](../docs/ibkr.md#setup)
- [Operational handoff](../docs/handoff.md)
- [Math catalog and domain notebooks](../docs/math/README.md)

The last cell writes a small demo summary beside the full artifacts. It contains
no account identifier and records whether paper execution was enabled.
''')
code('''
demo_summary = {
    "data_source": result.config["data_source"],
    "price_basis": result.config["price_basis"],
    "data_identity": result.config["data_identity"],
    "symbols": SYMBOLS, "start": START, "end": END, "bar_size": BAR_SIZE,
    "baseline": summary("baseline", result),
    "offline_artifacts_identical": True,
    "paper_execution_enabled": ENABLE_PAPER_TRADING,
    "limitations": ["No dividend cash flows", "No full corporate-action accounting",
                    "Fixed universe", "No options backtest", "No implicit liquidation"],
}
(RUN_DIR / "demo-summary.json").write_text(json.dumps(demo_summary, indent=2) + "\\n")
print("Demo complete. Research artifacts:", RUN_DIR.relative_to(ROOT))
display(pd.DataFrame([summary("baseline", result), summary("cost stress", stressed),
                      summary("same-size hold", hold), summary("holdout", holdout),
                      summary("scheduled", scheduled), summary("optimizer", optimized)]).set_index("run"))
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (quant)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
    "convexity": {"purpose": "comprehensive real-data demo", "paper_orders_default": False},
})
for index, cell in enumerate(notebook.cells):
    cell.id = f"demo-{index:02d}-{hashlib.sha256(cell.source.encode()).hexdigest()[:8]}"
    if cell.cell_type == "code":
        compile(cell.source, f"demo cell {index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
destination = ROOT / "examples" / "convexity_demo.ipynb"
nbf.validate(notebook)
nbf.write(notebook, destination)
print(f"Wrote {destination.relative_to(ROOT)}: {len(cells)} cells")
