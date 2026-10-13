# Convexity handoff

This is the operations and recovery guide. For the research API and a first
strategy, start with [event-driven research](backtesting.md). For standalone
DataFrames, see [historical data](data.md); for model APIs, see
[Convexity math](math/README.md).

Use Python 3.14 in the `quant` conda environment. Install the broker/calendar
extras and preserve the order journal, saved plans and schedules:

```sh
conda run -n quant python -m pip install -e '.[dev,ibkr,calendars]'
conda run -n quant python -m pytest -q
```

Historical research uses read-only connection settings (`IBKR_HOST`, `IBKR_PORT`,
`IBKR_CLIENT_ID`) and a persistent SQLite cache. Paper deployments additionally
set `IBKR_ACCOUNT` and `IBKR_PAPER_CONFIRMED=true` or construct
`PaperAccountConfig` in Python. Customize strategy parameters in Python or the
example's CLI; the combined paper smoke example keeps its defaults in `prepare`.

## Your own strategies

Use `convexity.Research` and `convexity.PaperTrading` with the same strategy
factories and matching observation cadence. Research takes dates; strategies
declare their instruments and subscriptions. Research fetches and validates
history, caches it, warms up the strategy, and runs the backtest.
`PaperTrading` owns live data, broker callbacks, schedules,
checkpoints and bounded reconnects. See the [Python API](strategies.md) and
[`examples/strategy_workflow.py`](../examples/strategy_workflow.py); no custom
polling loop is needed. The three-strategy acceptance runner below remains a
specialized historical demonstration and recovery tool.

```sh
conda run -n quant python examples/strategy_workflow.py --symbols IWM \
  --start 2025-01-02 --end 2025-01-03 --warmup-sessions 1 \
  --output out/custom-backtest
# Explicitly submits actual paper orders if the strategy signal is active:
conda run -n quant python examples/strategy_workflow.py --mode paper --symbol IWM \
  --output out/custom-paper --seconds 300
```

Keep the same deployment directory and strategy version when resuming. Unknown
broker effects or interrupted callbacks stay blocked for explicit reconciliation;
normal transport reconnects are automatic. Shutdown does not flatten positions.

## Research and math

```sh
conda run -n quant python examples/backtest_stocks.py \
  --symbols SPY QQQ --start 2025-01-02 --end 2025-02-07 --output out/backtest-baseline
conda run -n quant python examples/backtest_stocks.py \
  --symbols SPY QQQ --start 2025-01-02 --end 2025-02-07 \
  --execution conservative --output out/backtest-costly
conda run -n quant python examples/backtest_math.py \
  --symbols SPY QQQ --start 2025-01-02 --end 2025-02-07
```

These commands retrieve real daily observations. They do not substitute fixture
prices. Covered requests replay offline, and `--refresh` reacquires requested
history. Read-only acquisition submits no broker orders;
backtest fills are modeled. IBKR history uses split-adjusted prices and omits
dividend cash flows; results describe price-return research in adjusted units.

The result directory saves bars, sessions, provenance/hashes, parameters, modeled
costs, orders/fills, positions, equity and diagnostics. Inspect rejections and open
positions as well as returns. See [research API and limitations](backtesting.md).
The optimizer example calls `convexity.math` using only observed returns. Daily
research and one-minute paper bars are different signal cadences; use matching
bars/aggregation and schedules before deploying a researched strategy.

## Actual paper session

These commands submit actual paper orders. The default combined run is a
10-minute session; `--session` instead exits 30 minutes before the broker's
current session close, including early closes. Contract discovery must succeed
with live two-sided quotes. Existing manual holdings remain unassigned.

```sh
conda run --no-capture-output -n quant python scripts/ibkr_probe.py --redact
conda run --no-capture-output -n quant python scripts/ibkr_status.py
conda run --no-capture-output -n quant python examples/paper_mvp.py --run-id my-paper-session --output out/my-paper-session --seconds 600 --restart
```

Use a new run ID/output directory for a deliberately new session. Reuse both
when restarting an interrupted session: the saved plan fixes contract identities,
parameters and times; the journal fixes order ownership and attempt counts.
`--restart` disconnects and reconstructs the runtime once while positions are
held, then continues to the saved cutoff. Timer schedules persist across that
restart. Replaying a completed entry does not resubmit its broker intent.

The runner exports `plan.json`, `result.json`, SQLite schedules and terminal JSON
containing orders, leg executions, decisions, faults and schedule status. A
successful result requires actual long/short stock openings, a weekly-spread
round trip, SPY and QQQ same-day round trips, requested restart, no unresolved
orders, and flat owned positions reconciled with IBKR. A skip, rejection, missing
fill or unclosed position produces an incomplete result and nonzero exit.

The regular monthly calendar is independently persisted from the labeled
one-shot test rebalance. Production timers use first-session open plus five
minutes. Full exchange calendars handle holidays, DST and early closes; the
broker's liquid session determines today's paper cutoff. Old plan files retain
their original stored schedule.

## Recovery and explicit exits

Unknown submission outcomes block dependent orders. Reconnect using the same
journal and inspect `ibkr_status.py`; never invent fills or clear an unresolved
intent merely to retry. An interrupted durable callback remains uncertain until
its broker/journal effects are reconciled and `DurableSchedule.resolve` records
that decision. A successfully delivered exit timer does not imply a fill.

For a saved session, explicitly dispatch exits without new entries:

```sh
conda run --no-capture-output -n quant python examples/paper_mvp.py --run-id my-paper-session --output out/my-paper-session --close-only
```

Normal exit pricing requires fresh quotes. For a data outage, the separate
operator command closes only already-owned, balanced vertical legs at an
explicit minimum credit:

```sh
conda run --no-capture-output -n quant python scripts/ibkr_vertical_close.py --entry-id EXISTING_ENTRY_ID --id UNIQUE_CLOSE_ID --minimum-credit 0.25
```

That number is a caller-chosen limit, not a recommendation or fresh quote. The
command never opens naked legs or blindly replaces an order. Broker price
precautions can reject it; inspect the terminal outcome before selecting any
replacement. Shutdown itself never liquidates positions. Quote error 10197 means
IBKR market-data access is held by a competing session; release that session
before attempting new strategy entries.

[Actual execution evidence](paper-evidence.md) and [completed task acceptance](tasks.md)
record the ten-minute combined session: ten filled orders, sixteen executions,
a successful restart, no duplicates and all test exposures flat. The final full
suite passed 2,216 tests; expanded MVP type checking and independent reviews passed. Local
raw logs, journal and exports are in `out/` and are gitignored.
