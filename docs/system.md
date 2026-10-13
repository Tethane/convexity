# Convexity MVP

Build one Python runtime running three strategies on one Interactive Brokers paper
account, plus a configurable stock backtester. Completion means observable paper
trades and a working research-to-trading loop, not demonstrated profitability.

Use the existing `convexity.math` library inside strategy callbacks. Both the
research loop and actual paper execution are implemented. The configured
acceptance session is complete, with saved evidence in [the tasks](tasks.md).

The stock research implementation is now runnable: [backtesting API and examples](backtesting.md).

The strategy-facing entry points are `convexity.Research` and
`convexity.PaperTrading`. Both accept the same stock strategy factories;
`Research(start, end).run(MyStrategy)` retrieves and validates historical prices
for instruments and subscriptions declared by the strategy
and dates, caches the observations, warms up, replays and exports its inputs/results;
`PaperTrading` owns observations, warmup, scheduling, checkpoints, broker tracking
and reconnects. See the [Python API](strategies.md) and
[same-strategy example](../examples/strategy_workflow.py). The lower-level runtime
and broker adapters remain available for specialized integrations.

## Done means

- A monthly long-short stock portfolio rebalances on schedule.
- An options spread strategy opens and closes paper positions.
- A 0-DTE SPY/QQQ strategy trades intraday and exits before its session cutoff.
- All three run together in one process on the paper account. Orders, fills,
  positions, decisions, and failures are visible in the terminal and saved locally.
  Executions can be verified in IBKR.
- The same stock strategy runs through historical replay and a simulated exchange.
  Execution assumptions are configurable and saved with results.
- Schedules respect holidays, early closes, and daylight saving changes. Restarting
  or reconnecting does not blindly repeat orders.

Options trading is included. Options backtesting is excluded.

## Smallest system

| Piece | Responsibility |
| --- | --- |
| Runtime | One event loop, serial strategy callbacks, schedules, order tracking, basic limits |
| Strategy | Receives an event and current state; returns submit/cancel requests |
| IBKR adapter | One paper account; quotes/bars, stock/options contracts, orders, fills, account snapshots |
| Simulated exchange | Stock orders and fills from historical bars with configurable execution assumptions |
| Backtest runner | Advances the clock, calls the same runtime, saves orders/fills/equity |
| Local journal | SQLite records for order intent, fills, strategy ownership, schedule runs, recovery |

Use strategy contracts and explicit deployment settings for paper trading,
backtesting, and status. Select a maintained IBKR client compatible with the project's Python
version. Use whichever of TWS or IB Gateway is available for the paper account.
Prove the connection before building around it.

## Runtime essentials

- Only bars, top-of-book quotes, timers, order updates, fills, and account snapshots.
- Configure strategies at startup. Keep callbacks short; serialize state updates
  and order acceptance. No dynamic plugin or worker framework.
- Resolve broker contract identities, including option expiry, strike, right,
  multiplier, and spread legs. Preserve exact order and monetary precision.
- Track partial fills, pending cancels, and unknown submission outcomes. Apply each
  execution once. Attribute orders/positions to strategies; keep manual activity
  unassigned.
- IBKR is authoritative for paper balances, positions, and margin. Reconcile local
  records against it; do not build a derivatives margin engine.
- Check the expected paper account, quote freshness, per-strategy budgets,
  order/exposure limits, and pending commitments before submission. Strategies
  cannot spend the same capacity twice. Surface broker rejections.
- Persist intent before submission. Reconcile uncertain outcomes before retrying.
  Pause new entries while data or broker state is uncertain.
- Use the same exchange-session schedules with real and simulated clocks. Persist
  completed schedule keys. Skip missed entries/rebalances on restart; reconcile
  and handle overdue exits as soon as possible.
- Failed strategies stop producing entries while order tracking continues. Provide
  explicit cancel/close actions. Shutdown does not implicitly liquidate positions.

## Three strategies

Each needs one documented deterministic rule set: inputs, signal, sizing, entry,
exit, schedule, and limits. The configurable demonstration rules are specified in [Building strategies](strategies.md),
with parameters saved for each paper session. They demonstrate execution;
completion does not require profitable trading or model training.

| Strategy | Initial boundary |
| --- | --- |
| Monthly long-short stocks | Fixed universe, one ranking rule, target weights, gross/net limits, monthly rebalance time, unavailable-short behavior |
| Options spread | One defined-risk vertical template, underlying, expiry/strike selection, entry/exit rules, contract limit; broker combo submission without automatic legging fallback |
| 0-DTE SPY/QQQ | One bar/quote-based signal, same-day contract selection, bounded exposure, entry window, max trades/day, forced exit before session close |

The vertical template and bounded-exposure 0-DTE structure are proposed MVP choices.
Write exact rules before enabling each strategy. Missing contracts, permissions,
or usable quotes cause explicit skips/rejections. Unfilled exits remain visible
and use a bounded retry policy; a timer does not prove a position is flat.
Unexpected assignment/exercise or position differences pause the affected strategy
for reconciliation.

## Stock research loop

Accept stocks and inclusive research dates, acquire actual historical bars from
IBKR, validate complete exchange-session coverage, and cache immutable snapshots.
Never replace unavailable observations with generated or interpolated prices.
Keep local-data replay available. Record timestamps, requests, contract IDs, source
rows and hashes with results. Warmup uses only prior sessions without trading.
IBKR TRADES prices are split-adjusted; declare price-return accounting with no
dividend cash flows rather than claim full corporate-action accounting. A fixed
universe is supported with its survivorship limitation disclosed.

Use the same stock strategy, schedules, order tracking, and fill application in
paper trading and replay. Simulation supplies the clock, events, fills, and a simple
stock cash/position ledger; it does not reproduce IBKR's complete margin model.

Expose these settings directly:

- Market fills at the next eligible bar open; limit fills using touch or strict
  penetration of subsequent bars, with an explicit gap-price rule.
- Spread/slippage in basis points; commission per share with a minimum fee.
- Submission/cancel delay in whole bars; explicit ordering of cancels and fills.
- Bar-volume participation caps, partial fills, shared liquidity across orders.
- Initial cash, gross leverage limit, short availability, flat annual borrow cost.

An order based on a completed bar cannot fill inside that bar. Strategies cannot
read future bars. Equal-time events have a fixed documented order. Identical inputs
and configuration produce identical results. Reject unsupported instruments/order
types; no L2 queue, market-impact, or options simulator is required.
Stock stops, amendments, bracket/OCO orders, and dynamic subscription lifecycles
are described in [the order guide](order-lifecycle.md).

Save resolved configuration, data identity, orders, fills, positions, fees, equity,
and a small return/drawdown summary. Mark failures incomplete and show remaining
orders/open positions. Never silently liquidate at the end of a backtest.

## Outside this MVP

Multiple brokers/accounts/currencies, crypto/futures, L2/L3 books, queue simulation,
options backtests, distributed services, custom margin engines, hot loading,
general event/schema frameworks, dashboards, parameter-search platforms, and live
money deployment. None is a prerequisite for [the tasks](tasks.md).
