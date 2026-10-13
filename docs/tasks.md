# MVP tasks

All 26 tasks are complete for the configured MVP acceptance session on
2026-10-09. Scope: [Convexity MVP](system.md). Evidence uses actual paper
executions, not mocked fills. Existing mathematical-library work remains separate.

The public Python interfaces are `Strategy`, `Context`, events and commands,
`Runtime` / `PaperRuntime`, `PaperTrading`, `Research`, `IBKRHistoricalData`,
`SimulatedExchange`, `Backtest`,
`SessionSchedule`, `DurableSchedule`,
`OrderJournal`, and the paper broker adapters. [Building strategies](strategies.md),
[backtesting](backtesting.md), [broker setup](ibkr.md), [handoff](handoff.md).

## Paper account and execution

| Done | Task | Completion evidence |
| --- | --- | --- |
| [x] | 01. Read the paper account | Verified configured account, balances, positions, orders and clean disconnect. Redacted `out/paper-account-current.json`. |
| [x] | 02. Resolve and quote one stock | Actual SPY identity and live two-sided quotes; missing/stale/non-live cases explicit. `out/spy-quote.json`. |
| [x] | 03. Submit one paper stock order | Durable intents, actual paper acknowledgment, rejection and executions. `out/paper-regular-buy.jsonl`. |
| [x] | 04. Observe fills and cancellation | SPY buy/sell, resting close canceled before replacement, IWM short/cover. [Execution evidence](paper-evidence.md). |
| [x] | 05. Persist order intent and fills | SQLite intent before transport; stable IDs, executions, commissions, audit events, leg attribution and replay reconciliation. `out/orders.sqlite`, journal/broker tests. |

## Runtime and monthly stocks

| Done | Task | Completion evidence |
| --- | --- | --- |
| [x] | 06. Run strategy callbacks through the runtime | Serial callbacks on bars, timers, orders and executions; shared Strategy/Context contract for research and paper; actual option entry via PaperRuntime. |
| [x] | 07. Maintain trading state | Simulated accounting and owned positions; paper balances authoritative, stock/option ownership derived from journal, manual assets remain separate. |
| [x] | 08. Gate strategy orders | Account capacity, owner budgets, pending commitments, quote age, bounded stock limits, debit-vertical width exposure and unresolved-order gates; behavioral regression tests. |
| [x] | 09. Fire durable session schedules | Full exchange calendar plus actual persisted delivery: 7 entry/recheck timers and cutoff completed; 10 missed monthly timers skipped, 2 future timers pending, no uncertain runs. `out/paper-mvp-b/schedules.sqlite`; holiday/DST/early-close tests. |
| [x] | 10. Reconnect and restart | Combined session disconnected and reconstructed with all positions held; journal/broker reconciliation, no duplicate intents. `out/paper-mvp-b/result.json`: restarted true, generation 1. |
| [x] | 11. Specify the stock rule | Configurable `StockRule` and `MonthlyLongShort`, exact ranking, tie/short policy, sizing, scheduled rebalance and explicit exit. Rules authorized for paper testing. |
| [x] | 12. Compute rebalance orders | Owned plus pending quantities; reductions before additions, close before reversing; planner tests and same public rule in research/paper. |
| [x] | 13. Run a paper rebalance | Actual runtime-ranked IWM short 3 at 279.36 and QQQ long 1 at 750.92; both closed by the scheduled exit. One-shot demonstration distinct from monthly calendar. |

## Stock research

| Done | Task | Completion evidence |
| --- | --- | --- |
| [x] | 14. Replay a stock dataset | Validated aligned OHLC panels and explicit price convention; no future history exposed. |
| [x] | 15. Fill simulated market orders | Next eligible bar open; event accounting tests verify positions/cash/equity and no signal-bar fill. |
| [x] | 16. Configure costs and shorting | Spread/slippage, commissions, volume, borrow, short availability, capacity and leverage assumptions recorded. |
| [x] | 17. Simulate limits, delays, partial fills | Touch/penetration, gaps, rounding, submission/cancel delays, partials and shared FIFO liquidity tested. |
| [x] | 18. Export one stock backtest | Same stock strategy: baseline versus higher costs, deterministic exports, orders/fills/positions/fees/equity/configuration. Math optimizer example uses observed returns. |

## Paper options

| Done | Task | Completion evidence |
| --- | --- | --- |
| [x] | 19. Discover and quote options | Actual SPY/QQQ chains and same-day option quotes; exact expiry/right/strike/trading-class/multiplier identities. `out/*option*quote.json`. |
| [x] | 20. Trade and track a vertical combo | Actual SPY 777/778 call vertical round trip, two leg executions per combo, no BAG double counting; broker and journal flat. `out/paper-spy-vertical.jsonl`. |
| [x] | 21. Specify and run the spread strategy | Saved weekly SPY October 16 778/779 call rule; actual scheduled entry debit .58 and exit credit .54, one combo, both legs flat. |
| [x] | 22. Specify the 0-DTE strategy | `VerticalRule`/`ZeroDTE`: same-day contracts, bounded liquid-debit signal, explicit window/cutoff, one trade/day per underlying, two bounded exits, restart-safe intent counts. [Exact rules](strategies.md). |
| [x] | 23. Run 0-DTE entries and exits | Actual scheduled SPY 778/779 and QQQ 751/752 same-day entries and exits, one combo each; journal and broker legs flat. `out/paper-mvp-b/result.json`. |

## Finish

| Done | Task | Completion evidence |
| --- | --- | --- |
| [x] | 24. Show terminal status | JSON live order/decision/error/schedule status, account status with manual/owned attribution, runtime research status. `out/paper-mvp-a.jsonl`, `out/paper-status-after-close.txt`. |
| [x] | 25. Run all three together | Completed configured 10-minute paper session, one process/account, all categories filled, one restart, no duplicate intents or acceptance faults; all test exposures flat. This is not a full market day. [Evidence](paper-evidence.md). |
| [x] | 26. Save the handoff recipe | [Handoff](handoff.md): one TOML, API composition, research/paper/status commands, restart and explicit close recovery. Final suite: 2,216 passed; expanded MVP type check: zero errors; independent code and execution-evidence reviews passed. |

Local evidence is gitignored. [Paper evidence](paper-evidence.md) records the
verified outcomes and failed attempts without account identifiers. Missing data,
rejected orders and unfilled exits remain faults, never fabricated successes.

Independent critical review covers hardcoded outcomes, causal simulation,
accounting, order identity, ownership, API clarity and recovery. Findings fixed
include future/replayed bars, terminal-state regressions, overfills, duplicate
commissions, execution timestamp decoding, command acknowledgment races, stale
quote reuse, combo-leg replay gaps and explicit closes during data outages.

Final validation: `out/mvp-final-tests.txt` (2,216 passed, 70 math/dependency
warnings), `out/review-types.txt` (zero errors/warnings across MVP packages and
tests), and an independent read-only journal/result/schedule audit. All ten
combined-session orders were filled, all sixteen executions and commissions
matched, and the seven original account holdings were preserved exactly.

## Strategy API audit and completion

The original acceptance session proved the configured examples, but left library
integration gaps. The subsequent API audit required and implemented:

- A managed `PaperTrading` runner owning observations, warmup, schedules,
  broker updates, shutdown and bounded transport reconnects.
- Root imports and a custom strategy example that runs through either `Backtest`
  or `PaperTrading`, without an application-owned event pump.
- Matching public stock order identities, cancel/rejection behavior and DAY/GTC
  lifetimes; durable comparison against the complete original submission intent.
- Durable strategy checkpoints and callback claims, including exactly-once
  delivery of previously unseen reconciled fills and late commissions.
- Explicit schedule failure records, per-owner delivery, recurring exchange
  calendars and year rollover. Failed strategies remain paused while healthy
  owners and broker tracking continue.
- Data continuity checks, decision/quote freshness rechecked at broker transmission,
  and bounded observation/history buffers with durable execution records.

The managed runner was also connected to the configured IBKR paper session with
a strategy returning no commands: 30 real SPY warmup bars, 17 quote and 17 account
callbacks, no errors or owned orders, and clean shutdown. This was a connection
and data-path check; the actual order-execution evidence above is from the
original acceptance session.

A second real-session probe deliberately failed one strategy's startup callback.
It stayed paused while the healthy strategy received quotes, account updates and
its timer; the healthy timer completed and the paused owner's timer stayed
unclaimed. Neither strategy submitted an order. Both probes used separate
temporary journals and left the acceptance deployment intact.

The built wheel was extracted outside the repository and imported in isolated
Python mode. All root exports resolved, and a custom stock strategy completed a
backtest with one simulated fill. The documented example separately completed
with three fills. Type checking across the runtime, brokers, backtester,
strategies, trading, public facade, examples and their tests reported zero errors
and zero warnings.

Final API-completion validation: **2,278 tests passed**, with 70 existing
math/dependency warnings (`out/api-final-tests.txt`); the targeted type check
reported zero errors/warnings (`out/api-final-types.txt`). Independent review
reran 258 runtime/backtest/broker/strategy/trading/example tests after the final
failure-isolation changes, exercised paused-owner restart separately, and found
no remaining blockers. Package and real-session observation/isolation evidence
is saved in `out/api-package.txt`, `out/api-live-observations.txt`, and
`out/api-live-isolation.txt`. No new broker orders were placed during this API
completion work. Options backtesting remains excluded; ambiguous broker effects
and interrupted callbacks still require explicit reconciliation.

## Real-data research workflow

The preceding API completion did not include automatic date-range historical
acquisition; its default research examples still used synthetic fixtures. That
gap is now addressed by `Research`: explicit symbols/dates and strategy factories
drive real IBKR acquisition, exact exchange-session validation, immutable caching,
prior-session warmup, simulated execution, and export of the input and results.
The primary research examples require stocks and dates and never fall back to
generated prices. `Backtest` remains available for explicit local-data replay.

Daily and one-minute bars are supported. Shared `SessionSchedule` declarations
resolve full months correctly before filtering the research interval. Warmup
never invokes trading callbacks. Missing sessions or bars, wrong resolution,
conflicting observations, corrupted caches and unavailable data fail explicitly.
Defaults include modeled spread/slippage, commissions and borrow costs. Strategy
failures raise with saved partial results; rejections and open orders remain
visible. The shared research/paper example defaults to one-minute observations
and rejects incompatible daily paper mode.

Read-only real-data acceptance runs, with no broker orders:

| Workflow | Requested history | Observed outcome |
| --- | --- | --- |
| Shared strategy, daily | SPY/QQQ, 2025-01-02 through 2025-02-07, five prior warmup sessions | 14 simulated fills, zero rejections, two orders still open at the end; final modeled equity 99,954.67 from 100,000 |
| Shared strategy, minute | SPY/QQQ, 2025-01-02 through 2025-01-03, one prior warmup session | 531 simulated fills, zero rejections/open orders; final modeled equity 99,418.67 |
| Monthly long/short | Same daily period and actual exchange monthly timers | Six simulated fills, zero open orders; final modeled equity 100,145.104608 |
| Mathematical optimizer | Same daily period with twenty prior warmup sessions | Two simulated fills; final modeled equity 101,779.23 |

These are integration outcomes, not profitability evidence. Artifacts are in
`out/research-acceptance-final`, `out/research-minute-acceptance`,
`out/research-monthly-acceptance`, and `out/research-math-acceptance`. A second run
with broker client creation deliberately forbidden produced byte-identical
`result.json`, bars, sessions, provenance, fills, equity and cash-flow files in
`out/research-offline-acceptance-final`. Independent review separately replayed
the actual snapshot with network disabled and matched every order/fill/metric.

This is explicitly split-adjusted **price-return** research using real broker
observations. Dividend cash flows, full point-in-time corporate-action accounting,
historical universe reconstruction and options backtesting remain unsupported.
The [research guide](backtesting.md) explains those limits and reproducible usage.

Research verification: the full repository suite passed **2,315 tests** with 70
existing math/dependency warnings. After the final canonical-JSON export fix,
**169 focused tests** passed, including a new regression comparing every exported
artifact across a cache roundtrip. Expanded type checking reported zero errors
and warnings. Independent review passed and separately reproduced the real-data
offline replay. A built wheel was imported outside the repository and ran a
custom strategy through `Research` against actual cached observations with
network access disabled. Logs: `out/research-final-tests.txt`,
`out/research-final-focused.txt`, `out/research-final-types.txt`,
`out/research-offline-replay.txt`, and `out/research-package.txt`.
