# Paper execution evidence — 2026-10-09

## Combined acceptance session completed

`mvp-20261009b` ran from 13:59 to 14:09 Eastern on the configured IBKR paper
account. This was a complete, explicitly configured ten-minute acceptance
session, not a full market day. One process ran the stock, weekly-spread and
SPY/QQQ 0-DTE strategies, deliberately disconnected/reconstructed once with
positions held, then closed all positions through scheduled runtime callbacks.

| Strategy | Quantity | Actual opening fill | Actual closing fill |
| --- | --- | --- | --- |
| Monthly ranked stocks: IWM short | 3 shares | Sell 279.36 | Buy 279.29 |
| Monthly ranked stocks: QQQ long | 1 share | Buy 750.92 | Sell 750.94 |
| SPY October 16 778/779 call vertical | 1 combo | Debit .58 | Credit .54 |
| SPY October 9 778/779 call vertical | 1 combo | Debit .47 | Credit .46 |
| QQQ October 9 751/752 call vertical | 1 combo | Debit .34 | Credit .33 |

Ten unique orders, all filled; sixteen actual executions with commissions.
No duplicate entry after restart, no runtime errors, no paused strategies and
no acceptance faults. Broker/journal reconciliation confirms zero test-owned
positions. A separate final account read confirmed all seven original nonzero
holdings unchanged and no open orders. A completed callback alone never counted
as a fill.

Regular monthly schedules were persisted separately: ten missed historical
rebalance dates skipped, November/December pending. Seven one-shot entry/recheck
timers and the cutoff timer completed. The stock demonstration was explicitly
labeled; it does not claim a trade on an actual monthly boundary.

Local evidence: `out/paper-mvp-b/plan.json`, `result.json`, `schedules.sqlite`,
`out/paper-mvp-b.jsonl`, `out/paper-account-final.json`, and
`out/paper-final-reconciliation.json`. The result includes broker positions,
order IDs, leg executions, exact limits, decisions, faults and schedule state.
All exports are local/gitignored; account identifiers are redacted.

## Earlier smoke tests and recovered failures


These are actual IBKR paper executions recorded in `out/orders.sqlite`, not
simulator fills. Account identifiers are redacted in exported logs. All quantities
below are one share or one vertical; option premiums are per underlying share,
with the standard 100 multiplier. Commissions are retained separately.

| Owner / order prefix | Actual execution | Final state |
| --- | --- | --- |
| `paper-smoke` / `smoke-20261009-regular` | SPY buy 778.18; sell 778.06 | Flat; original manual SPY holding preserved |
| `short-smoke` / `smoke-20261009-iwm` | IWM short 278.95; cover 278.98 | Broker and owned position zero |
| `spread-smoke` / `spread-20261009` | SPY October 9 777/778 call vertical: buy debit .68, sell credit .65 | Both broker and owned legs zero |
| `zero-dte-mvp-20261009a` | Runtime SPY October 9 778/779 call vertical: buy debit .35, explicit close credit .37 | Both broker and owned legs zero |

The first SPY stock close rested and was canceled before a bounded replacement.
The last spread's first explicit close was canceled by IBKR's price precaution;
only after that terminal cancellation was a new .25 minimum-credit order sent.
It filled at .37. These failures and replacements remain in the journal.

Local raw evidence:

- `out/paper-regular-buy.jsonl`, `out/paper-regular-sell.jsonl`, `out/paper-regular-sell2.jsonl`
- `out/paper-iwm-short.jsonl`, `out/paper-spy-vertical.jsonl`
- `out/paper-mvp-a/plan.json`, `out/paper-mvp-a/schedules.sqlite`
- `out/paper-mvp-a-explicit-close.jsonl`, `out/paper-mvp-a-explicit-close2.jsonl`

The first combined session is **incomplete**: only its SPY 0-DTE entry traded.
It exposed an acknowledgment race between serial command submissions and stale
snapshots during sequential quote retrieval. Those issues have regression tests
and fixes. The subsequent combined session above met the independent reviewer’s criteria:
actual long/short stock fills, a weekly-spread round trip, both SPY and QQQ
0-DTE round trips, runtime restart and broker-verified flatness.

Subsequent quote requests returned IBKR 10197, “No market data during competing
live session.” Saved diagnostic: `out/spy-quote-retry.json`. No stale or invented
quote was substituted. An explicit owned-position close was used during that
outage, with a caller-specified minimum credit and the normal ownership,
contract-identity, pending-order, and uncertainty checks.

The user released the competing session before run B; live quotes resumed.
