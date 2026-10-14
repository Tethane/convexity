# Explicit corporate actions in historical replay

`Backtest(..., corporate_actions=(...))` and `Research.run(...,
corporate_actions=(...))` apply sourced events to the existing runtime account.
They do not submit fictitious trades or create an alternate portfolio simulator.
Dynamic-universe events use stable security IDs in `symbol` and `target_symbol`;
fixed-instrument replays use their declared symbols.

```python
from datetime import datetime, timezone
from convexity import CorporateAction

split = CorporateAction(
    action_id="verified-event-123",
    symbol="security-id",
    kind="split",
    effective_at=datetime(2024, 6, 10, 13, 30, tzinfo=timezone.utc),
    known_at=datetime(2024, 5, 22, 20, 0, tzinfo=timezone.utc),
    source="your verified announcement URL or event dataset record",
    ratio=10,
)
```

No corporate-action feed is bundled or inferred from price jumps. The caller
must provide verified terms, provenance, announcement time and effective time.
Events with publication after their effective time fail rather than being
retroactively introduced into a strategy's information set. Event records and
cash movements are included in backtest exports.

Supported events:

- Splits and reverse splits change quantities, per-share cost and the last mark
  inversely, preserving value. Non-integral entitlements fail; no cash-in-lieu
  price is guessed.
- Dividends establish a receivable on `effective_at`, using holdings at that
  timestamp. `pay_at` releases that entitlement to cash even if holdings have
  since changed. Short holdings owe the dividend. An explicit sourced `pay_at`
  is required (it may equal the effective timestamp). For held securities,
  supply `valuation_price` and `valuation_at=effective_at` from a verified ex-date
  observation. No ex-date price is inferred by subtracting a dividend from an
  old quote; missing valuation fails before changing the account.
- Mergers accept explicit `cash_per_share`, `target_symbol` and `ratio` terms.
  Share consideration needs an already observed destination mark and the same
  currency. The old holding is closed, its economic P&L is realized at the
  consideration value, and incoming shares receive the destination mark as
  their accounting basis. This is economic P&L accounting, not tax-lot reporting.

Unknown delisting, bankruptcy and spinoff settlements fail visibly. Missing
terminal prices never imply a zero-price liquidation. Open orders involving an
affected security must be reconciled before the action; the engine does not
guess a broker's order-adjustment policy. At a shared bar/action boundary the
preceding bar completes first, then the action, then scheduled decisions. A bar
callback that creates a conflicting working order causes an explicit failure.
Simultaneous actions on one security require verified entitlement ordering and
are rejected; action IDs never determine payment economics. Events cannot occur inside execution bars; supply finer bars for those events.

All explicit events require `unadjusted_no_corporate_actions` data. The existing
label names the raw input convention; supplied events add the accounting. A
split-adjusted replay uses adjusted share units, so nominal dividends cannot
safely be applied without a verified conversion factor. Mixing those units is
rejected. Dividend-adjusted/total-return data remains unsupported rather than
being double-counted. Alpaca distinguishes raw, split and dividend adjustments
in its [historical bars API](https://docs.alpaca.markets/us/reference/stockbars).
IBKR documents `TRADES` as split-adjusted, not dividend-adjusted in its
[historical bars documentation](https://www.interactivebrokers.com/docs/tws-api/doc/md/historical-bars).

Paper broker corporate actions remain broker-authoritative. This change adds
explicit historical accounting, not inferred live broker entitlements or
corporate-action reconciliation for existing order journals.
