# Historical universes and portfolio targets

Declare a historical universe without listing every constituent:

```python
import convexity as cx
from datetime import date

class Portfolio(cx.TargetWeightStrategy):
    metadata = cx.StrategyMetadata("historical-portfolio")
    contract = cx.StrategyContract(
        universes=(cx.UniverseSubscription("SP500", store_path=".convexity/universes.sqlite3"),),
        subscriptions=(cx.EventSubscription(cx.Timer, schedule=cx.SessionSchedule(
            "monthly-portfolio", "rebalance", start=date(2015, 1, 1),
            frequency="monthly", at="open")),),
        warmup_sessions=127,
    )

    def __init__(self):
        super().__init__(lambda context: cx.equal_weights(
            member.security_id for member in context.universe("SP500").members
            if len(context.history.get(member.security_id, ())) >= 127))

result = cx.Research("2015-01-01", "2016-01-01").run(
    Portfolio, missing_data="exclude", output="out/portfolio")
```

This requires a populated universe store with verified membership and historical ticker
mappings. Public reconstructed imports require explicit `allow_reconstructed=True` in
the declaration; that permission does not establish contemporaneous publication
evidence. The default remains strict. `context.universe(name)` resolves effective
membership and information availability at the callback timestamp. Dynamic history,
positions and orders use stable security IDs; snapshot members retain historical
provider ticker symbols. Names can change without creating another position, and reused
tickers cannot merge distinct security IDs.

Research loads the historical set internally, retrieves each historical ticker interval
through the existing provider/cache interfaces, then replays the ordinary backtester. It
does not expose the future constituent list to callbacks. Ticker mappings outside their
verified intervals are not extended to invent a feature lookback. Missing data never
removes the membership record. `missing_data="fail"` (the default) fails on missing
eligible daily prices; `"exclude"` retains explicit dated eligible/observable counts,
coverage ratios, missing IDs and provider errors in
`result.config["metadata"]["research"]`. Strategies must additionally report
feature/candidate coverage, because having today's price does not establish a six-month
history.

Dynamic research currently uses daily observations and session-boundary schedules on one
exchange calendar. The default provider is native Alpaca daily history using the same
cache as bulk prefetch. Native daily observations are conservatively published at the
following session open, and require open-time schedules. Native daily volume includes
extended hours; daily OHLC execution is an approximation, and participation limits can
overestimate regular-session capacity. Explicit regular-session providers publish
completed bars at the close. Intraday work remains available through the existing
explicit historical and replay APIs. Automatic dynamic-universe broker subscriptions in
managed paper trading are not implemented.

`equal_weights(ids, invested=0.98)` and custom nonnegative mappings are supported.
`target_weight_orders(context, weights)` returns a `RebalancePlan` containing ordinary
orders, integer targets, observed allocations, estimated costs, gross
traded-notional/equity turnover and diagnostics. Weights sum to at most one. The default
1% cash reserve, optional basis-point cost estimate and per-order commission estimate
bound planned buys by actual cash and available funds. Execution fees and slippage still
come from the execution model; planning estimates do not replace them.

Omitted held securities target zero. Reductions are submitted first and must fill before
their proceeds fund purchases. Working orders prevent duplicate reservations.
`TargetWeightStrategy` freezes integer targets at the scheduled decision and pursues
them on subsequent bar observations; it also removes targets for securities that leave
eligibility. It exposes plans and rejected-order messages (subscribe to `OrderUpdate`
and `Rejection` events to collect venue rejection messages). Missing, future or stale
observations are reported and skipped; stale holdings remain visible for valuation and
cannot be liquidated without a usable price. The default maximum mark age is four days
and can be changed with `max_price_age=timedelta(...)`.

Current order/fill contracts only support whole shares. `fractional=True` raises
explicitly. Account equity is shared across strategies, so allocate disjoint weight
budgets or run portfolio strategies independently. Submitted prices are observations,
not promised fills: ordinary runtime risk checks, next eligible execution bars,
participation limits and rejections still apply. Turnover reported by a plan is
estimated submitted notional; realized turnover must use actual fills.

See `examples/monthly_momentum.py` for the six-month momentum research demonstration and
SPY comparison. Incomplete observable history and retrospectively reconstructed
membership must never be described as a fully survivorship-bias-free backtest.

Run the momentum example after importing reference membership and prefetching daily history:

```bash
python examples/monthly_momentum.py --start 2015-07-01 --end 2026-08-17 \
  --allow-reconstructed --missing-data exclude --output out/monthly_momentum
```

The end date must lie within the imported source's verified coverage; do not extend
current snapshots through the present by assumption. Inspect the importer provenance
report for the actual bound and source discrepancies.

In exclusion mode, one missing daily observation does not discard an otherwise usable
security history. Research first recovers validated cached session runs, then bisects
coverage failures with at most 64 additional requests per ticker interval. Credential,
permission, network and other infrastructure failures do not trigger this subdivision.
Unresolved intervals and exhausted acquisition budgets remain explicit in
provider-failure reports.

For a fixed ETF universe, see [portfolio rebalancing](../examples/portfolio_rebalancing.ipynb).
