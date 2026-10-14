"""Monthly historical S&P 500 momentum research using normal runtime orders.

Run after importing membership and prefetching history; see docs/portfolio.md.
This is an infrastructure demonstration, not evidence of profitable performance.
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
from datetime import date, datetime, timedelta
from decimal import Decimal
import json
import math
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

import convexity as cx
from convexity.strategies.portfolio import TargetWeightStrategy, equal_weights

NY = ZoneInfo("America/New_York")


def six_months_before(day: date) -> date:
    month = day.year * 12 + day.month - 1 - 6
    year, month = divmod(month, 12)
    month += 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


class MonthlyMomentum(TargetWeightStrategy):
    """Rank six calendar months of completed daily closes; buy top 20 equally."""

    metadata = cx.StrategyMetadata("monthly_sp500_momentum", version="1")

    def __init__(
        self,
        *,
        start: date,
        store_path=".convexity/universes.sqlite3",
        allow_reconstructed=False,
        count=20,
    ):
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer.")
        self.count = count
        self.candidate_coverage = []
        self.config = dict(
            count=count,
            lookback="six calendar months",
            cash_buffer="0.02",
            membership="SP500",
            allow_reconstructed=allow_reconstructed,
        )
        self.contract = cx.StrategyContract(
            instruments=(),
            universes=(
                cx.UniverseSubscription(
                    "SP500",
                    store_path=str(store_path),
                    allow_reconstructed=allow_reconstructed,
                ),
            ),
            warmup_sessions=140,
            subscriptions=(
                cx.EventSubscription(
                    cx.Timer,
                    schedule=cx.SessionSchedule(
                        "monthly_momentum",
                        "rebalance",
                        start=start,
                        frequency="monthly",
                        at="open",
                    ),
                ),
                cx.EventSubscription(cx.Rejection),
                cx.EventSubscription(cx.OrderUpdate),
            ),
        )
        super().__init__(
            self.rank,
            cash_buffer=Decimal(".02"),
            cost_bps=5,
            commission_per_order=1,
            max_price_age=timedelta(days=5),
        )

    def rank(self, context):
        snapshot = context.universe("SP500")
        decision_date = context.time.astimezone(NY).date()
        cutoff = six_months_before(decision_date)
        ranked, missing = [], []
        for member in snapshot.members:
            # Strictly prior observations only, including when a caller chooses
            # a close-time timer. Ticker renames share the stable history key.
            history = sorted(
                (
                    bar
                    for bar in context.history.get(member.security_id, ())
                    if bar.end < context.time
                ),
                key=lambda bar: bar.end,
            )
            anchor = [bar for bar in history if bar.end.astimezone(NY).date() <= cutoff]
            if not history or not anchor:
                missing.append(
                    dict(
                        security_id=member.security_id,
                        symbol=member.symbol,
                        reason="Missing six-month trailing daily history",
                    )
                )
                continue
            latest, first = history[-1], anchor[-1]
            if (decision_date - latest.end.astimezone(NY).date()).days > 5:
                reason = "Stale latest daily price"
            elif (cutoff - first.end.astimezone(NY).date()).days > 5:
                reason = "Missing price near six-month boundary"
            else:
                ranked.append((latest.close / first.close - 1, member.security_id))
                continue
            missing.append(
                dict(
                    security_id=member.security_id, symbol=member.symbol, reason=reason
                )
            )
        ranked.sort(key=lambda row: (-row[0], row[1]))
        selected = [identity for _, identity in ranked[: self.count]]
        self.candidate_coverage.append(
            dict(
                time=context.time.isoformat(),
                universe="SP500",
                eligible=len(snapshot.members),
                observable=len(ranked),
                selected=len(selected),
                selected_ids=selected,
                ratio=len(ranked) / len(snapshot.members) if snapshot.members else None,
                missing=missing,
                warnings=list(snapshot.warnings),
            )
        )
        return equal_weights(selected)


def performance(result) -> dict:
    """End-of-session price returns, annualized at 252 sessions and zero RF."""
    by_day = {}
    for point in result.equity:
        by_day[point["time"].astimezone(NY).date()] = float(point["equity"])
    initial = float(result.config["initial_cash"])
    values = [initial, *by_day.values()]
    returns = [b / a - 1 for a, b in zip(values, values[1:]) if a > 0]
    sigma = stdev(returns) if len(returns) > 1 else 0
    average_equity = mean(values)
    traded = sum(float(fill.quantity * fill.price) for fill in result.fills)
    return dict(
        total_return=float(result.metrics["total_return"]),
        annualized_volatility=sigma * math.sqrt(252),
        sharpe=mean(returns) / sigma * math.sqrt(252) if sigma else None,
        max_drawdown=float(result.metrics["max_drawdown"]),
        two_way_turnover=traded / average_equity if average_equity else None,
        fees=float(result.metrics["fees"]),
        sessions=len(returns),
    )


async def run_example(args, *, provider=None):
    # Dynamic Research defaults to the same native daily cache namespace as the
    # prefetch CLI, and delays daily strategy observations until the next open.
    instances = []

    def factory():
        strategy = MonthlyMomentum(
            start=args.start,
            store_path=args.store,
            allow_reconstructed=args.allow_reconstructed,
        )
        instances.append(strategy)
        return strategy

    result = await cx.Research(args.start, args.end).run_async(
        factory,
        provider=provider,
        cache=args.cache,
        missing_data=args.missing_data,
        execution="standard",
        initial_cash=100_000,
        output=Path(args.output) / "portfolio",
    )
    # Use the same native daily provider for SPY, with ordinary buy/hold orders.
    if provider is None:
        from convexity.data.prefetch import daily_provider

        provider = daily_provider(cache=args.cache)

    class Benchmark(cx.Strategy):
        metadata = cx.StrategyMetadata("spy_price_benchmark")
        contract = cx.StrategyContract(instruments=("SPY",))

        def on_start(self, context):
            quantity = int(context.equity * Decimal(".98") / context.price("SPY"))
            return cx.SubmitOrder.market("SPY", "BUY", quantity) if quantity else None

    benchmark = await cx.Research(args.start, args.end).run_async(
        Benchmark,
        provider=provider,
        cache=args.cache,
        execution="standard",
        initial_cash=100_000,
        output=Path(args.output) / "benchmark",
    )
    report = dict(
        portfolio=performance(result),
        spy=performance(benchmark),
        candidate_coverage=instances[0].candidate_coverage,
        daily_coverage=result.config["metadata"]["research"]["coverage"],
        provider_failures=result.config["metadata"]["research"]["provider_failures"],
        rejections=instances[0].rejections,
        limitations=[
            "Price-return comparison; dividends and unsupported corporate actions are not invented.",
            "Incomplete delisted-security data causes survivorship/selection bias.",
            "Reconstructed membership lacks verified contemporaneous publication evidence.",
            "Integer shares, 2% cash reserve, modeled spread/slippage and commissions.",
            "Native daily execution volume is an aggregate capacity approximation.",
        ],
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("portfolio", "spy", "limitations")}, indent=2
        )
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--store", default=".convexity/universes.sqlite3")
    parser.add_argument("--cache", default=".convexity/history.sqlite3")
    parser.add_argument("--allow-reconstructed", action="store_true")
    parser.add_argument("--missing-data", choices=("exclude", "fail"), default="fail")
    parser.add_argument("--output", default="out/monthly_momentum")
    args = parser.parse_args(argv)
    asyncio.run(run_example(args))


if __name__ == "__main__":
    main()
