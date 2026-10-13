"""Run monthly long-short stock research through the public Convexity API.

Requires a stock universe and dates. Fetches real IBKR historical observations,
validates complete sessions, then replays simulated orders. No synthetic fallback.
"""
import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.backtest import Research
from convexity.runtime import BarSubscription, SessionSchedule, StrategyContract, StrategyMetadata
from convexity.strategies import MonthlyLongShort, StockRule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--bar-size", default="1 day", help="Duration, e.g. '7 mins', '90 secs', '2 hours', '1 week'")
    parser.add_argument("--warmup-sessions", type=int, default=5)
    parser.add_argument("--cache", type=Path, default=Path(".convexity/history.sqlite3"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("out/backtest"))
    parser.add_argument("--initial-cash", type=Decimal, default=Decimal(100_000))
    parser.add_argument("--execution", choices=("standard", "frictionless", "conservative"), default="standard")
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--leg-weight", type=Decimal, default=Decimal(".30"))
    args = parser.parse_args()
    class MonthlyResearch(MonthlyLongShort):
        metadata = StrategyMetadata("monthly", version="1")
        contract = StrategyContract(
            instruments=tuple(args.symbols), warmup_sessions=args.warmup_sessions,
            subscriptions=(BarSubscription(tuple(args.symbols), bar_size=args.bar_size),
                           SessionSchedule("monthly", "rebalance", start=args.start,
                                           frequency="monthly", at="close")))
        def __init__(self):
            super().__init__(StockRule(tuple(args.symbols), lookback=args.lookback, leg_weight=args.leg_weight))

    result = Research(args.start, args.end).run(
        MonthlyResearch, cache=args.cache, output=args.output, refresh=args.refresh,
        initial_cash=args.initial_cash,
        execution=args.execution,
        metadata={"strategy": {"lookback": args.lookback, "leg_weight": str(args.leg_weight)},
                  "dataset_notice": "Real IBKR split-adjusted prices; dividends and survivorship controls excluded."},
    )
    print(f"{result.status}: {len(result.orders)} orders, {len(result.fills)} fills, {len(result.open_orders)} open orders")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.6f}")
    for diagnostic in result.diagnostics:
        print(diagnostic)
    print(f"Price basis: {result.config['price_basis']}; dividend cash flows are not included")
    print(f"Dataset SHA-256: {result.config['data_identity']}")
    print(f"Results: {args.output / 'result.json'}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
