"""Use Convexity's quadratic optimizer inside the public strategy callback.

Fits minimum-variance long-only weights to observed real IBKR returns.
Specify symbols and inclusive research dates; observations are fetched and cached.
This demonstrates API composition, not a model recommendation. A failed solver pauses the strategy through Runtime diagnostics.
"""

import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from convexity.backtest import Research
from convexity.runtime import (
    BarSubscription,
    SessionSchedule,
    StrategyContract,
    StrategyMetadata,
)
from convexity.math.optimization import QuadraticProgram, solve
from convexity.runtime import Context
from convexity.strategies import TargetStrategy


def minimum_variance_targets(
    context: Context, lookback: int = 20
) -> dict[str, int] | None:
    symbols = sorted(context.history)
    if any(len(context.history[s]) < lookback for s in symbols):
        return None
    closes = np.array(
        [[float(b.close) for b in context.history[s][-lookback:]] for s in symbols]
    )
    returns = closes[:, 1:] / closes[:, :-1] - 1
    covariance = np.cov(returns) + np.eye(len(symbols)) * 1e-6
    result = solve(
        QuadraticProgram(
            linear=np.zeros(len(symbols)),
            quadratic=2 * covariance,
            equality=np.ones((1, len(symbols))),
            target=[1],
            lower_bounds=0,
            upper_bounds=1,
        )
    )
    assert result.solution is not None
    return {
        s: int(
            context.equity
            * Decimal(".8")
            * Decimal(str(max(0, float(w))))
            / context.marks[s]
        )
        for s, w in zip(symbols, result.solution)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument(
        "--cache", type=Path, default=Path(".convexity/history.sqlite3")
    )
    parser.add_argument("--output", type=Path, default=Path("out/backtest-math"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.lookback < 3 or len(set(args.symbols)) < 2:
        parser.error("Use lookback >= 3 and at least two distinct stocks")

    class MinimumVariance(TargetStrategy):
        metadata = StrategyMetadata("minimum-variance", version="1")
        contract = StrategyContract(
            instruments=tuple(args.symbols),
            warmup_sessions=args.lookback,
            subscriptions=(
                BarSubscription(tuple(args.symbols)),
                SessionSchedule(
                    "monthly",
                    "rebalance",
                    start=args.start,
                    frequency="monthly",
                    at="close",
                ),
            ),
        )

        def __init__(self):
            super().__init__(
                lambda context: minimum_variance_targets(context, args.lookback)
            )

    result = Research(args.start, args.end).run(
        MinimumVariance,
        cache=args.cache,
        output=args.output,
        refresh=args.refresh,
        metadata={
            "rule": "minimum variance long-only",
            "lookback": args.lookback,
            "equity_fraction": ".8",
            "covariance_ridge": "1e-6",
        },
    )
    print(
        f"{result.status}: {len(result.fills)} fills; equity {result.metrics['final_equity']}"
    )
    print(
        f"Price basis: {result.config['price_basis']}; dividend cash flows are not included"
    )
    print(f"Results: {args.output / 'result.json'}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
