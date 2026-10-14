"""One user strategy, historical replay or managed IBKR paper execution.

Backtests require explicit symbols and dates, fetch Alpaca history (IBKR fallback), and
cache the validated observations. No synthetic fallback exists. --mode paper
sends actual paper orders; stopping never implies liquidation.
"""

import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import TypedDict
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convexity as cx
from convexity.backtest.bar_size import parse_bar_size
from convexity.strategies import rebalance_orders


class MomentumConfig(TypedDict):
    symbol: str
    lookback: int
    shares: int


class Momentum(cx.Strategy):
    """Hold a fixed number of shares while the observed lookback return is positive."""

    def __init__(
        self,
        symbol: str,
        lookback: int = 3,
        shares: int = 1,
        bar_size: str = "1 min",
        warmup_sessions: int = 5,
    ):
        if (
            not symbol
            or type(lookback) is not int
            or lookback < 2
            or type(shares) is not int
            or shares < 1
        ):
            raise ValueError(
                "Supply a symbol, lookback >= 2, and positive whole shares."
            )
        self.metadata = cx.StrategyMetadata(f"momentum_{symbol}", version="1")
        self.contract = cx.StrategyContract(
            instruments=(symbol,),
            subscriptions=(cx.BarSubscription((symbol,), bar_size=bar_size),),
            warmup_sessions=warmup_sessions,
        )
        self.config: MomentumConfig = {
            "symbol": symbol,
            "lookback": lookback,
            "shares": shares,
        }

    def on_event(self, event, context):
        if not isinstance(event, cx.Bars):
            return None
        symbol = self.config["symbol"]
        history = context.history.get(symbol, ())
        if len(history) < self.config["lookback"]:
            return None
        target = (
            self.config["shares"]
            if history[-1].close > history[-self.config["lookback"]].close
            else 0
        )
        return rebalance_orders(context, {symbol: target})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("backtest", "paper"), default="backtest")
    symbols = parser.add_mutually_exclusive_group(required=True)
    symbols.add_argument("--symbol", help="One stock symbol")
    symbols.add_argument(
        "--symbols", nargs="+", help="Stock universe, for example SPY QQQ"
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        help="First research session date (inclusive)",
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, help="Last research session date (inclusive)"
    )
    parser.add_argument(
        "--bar-size",
        default="1 min",
        help="1 min matches paper callbacks; other intervals are research-only in this example",
    )
    parser.add_argument("--warmup-sessions", type=int, default=5)
    parser.add_argument(
        "--cache", type=Path, default=Path(".convexity/history.sqlite3")
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch a new real-data snapshot instead of using cache",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--shares", type=int, default=1)
    parser.add_argument("--initial-cash", type=Decimal, default=Decimal(100_000))
    parser.add_argument("--budget", type=Decimal, default=Decimal(1000))
    parser.add_argument("--order-cap", type=Decimal, default=Decimal(1000))
    parser.add_argument("--seconds", type=float, help="Optional paper-session duration")
    args = parser.parse_args(argv)
    try:
        args.bar_size = parse_bar_size(args.bar_size).label
    except ValueError as exc:
        parser.error(str(exc))
    universe = tuple(args.symbols or (args.symbol,))
    if len(set(universe)) != len(universe):
        parser.error("Stock symbols must be distinct")
    if args.mode == "backtest" and (args.start is None or args.end is None):
        parser.error(
            "Backtests require --start and --end; prices are fetched from IBKR"
        )
    if args.mode == "paper" and args.bar_size != "1 min":
        parser.error(
            "This paper strategy consumes 1-minute bars; other research intervals require matching signal aggregation"
        )
    if args.seconds is not None and (args.seconds <= 0 or args.mode != "paper"):
        parser.error("--seconds must be positive and is only valid in paper mode")
    factories = {
        f"momentum_{symbol}": (
            lambda symbol=symbol: Momentum(
                symbol, args.lookback, args.shares, args.bar_size, args.warmup_sessions
            )
        )
        for symbol in universe
    }
    if args.mode == "backtest":
        result = cx.Research(start=args.start, end=args.end).run(
            tuple(factories.values()),
            refresh=args.refresh,
            cache=args.cache,
            output=args.output or "out/custom-research",
            initial_cash=args.initial_cash,
            limits=cx.RiskLimits(
                max_order_notional=args.order_cap,
                strategy_gross_limits={owner: args.budget for owner in factories},
            ),
        )
        print(
            f"{result.status}: {len(result.fills)} fills; equity {result.metrics['final_equity']}"
        )
        print(
            f"Orders: {len(result.open_orders)} still open; {sum(o.status == 'rejected' for o in result.orders)} rejected"
        )
        print(f"Real historical data: {result.config['data_source']}")
        print(
            f"Price basis: {result.config['price_basis']}; dividend cash flows are not included"
        )
        print(f"Dataset SHA-256: {result.config['data_identity']}")
        return 0 if result.status == "completed" else 1
    app = cx.PaperTrading(
        config=cx.PaperAccountConfig.from_env(),
        strategies=tuple(factories.values()),
        strategy_versions={
            owner: f"momentum-v1:{symbol}:{args.lookback}:{args.shares}"
            for owner, symbol in zip(factories, universe)
        },
        path=args.output or "out/custom-paper",
        budgets={owner: args.budget for owner in factories},
        max_order_notional=args.order_cap,
        market_collar_bps=Decimal(25),
    )
    until = (
        datetime.now(timezone.utc) + timedelta(seconds=args.seconds)
        if args.seconds is not None
        else None
    )
    try:
        asyncio.run(app.run(until=until))
    except KeyboardInterrupt:
        pass
    print(app.status().format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
