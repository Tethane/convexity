"""Standalone data -> pandas research. No event runtime or backtest.

python examples/data_research.py --provider yahoo --symbols SPY QQQ
python examples/data_research.py --provider binance --symbols BTCUSDT --interval 1h
python examples/data_research.py --provider deribit --symbols BTC-PERPETUAL
python examples/data_research.py --provider ibkr --symbols SPY --port 7497
"""
import argparse

from convexity.data import history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["yahoo", "binance", "deribit", "ibkr"], default="yahoo")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-02-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=71)
    parser.add_argument("--csv", help="Optional output CSV path")
    args = parser.parse_args()
    options = {"host": args.host, "port": args.port, "client_id": args.client_id} if args.provider == "ibkr" else {}
    df = history(args.symbols, provider=args.provider, start=args.start, end=args.end,
                 interval=args.interval, **options)
    print(df.head())
    closes = df.pivot(columns="symbol", values="close")
    print(closes.pct_change(fill_method=None).describe())
    print(df.attrs)
    if args.csv:
        df.to_csv(args.csv)


if __name__ == "__main__":
    main()
