"""Yahoo Finance adapter using the optional yfinance package."""
from typing import ClassVar

import pandas as pd

from ._http import positive
from .core import DataError, empty_frame


class YahooFinance:
    name = "yahoo"
    intervals: ClassVar[dict[str, str]] = {"1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
                 "1h": "60m", "90m": "90m", "1d": "1d", "5d": "5d", "1w": "1wk",
                 "1mo": "1mo", "3mo": "3mo"}

    def __init__(self, *, auto_adjust: bool = False, actions: bool = True,
                 prepost: bool = False, timeout: float = 30):
        positive(timeout, "timeout")
        self.auto_adjust, self.actions, self.prepost, self.timeout = auto_adjust, actions, prepost, timeout

    def read(self, symbol, start, end, interval):
        if interval not in self.intervals:
            raise ValueError(f"Yahoo intervals: {', '.join(self.intervals)}")
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Install Yahoo support: pip install 'convexity[yahoo]'") from exc
        # Calendar bars are labeled by exchange date. Supply date strings so
        # yfinance interprets boundaries in the exchange timezone.
        calendar = interval in {"1d", "5d", "1w", "1mo", "3mo"}
        if calendar and (start != start.normalize() or end != end.normalize()):
            raise ValueError("Yahoo calendar-bar boundaries must be midnight UTC dates.")
        try:
            frame = yf.Ticker(symbol).history(
                start=start.strftime("%Y-%m-%d") if calendar else start.to_pydatetime(),
                end=end.strftime("%Y-%m-%d") if calendar else end.to_pydatetime(),
                interval=self.intervals[interval], auto_adjust=self.auto_adjust,
                actions=self.actions, prepost=self.prepost, timeout=self.timeout,
                raise_errors=True, keepna=True)
        except Exception as exc:
            raise DataError(f"Yahoo history failed for {symbol}: {exc}") from exc
        if frame.empty:
            frame = empty_frame()
        else:
            frame = frame.rename(columns=lambda c: c.lower().replace(" ", "_"))
            if calendar:
                frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).tz_localize("UTC")
        frame.attrs = {"price_basis": "dividend_and_split_adjusted" if self.auto_adjust else "yahoo_split_adjusted",
                       "auto_adjust": self.auto_adjust, "prepost": self.prepost,
                       "volume_unit": "provider_native", "timestamp_kind": "session_date" if calendar else "bar_open"}
        return frame
