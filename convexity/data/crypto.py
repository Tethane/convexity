"""Public Binance spot and Deribit instrument candle adapters."""
from typing import ClassVar

import pandas as pd

from ._http import JSONClient
from .core import OHLCV, DataError, empty_frame


def _milliseconds(value):
    return value.value // 1_000_000


class Binance:
    """Binance spot candles; symbols are native pairs such as BTCUSDT."""
    name = "binance"
    intervals: ClassVar[set[str]] = {"1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h",
                 "8h", "12h", "1d", "3d", "1w", "1mo"}

    def __init__(self, *, base_url: str = "https://data-api.binance.vision", **http_options):
        self.http = JSONClient(base_url, **http_options)

    def read(self, symbol, start, end, interval):
        if interval not in self.intervals:
            raise ValueError(f"Binance intervals: {', '.join(sorted(self.intervals))}")
        cursor, stop = _milliseconds(start), _milliseconds(end)
        rows = []
        while cursor < stop:
            page = self.http.get("/api/v3/klines", {
                "symbol": symbol, "interval": "1M" if interval == "1mo" else interval,
                "startTime": cursor, "endTime": stop - 1, "limit": 1000})
            if not isinstance(page, list):
                raise DataError(f"Invalid Binance kline response: {page}")
            if not page:
                break
            if any(not isinstance(row, list) or len(row) < 11 for row in page):
                raise DataError("Malformed Binance candle.")
            next_cursor = max(int(row[0]) for row in page) + 1
            if next_cursor <= cursor:
                raise DataError("Binance pagination did not advance.")
            rows.extend(page)
            cursor = next_cursor
            if len(page) < 1000:
                break
        if rows:
            frame = pd.DataFrame([[r[0], *r[1:6], r[7], r[8], r[9], r[10]] for r in rows],
                                 columns=["timestamp", *OHLCV, "quote_volume", "trade_count",
                                          "taker_buy_volume", "taker_buy_quote_volume"])
            frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
            frame = frame.set_index("timestamp")
        else:
            frame = empty_frame()
        frame.attrs = {"market": "spot", "price_basis": "exchange_native", "volume_unit": "base_asset",
                       "quote_volume_unit": "quote_asset", "timestamp_kind": "bar_open", "endpoint": self.http.base_url}
        return frame


class Deribit:
    """Deribit chart candles for a native instrument, e.g. BTC-PERPETUAL."""
    name = "deribit"
    intervals: ClassVar[dict[str, str]] = {"1m": "1", "3m": "3", "5m": "5", "10m": "10", "15m": "15", "30m": "30",
                 "1h": "60", "2h": "120", "3h": "180", "6h": "360", "12h": "720", "1d": "1D"}

    def __init__(self, *, base_url: str = "https://www.deribit.com", **http_options):
        self.http = JSONClient(base_url, **http_options)

    def read(self, symbol, start, end, interval):
        if interval not in self.intervals:
            raise ValueError(f"Deribit intervals: {', '.join(self.intervals)}")
        if symbol in {"BTCUSDT", "ETHUSDT"}:
            native = symbol.removesuffix("USDT") + "-PERPETUAL"
            raise DataError(
                f"{symbol!r} is a Binance symbol, not a Deribit instrument name. "
                f"Use provider='binance' for that spot pair, or explicitly request "
                f"{native!r} for the Deribit perpetual future. These are different instruments.")
        resolution = self.intervals[interval]
        step = (1440 if resolution == "1D" else int(resolution)) * 60_000
        # Bounded windows keep responses small; advance by returned bars to
        # handle a server returning fewer bars than the requested window.
        cursor, stop = _milliseconds(start), _milliseconds(end)
        frames = []
        while cursor < stop:
            window_end = min(stop, cursor + step * 1000)
            try:
                payload = self.http.get("/api/v2/public/get_tradingview_chart_data", {
                    "instrument_name": symbol, "start_timestamp": cursor,
                    "end_timestamp": window_end - 1, "resolution": resolution})
            except DataError as exc:
                raise DataError(f"Deribit history failed for {symbol!r}: {exc}") from exc
            if not isinstance(payload, dict) or "error" in payload or not isinstance(payload.get("result"), dict):
                raise DataError(f"Deribit request failed for {symbol!r}: {payload}")
            result = payload["result"]
            if result.get("status") == "no_data":
                cursor = window_end
                continue
            if result.get("status") != "ok":
                raise DataError(f"Unexpected Deribit chart status: {result.get('status')}")
            keys = ["ticks", *OHLCV, "cost"]
            if any(not isinstance(result.get(k), list) for k in keys) or len({len(result[k]) for k in keys}) != 1:
                raise DataError("Deribit returned missing or unequal candle arrays.")
            ticks = result["ticks"]
            if not ticks:
                raise DataError("Deribit returned an empty 'ok' response.")
            frame = pd.DataFrame({k: result[k] for k in keys}).rename(columns={"cost": "cost_volume"})
            frame.index = pd.to_datetime(frame.pop("ticks"), unit="ms", utc=True)
            frames.append(frame)
            next_cursor = max(ticks) + step
            if next_cursor <= cursor:
                raise DataError("Deribit pagination did not advance.")
            cursor = min(window_end, next_cursor)
        frame = pd.concat(frames) if frames else empty_frame()
        frame.attrs = {"price_basis": "exchange_native", "volume_unit": "provider_native",
                       "cost_volume_unit": "provider_native", "timestamp_kind": "bar_open",
                       "endpoint": self.http.base_url}
        return frame
