import asyncio
import io
import json
import subprocess
import sys
from datetime import date
from types import SimpleNamespace
from urllib.error import HTTPError

import pandas as pd
import pytest

from convexity.data import (
    IBKR,
    Binance,
    DataClient,
    DataError,
    Deribit,
    NoDataError,
    YahooFinance,
    ahistory,
    history,
    split_history,
)
from convexity.data._http import JSONClient
from convexity.data.core import OHLCV, empty_frame


def candles(timestamps, closes=None):
    n = len(timestamps)
    return pd.DataFrame({"open": [10] * n, "high": [12] * n, "low": [9] * n,
                         "close": closes or [11] * n, "volume": [100] * n},
                        index=pd.to_datetime(timestamps, utc=True))


class Source:
    name = "test"

    def read(self, symbol, start, end, interval):
        frame = candles(["2025-01-03", "2025-01-01", "2025-01-02", "2025-01-02"])
        frame.attrs = {"currency": "USD"}
        return frame


def test_tidy_panel_dates_duplicates_metadata_and_pivot():
    df = history(["SPY", "QQQ"], start="2025-01-01", end="2025-01-03", provider=Source())
    assert df.columns.tolist() == ["symbol", *OHLCV]
    assert str(df.index.tz) == "UTC" and df.index.name == "timestamp"
    assert len(df) == 4
    assert df.symbol.tolist() == ["QQQ", "SPY", "QQQ", "SPY"]
    assert all(df[c].dtype == "float64" for c in OHLCV)
    assert df.pivot(columns="symbol", values="close").shape == (2, 2)
    assert df.attrs["sources"]["SPY"]["currency"] == "USD"
    assert df.attrs["end_exclusive"] is True


def test_split_returns_frames_in_request_order_with_own_metadata():
    spy, qqq = history(["SPY", "QQQ"], start="2025-01-01", end="2025-01-03",
                       provider=Source(), split=True)
    assert spy.symbol.unique().tolist() == ["SPY"]
    assert qqq.symbol.unique().tolist() == ["QQQ"]
    assert spy.attrs["symbols"] == ["SPY"]
    assert qqq.attrs["sources"] == {"QQQ": {"currency": "USD"}}
    assert spy.index.is_unique and str(spy.index.tz) == "UTC"
    assert all(spy[c].dtype == "float64" for c in OHLCV)
    spy.iloc[0, spy.columns.get_loc("close")] = -100
    spy.attrs["sources"]["SPY"]["currency"] = "CHANGED"
    assert qqq.close.iloc[0] == 11
    assert qqq.attrs["sources"]["QQQ"]["currency"] == "USD"


def test_split_existing_frame_does_not_fetch_and_makes_independent_copies():
    df = history(["SPY", "QQQ"], start="2025-01-01", end="2025-01-03", provider=Source())
    tables = split_history(df)
    assert list(tables) == ["SPY", "QQQ"]
    tables["SPY"].attrs["sources"]["SPY"]["currency"] = "CHANGED"
    tables["SPY"].iloc[0, tables["SPY"].columns.get_loc("close")] = -100
    assert df.attrs["sources"]["SPY"]["currency"] == "USD"
    assert df.close.eq(11).all()
    df.attrs = {}
    assert list(split_history(df)) == ["QQQ", "SPY"]
    with pytest.raises(ValueError, match="symbol column"):
        split_history(df.drop(columns="symbol"))


def test_split_preserves_empty_symbols_and_single_symbol_tuple():
    provider = Source()
    provider.read = lambda symbol, *a: empty_frame() if symbol == "EMPTY" else candles(["2025-01-01"])
    client = DataClient(provider)
    empty, spy = client.history(["EMPTY", "SPY"], start="2025-01-01", end="2025-01-02",
                                split=True, allow_empty=True)
    assert empty.empty and empty.attrs["symbols"] == ["EMPTY"]
    assert str(empty.index.tz) == "UTC" and len(spy) == 1
    single = client.history("SPY", start="2025-01-01", end="2025-01-02", split=True)
    assert isinstance(single, tuple) and len(single) == 1
    with pytest.raises(TypeError, match="split must"):
        client.history("SPY", start="2025-01-01", end="2025-01-02", split="yes")
    with pytest.raises(NoDataError):
        client.history(["SPY", "EMPTY"], start="2025-01-01", end="2025-01-02", split=True)


def test_async_split_top_level_and_client():
    async def query():
        options = {"start": "2025-01-01", "end": "2025-01-03", "split": True}
        first = await ahistory(["SPY", "QQQ"], provider=Source(), **options)
        second = await DataClient(Source()).ahistory(["QQQ", "SPY"], **options)
        return first, second
    first, second = asyncio.run(query())
    assert [frame.attrs["symbols"][0] for frame in first] == ["SPY", "QQQ"]
    assert [frame.attrs["symbols"][0] for frame in second] == ["QQQ", "SPY"]


@pytest.mark.parametrize("symbol,native", [("BTCUSDT", "BTC-PERPETUAL"), ("ETHUSDT", "ETH-PERPETUAL")])
def test_deribit_binance_symbols_get_actionable_error_without_network(symbol, native):
    provider = Deribit()
    def unexpected(*args):
        pytest.fail("Invalid symbols should fail before making an HTTP request")
    provider.http.get = unexpected
    with pytest.raises(DataError, match=native) as exc:
        history(symbol, start="2024-01-01", end="2024-01-03", provider=provider)
    assert "different instruments" in str(exc.value)


def test_deribit_http_failure_identifies_symbol():
    provider = Deribit()
    def fail(*args):
        raise DataError("HTTP 400: instrument not found")
    provider.http.get = fail
    with pytest.raises(DataError, match="BAD-INSTRUMENT.*instrument not found"):
        history("BAD-INSTRUMENT", start="2024-01-01", end="2024-01-03", provider=provider)


@pytest.mark.parametrize("kwargs", [
    {"symbols": []}, {"symbols": ["SPY", "SPY"]}, {"symbols": " "},
    {"symbols": [1]}, {"start": "NaT"}, {"start": "2025-01-04"},
    {"start": 123}, {"interval": ""}, {"provider": "unknown"},
])
def test_validation(kwargs):
    args = {"symbols": "SPY", "start": "2025-01-01", "end": "2025-01-03", "provider": Source()}
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        history(**args)


def test_timezone_boundaries():
    df = history("SPY", start="2024-12-31T19:00:00-05:00", end="2025-01-01T19:00:00-05:00", provider=Source())
    assert df.index.tolist() == [pd.Timestamp("2025-01-01", tz="UTC")]


def test_missing_empty_and_conflicting_data():
    provider = Source()
    provider.read = lambda *a: empty_frame()
    with pytest.raises(NoDataError):
        history("SPY", start="2025-01-01", end="2025-01-03", provider=provider)
    empty = history("SPY", start="2025-01-01", end="2025-01-03", provider=provider, allow_empty=True)
    assert empty.empty and str(empty.index.tz) == "UTC"
    provider.read = lambda *a: candles(["2025-01-01", "2025-01-01"], [10, 11])
    with pytest.raises(DataError, match="conflicting"):
        history("SPY", start="2025-01-01", end="2025-01-03", provider=provider)


def test_gaps_and_nan_are_not_filled():
    provider = Source()
    provider.read = lambda *a: candles(["2025-01-01", "2025-01-03"], [11, float("nan")])
    df = history("SPY", start="2025-01-01", end="2025-01-04", provider=provider)
    assert len(df) == 2 and pd.isna(df.close.iloc[-1])


def test_failure_does_not_return_partial_symbol_panel():
    provider = Source()
    def read(symbol, *args):
        if symbol == "BAD":
            raise OSError("connection lost")
        return candles(["2025-01-01"])
    provider.read = read
    with pytest.raises(DataError, match="BAD.*connection lost"):
        history(["SPY", "BAD"], start="2025-01-01", end="2025-01-03", provider=provider)


def test_async_api():
    df = asyncio.run(ahistory("SPY", start="2025-01-01", end="2025-01-03", provider=Source()))
    assert len(df) == 2


def test_reusable_client_empty_symbol_and_options():
    provider = Source()
    provider.read = lambda symbol, *a: empty_frame() if symbol == "EMPTY" else candles(["2025-01-01"])
    client = DataClient(provider)
    df = client.history(["SPY", "EMPTY"], start="2025-01-01", end="2025-01-02", allow_empty=True)
    assert df.symbol.tolist() == ["SPY"]
    assert df.attrs["symbols"] == ["SPY", "EMPTY"]
    assert len(asyncio.run(client.ahistory("SPY", start="2025-01-01", end="2025-01-02"))) == 1
    with pytest.raises(TypeError, match="provider instance"):
        DataClient(provider, timeout=5)


@pytest.mark.parametrize("bad", ["missing_column", "bad_number", "infinite", "missing_time"])
def test_bad_provider_values_fail_explicitly(bad):
    frame = candles(["2025-01-01"])
    if bad == "missing_column":
        frame = frame.drop(columns="close")
    elif bad == "bad_number":
        frame["close"] = ["not a price"]
    elif bad == "infinite":
        frame["close"] = [float("inf")]
    else:
        frame.index = pd.DatetimeIndex([pd.NaT], tz="UTC")
    provider = Source()
    provider.read = lambda *args: frame
    with pytest.raises(DataError):
        history("SPY", start="2025-01-01", end="2025-01-02", provider=provider)


def kline(t):
    return [t, "10", "12", "9", "11", "100", t + 59999, "1000", 20, "50", "500", "0"]


def test_binance_pagination_and_exclusive_end():
    provider = Binance()
    start = pd.Timestamp("2025-01-01", tz="UTC").value // 1_000_000
    calls = []
    def get(path, params):
        calls.append(params)
        return [kline(start + i * 60000) for i in range(1000)] if len(calls) == 1 else [kline(start + 1000 * 60000)]
    provider.http.get = get
    df = history("BTCUSDT", start="2025-01-01", end="2025-01-02", interval="1m", provider=provider)
    assert len(df) == 1001 and len(calls) == 2
    assert calls[1]["startTime"] == start + 999 * 60000 + 1
    assert calls[0]["endTime"] == start + 86400000 - 1
    assert df.quote_volume.iloc[0] == 1000
    assert df.attrs["sources"]["BTCUSDT"]["volume_unit"] == "base_asset"


def test_binance_error_and_no_progress():
    provider = Binance()
    provider.http.get = lambda *a: {"code": -1121, "msg": "Invalid symbol"}
    with pytest.raises(DataError, match="Invalid symbol"):
        history("BAD", start="2025-01-01", end="2025-01-02", provider=provider)
    provider.http.get = lambda *a: [kline(0)] * 1000
    with pytest.raises(DataError, match="did not advance"):
        history("BTCUSDT", start="2025-01-01", end="2025-01-02", provider=provider)


def chart(ticks):
    return {"result": {"status": "ok", "ticks": ticks,
                        **{k: [10] * len(ticks) for k in [*OHLCV, "cost"]}}}


def test_deribit_paginates_short_pages_and_preserves_cost():
    provider = Deribit()
    calls = []
    def get(path, params):
        calls.append(params)
        # Simulate a server response capped below the requested window.
        return chart([params["start_timestamp"]])
    provider.http.get = get
    df = history("BTC-PERPETUAL", start="2025-01-01", end="2025-01-01T03:00Z", interval="1h", provider=provider)
    assert len(df) == len(calls) == 3
    assert calls[1]["start_timestamp"] - calls[0]["start_timestamp"] == 3600000
    assert calls[0]["resolution"] == "60" and "cost_volume" in df


def test_deribit_empty_window_and_bad_arrays():
    provider = Deribit()
    calls = []
    def get(path, params):
        calls.append(params)
        return {"result": {"status": "no_data"}}
    provider.http.get = get
    df = history("BTC-PERPETUAL", start="2025-01-01", end="2025-01-03", interval="1m", provider=provider, allow_empty=True)
    assert df.empty and len(calls) == 3
    payload = chart([1735689600000])
    payload["result"]["close"] = []
    provider.http.get = lambda *a: payload
    with pytest.raises(DataError, match="unequal"):
        history("BTC-PERPETUAL", start="2025-01-01", end="2025-01-02", provider=provider)


def test_yahoo_adjustments_actions_and_daily_timezone(monkeypatch):
    calls = []
    def read(**kwargs):
        calls.append(kwargs)
        df = candles(["2025-01-02"])
        df.index = pd.DatetimeIndex(["2025-01-02"], tz="Asia/Tokyo")
        df.columns = [c.title() for c in df]
        df["Adj Close"] = 10.5
        df["Stock Splits"] = 0
        return df
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda s: SimpleNamespace(history=read)))
    df = history("7203.T", start="2025-01-02", end="2025-01-03", provider="yahoo")
    assert df.index[0] == pd.Timestamp("2025-01-02", tz="UTC")
    assert df.adj_close.iloc[0] == 10.5 and "stock_splits" in df
    assert calls[0]["auto_adjust"] is False and calls[0]["raise_errors"] is True
    assert calls[0]["start"] == "2025-01-02"


def test_yahoo_failure_is_not_empty(monkeypatch):
    def read(**kwargs):
        raise OSError("rate limited")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda s: SimpleNamespace(history=read)))
    with pytest.raises(DataError, match="rate limited"):
        history("SPY", start="2025-01-01", end="2025-01-02", allow_empty=True)


class FakeIB:
    def __init__(self):
        self.disconnected = False
        self.requests = []
        self.resolved = SimpleNamespace(conId=123, secType="STK", currency="USD", exchange="SMART")

    async def connectAsync(self, *args, **kwargs):
        self.connection = kwargs

    async def reqContractDetailsAsync(self, contract):
        self.contract = contract
        return [SimpleNamespace(contract=self.resolved)]

    async def reqHistoricalDataAsync(self, contract, **kwargs):
        self.requests.append(kwargs)
        return [SimpleNamespace(date=date(2025, 1, 2), open=10, high=12, low=9, close=11, volume=100)]

    def isConnected(self):
        return True

    def disconnect(self):
        self.disconnected = True


def fake_ib_module(monkeypatch, client):
    monkeypatch.setitem(sys.modules, "ib_async", SimpleNamespace(
        IB=lambda: client, StartupFetchNONE=0,
        Stock=lambda s, e, c: SimpleNamespace(symbol=s, exchange=e, currency=c)))


def test_ibkr_readonly_chunks_contract_and_cleanup(monkeypatch):
    client = FakeIB()
    fake_ib_module(monkeypatch, client)
    custom = SimpleNamespace(exchange="CME")
    df = history("ES", start="2025-01-01", end="2025-03-05", provider="ibkr",
                 contracts={"ES": custom}, request_interval=0)
    assert client.contract is custom
    assert client.connection["readonly"] is True and client.connection["fetchFields"] == 0
    assert len(client.requests) == 1 and client.disconnected
    assert all(r["timeout"] == 0 and r["formatDate"] == 2 for r in client.requests)
    assert len(df) == 1 and df.attrs["sources"]["ES"]["contract_id"] == 123


def test_ibkr_timeout_raises_and_disconnects(monkeypatch):
    client = FakeIB()
    async def hang(*args, **kwargs):
        await asyncio.sleep(10)
    client.reqHistoricalDataAsync = hang
    fake_ib_module(monkeypatch, client)
    with pytest.raises(DataError, match="TimeoutError"):
        history("SPY", start="2025-01-01", end="2025-01-03", provider="ibkr", timeout=0.01)
    assert client.disconnected


def test_ibkr_sync_call_inside_notebook_loop(monkeypatch):
    client = FakeIB()
    fake_ib_module(monkeypatch, client)
    async def notebook():
        return history("SPY", start="2025-01-01", end="2025-01-03", provider="ibkr")
    assert len(asyncio.run(notebook())) == 1
    assert client.disconnected


def test_ibkr_intraday_pacing_utc_and_missing_volume(monkeypatch):
    client = FakeIB()
    pauses = []
    async def read(contract, **kwargs):
        client.requests.append(kwargs)
        return [SimpleNamespace(date=kwargs["endDateTime"] - pd.Timedelta(seconds=1),
                                open=10, high=12, low=9, close=11, volume=-1)]
    async def sleep(seconds):
        pauses.append(seconds)
    client.reqHistoricalDataAsync = read
    fake_ib_module(monkeypatch, client)
    monkeypatch.setattr("convexity.data.ibkr.asyncio.sleep", sleep)
    df = history("SPY", start="2025-01-02T15:00Z", end="2025-01-02T16:00Z",
                 provider="ibkr", interval="1s", request_interval=0)
    assert len(df) == 2 and df.volume.isna().all()
    assert pauses == [10.1]
    assert all(r["durationStr"] == "1800 S" for r in client.requests)


def test_ibkr_ambiguous_contract_disconnects_without_history(monkeypatch):
    client = FakeIB()
    async def ambiguous(contract):
        return [SimpleNamespace(contract=client.resolved)] * 2
    client.reqContractDetailsAsync = ambiguous
    fake_ib_module(monkeypatch, client)
    with pytest.raises(DataError, match="uniquely resolve"):
        history("SPY", start="2025-01-01", end="2025-01-03", provider="ibkr")
    assert client.disconnected and not client.requests


def test_ibkr_four_year_hourly_query_uses_36_requests_without_losing_bars(monkeypatch):
    client = FakeIB()
    pauses = []
    async def read(contract, **kwargs):
        client.requests.append(kwargs)
        amount, unit = kwargs["durationStr"].split()
        end = pd.Timestamp(kwargs["endDateTime"])
        begin = end - pd.Timedelta(**{"days" if unit == "D" else "seconds": int(amount)})
        return [SimpleNamespace(date=t, open=10, high=12, low=9, close=11, volume=100)
                for t in pd.date_range(begin, end, freq="h", inclusive="left")]
    async def sleep(seconds):
        pauses.append(seconds)
    client.reqHistoricalDataAsync = read
    fake_ib_module(monkeypatch, client)
    monkeypatch.setattr("convexity.data.ibkr.asyncio.sleep", sleep)
    df = history(["SPY", "QQQ"], start="2020-01-01", end="2024-01-01", interval="1h", provider="ibkr")
    assert len(client.requests) == 36
    assert pauses == [0.35] * 34
    expected = pd.date_range("2020-01-01", "2024-01-01", freq="h", inclusive="left", tz="UTC", name="timestamp")
    for symbol, frame in split_history(df).items():
        pd.testing.assert_index_equal(frame.index, expected)
        assert frame.attrs["sources"][symbol]["request_count"] == 18
        assert frame.attrs["sources"][symbol]["elapsed_seconds"] >= 0


@pytest.mark.parametrize("interval,seconds", [
    ("1m", 60), ("2m", 120), ("3m", 180), ("5m", 300), ("10m", 600),
    ("15m", 900), ("20m", 1200), ("30m", 1800), ("1h", 3600),
    ("2h", 7200), ("3h", 10800), ("4h", 14400), ("8h", 28800), ("1d", 86400),
])
def test_ibkr_larger_windows_respect_duration_and_bar_budget(interval, seconds):
    from convexity.data.ibkr import _chunks
    _, window = IBKR.intervals[interval]
    assert window <= 365 * 86400 and window / seconds <= 2000
    start, end = pd.Timestamp("2020-02-29T17:23:11Z"), pd.Timestamp("2024-03-11T19:00:00Z")
    chunks = list(_chunks(start, end, window))
    assert chunks[0][1] == end and chunks[-1][0] == start
    for (lower, upper, duration), next_chunk in zip(chunks, [*chunks[1:], None]):
        if next_chunk:
            assert lower == next_chunk[1]
        n, unit = duration.split()
        requested = int(n) * (86400 if unit == "D" else 1)
        assert 0 < (upper - lower).total_seconds() <= requested <= window
        assert requested / seconds <= 2000


def test_ibkr_final_chunk_overfetch_is_clipped(monkeypatch):
    client = FakeIB()
    async def read(contract, **kwargs):
        client.requests.append(kwargs)
        return [SimpleNamespace(date=pd.Timestamp(t), open=10, high=12, low=9, close=11, volume=100)
                for t in ["2024-03-01T14:00Z", "2024-03-01T15:00Z", "2024-03-02T14:00Z", "2024-03-02T15:00Z"]]
    client.reqHistoricalDataAsync = read
    fake_ib_module(monkeypatch, client)
    df = history("SPY", start="2024-03-01T14:30Z", end="2024-03-02T15:00Z", interval="1h", provider="ibkr")
    assert client.requests[0]["durationStr"] == "2 D"
    assert df.index.tolist() == [pd.Timestamp("2024-03-01T15:00Z"), pd.Timestamp("2024-03-02T14:00Z")]


def test_ibkr_small_bar_bid_ask_pacing_counts_double(monkeypatch):
    client = FakeIB()
    pauses = []
    async def read(contract, **kwargs):
        client.requests.append(kwargs)
        return []
    async def sleep(seconds):
        pauses.append(seconds)
    client.reqHistoricalDataAsync = read
    fake_ib_module(monkeypatch, client)
    monkeypatch.setattr("convexity.data.ibkr.asyncio.sleep", sleep)
    history("SPY", start="2025-01-02T15:00Z", end="2025-01-02T16:00Z", interval="1s",
            provider="ibkr", what_to_show="BID_ASK", request_interval=0, allow_empty=True)
    assert pauses == [20.1]


@pytest.mark.parametrize("provider", [Binance(), Deribit(), YahooFinance(), IBKR()])
def test_unsupported_interval_fails_before_network(provider):
    with pytest.raises(ValueError, match="intervals"):
        history("SPY", start="2025-01-01", end="2025-01-03", interval="7m", provider=provider)


def test_http_retry_and_permanent_error(monkeypatch):
    from convexity.data import _http
    calls = []
    monkeypatch.setattr(_http.time, "sleep", lambda s: None)
    def open_request(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "rate limited", {"Retry-After": "0"}, io.BytesIO(b"limited"))
        return io.BytesIO(json.dumps({"ok": True}).encode())
    monkeypatch.setattr(_http, "urlopen", open_request)
    assert JSONClient("https://example.com").get("/history", {"symbol": "A&B"}) == {"ok": True}
    assert len(calls) == 2 and "A%26B" in calls[0].full_url
    def denied(request, timeout):
        raise HTTPError(request.full_url, 403, "forbidden", {}, io.BytesIO(b"region unavailable"))
    monkeypatch.setattr(_http, "urlopen", denied)
    with pytest.raises(DataError, match="403.*region unavailable"):
        JSONClient("https://example.com").get("/history", {})


def test_import_independent_of_runtime_backtest_and_optional_providers():
    result = subprocess.run([sys.executable, "-c", """
import sys
from convexity import data
assert data.DataClient
assert not any(m in sys.modules for m in ('convexity.runtime', 'convexity.backtest', 'convexity.brokers', 'yfinance', 'ib_async'))
"""], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
