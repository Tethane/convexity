import asyncio
from datetime import date

import pandas as pd
import pytest

import convexity as cx
from convexity.backtest import DataFrameHistoricalProvider, HistoryError, HistoryRequest


@pytest.fixture
def frame():
    data = pd.DataFrame(
        {
            "symbol": ["SPY"] * 4,
            "open": [99, 100, 101, 102],
            "high": [101, 102, 103, 104],
            "low": [98, 99, 100, 101],
            "close": [100, 101, 102, 103],
            "volume": [100_000] * 4,
        },
        index=pd.to_datetime(
            ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"], utc=True
        ),
    )
    data.attrs = {
        "provider": "test-fixture",
        "interval": "1d",
        "sources": {
            "SPY": {
                "price_basis": "yahoo_split_adjusted",
                "timestamp_kind": "session_date",
            }
        },
    }
    return data


def provider(frame):
    return DataFrameHistoricalProvider(frame, price_basis="split_adjusted_price_return")


def test_fill_and_stop_callbacks_cannot_read_unpublished_close(frame):
    seen = {}

    class Inspect(cx.Strategy):
        metadata = cx.StrategyMetadata("inspect")
        contract = cx.StrategyContract(
            ("SPY",), (cx.BarSubscription(("SPY",)), cx.EventSubscription(cx.FillEvent))
        )

        def on_start(self, context):
            return cx.SubmitOrder.market("SPY", "BUY", 1)

        def on_fill(self, event, context):
            seen["fill"] = context

        def on_bar(self, event, context):
            seen[event.time.date()] = context

        def on_stop(self, context):
            seen["stop"] = context

    result = cx.Research("2024-01-08", "2024-01-10").run(
        Inspect, provider=provider(frame), cache=None, execution=cx.ExecutionModel()
    )
    filled = seen["fill"]
    assert filled.time.date() == date(2024, 1, 8)
    assert filled.price("SPY") == 100  # Jan 5 close, not Jan 8 close (101).
    assert filled.price_info("SPY").time.date() == date(2024, 1, 5)
    assert filled.equity == filled.account.equity == 100_000

    published = seen[date(2024, 1, 9)]
    assert published.price("SPY") == 101
    assert published.equity == published.account.equity == 100_001
    assert seen["stop"].price("SPY") == 102
    # Reporting still marks the terminal account with the final observed close.
    assert result.metrics["final_equity"] == 100_003


def test_daily_observations_arrive_before_next_open_orders(frame):
    seen = []

    class Trade(cx.Strategy):
        metadata = cx.StrategyMetadata("daily")
        contract = cx.StrategyContract(
            ("SPY",), (cx.BarSubscription(("SPY",)),), warmup_sessions=1
        )

        def on_start(self, context):
            assert context.history["SPY"][-1].end.date() == date(2024, 1, 5)

        def on_bar(self, event, context):
            seen.append((event.time.date(), event.bars["SPY"].end.date()))
            assert event.delayed
            if not context.position("SPY") and not context.pending("SPY"):
                return cx.SubmitOrder.market("SPY", "BUY", 1)

    result = cx.Research("2024-01-08", "2024-01-10").run(
        Trade, provider=provider(frame), cache=None, execution=cx.ExecutionModel()
    )
    assert seen == [
        (date(2024, 1, 9), date(2024, 1, 8)),
        (date(2024, 1, 10), date(2024, 1, 9)),
    ]
    assert len(result.fills) == 1
    assert result.fills[0].occurred_at.date() == date(2024, 1, 9)
    assert result.fills[0].price == 101
    assert result.config["metadata"]["research"]["history"]["provenance"] == frame.attrs


def test_snapshot_owns_prices_and_metadata_and_has_distinct_cache_identity(frame):
    original = provider(frame)
    frame.iloc[0, frame.columns.get_loc("close")] = 99
    frame.attrs["sources"]["SPY"]["note"] = "changed"
    revised = provider(frame)
    assert original.cache_identity != revised.cache_identity

    request = HistoryRequest(("SPY",), "2024-01-05", "2024-01-10")
    result = asyncio.run(original.fetch(request))
    assert result.data.bars[0].close == 100
    assert "note" not in result.manifest["provenance"]["sources"]["SPY"]


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "volume", "time", "adjustment"]
)
def test_invalid_daily_data_is_rejected(frame, problem):
    if problem == "missing":
        frame = frame.drop(frame.index[1])
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "volume":
        frame["volume"] = 1.5
    elif problem == "time":
        frame.index = frame.index + pd.Timedelta(hours=1)
    else:
        frame.attrs["sources"]["SPY"]["price_basis"] = "dividend_and_split_adjusted"

    request = HistoryRequest(("SPY",), "2024-01-05", "2024-01-10")
    with pytest.raises((ValueError, HistoryError)):
        asyncio.run(provider(frame).fetch(request))


@pytest.mark.parametrize("nested", [True, False])
def test_ibkr_split_adjusted_history_cannot_be_replayed_as_raw(frame, nested):
    source = {
        "price_basis": "ibkr_trades",
        "timestamp_kind": "session_date",
        "security_type": "STK",
        "currency": "USD",
    }
    frame.attrs = {"sources": {"SPY": source}} if nested else source
    request = HistoryRequest(("SPY",), "2024-01-05", "2024-01-10")
    raw = DataFrameHistoricalProvider(
        frame, price_basis="unadjusted_no_corporate_actions"
    )

    with pytest.raises(ValueError, match="ibkr_trades.*split_adjusted_price_return"):
        asyncio.run(raw.fetch(request))

    result = asyncio.run(provider(frame).fetch(request))
    assert result.data.price_basis == "split_adjusted_price_return"
    assert result.manifest["provenance"] == frame.attrs


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("currency", "CAD", "supports USD stocks"),
        ("security_type", "FUT", "supports stocks"),
        ("price_basis", "ibkr_adjusted_last", "Dividend-adjusted prices"),
    ],
)
def test_known_source_semantics_cannot_be_silently_reinterpreted(
    frame, field, value, message
):
    frame.attrs["sources"]["SPY"][field] = value
    request = HistoryRequest(("SPY",), "2024-01-05", "2024-01-10")

    with pytest.raises(ValueError, match=message):
        asyncio.run(provider(frame).fetch(request))
