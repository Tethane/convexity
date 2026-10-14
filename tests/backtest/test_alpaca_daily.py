import asyncio
from email.message import Message
from urllib.error import HTTPError

import pytest

from convexity.backtest.alpaca import AlpacaCredentials
from convexity.backtest.alpaca_daily import AlpacaDailyHistoricalData
from convexity.backtest import HistoryError, HistoryRequest


def test_daily_pagination_never_downloads_minutes(monkeypatch):
    source = AlpacaDailyHistoricalData(
        AlpacaCredentials("key", "secret"), request_interval=0
    )
    calls = []
    pages = [
        dict(
            bars={
                "OLD": [
                    dict(t="2024-11-27T05:00:00Z", o=100, h=102, l=99, c=101, v=1000)
                ]
            },
            next_page_token="2",
        ),
        dict(
            bars={
                "OLD": [
                    dict(t="2024-11-29T05:00:00Z", o=101, h=103, l=100, c=102, v=2000)
                ]
            },
            next_page_token=None,
        ),
    ]

    def page(params, credentials):
        calls.append(dict(params))
        return pages.pop(0)

    monkeypatch.setattr(source, "_request_page", page)
    result = asyncio.run(
        source.fetch(HistoryRequest(("OLD",), "2024-11-27", "2024-11-29"))
    )
    assert [b.volume for b in result.data.bars] == [1000, 2000]
    assert all(p["timeframe"] == "1Day" and p["asof"] == "-" for p in calls)
    assert result.manifest["availability"] == "next_new_york_midnight"
    assert result.manifest["aggregation"] == "native_daily"
    assert source.source != "Alpaca:SIP:split:RTH:historical:v1"


def test_native_daily_rejects_intraday():
    source = AlpacaDailyHistoricalData(AlpacaCredentials("key", "secret"))
    with pytest.raises(HistoryError, match="only supports"):
        asyncio.run(
            source.fetch(HistoryRequest(("SPY",), "2024-11-27", "2024-11-29", "1 min"))
        )


def test_retry_after_is_not_shortened(monkeypatch):
    source = AlpacaDailyHistoricalData(
        AlpacaCredentials("key", "secret"), request_interval=0
    )
    calls, delays = [], []

    def page(params, credentials):
        calls.append(None)
        if len(calls) == 1:
            headers = Message()
            headers["Retry-After"] = "7"
            raise HTTPError(
                "https://example.invalid", 429, "rate limited", headers, None
            )
        return dict(
            bars={
                "SPY": [
                    dict(t="2024-11-27T05:00:00Z", o=100, h=102, l=99, c=101, v=1000)
                ]
            }
        )

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(source, "_request_page", page)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(source.fetch(HistoryRequest(("SPY",), "2024-11-27", "2024-11-27")))
    assert delays == [7]
