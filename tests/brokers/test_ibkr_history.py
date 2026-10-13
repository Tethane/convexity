"""Long strategy warmups use bounded one-minute history requests."""
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import cast

from convexity.brokers import ibkr_data
from convexity.brokers.ibkr_trading import PaperBroker


def test_long_warmup_chunks_history_and_merges_ordered_overlaps(monkeypatch):
    stamp = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(days=2)
    def row(time):
        return SimpleNamespace(date=time, open=100, high=100, low=100, close=100, volume=100)
    calls = []
    async def detail(broker, symbol):
        return SimpleNamespace(contract='contract')
    async def history(contract, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return [row(stamp+timedelta(days=1)), row(datetime.now(timezone.utc))]
        if len(calls) == 2:
            return []  # An empty nontrading slice must not terminate the request.
        return [row(stamp), row(stamp+timedelta(days=1))]
    monkeypatch.setattr(ibkr_data, '_stock_detail', detail)
    broker = cast(PaperBroker, SimpleNamespace(config=SimpleNamespace(timeout=10),
                                              _ib=SimpleNamespace(reqHistoricalDataAsync=history)))
    data = asyncio.run(ibkr_data.read_stock_history(broker, 'A', seconds=2*86400+1800))
    assert [call['durationStr'] for call in calls] == ['86400 S', '86400 S', '1800 S']
    assert calls[0]['endDateTime'] == ''
    assert calls[1]['endDateTime'] > calls[2]['endDateTime']
    assert len(data.bars) == 2
    assert data.bars[0].start == stamp
    assert data.bars[0].close == Decimal(100)
