from decimal import Decimal
from datetime import datetime, timedelta, timezone

import pytest

from convexity.backtest import Backtest, HistoricalData, Session, SessionCalendar
from convexity.runtime import Bar, Strategy

T = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)


def bar(symbol, offset):
    start = T + timedelta(days=offset)
    return Bar(
        symbol,
        start,
        start + timedelta(hours=6),
        Decimal(10),
        Decimal(10),
        Decimal(10),
        Decimal(10),
        1000,
    )


def test_sparse_is_explicit_and_retains_missing_rows_without_interpolation():
    rows = (bar("A", 0), bar("B", 0), bar("A", 2))
    with pytest.raises(ValueError, match="Incomplete bar panel"):
        HistoricalData(rows)
    data = HistoricalData(rows, sparse=True)
    assert len(data.bars) == 3
    assert set(data.panels()[-1].bars) == {"A"}
    sessions = SessionCalendar(
        tuple(
            Session(
                (T + timedelta(days=i)).date(),
                T + timedelta(days=i),
                T + timedelta(days=i, hours=6),
            )
            for i in range(4)
        )
    )
    result = Backtest(
        data=data,
        strategies={"idle": Strategy},
        calendar=sessions,
        start=T,
        end=T + timedelta(days=3, hours=6),
    ).run()
    assert result.status == "completed"
    assert not result.fills
    assert result.metrics["final_equity"] == 100000


def test_native_daily_cannot_publish_same_close_through_low_level_backtest():
    data = HistoricalData((bar("A", 0),), source="Alpaca:SIP:split:native-daily:v1")
    with pytest.raises(ValueError, match="deferred"):
        Backtest(data=data, strategies={"idle": Strategy})
