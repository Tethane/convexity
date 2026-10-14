"""Replay daily DataFrames acquired through ``convexity.data``."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from convexity.runtime import Bar
from convexity.runtime.types import decimal

from .data import HistoricalData, SessionCalendar
from .history import HistoricalDataset, HistoryError, HistoryRequest, validate_coverage


def daily_bars(
    frame, calendar: SessionCalendar, *, price_basis: str
) -> tuple[Bar, ...]:
    import numpy as np
    import pandas as pd

    required = {"symbol", "open", "high", "low", "close", "volume"}
    if not isinstance(frame, pd.DataFrame) or not isinstance(
        frame.index, pd.DatetimeIndex
    ):
        raise TypeError("Supply a daily history DataFrame with a DatetimeIndex.")
    if not frame.columns.is_unique or not required.issubset(frame.columns):
        raise ValueError("Daily history requires unique symbol and OHLCV columns.")
    if frame.empty or frame.index.hasnans:
        raise ValueError("Daily history must contain dated observations.")
    if frame.attrs.get("interval", "1d") != "1d":
        raise ValueError("Only daily (1d) history can be mapped to exchange sessions.")
    if frame.index.tz is not None and str(frame.index.tz) != "UTC":
        raise ValueError("Daily session labels must be timezone-naive or midnight UTC.")
    labels = frame.index.to_numpy(dtype="datetime64[ns]")
    if not (labels == labels.astype("datetime64[D]")).all():
        raise ValueError("Daily history must use midnight session-date labels.")

    # DataClient nests per-symbol metadata; adapter.read() returns it directly.
    sources = [frame.attrs, *frame.attrs.get("sources", {}).values()]
    for source in sources:
        if source.get("timestamp_kind", "session_date") != "session_date":
            raise ValueError("Daily history must be labeled by session date.")

        currency = source.get("currency")
        if currency is not None and currency != "USD":
            raise ValueError(
                f"Daily DataFrame replay supports USD stocks; source currency is {currency!r}."
            )
        security_type = source.get("security_type")
        if security_type is not None and security_type != "STK":
            raise ValueError(
                "Daily DataFrame replay supports stocks (STK); "
                f"source security_type is {security_type!r}."
            )

        basis = source.get("price_basis")
        if basis in ("dividend_and_split_adjusted", "ibkr_adjusted_last"):
            raise ValueError(
                "Dividend-adjusted prices cannot be used as execution prices."
            )
        if (
            basis in ("yahoo_split_adjusted", "ibkr_trades")
            and price_basis != "split_adjusted_price_return"
        ):
            raise ValueError(
                f"Source basis {basis!r} requires price_basis='split_adjusted_price_return'."
            )

    sessions = {session.label: session for session in calendar.sessions}
    bars = []
    columns = ["symbol", "open", "high", "low", "close", "volume"]
    for stamp, symbol, opening, high, low, close, amount in frame[columns].itertuples(
        index=True, name=None
    ):
        session = sessions.get(stamp.date())
        if session is None:
            raise HistoryError(
                f"No exchange session for daily observation {stamp.date()}."
            )
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Daily observations require nonempty symbol labels.")
        volume = float(amount)
        if not np.isfinite(volume) or volume < 0 or not volume.is_integer():
            raise ValueError(
                "Daily stock volume must be a nonnegative whole number of shares."
            )

        bars.append(
            Bar(
                symbol,
                session.open,
                session.close,
                decimal(opening),
                decimal(high),
                decimal(low),
                decimal(close),
                int(volume),
            )
        )

    data = HistoricalData(tuple(bars), price_basis=price_basis)
    request = HistoryRequest(
        tuple(sorted(set(frame["symbol"]))),
        calendar.sessions[0].label,
        calendar.sessions[-1].label,
    )
    validate_coverage(data, calendar, request)
    return data.bars


class DataFrameHistoricalProvider:
    """Replay an owned daily USD-stock snapshot without downloading.

    The frame must cover the trading period and declared warmup. Its original
    provenance is retained in the result. Observations are published at the next
    session open; execution uses the supplied daily OHLCV with zero latency.
    This is price-return research, unless separate corporate actions are supplied.
    Known source metadata must agree with the stock/currency/adjustment contract.
    When metadata is absent, the caller is responsible for those assumptions.
    """

    def __init__(self, frame, *, price_basis: str, calendar: str = "XNYS"):
        import pandas as pd

        if not isinstance(frame, pd.DataFrame) or not isinstance(
            frame.index, pd.DatetimeIndex
        ):
            raise TypeError("Supply a daily history DataFrame with a DatetimeIndex.")
        self._frame = frame.copy(deep=True)
        self._frame.attrs = deepcopy(frame.attrs)
        self.price_basis = price_basis
        self.calendar = calendar
        self.source = f"{frame.attrs.get('provider', 'dataframe')}:native-daily"
        content = self._frame.to_csv().encode()
        metadata = json.dumps(frame.attrs, sort_keys=True, default=str).encode()
        self.cache_identity = (
            f"{self.source}:{calendar}:{hashlib.sha256(content + metadata).hexdigest()}"
        )

    async def fetch(
        self, request: HistoryRequest, *, refresh: bool = False
    ) -> HistoricalDataset:
        import numpy as np

        if refresh:
            raise ValueError(
                "A DataFrame snapshot cannot refresh; fetch new history and replace the provider."
            )
        if request.bar_size != "1 day" or request.calendar != self.calendar:
            raise HistoryError(
                "DataFrame history supports daily requests on its declared calendar only."
            )

        dates = self._frame.index.to_numpy(dtype="datetime64[D]")
        selected = self._frame.loc[
            self._frame["symbol"].isin(request.symbols)
            & (dates >= np.datetime64(request.start))
            & (dates <= np.datetime64(request.end))
        ]
        calendar = SessionCalendar.from_exchange(
            request.calendar, request.start, request.end
        )
        data = HistoricalData.from_frame(
            selected, calendar, price_basis=self.price_basis
        )
        validate_coverage(data, calendar, request)

        manifest = {
            "source": self.source,
            "request": request.to_dict(),
            "aggregation": "native_daily",
            "availability": "next_session_open",
            "price_basis": self.price_basis,
            "provenance": deepcopy(self._frame.attrs),
            "data_identity": data.identity,
        }
        return HistoricalDataset(data, calendar, manifest, Path("."))
