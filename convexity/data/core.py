"""Provider-independent query and DataFrame contract."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from datetime import date, datetime
from typing import Literal, Protocol, cast, overload

import numpy as np

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "Install data dependencies: pip install 'convexity[data]'"
    ) from exc

DateLike = str | date | datetime | pd.Timestamp
OHLCV = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """A provider request or response failed; partial results are not returned."""


class NoDataError(DataError):
    """A requested symbol has no observations in the requested range."""


class DataProvider(Protocol):
    """Implement read() to add a source without changing the query API.

    Return a DatetimeIndex frame with lowercase OHLCV columns. Put provenance
    and unit information in frame.attrs. Daily date labels use midnight UTC.
    """

    name: str

    def read(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str
    ) -> pd.DataFrame: ...


def timestamp(value: DateLike) -> pd.Timestamp:
    if not isinstance(value, (str, date, datetime, pd.Timestamp)):
        raise TypeError(
            "Dates must be ISO strings, date, datetime, or Timestamp objects."
        )
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Dates cannot be NaT.")
    return cast(
        pd.Timestamp,
        result.tz_localize("UTC")
        if result.tzinfo is None
        else result.tz_convert("UTC"),
    )


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in OHLCV},
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )


def normalize(
    frame: pd.DataFrame, symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not isinstance(
        frame.index, pd.DatetimeIndex
    ):
        raise DataError("Provider must return a DataFrame with a DatetimeIndex.")
    if not frame.columns.is_unique or not set(OHLCV).issubset(frame.columns):
        raise DataError(
            "Provider response is missing OHLCV columns or has duplicate columns."
        )
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).rename("timestamp")
    if frame.index.hasnans:
        raise DataError("Provider returned missing timestamps.")
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    for column in frame.columns:
        try:
            frame[column] = cast(
                pd.Series, pd.to_numeric(frame[column], errors="raise")
            ).astype("float64")
        except (TypeError, ValueError) as exc:
            raise DataError(f"Provider returned nonnumeric {column} values.") from exc
    if np.isinf(frame.to_numpy()).any():
        raise DataError("Provider returned infinite values.")
    # Exact overlap is harmless. Conflicting overlap must not select a price silently.
    frame = frame.reset_index().drop_duplicates().set_index("timestamp")
    if frame.index.has_duplicates:
        raise DataError(f"Provider returned conflicting observations for {symbol}.")
    frame.insert(0, "symbol", symbol)
    return frame.sort_index()


def split_history(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separate an existing history frame without downloading again.

    Returns independent frames keyed by symbol, in original request order when
    history metadata is available, otherwise first-observed order. Each frame
    retains its symbol column and only that symbol's source metadata. Empty
    requested symbols are retained when allow_empty=True was used.
    """
    if "symbol" not in frame.columns:
        raise ValueError(
            "split_history requires a history DataFrame with a symbol column."
        )
    observed = frame["symbol"].drop_duplicates().tolist()
    if any(not isinstance(symbol, str) for symbol in observed):
        raise ValueError("History symbol labels must be strings.")
    requested = frame.attrs.get("symbols", observed)
    names = list(dict.fromkeys([*requested, *observed]))
    result = {}
    for symbol in names:
        part = frame.loc[frame["symbol"] == symbol].copy()
        part.attrs = deepcopy(frame.attrs)
        part.attrs["symbols"] = [symbol]
        sources = part.attrs.get("sources", {})
        part.attrs["sources"] = {symbol: sources[symbol]} if symbol in sources else {}
        result[symbol] = part
    return result


def _provider(provider: str | DataProvider, options: dict) -> DataProvider:
    if not isinstance(provider, str):
        if options:
            raise TypeError("Configure options on the provider instance itself.")
        return provider
    from .crypto import Binance, Deribit
    from .ibkr import IBKR
    from .yahoo import YahooFinance

    providers = {
        "yahoo": YahooFinance,
        "binance": Binance,
        "deribit": Deribit,
        "ibkr": IBKR,
    }
    if provider.lower() not in providers:
        raise ValueError(
            f"Unknown provider {provider!r}; choose {', '.join(providers)}."
        )
    return providers[provider.lower()](**options)


class DataClient:
    """Reusable provider configuration; query combined or per-symbol DataFrames."""

    def __init__(self, provider: str | DataProvider = "yahoo", **provider_options):
        self.provider = _provider(provider, provider_options)

    @overload
    def history(
        self,
        symbols: str | Sequence[str],
        *,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
        allow_empty: bool = False,
        split: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @overload
    def history(
        self,
        symbols: str | Sequence[str],
        *,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
        allow_empty: bool = False,
        split: Literal[True],
    ) -> tuple[pd.DataFrame, ...]: ...

    @overload
    def history(
        self,
        symbols: str | Sequence[str],
        *,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
        allow_empty: bool = False,
        split: bool,
    ) -> pd.DataFrame | tuple[pd.DataFrame, ...]: ...

    def history(
        self,
        symbols: str | Sequence[str],
        *,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
        allow_empty: bool = False,
        split: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
        """Read bars in [start, end), sorted by UTC timestamp then symbol.

        Naive boundaries mean UTC. Daily timestamps are session-date labels,
        not close times. Missing observations are never filled. Provider limits
        and unavailable history still apply; this is not a completeness check.
        Set split=True for a tuple of independent DataFrames in request order,
        including a one-element tuple for a single symbol.
        """
        if not isinstance(split, bool):
            raise TypeError("split must be True or False.")
        names = [symbols] if isinstance(symbols, str) else list(symbols)
        if not names or any(
            not isinstance(s, str) or not s.strip() or s != s.strip() for s in names
        ):
            raise ValueError(
                "symbols must contain nonempty symbols without surrounding whitespace."
            )
        if len(set(names)) != len(names):
            raise ValueError("symbols must be unique.")

        first, stop = timestamp(start), timestamp(end)
        if first >= stop:
            raise ValueError("start must precede the exclusive end.")
        if not isinstance(interval, str) or not interval:
            raise ValueError(
                "interval must be a nonempty string, for example '1d' or '1h'."
            )

        frames, metadata = [], {}
        for symbol in names:
            try:
                raw = self.provider.read(symbol, first, stop, interval)
                metadata[symbol] = dict(raw.attrs)
                frame = normalize(raw, symbol, first, stop)
            except DataError, ImportError, ValueError:
                raise
            except Exception as exc:
                raise DataError(
                    f"{self.provider.name} history failed for {symbol}: {exc}"
                ) from exc
            if frame.empty and not allow_empty:
                raise NoDataError(
                    f"{self.provider.name} returned no data for {symbol} in [{first}, {stop})."
                )
            frame.attrs = {}
            frames.append(frame)

        result = (
            pd.concat(frames)
            .reset_index()
            .sort_values(["timestamp", "symbol"])
            .set_index("timestamp")
        )
        result = result.reindex(
            columns=[
                "symbol",
                *OHLCV,
                *[c for c in result if c not in ["symbol", *OHLCV]],
            ]
        )
        result.attrs = {
            "provider": self.provider.name,
            "symbols": names,
            "interval": interval,
            "start": first.isoformat(),
            "end": stop.isoformat(),
            "end_exclusive": True,
            "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "sources": metadata,
        }
        return tuple(split_history(result).values()) if split else result

    @overload
    async def ahistory(
        self,
        symbols: str | Sequence[str],
        *,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
        allow_empty: bool = False,
        split: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @overload
    async def ahistory(
        self,
        symbols: str | Sequence[str],
        *,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
        allow_empty: bool = False,
        split: Literal[True],
    ) -> tuple[pd.DataFrame, ...]: ...

    @overload
    async def ahistory(
        self,
        symbols: str | Sequence[str],
        *,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
        allow_empty: bool = False,
        split: bool,
    ) -> pd.DataFrame | tuple[pd.DataFrame, ...]: ...

    async def ahistory(
        self,
        symbols: str | Sequence[str],
        *,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
        allow_empty: bool = False,
        split: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
        """Run blocking acquisition in a worker; useful in notebooks/async apps.

        Cancelling the await does not stop an in-flight blocking request.
        """
        return await asyncio.to_thread(
            self.history,
            symbols,
            start=start,
            end=end,
            interval=interval,
            allow_empty=allow_empty,
            split=split,
        )


@overload
def history(
    symbols: str | Sequence[str],
    *,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    provider: str | DataProvider = "yahoo",
    allow_empty: bool = False,
    split: Literal[False] = False,
    **provider_options,
) -> pd.DataFrame: ...


@overload
def history(
    symbols: str | Sequence[str],
    *,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    provider: str | DataProvider = "yahoo",
    allow_empty: bool = False,
    split: Literal[True],
    **provider_options,
) -> tuple[pd.DataFrame, ...]: ...


@overload
def history(
    symbols: str | Sequence[str],
    *,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    provider: str | DataProvider = "yahoo",
    allow_empty: bool = False,
    split: bool,
    **provider_options,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]: ...


def history(
    symbols: str | Sequence[str],
    *,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    provider: str | DataProvider = "yahoo",
    allow_empty: bool = False,
    split: bool = False,
    **provider_options,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    """One-call historical OHLCV query. See DataClient.history for the schema."""
    return DataClient(provider, **provider_options).history(
        symbols,
        start=start,
        end=end,
        interval=interval,
        allow_empty=allow_empty,
        split=split,
    )


@overload
async def ahistory(
    symbols: str | Sequence[str],
    *,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    provider: str | DataProvider = "yahoo",
    allow_empty: bool = False,
    split: Literal[False] = False,
    **provider_options,
) -> pd.DataFrame: ...


@overload
async def ahistory(
    symbols: str | Sequence[str],
    *,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    provider: str | DataProvider = "yahoo",
    allow_empty: bool = False,
    split: Literal[True],
    **provider_options,
) -> tuple[pd.DataFrame, ...]: ...


@overload
async def ahistory(
    symbols: str | Sequence[str],
    *,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    provider: str | DataProvider = "yahoo",
    allow_empty: bool = False,
    split: bool,
    **provider_options,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]: ...


async def ahistory(
    symbols: str | Sequence[str],
    *,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    provider: str | DataProvider = "yahoo",
    allow_empty: bool = False,
    split: bool = False,
    **provider_options,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    """Asynchronous counterpart to history(), with the same result and options."""
    return await DataClient(provider, **provider_options).ahistory(
        symbols,
        start=start,
        end=end,
        interval=interval,
        allow_empty=allow_empty,
        split=split,
    )
