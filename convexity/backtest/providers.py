"""Provider selection with separate durable caches and visible fallback."""

from dataclasses import replace
from pathlib import Path
import warnings

from .alpaca import AlpacaHistoricalData
from .cache import CachedHistoricalProvider
from .history import HistoryError, IBKRHistoricalData


class FallbackHistoricalProvider:
    """Retry the entire request on IBKR; never splice different feeds together.

    Cache each provider individually, inside this selector. An Alpaca failure is
    reported as a warning and in the returned manifest. Subsequent calls try
    Alpaca again; fallback observations cannot poison Alpaca's cache namespace.
    """

    def __init__(
        self,
        primary=None,
        fallback=None,
        *,
        cache: str | Path | None = ".convexity/history.sqlite3",
    ):
        primary = primary if primary is not None else AlpacaHistoricalData()
        fallback = fallback if fallback is not None else IBKRHistoricalData()
        self.primary = (
            CachedHistoricalProvider(primary, cache) if cache is not None else primary
        )
        self.fallback = (
            CachedHistoricalProvider(fallback, cache) if cache is not None else fallback
        )

    async def fetch(self, request, *, refresh=False):
        try:
            return await self.primary.fetch(request, refresh=refresh)
        except HistoryError as exc:
            reason = str(exc)
        warnings.warn(
            f"Alpaca history unavailable: {reason} Falling back to IBKR for the whole request; "
            "uncached IBKR history may be slow.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = await self.fallback.fetch(request, refresh=refresh)
        return replace(
            result,
            manifest=dict(
                result.manifest,
                fallback=dict(preferred="Alpaca", selected="IBKR", reason=reason),
            ),
        )
