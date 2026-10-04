from collections.abc import Iterator
"""Shared financial conventions and scoped QuantLib settings.

All rates/volatilities are decimals; times are years. Date-based calculations
use an explicit day counter. The lock serializes Convexity's QuantLib calls;
external code modifying QuantLib Settings must use this context too.
"""

from typing import Any
from contextlib import contextmanager
from datetime import date, datetime
from threading import RLock
import QuantLib as ql
from .._validation import choice

_LOCK = RLock()


def qdate(value: str | date | datetime | ql.Date) -> ql.Date:
    """Convert an ISO date, Python date/datetime or native date to QuantLib Date.
    
    Dates are calendar dates; intraday components are not retained."""
    if isinstance(value, ql.Date):
        if value == ql.Date():
            raise ValueError("date must not be null")
        return ql.Date(value.serialNumber())
    if isinstance(value, str):
        value = date.fromisoformat(value)
    if isinstance(value, (date, datetime)):
        return ql.Date(value.day, value.month, value.year)
    raise TypeError("date must be datetime.date, ISO date, or QuantLib.Date")


def day_count(name: str | ql.DayCounter = "actual365") -> ql.DayCounter:
    """Return the native QuantLib day counter for a supported convention name.
    
    This is an explicit interoperability helper; year_fraction is the scalar facade."""
    constructors = {
        "actual365": ql.Actual365Fixed,
        "actual360": ql.Actual360,
        "actual_actual": lambda: ql.ActualActual(ql.ActualActual.ISDA),
        "30_360": lambda: ql.Thirty360(ql.Thirty360.BondBasis),
    }
    if isinstance(name, ql.DayCounter):
        return name
    return constructors[choice(name, tuple(constructors), "day count")]()


@contextmanager
def evaluation_date(value: str | date | datetime | ql.Date) -> Iterator[None]:
    """Restore all QuantLib settings, including on exceptions and nested calls."""
    with _LOCK:
        with ql.SavedSettings():
            ql.Settings.instance().evaluationDate = qdate(value)
            yield


def option_type(kind):
    return (
        ql.Option.Call
        if choice(kind, ("call", "put"), "option kind") == "call"
        else ql.Option.Put
    )


def calendar(value=None):
    if isinstance(value, ql.Calendar):
        return value
    names = {
        "none": ql.NullCalendar,
        "weekends": ql.WeekendsOnly,
        "target": ql.TARGET,
        "us_settlement": lambda: ql.UnitedStates(ql.UnitedStates.Settlement),
        "us_government_bond": lambda: ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        "nyse": lambda: ql.UnitedStates(ql.UnitedStates.NYSE),
        "uk": ql.UnitedKingdom,
        "canada": ql.Canada,
    }
    return names[choice("none" if value is None else value, tuple(names), "calendar")]()


def adjustment(value=None):
    names = {
        "unadjusted": ql.Unadjusted,
        "following": ql.Following,
        "modified_following": ql.ModifiedFollowing,
        "preceding": ql.Preceding,
        "modified_preceding": ql.ModifiedPreceding,
    }
    if value is None:
        return ql.ModifiedFollowing
    if isinstance(value, str):
        return names[choice(value, tuple(names), "business-day adjustment")]
    if isinstance(value, bool) or value not in names.values():
        raise ValueError("invalid business-day adjustment")
    return value


def frequency(value=None, *, default=2):
    from .._validation import integer

    names = {"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12}
    value = default if value is None else value
    if isinstance(value, str):
        value = names[choice(value, tuple(names), "frequency")]
    value = integer(value, "frequency", 1)
    if value not in (1, 2, 3, 4, 6, 12):
        raise ValueError("frequency must be payments per year: 1, 2, 3, 4, 6 or 12")
    return value


def compounding(value=None):
    names = {
        "simple": ql.Simple,
        "continuous": ql.Continuous,
        "periodic": ql.Compounded,
    }
    if value is None:
        return ql.Compounded
    if isinstance(value, str):
        return names[choice(value, tuple(names), "compounding")]
    if isinstance(value, bool) or value not in names.values():
        raise ValueError("invalid compounding convention")
    return value


def period(value):
    import re

    if isinstance(value, ql.Period):
        if value.length() <= 0:
            raise ValueError("tenor must be positive")
        return value
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*[DWMY]", value):
        raise ValueError("tenor must be a positive D/W/M/Y period, e.g. '6M' or '5Y'")
    return ql.Period(value)


def currency(value):
    names = {
        "USD": ql.USDCurrency,
        "EUR": ql.EURCurrency,
        "GBP": ql.GBPCurrency,
        "CAD": ql.CADCurrency,
        "JPY": ql.JPYCurrency,
        "CHF": ql.CHFCurrency,
    }
    return names[choice(value, tuple(names), "currency")]()


def attach_fixings(instrument, *indices):
    # Retain Python index adapters and their owned fixing snapshots for valuation.
    instrument._convexity_indices = tuple(indices)
    if indices:
        instrument._convexity_currency = indices[0].currency().code()
    return instrument


def set_fixings(index, fixings):
    from types import MappingProxyType
    from .._validation import scalar

    if fixings is not None:
        values = {qdate(d): scalar(v, "fixing") for d, v in dict(fixings).items()}
        index._convexity_fixings = MappingProxyType(values)
    return index


@contextmanager
def fixing_context(instrument):
    """Apply owned index histories only during pricing, restoring external state."""
    indices = getattr(instrument, "_convexity_indices", ())
    with _LOCK:
        saved = []
        try:
            for index in indices:
                fixings = getattr(index, "_convexity_fixings", None)
                if fixings is None:
                    continue
                history = index.timeSeries()
                saved.append((index, list(history.dates()), list(history.values())))
                index.clearFixings()
                for d, v in fixings.items():
                    index.addFixing(d, v, True)
            yield
        finally:
            for index, dates, values in reversed(saved):
                index.clearFixings()
                if dates:
                    index.addFixings(dates, values, True)
