"""Public access to the shared QuantLib date and valuation conventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..._typing import DateLike

if TYPE_CHECKING:
    from datetime import date

    from ..._finance_types import (
        NativeDate,
        NativeSchedule,
    )


from ..._finance import day_count, evaluation_date
from ..._finance import qdate as to_quantlib_date

__all__ = ["evaluation_date", "day_count", "to_quantlib_date"]


def to_date(value: DateLike | NativeDate) -> date:
    """Return a standard datetime.date from an ISO, Python or native date."""
    from .._backends.quantlib import date_value

    return date_value(value)


def schedule_dates(
    value: NativeSchedule | Sequence[DateLike | NativeDate],
) -> tuple[date, ...]:
    """Inspect a schedule as a tuple of standard Python dates."""
    return tuple(to_date(d) for d in value)


__all__ += ["to_date", "schedule_dates"]
