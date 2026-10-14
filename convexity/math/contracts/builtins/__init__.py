"""Concrete specifications and convenience functions for contracts.

Ordinary workflows may import these names directly from convexity.math.contracts.
"""

from .cashflows import (
    payoff,
    schedule,
    year_fraction,
    interest_factor,
    fixed_cashflows,
    present_value,
    split_holdings,
    dividend_adjustment,
)
from .conventions import evaluation_date, day_count, to_quantlib_date
from .conventions import to_date, schedule_dates

__all__ = [
    "evaluation_date",
    "day_count",
    "to_quantlib_date",
    "payoff",
    "schedule",
    "year_fraction",
    "interest_factor",
    "fixed_cashflows",
    "present_value",
    "split_holdings",
    "dividend_adjustment",
    "to_date",
    "schedule_dates",
]
