"""Realized variance from supplied intraperiod returns; no data acquisition."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray
from ...forecasting import _validation as v


def realized_variance(returns: ArrayLike, *, axis: int = -1) -> FloatArray:
    """Sum squared intraperiod returns along axis, without demeaning or annualizing.

    For (period,intraperiod) returns the default returns one variance per period.
    Sampling intervals/session boundaries are caller-supplied; NaNs are rejected.
    """
    x = v.array(returns, "returns")
    if x.ndim == 0:
        raise ValueError("returns must have an intraperiod axis")
    if (
        isinstance(axis, (bool, np.bool_))
        or not isinstance(axis, (int, np.integer))
        or not -x.ndim <= axis < x.ndim
    ):
        raise ValueError("axis must identify an intraperiod dimension")
    return v.array(np.sum(x**2, axis=axis), "realized variance")


def realized_volatility(returns: ArrayLike, *, axis: int = -1) -> FloatArray:
    """Square root of realized_variance in the supplied return units."""
    return v.array(
        np.sqrt(realized_variance(returns, axis=axis)), "realized volatility"
    )
