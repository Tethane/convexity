"""Owned account snapshots in one valuation currency and marked asset units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike

from ... import _decision as v
from ..._typing import FloatArray


@dataclass(frozen=True)
class Account:
    """Wealth/account state under explicit holdings, cash and valuation conventions."""

    holdings: FloatArray
    cash: float = 0.0
    collateral: float = 0.0
    pending_holdings: FloatArray = cast("FloatArray", None)
    pending_cash: float = 0.0

    def __post_init__(self) -> None:
        h = v.array(self.holdings, "holdings", ndim=1)
        p = (
            np.zeros_like(h)
            if self.pending_holdings is None
            else v.array(self.pending_holdings, "pending holdings", ndim=1)
        )
        if p.shape != h.shape:
            raise ValueError("pending holdings must align with settled holdings")
        for name in ("cash", "collateral", "pending_cash"):
            v.scalar(getattr(self, name), name)
        if self.collateral < 0:
            raise ValueError("posted collateral must be nonnegative")
        object.__setattr__(self, "holdings", h)
        object.__setattr__(self, "pending_holdings", v.array(p, "pending holdings"))

    def wealth(self, marks: ArrayLike) -> float:
        x = v.array(marks, "marks", ndim=1)
        if x.shape != self.holdings.shape:
            raise ValueError("marks must align with holdings")
        return float(
            self.cash
            + self.collateral
            + self.pending_cash
            + (self.holdings + self.pending_holdings) @ x
        )

    def __init__(
        self,
        holdings: ArrayLike,
        cash: float = 0.0,
        collateral: float = 0.0,
        pending_holdings: ArrayLike | None = None,
        pending_cash: float = 0.0,
    ) -> None:
        object.__setattr__(self, "holdings", holdings)
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "collateral", collateral)
        object.__setattr__(self, "pending_holdings", pending_holdings)
        object.__setattr__(self, "pending_cash", pending_cash)
        self.__post_init__()


@dataclass(frozen=True)
class Attribution:
    """Decomposition of wealth change into the components declared by the attribution calculation."""

    exact: float
    first_order: float
    second_order: float
    residual: float

    def __post_init__(self) -> None:
        for name in ("exact", "first_order", "second_order", "residual"):
            v.scalar(getattr(self, name), name)


class WealthMap(Protocol):
    """Transform a supplied account state under explicitly modeled actions and market outcomes."""

    def __call__(self, outcome: ArrayLike) -> float: ...
