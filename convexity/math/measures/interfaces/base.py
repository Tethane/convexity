"""Explicit physical/pricing laws; no inferred risk-neutral conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..._finance import array, choice, positive, scalar


class Numeraire(Protocol):
    """Positive value process used as a unit of account for a declared pricing measure."""
    def value(self, time: float) -> ArrayLike: ...


@dataclass(frozen=True)
class MeasureModel[ModelT = object]:
    """Model plus explicitly declared probability measure and optional numeraire; a tag is not a measure transformation."""

    model: ModelT
    measure: str
    numeraire: Numeraire | None = None

    def __post_init__(self) -> None:
        choice(self.measure, ("P", "Q"), "measure")
        if self.measure == "Q" and self.numeraire is None:
            raise ValueError("Q models require their pricing numeraire")

    def value(self, cashflow, maturity: float, *, expectation):
        """N(0) E^Q[cashflow/N(T)], expectation supplied by the joint law."""
        if self.measure != "Q":
            raise ValueError("valuation requires a pricing measure")
        maturity = scalar(maturity, "maturity", minimum=0)
        assert self.numeraire is not None
        terminal = array(self.numeraire.value(maturity), "terminal numeraire")
        if np.any(terminal <= 0):
            raise ValueError("terminal numeraire must be strictly positive")
        return positive(self.numeraire.value(0.0), "initial numeraire") * expectation(
            array(cashflow, "cashflow") / terminal
        )
