"""Discount curve structural contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..._finance_types import NativeDate
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from ..._typing import DateLike
from ...calibration import CalibrationResult


@dataclass(frozen=True)
class CurveFitResult:
    """Owned discount-model fit; cashflow times are years from reference_date.

    Parameters belong to the caller's supplied discount model. Diagnostics
    describe a local least-squares solution, not a global optimum certificate.
    """

    reference_date: date
    diagnostics: CalibrationResult

    @property
    def parameters(self):
        return self.diagnostics.parameters

    @property
    def success(self):
        return self.diagnostics.success

    @property
    def x(self):
        """Compatibility alias; prefer parameters."""
        return self.parameters


class DiscountCurve(Protocol):
    """Discount-factor term structure measured from its reference date under an explicit day-count convention."""

    @property
    def reference_date(self) -> DateLike | NativeDate: ...
    def time(self, maturity: float | DateLike | NativeDate) -> float: ...
    def discount(self, maturity: float | DateLike | NativeDate) -> float: ...
    def zero_rate(self, maturity: float | DateLike | NativeDate) -> float: ...
    def forward_rate(
        self, start: float | DateLike | NativeDate, end: float | DateLike | NativeDate
    ) -> float: ...
