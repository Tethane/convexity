"""Contract conventions independent of pricing engines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import (
    DateLike,
)

if TYPE_CHECKING:
    from ..._finance_types import NativeDate

from dataclasses import dataclass
from typing import Protocol

from ..._finance import choice, option_type, positive, qdate, scalar


class Payoff(Protocol):
    """Callable terminal payoff from supplied underlying values; amounts inherit the contract's unit convention."""

    def __call__(self, value: ArrayLike) -> ArrayLike: ...


@dataclass(frozen=True)
class Exercise:
    """Exercise style and allowed dates; calendar dates are distinct from model-time year fractions."""

    style: str
    dates: tuple[NativeDate, ...]

    def __init__(self, style: str, dates: Sequence[DateLike | NativeDate]) -> None:
        object.__setattr__(self, "style", style)
        object.__setattr__(self, "dates", dates)
        self.__post_init__()

    def __post_init__(self) -> None:
        choice(self.style, ("european", "american", "bermudan"), "exercise")
        dates = tuple((qdate(d) for d in self.dates))
        if not dates or any((b <= a for a, b in zip(dates, dates[1:]))):
            raise ValueError("exercise dates must be strictly increasing")
        if (
            self.style == "european"
            and len(dates) != 1
            or (self.style == "american" and len(dates) != 2)
        ):
            raise ValueError(
                "European requires expiry; American requires start and expiry"
            )
        object.__setattr__(self, "dates", dates)

    def to_quantlib(self):
        from .._backends import quantlib as _quantlib

        return _quantlib.exercise(self)


@dataclass(frozen=True)
class VanillaContract:
    """Plain option payoff and exercise specification, independent of the pricing model and engine."""

    strike: float
    exercise: Exercise
    kind: str = "call"
    multiplier: float = 1.0
    settlement: str = "cash"
    currency: str = "USD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "strike", positive(self.strike, "strike"))
        object.__setattr__(self, "multiplier", positive(self.multiplier, "multiplier"))
        option_type(self.kind)
        choice(self.settlement, ("cash", "physical"), "settlement")
        if not isinstance(self.exercise, Exercise):
            raise TypeError("exercise must be Exercise")
        if not isinstance(self.currency, str) or not self.currency:
            raise ValueError("currency must be nonempty")

    def to_quantlib(self):
        from .._backends import quantlib as _quantlib

        return _quantlib.vanilla_contract(self)

    def delivery(self, spot: float):
        """(underlying units, cash) at exercise; zero when out of the money."""
        spot = scalar(spot, "spot", minimum=0)
        sign = 1 if self.kind == "call" else -1
        if sign * (spot - self.strike) <= 0:
            return (0.0, 0.0)
        if self.settlement == "cash":
            return (0.0, self.multiplier * sign * (spot - self.strike))
        return (sign * self.multiplier, -sign * self.multiplier * self.strike)


@dataclass(frozen=True)
class CashFlow:
    """Signed amount payable on a calendar date; currency/unit consistency belongs to the containing contract."""

    date: NativeDate
    amount: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", qdate(self.date))
        object.__setattr__(self, "amount", scalar(self.amount, "amount"))
        if not isinstance(self.currency, str) or not self.currency:
            raise ValueError("currency must be nonempty")

    def to_quantlib(self):
        from .._backends import quantlib as _quantlib

        return _quantlib.cashflow(self)

    def __init__(
        self,
        date: DateLike | NativeDate,
        amount: float,
        currency: str = "USD",
    ) -> None:
        object.__setattr__(self, "date", date)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "currency", currency)
        self.__post_init__()
