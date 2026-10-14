"""Engine-independent pricing problems and validated results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..._typing import DateLike

if TYPE_CHECKING:
    from .. import PricingProblem


from typing import Protocol

if TYPE_CHECKING:
    from ..._finance_types import NativeDate


class PricingModel(Protocol):
    """A supplied pricing-measure model; each engine checks its required capabilities."""

    @property
    def measure(self) -> str: ...


class NativePricingModel(PricingModel, Protocol):
    """Explicit native-engine extension; ordinary custom engines need only PricingModel."""

    def to_quantlib(self) -> object: ...


from dataclasses import dataclass, field
from types import MappingProxyType

from ..._finance import qdate, scalar


@dataclass(frozen=True)
class PricingProblem[ContractT = object, ModelT: PricingModel = PricingModel]:
    """Contract, pricing model and valuation date. Generic types retain custom models without requiring native conversion."""

    contract: ContractT
    model: ModelT
    valuation_date: NativeDate

    def __post_init__(self) -> None:
        object.__setattr__(self, "valuation_date", qdate(self.valuation_date))
        if getattr(self.model, "measure", None) != "Q":
            raise ValueError("pricing requires a Q model")

    def __init__(
        self,
        contract: ContractT,
        model: ModelT,
        valuation_date: DateLike | NativeDate,
    ) -> None:
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "valuation_date", valuation_date)
        self.__post_init__()


@dataclass(frozen=True)
class PricingResult:
    """Valuation with named Greeks and engine information; absent Greeks are not silently replaced by zeros."""

    value: float
    greeks: Mapping[str, float] = field(default_factory=dict)
    error_estimate: float | None = None
    engine: str = ""
    currency: str = "USD"

    def __post_init__(self) -> None:
        scalar(self.value, "price")
        object.__setattr__(
            self,
            "greeks",
            MappingProxyType({k: scalar(v, k) for k, v in self.greeks.items()}),
        )
        if self.error_estimate is not None:
            scalar(self.error_estimate, "error estimate", minimum=0)


class PricingEngine(Protocol):
    """Price a PricingProblem into an owned PricingResult; custom engines need no native provider inheritance."""

    def price(self, problem: PricingProblem) -> PricingResult: ...


def price(problem: PricingProblem, engine: PricingEngine) -> PricingResult:
    """Evaluate a PricingProblem using a built-in or custom PricingEngine.

    Return an owned PricingResult whose currency matches the contract. The model
    must explicitly declare Q; physical forecasts do not imply pricing probabilities."""
    result = engine.price(problem)
    if not isinstance(result, PricingResult):
        raise TypeError("engine must return PricingResult")
    currency = getattr(problem.contract, "currency", None)
    if currency is not None and result.currency != currency:
        raise ValueError("pricing result currency must match the contract")
    return result
