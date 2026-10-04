"""Declared QuantLib capabilities; unsupported combinations fail explicitly."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...contracts import CashFlow

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike, FloatArray

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeDate,
        NativePricingEngine,
    )
if TYPE_CHECKING:
    from ...measures import MeasureModel
    from ...simulation import PathScenarios

if TYPE_CHECKING:
    from .. import BlackScholesModel, PricingResult


from dataclasses import dataclass

import numpy as np

from ..._finance import choice, integer, positive, scalar
from .._backends import quantlib_engines as _quantlib_engines
from ..interfaces import PricingResult
from ..interfaces.base import NativePricingModel


def _result(instrument, name, multiplier=1.0, currency="USD"):
    return _quantlib_engines._result(instrument, name, multiplier, currency)


@dataclass(frozen=True)
class VanillaEngine:
    """Named built-in vanilla pricing method with explicit numerical settings and supported model restrictions."""

    method: str = "analytic"
    steps: int | np.integer = 300
    grid: int | np.integer = 300
    samples: int | np.integer = 32768
    seed: int | np.integer = 42
    dividends: tuple[CashFlow, ...] = ()

    def __post_init__(self) -> None:
        choice(
            self.method,
            ("analytic", "tree", "fd", "mc", "heston", "heston_fd", "lsmc"),
            "method",
        )
        for name, minimum in (("steps", 2), ("grid", 10), ("samples", 2), ("seed", 1)):
            object.__setattr__(self, name, integer(getattr(self, name), name, minimum))
        object.__setattr__(self, "dividends", tuple(self.dividends))

    @property
    def capabilities(self):
        return {
            "analytic": ("european",),
            "tree": ("european", "american", "bermudan"),
            "fd": ("european", "american", "bermudan"),
            "mc": ("european",),
            "heston": ("european",),
            "heston_fd": ("european", "american", "bermudan"),
            "lsmc": ("american",),
        }[self.method]

    def price(self, problem):
        return _quantlib_engines.price_vanilla(self, problem)


@dataclass(frozen=True)
class QuantLibEngine:
    """Extension adapter for native instruments and explicit engine factories.

    factory(model) creates a QuantLib engine. The caller owns instrument/engine
    compatibility; evaluation and all result extraction remain date-scoped.
    """

    factory: Callable[[NativePricingModel], NativePricingEngine]
    name: str = "QuantLib"

    def price(self, problem):
        return _quantlib_engines.price_native(self, problem)


def forward_option(
    forward: float,
    strike: float,
    volatility: float,
    maturity: float,
    *,
    discount: float = 1.0,
    kind: str = "call",
    model: str = "black76",
    displacement: float = 0.0,
) -> float:
    """Black-76/normal prices; normal volatility has price/sqrt(year) units."""
    return _quantlib_engines.forward_option(
        forward,
        strike,
        volatility,
        maturity,
        discount=discount,
        kind=kind,
        model=model,
        displacement=displacement,
    )


def asian_price(
    model: BlackScholesModel,
    valuation_date: DateLike | NativeDate,
    strike: float,
    fixing_dates: Sequence[DateLike | NativeDate],
    *,
    kind: str = "call",
    average: str = "arithmetic",
    samples: int = 32768,
    seed: int = 42,
    past_fixings: int = 0,
    running_accumulator: float | None = None,
) -> PricingResult:
    """Discrete European average-price option; running sum/product is explicit."""
    return _quantlib_engines.asian_price(
        model,
        valuation_date,
        strike,
        fixing_dates,
        kind=kind,
        average=average,
        samples=samples,
        seed=seed,
        past_fixings=past_fixings,
        running_accumulator=running_accumulator,
    )


def basket_price(
    models: Sequence[BlackScholesModel],
    correlation: ArrayLike,
    weights: ArrayLike,
    valuation_date: DateLike | NativeDate,
    expiry: DateLike | NativeDate,
    strike: float,
    *,
    kind: str = "call",
    samples: int = 32768,
    seed: int = 42,
) -> PricingResult:
    """European weighted arithmetic basket; correlated Q GBM processes."""
    return _quantlib_engines.basket_price(
        models,
        correlation,
        weights,
        valuation_date,
        expiry,
        strike,
        kind=kind,
        samples=samples,
        seed=seed,
    )


def barrier_price(
    model: BlackScholesModel,
    valuation_date: DateLike | NativeDate,
    expiry: DateLike | NativeDate,
    strike: float,
    barrier: float,
    *,
    barrier_type: str = "down_out",
    rebate: float = 0.0,
    kind: str = "call",
) -> PricingResult:
    """Price a continuously monitored Black-Scholes barrier option under Q.

    Expiry/valuation are calendar dates; strike, barrier and rebate use spot-price units.
    barrier_type selects down_out, down_in, up_out or up_in."""
    return _quantlib_engines.barrier_price(
        model,
        valuation_date,
        expiry,
        strike,
        barrier,
        barrier_type=barrier_type,
        rebate=rebate,
        kind=kind,
    )


def scenario_price[ModelT](
    scenarios: PathScenarios,
    payoff: Callable[[FloatArray], ArrayLike],
    measure_model: MeasureModel[ModelT],
    *,
    currency: str = "USD",
) -> PricingResult:
    """Value a scalar terminal/path payoff on explicitly Q joint scenarios.

    The payoff consumes a single (time[,state]) path. This adapter requires a
    deterministic numeraire; use joint state/numeraire payoffs for stochastic
    discounting. It preserves the simulation family's sampling-error semantics.
    """
    from ...measures import MeasureModel
    from ...simulation import estimate_scenarios

    if not isinstance(measure_model, MeasureModel) or measure_model.measure != "Q":
        raise ValueError("scenario valuation requires an explicit Q measure model")
    if scenarios.times[0] != 0:
        raise ValueError("valuation scenarios must start at time zero")
    assert measure_model.numeraire is not None
    n0 = positive(measure_model.numeraire.value(0.0), "initial numeraire")
    nt = positive(
        measure_model.numeraire.value(scenarios.times[-1]), "terminal numeraire"
    )
    result = estimate_scenarios(
        scenarios, lambda path: scalar(payoff(path), "payoff") * n0 / nt
    )
    return PricingResult(
        float(result.value),
        error_estimate=None
        if result.standard_error is None
        else float(result.standard_error),
        engine="scenario-expectation",
        currency=currency,
    )
