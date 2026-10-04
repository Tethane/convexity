"""Rate option valuation with native contracts and explicit engine selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeInflationSwap,
        NativeInstrument,
    )
    from ...curves import YieldCurve
    from ...pricing import PricingResult


if TYPE_CHECKING:
    from .. import ShortRateModel


from .._backends import quantlib as _quantlib


def rate_option_price(
    instrument: NativeInstrument,
    curve: YieldCurve,
    *,
    kind: str | None = None,
    method: str = "black",
    volatility: float | None = None,
    model: ShortRateModel | None = None,
    steps: int = 300,
    displacement: float = 0.0,
) -> PricingResult:
    """Price a supported native rate option under an explicit discount curve.

    method selects Black, normal, tree or model-based valuation; volatility units
    follow that method. Instrument/model compatibility is checked by the engine."""
    return _quantlib.rate_option_price(
        instrument,
        curve,
        kind=kind,
        method=method,
        volatility=volatility,
        model=model,
        steps=steps,
        displacement=displacement,
    )


def inflation_swap_price(
    instrument: NativeInflationSwap, discount: YieldCurve
) -> PricingResult:
    """Value a native inflation swap using an explicit nominal discount curve.

    The instrument must already carry its inflation index and forecast curve."""
    return _quantlib.inflation_swap_price(instrument, discount)
