"""Type-only names for explicit native financial interoperability."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from QuantLib import (
        Date as NativeDate,
        Instrument as NativeInstrument,
        YieldTermStructure as NativeYieldCurve,
        DefaultProbabilityTermStructure as NativeDefaultCurve,
        PricingEngine as NativePricingEngine,
        DayCounter as NativeDayCounter,
        BlackVolTermStructure as NativeVolatility,
        Calendar as NativeCalendar,
        Schedule as NativeSchedule,
        Bond as NativeBond,
    )
    from QuantLib import RateHelper as NativeRateHelper
    from QuantLib import DefaultProbabilityHelper as NativeDefaultHelper
    from QuantLib import IborIndex as NativeIborIndex
    from QuantLib import InflationIndex as NativeInflationIndex
    from QuantLib import InflationTermStructure as NativeInflationCurve
    from QuantLib import VanillaSwap as NativeSwap
    from QuantLib import Swaption as NativeSwaption
    from QuantLib import CapFloor as NativeCapFloor
    from QuantLib import ZeroCouponInflationSwap as NativeInflationSwap
    from QuantLib import Gsr as NativeGSR
    from QuantLib import MarketModel as NativeMarketModel
    from QuantLib import CalibratedModel as NativeCalibratedModel
    from QuantLib import CalibrationHelper as NativeCalibrationHelper
    from QuantLib import Callability as NativeCallability
