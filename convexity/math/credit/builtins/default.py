"""QuantLib survival curves, reduced-form valuation and statistical adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike, FloatArray

if TYPE_CHECKING:
    from ..._finance_types import (
        NativeDate,
        NativeDefaultHelper,
        NativeSchedule,
    )
    from ...curves import YieldCurve


if TYPE_CHECKING:
    from .. import StructuralCreditResult


if TYPE_CHECKING:
    from .. import HazardCurve


if TYPE_CHECKING:
    from .. import CDSResult


import numpy as np

if TYPE_CHECKING:
    from ..._finance_types import NativeDefaultCurve

from dataclasses import dataclass, field
from numbers import Real

from ..._finance import array, choice, qdate, scalar
from .._backends import quantlib as _quantlib
from .._backends import scipy as _scipy
from ..interfaces import HazardEstimate


@dataclass(frozen=True)
class HazardCurve:
    """Term structure of default intensity/survival with explicit reference date and day-count convention."""

    structure: NativeDefaultCurve
    measure: str = "Q"
    _calibration_source: object | None = field(default=None, repr=False, compare=False)

    calibration_errors: FloatArray | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        return _quantlib.validate_curve(self)

    @property
    def reference_date(self):
        return _quantlib.reference_date(self)

    @property
    def handle(self):
        return _quantlib.handle(self)

    def _arg(self, maturity):
        if isinstance(maturity, Real):
            return scalar(maturity, "maturity", minimum=0)
        d = qdate(maturity)
        if d < self.reference_date:
            raise ValueError("maturity precedes reference")
        return d

    def survival(self, maturity: float | DateLike | NativeDate):
        return _quantlib.survival(self, maturity)

    def hazard(self, maturity: float | DateLike | NativeDate):
        return _quantlib.hazard(self, maturity)

    def density(self, maturity: float | DateLike | NativeDate):
        return _quantlib.density(self, maturity)

    def default_probability(self, start, end):
        a, b = (self._arg(start), self._arg(end))
        if type(a) is not type(b) or b < a:
            raise ValueError(
                "default interval must be ordered with matching date/time types"
            )
        return self.survival(a) - self.survival(b)

    def sample(self, draws: int, *, horizon, rng: np.random.Generator):
        """Default times up to horizon; +inf means survival beyond horizon."""
        return _scipy.sample_default_times(self, draws, horizon=horizon, rng=rng)


def flat_hazard(
    reference_date: DateLike | NativeDate,
    hazard: float,
    *,
    measure: str = "Q",
    convention: str = "actual365",
) -> HazardCurve:
    """Construct constant default intensity per year under an explicit P or Q label.

    Survival at time t is exp(-hazard*t); the measure label does not transform a model."""
    return _quantlib.flat_hazard(
        reference_date, hazard, measure=measure, convention=convention
    )


def survival_curve(
    dates: Sequence[DateLike | NativeDate],
    probabilities: ArrayLike,
    *,
    measure: str = "Q",
    convention: str = "actual365",
    extrapolate: bool = False,
) -> HazardCurve:
    """Interpolate nonincreasing survival probabilities at increasing dates.

    Probabilities are unconditional survival from the reference date; P/Q labeling
    is explicit and extrapolation must be enabled deliberately."""
    return _quantlib.survival_curve(
        dates,
        probabilities,
        measure=measure,
        convention=convention,
        extrapolate=extrapolate,
    )


def bootstrap_cds(
    reference_date: DateLike | NativeDate,
    helpers: Sequence[NativeDefaultHelper],
    *,
    convention: str = "actual365",
    extrapolate: bool = False,
) -> HazardCurve:
    """Bootstrap a Q default curve from native CDS calibration helpers.

    Helpers own spreads, recoveries, schedules and discount curves; the returned
    curve retains repricing diagnostics. Extrapolation is explicit."""
    return _quantlib.bootstrap_cds(
        reference_date, helpers, convention=convention, extrapolate=extrapolate
    )


def cds_price(
    schedule: NativeSchedule,
    spread: float,
    discount: YieldCurve,
    default: HazardCurve,
    *,
    notional: float = 1.0,
    recovery: float = 0.4,
    buyer: bool = True,
    protection_start: DateLike | NativeDate | None = None,
    trade_date: DateLike | NativeDate | None = None,
    cash_settlement_days: int = 3,
) -> CDSResult:
    """Value scheduled CDS protection and premium legs under Q discount/default curves.

    Spread is annual decimal premium, recovery a fraction, notional a currency amount.
    Protection/trade dates and cash settlement lag are explicit."""
    return _quantlib.cds_price(
        schedule,
        spread,
        discount,
        default,
        notional=notional,
        recovery=recovery,
        buyer=buyer,
        protection_start=protection_start,
        trade_date=trade_date,
        cash_settlement_days=cash_settlement_days,
    )


def risky_cashflows(
    times: ArrayLike,
    amounts: ArrayLike,
    discount: YieldCurve,
    default: HazardCurve,
    *,
    recovery: float = 0.0,
    recovery_notional: float = 1.0,
    integration_points: ArrayLike | None = None,
) -> float:
    """Recovery of par paid at default through final payment, continuous integral.

    Float times use each curve's year convention; matching conventions required.
    """
    return _scipy.risky_cashflows(
        times,
        amounts,
        discount,
        default,
        recovery=recovery,
        recovery_notional=recovery_notional,
        integration_points=integration_points,
    )


def estimate_hazard(exposure_times: ArrayLike, defaults: ArrayLike) -> HazardEstimate:
    """Constant P intensity MLE with right-censoring; times are at-risk durations."""
    t = array(exposure_times, "exposure", ndim=1)
    d = np.asarray(defaults)
    if d.dtype.kind != "b" or d.shape != t.shape or np.any(t < 0) or (t.sum() <= 0):
        raise ValueError(
            "nonnegative exposures and aligned boolean default indicators required"
        )
    return HazardEstimate(
        hazard=float(d.sum() / t.sum()),
        events=int(d.sum()),
        exposure=float(t.sum()),
        standard_error=float(np.sqrt(d.sum()) / t.sum()) if d.sum() else None,
        measure="P",
    )


@dataclass(frozen=True)
class MigrationModel:
    """Continuous-time rating transitions with absorbing final default state."""

    generator: FloatArray
    measure: str = "P"

    def __post_init__(self) -> None:
        g = array(self.generator, "generator", ndim=2)
        if g.shape[0] != g.shape[1] or len(g) < 2:
            raise ValueError("generator must be square with at least two states")
        off = g - np.diag(np.diag(g))
        if (
            g.shape[0] != g.shape[1]
            or np.any(off < 0)
            or np.any(np.diag(g) > 0)
            or np.any(abs(g.sum(axis=1)) > 1e-12 * np.max(abs(g), axis=1))
            or np.any(g[-1] != 0)
        ):
            raise ValueError(
                "generator needs nonnegative off-diagonals, zero row sums, absorbing default"
            )
        object.__setattr__(self, "generator", g)
        choice(self.measure, ("P", "Q"), "measure")

    def transition(self, time: float):
        return _scipy.transition(self, time)

    @classmethod
    def fit(cls, transition_counts, exposures, *, measure: str = "P"):
        c = array(transition_counts, "counts", ndim=2)
        t = array(exposures, "exposures", ndim=1)
        if (
            c.shape != (len(t), len(t))
            or np.any(c < 0)
            or np.any(c != np.floor(c))
            or np.any(t < 0)
            or np.any(t[:-1] <= 0)
            or np.any(c[-1] != 0)
        ):
            raise ValueError(
                "integer counts and positive nondefault state exposure required"
            )
        g = np.zeros_like(c)
        g[:-1] = c[:-1] / t[:-1, None]
        np.fill_diagonal(g, 0)
        np.fill_diagonal(g, -g.sum(axis=1))
        return cls(g, measure)

    def __init__(
        self,
        generator: ArrayLike,
        measure: str = "P",
    ) -> None:
        object.__setattr__(self, "generator", generator)
        object.__setattr__(self, "measure", measure)
        self.__post_init__()


def merton_structural(
    asset_value: float,
    debt: float,
    asset_volatility: float,
    maturity: float,
    rate: float,
    *,
    physical_drift: float | None = None,
) -> StructuralCreditResult:
    """Merton terminal default model; Q debt valuation and optional P probability.

    This is terminal insolvency, not a first-passage default-time intensity model.
    """
    return _scipy.merton_structural(
        asset_value,
        debt,
        asset_volatility,
        maturity,
        rate,
        physical_drift=physical_drift,
    )


def cds_quote(
    spread: float,
    tenor: str,
    discount: YieldCurve,
    *,
    recovery: float = 0.4,
    settlement_days: int = 0,
    calendar: str = "none",
    adjustment: str = "unadjusted",
    frequency: str = "quarterly",
    convention: str = "actual360",
) -> NativeDefaultHelper:
    """Par CDS running spread for bootstrap, using forward schedule generation."""
    return _quantlib.cds_quote(
        spread,
        tenor,
        discount,
        recovery=recovery,
        settlement_days=settlement_days,
        calendar=calendar,
        adjustment=adjustment,
        frequency=frequency,
        convention=convention,
    )
