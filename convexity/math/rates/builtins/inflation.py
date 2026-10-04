"""Lognormal CPI dynamics with deterministic nominal and real rates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..._finance import array, choice, integer, positive, scalar
from ..._validation import generator
from ...curves import YieldCurve
from ...distributions import lognormal
from ...pricing import forward_option
from ...simulation import PathScenarios


@dataclass(frozen=True)
class InflationDynamics:
    """P uses a supplied expected proportional CPI drift. Q uses D_real/D_nominal.

    Curves are deterministic; stochastic rate/inflation correlation and index
    publication lags are not implicit. Use the contract's observation dates and
    existing CPI fixings when combining this law with inflation instruments.
    """

    volatility: float
    measure: str = "P"
    drift: float | None = None
    nominal: YieldCurve | None = None
    real: YieldCurve | None = None

    def __post_init__(self) -> None:
        scalar(self.volatility, "volatility", minimum=0)
        choice(self.measure, ("P", "Q"), "measure")
        if self.measure == "P":
            if self.drift is None or self.nominal is not None or self.real is not None:
                raise ValueError(
                    "P inflation dynamics require drift and no valuation curves"
                )
            scalar(self.drift, "drift")
        else:
            if self.drift is not None or self.nominal is None or self.real is None:
                raise ValueError(
                    "Q inflation dynamics require nominal and real curves, no P drift"
                )
            if (
                self.nominal.reference_date != self.real.reference_date
                or self.nominal.structure.dayCounter()
                != self.real.structure.dayCounter()
            ):
                raise ValueError(
                    "nominal/real curve dates and time conventions must agree"
                )

    def growth(self, time: float):
        t = scalar(time, "time", minimum=0)
        if self.measure == "P":
            assert self.drift is not None  # validated on construction
            return np.exp(self.drift * t)
        assert self.real is not None and self.nominal is not None
        return self.real.discount(t) / self.nominal.discount(t)

    def transition(self, index, start, end):
        """Exact scalar CPI law conditional on the supplied starting index."""
        initial = positive(index, "index")
        start, end = scalar(start, "start", minimum=0), scalar(end, "end", minimum=0)
        if end < start:
            raise ValueError("end must be >= start")
        mean = initial * self.growth(end) / self.growth(start)
        variance = self.volatility**2 * (end - start)
        if variance == 0:
            from ...distributions import empirical

            return empirical([mean])
        return lognormal(np.log(mean) - 0.5 * variance, np.sqrt(variance))

    def sample(
        self, index, times, *, draws: int, rng: np.random.Generator
    ) -> PathScenarios:
        generator(rng)
        t = array(times, "times", ndim=1)
        if t[0] != 0 or np.any(np.diff(t) <= 0):
            raise ValueError("sampling times must start at zero and increase")
        n = integer(draws, "draws", 1)
        dt = np.diff(t)
        shocks = self.volatility * np.sqrt(dt) * rng.normal(size=(n, len(dt)))
        log_growth = np.log([self.growth(s) for s in t])
        logs = (
            np.log(positive(index, "index")) + log_growth - 0.5 * self.volatility**2 * t
        )
        values = np.exp(logs + np.c_[np.zeros(n), np.cumsum(shocks, axis=1)])
        return PathScenarios(
            t, values, sampling="iid", method="lognormal-CPI", exact=True
        )

    def option(self, index, strike: float, maturity: float, *, kind: str = "call"):
        """European option on the CPI level in nominal currency at maturity."""
        if self.measure != "Q":
            raise ValueError("CPI option valuation requires Q dynamics")
        assert self.nominal is not None
        t = positive(maturity, "maturity")
        return forward_option(
            positive(index, "index") * self.growth(t),
            positive(strike, "strike"),
            self.volatility,
            t,
            discount=self.nominal.discount(t),
            kind=kind,
        )
