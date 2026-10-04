"""Heterogeneous autoregressions of realized variance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Covariates, TimeSeries

from dataclasses import dataclass

from ...forecasting import _validation as v


@dataclass(frozen=True)
class HARForecaster:
    """OLS on lagged window means of variance or log variance.

    log models assume homoskedastic Gaussian log innovations for uncertainty
    and Jensen-corrected means. level models supply point forecasts only and
    reject negative predictions. Lags are counts of observations, not days.
    """

    lags: tuple[int, ...] = (1, 5, 22)
    transform: str = "log"
    method: str = "analytic"
    simulations: int = 5000
    covariance: str = "HC0"

    def __post_init__(self) -> None:
        lags = tuple(v.integer(x, "lag", 1) for x in self.lags)
        if not lags or any(a >= b for a, b in zip(lags, lags[1:])):
            raise ValueError("lags must be nonempty and strictly increasing")
        object.__setattr__(self, "lags", lags)
        v.choice(self.transform, ("log", "level"), "transform")
        v.choice(self.method, ("analytic", "simulation"), "method")
        v.choice(
            self.covariance, ("nonrobust", "HC0", "HC1", "HC2", "HC3"), "covariance"
        )
        v.integer(self.simulations, "simulations", 2)
        if self.transform == "level" and self.method == "simulation":
            raise ValueError(
                'positive realized-variance simulation requires transform="log"'
            )

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        from .._backends.har import fit_har

        return fit_har(self, data, covariates)
