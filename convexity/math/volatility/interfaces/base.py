"""A typed specialization of the shared fitted forecast artifact."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Covariates

from dataclasses import dataclass, field

from ...forecasting.interfaces import FittedForecaster
from ...forecasting.interfaces.base import ForecastAdapter
from .result import VolatilityForecastResult


@dataclass(frozen=True)
class FittedVolatilityForecaster(FittedForecaster):
    """Fitted conditional-variance model with retained return history and forecast capabilities."""

    adapter: ForecastAdapter = field(repr=False)

    def forecast(
        self,
        horizon,
        *,
        covariates: Covariates | ArrayLike | None = None,
        rng: np.random.Generator | None = None,
    ) -> VolatilityForecastResult:
        result = super().forecast(horizon, covariates=covariates, rng=rng)
        if not isinstance(result, VolatilityForecastResult):
            raise TypeError("volatility adapters must return VolatilityForecastResult")
        return result
