"""Classical specifications; statsmodels imports only during fitting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Covariates, TimeSeries

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypedDict, Unpack

from .. import _validation as v


class ArimaforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for ARIMAForecaster."""

    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class SarimaxforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for SARIMAXForecaster."""

    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class ArforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for ARForecaster."""

    seasonal_order: tuple[int, int, int, int]
    trend: str | None
    model_options: Mapping[str, object] | None
    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class ArmaforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for ARMAForecaster."""

    seasonal_order: tuple[int, int, int, int]
    trend: str | None
    model_options: Mapping[str, object] | None
    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class SarimaforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for SARIMAForecaster."""

    trend: str | None
    model_options: Mapping[str, object] | None
    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class ExponentialsmoothingforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for ExponentialSmoothingForecaster."""

    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class HoltforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for HoltForecaster."""

    seasonal: str | None
    seasonal_periods: int | None
    damped_trend: bool
    model_options: Mapping[str, object] | None
    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class HoltwintersforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for HoltWintersForecaster."""

    damped_trend: bool
    model_options: Mapping[str, object] | None
    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class EtsforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for ETSForecaster."""

    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class VarforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for VARForecaster."""

    model_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class VarmaxforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for VARMAXForecaster."""

    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class VecmforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for VECMForecaster."""

    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class StructuralforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for StructuralForecaster."""

    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


class DynamicfactorforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for DynamicFactorForecaster."""

    fit_options: Mapping[str, object]
    missing: str
    check_convergence: bool
    samples: int
    seed: int


@dataclass(frozen=True)
class StatsmodelsForecaster:
    """Finite catalog with native model/fit options passed to statsmodels.

    AR, ARMA and SARIMA are configurations of ARIMA/SARIMAX. Holt and
    Holt-Winters are configurations of exponential smoothing.
    """

    model: str = "arima"
    model_options: Mapping[str, object] = field(default_factory=dict)
    fit_options: Mapping[str, object] = field(default_factory=dict)
    missing: str = "raise"
    check_convergence: bool = True
    samples: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        v.choice(
            self.model,
            (
                "arima",
                "sarimax",
                "exponential_smoothing",
                "ets",
                "var",
                "varmax",
                "vecm",
                "structural",
                "dynamic_factor",
            ),
            "model",
        )
        v.choice(self.missing, ("raise", "ffill", "native"), "missing")
        v.boolean(self.check_convergence, "check_convergence")
        v.integer(self.samples, "samples", 1)
        v.seed(self.seed)
        if self.missing == "native" and self.model not in (
            "arima",
            "sarimax",
            "varmax",
            "structural",
            "dynamic_factor",
        ):
            raise ValueError("native missing values require a state-space model")
        forbidden = {
            "endog",
            "exog",
            "dates",
            "freq",
            "missing",
        } & self.model_options.keys()
        if self.model_options.get("simple_differencing", False):
            raise ValueError(
                "simple_differencing changes forecast target units; use state-space differencing"
            )
        if self.fit_options.get("return_params", False):
            raise ValueError(
                "forecasting requires a fitted backend result, not return_params=True"
            )
        if forbidden:
            raise ValueError(
                f"model_options cannot override data arguments: {forbidden}"
            )
        for name in ("model_options", "fit_options"):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        from .._backends.statsmodels import fit_classical

        return fit_classical(self, data, covariates)


def ARIMAForecaster(
    order: tuple[int, ...] = (1, 0, 0),
    *,
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    model_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[ArimaforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the ARIMA forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: fit_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "arima",
        {
            **(model_options or {}),
            "order": order,
            "seasonal_order": seasonal_order,
            "trend": trend,
        },
        **kwargs,
    )


def SARIMAXForecaster(
    order: tuple[int, ...] = (1, 0, 0),
    *,
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
    model_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[SarimaxforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the SARIMAX forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: fit_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "sarimax",
        {
            **(model_options or {}),
            "order": order,
            "seasonal_order": seasonal_order,
            "trend": trend,
        },
        **kwargs,
    )


def ARForecaster(
    lags: int = 1, **kwargs: Unpack[ArforecasterOptions]
) -> StatsmodelsForecaster:
    """Construct the AR forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: seasonal_order, trend, model_options, fit_options, missing, check_convergence, samples, seed."""
    return ARIMAForecaster((v.integer(lags, "lags", 1), 0, 0), **kwargs)


def ARMAForecaster(
    order: tuple[int, ...] = (1, 1), **kwargs: Unpack[ArmaforecasterOptions]
) -> StatsmodelsForecaster:
    """Construct the ARMA forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: seasonal_order, trend, model_options, fit_options, missing, check_convergence, samples, seed."""
    if len(order) != 2:
        raise ValueError("ARMA order must be (p,q)")
    return ARIMAForecaster((order[0], 0, order[1]), **kwargs)


def SARIMAForecaster(
    order: tuple[int, ...] = (1, 0, 0),
    *,
    seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 12),
    **kwargs: Unpack[SarimaforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the SARIMA forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: trend, model_options, fit_options, missing, check_convergence, samples, seed."""
    return ARIMAForecaster(order, seasonal_order=seasonal_order, **kwargs)


def ExponentialSmoothingForecaster(
    *,
    trend: str | None = None,
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    damped_trend: bool = False,
    model_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[ExponentialsmoothingforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the ExponentialSmoothing forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: fit_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "exponential_smoothing",
        {
            **(model_options or {}),
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "damped_trend": damped_trend,
        },
        **kwargs,
    )


def HoltForecaster(**kwargs: Unpack[HoltforecasterOptions]) -> StatsmodelsForecaster:
    """Construct the Holt forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: seasonal, seasonal_periods, damped_trend, model_options, fit_options, missing, check_convergence, samples, seed."""
    return ExponentialSmoothingForecaster(trend="add", **kwargs)


def HoltWintersForecaster(
    period: int = 12,
    *,
    seasonal: str = "add",
    **kwargs: Unpack[HoltwintersforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the HoltWinters forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: damped_trend, model_options, fit_options, missing, check_convergence, samples, seed."""
    return ExponentialSmoothingForecaster(
        trend="add", seasonal=seasonal, seasonal_periods=period, **kwargs
    )


def ETSForecaster(
    *,
    error: str = "add",
    trend: str | None = None,
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    model_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[EtsforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the ETS forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: fit_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "ets",
        {
            **(model_options or {}),
            "error": error,
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
        },
        **kwargs,
    )


def VARForecaster(
    lags: int = 1,
    *,
    fit_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[VarforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the VAR forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: model_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "var",
        fit_options={**(fit_options or {}), "maxlags": v.integer(lags, "lags", 1)},
        **kwargs,
    )


def VARMAXForecaster(
    order: tuple[int, ...] = (1, 0),
    *,
    model_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[VarmaxforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the VARMAX forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: fit_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "varmax", {**(model_options or {}), "order": order}, **kwargs
    )


def VECMForecaster(
    *,
    k_ar_diff: int = 1,
    coint_rank: int = 1,
    model_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[VecmforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the VECM forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: fit_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "vecm",
        {**(model_options or {}), "k_ar_diff": k_ar_diff, "coint_rank": coint_rank},
        **kwargs,
    )


def StructuralForecaster(
    level: str = "local level",
    *,
    model_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[StructuralforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the Structural forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: fit_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "structural", {**(model_options or {}), "level": level}, **kwargs
    )


def DynamicFactorForecaster(
    k_factors: int = 1,
    factor_order: int = 1,
    *,
    model_options: Mapping[str, object] | None = None,
    **kwargs: Unpack[DynamicfactorforecasterOptions],
) -> StatsmodelsForecaster:
    """Construct the DynamicFactor forecasting specification.

    Finite catalog with native model/fit options passed to statsmodels.
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: fit_options, missing, check_convergence, samples, seed."""
    return StatsmodelsForecaster(
        "dynamic_factor",
        {**(model_options or {}), "k_factors": k_factors, "factor_order": factor_order},
        **kwargs,
    )
