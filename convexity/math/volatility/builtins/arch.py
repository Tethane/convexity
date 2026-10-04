"""Immutable ARCH-family specifications; numerical work is delegated to arch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .. import Covariates, TimeSeries

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypedDict, Unpack

from ...forecasting import _validation as v


class ArchforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for ARCHForecaster."""

    o: int
    mean: str
    lags: int
    innovation: str
    method: str
    simulations: int
    decay: float
    rescale: bool
    check_convergence: bool
    fit_options: Mapping[str, object]


class GarchforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for GARCHForecaster."""

    o: int
    mean: str
    lags: int
    innovation: str
    method: str
    simulations: int
    decay: float
    rescale: bool
    check_convergence: bool
    fit_options: Mapping[str, object]


class GjrgarchforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for GJRGARCHForecaster."""

    mean: str
    lags: int
    innovation: str
    method: str
    simulations: int
    decay: float
    rescale: bool
    check_convergence: bool
    fit_options: Mapping[str, object]


class EgarchforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for EGARCHForecaster."""

    mean: str
    lags: int
    innovation: str
    method: str
    simulations: int
    decay: float
    rescale: bool
    check_convergence: bool
    fit_options: Mapping[str, object]


class ConstantvarianceforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for ConstantVarianceForecaster."""

    o: int
    mean: str
    lags: int
    innovation: str
    method: str
    simulations: int
    decay: float
    rescale: bool
    check_convergence: bool
    fit_options: Mapping[str, object]


class EwmaforecasterOptions(TypedDict, total=False):
    """Optional keyword settings for EWMAForecaster."""

    o: int
    mean: str
    lags: int
    innovation: str
    method: str
    simulations: int
    rescale: bool
    check_convergence: bool
    fit_options: Mapping[str, object]


@dataclass(frozen=True)
class ARCHModelForecaster:
    """Conditional-volatility specification with explicit mean, innovation and variance dynamics."""

    model: str = "garch"
    p: int = 1
    o: int = 0
    q: int = 1
    mean: str = "constant"
    lags: int = 0
    innovation: str = "normal"
    method: str = "auto"
    simulations: int = 5000
    decay: float = 0.94
    rescale: bool = True
    check_convergence: bool = True
    fit_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        v.choice(
            self.model,
            ("arch", "garch", "gjr_garch", "egarch", "constant", "ewma"),
            "model",
        )
        v.choice(self.mean, ("zero", "constant", "ar"), "mean")
        v.choice(
            self.innovation,
            ("normal", "student_t", "skew_student_t", "ged"),
            "innovation",
        )
        v.choice(self.method, ("auto", "analytic", "simulation", "bootstrap"), "method")
        for name in ("p", "o", "q", "lags"):
            v.integer(getattr(self, name), name)
        if self.model in ("arch", "garch", "gjr_garch", "egarch") and self.p < 1:
            raise ValueError("p must be positive")
        if self.model == "arch" and (self.o or self.q):
            raise ValueError("ARCH requires o=q=0")
        if self.model == "garch" and self.o:
            raise ValueError("use gjr_garch for asymmetric squared innovations")
        if self.model == "gjr_garch" and self.o < 1:
            raise ValueError("GJR-GARCH requires positive o")
        if self.model in ("constant", "ewma") and (self.p or self.o or self.q):
            raise ValueError("constant/EWMA variance requires p=o=q=0")
        if (self.mean == "ar") != (self.lags > 0):
            raise ValueError('positive lags are required exactly when mean="ar"')
        v.integer(self.simulations, "simulations", 2)
        v.boolean(self.rescale, "rescale")
        v.boolean(self.check_convergence, "check_convergence")
        decay = v.scalar(self.decay, "decay")
        if not 0 < decay < 1:
            raise ValueError("decay must lie strictly inside (0,1)")
        object.__setattr__(self, "decay", decay)
        allowed = {"tol", "options", "cov_type", "starting_values", "backcast"}
        if not isinstance(self.fit_options, dict) or set(self.fit_options) - allowed:
            raise ValueError(f"fit_options may contain only {sorted(allowed)}")
        object.__setattr__(self, "fit_options", deepcopy(self.fit_options))

    def fit(
        self, data: TimeSeries | ArrayLike, *, covariates: Covariates | None = None
    ):
        from .._backends.arch import fit_arch

        return fit_arch(self, data, covariates)


def ARCHForecaster(
    p: int = 1, **kwargs: Unpack[ArchforecasterOptions]
) -> ARCHModelForecaster:
    """Construct the ARCH forecasting specification.

    ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: o, mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options."""
    return ARCHModelForecaster("arch", p=p, q=0, **kwargs)


def GARCHForecaster(
    p: int = 1, q: int = 1, **kwargs: Unpack[GarchforecasterOptions]
) -> ARCHModelForecaster:
    """Construct the GARCH forecasting specification.

    ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: o, mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options."""
    return ARCHModelForecaster("garch", p=p, q=q, **kwargs)


def GJRGARCHForecaster(
    p: int = 1, o: int = 1, q: int = 1, **kwargs: Unpack[GjrgarchforecasterOptions]
) -> ARCHModelForecaster:
    """Construct the GJRGARCH forecasting specification.

    ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options."""
    return ARCHModelForecaster("gjr_garch", p=p, o=o, q=q, **kwargs)


def EGARCHForecaster(
    p: int = 1, o: int = 1, q: int = 1, **kwargs: Unpack[EgarchforecasterOptions]
) -> ARCHModelForecaster:
    """Construct the EGARCH forecasting specification.

    ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options."""
    return ARCHModelForecaster("egarch", p=p, o=o, q=q, **kwargs)


def ConstantVarianceForecaster(
    **kwargs: Unpack[ConstantvarianceforecasterOptions],
) -> ARCHModelForecaster:
    """Construct the ConstantVariance forecasting specification.

    ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: o, mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options."""
    return ARCHModelForecaster("constant", p=0, q=0, **kwargs)


def EWMAForecaster(
    decay: float = 0.94, **kwargs: Unpack[EwmaforecasterOptions]
) -> ARCHModelForecaster:
    """Construct the EWMA forecasting specification.

    ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
    Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
    are horizons and columns are targets; variances use squared target units.
    Optional settings: o, mean, lags, innovation, method, simulations, rescale, check_convergence, fit_options."""
    return ARCHModelForecaster("ewma", p=0, q=0, decay=decay, **kwargs)
