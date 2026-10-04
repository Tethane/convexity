"""Volatility quantities alongside the existing target-forecast contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import _FACTORY, FloatArray

if TYPE_CHECKING:
    from ...distributions import Distribution
from ...forecasting import ForecastResult
from ...forecasting import _validation as v
from ...forecasting.interfaces.data import ForecastHorizon


@dataclass(frozen=True)
class VolatilityForecastResult(ForecastResult):
    """Target forecasts plus E[future innovation variance] (or E[realized variance]).

    All variance quantities use squared input-return units per observation.
    conditional_variance_paths, when present, are (draw,horizon,target) and
    retain the same draw order as target paths. They are not squared returns.
    """

    conditional_variance: FloatArray = cast("FloatArray", None)
    conditional_variance_paths: FloatArray | None = cast("FloatArray | None", None)
    target_kind: str = "return"

    def __post_init__(self) -> None:
        super().__post_init__()
        v.choice(self.target_kind, ("return", "realized_variance"), "target_kind")
        variance = v.matrix(self.conditional_variance, "conditional_variance")
        if variance.shape != self.shape or np.any(variance < 0):
            raise ValueError(
                "conditional_variance must be nonnegative and match target shape"
            )
        object.__setattr__(self, "conditional_variance", variance)
        if self.conditional_variance_paths is not None:
            paths = v.array(
                self.conditional_variance_paths, "conditional_variance_paths", ndim=3
            )
            if paths.shape[1:] != self.shape or np.any(paths < 0):
                raise ValueError(
                    "conditional variance paths must be nonnegative with shape (draw,horizon,target)"
                )
            if self.paths is None or paths.shape != self.paths.shape:
                raise ValueError(
                    "conditional variance paths require aligned target paths"
                )
            object.__setattr__(self, "conditional_variance_paths", paths)

    def volatility(self, *, periods_per_year: float = 1.0) -> FloatArray:
        """sqrt(periods_per_year * E[variance]); not E[sqrt(variance)]."""
        periods = v.scalar(periods_per_year, "periods_per_year", minimum=0, strict=True)
        return v.array(np.sqrt(periods * self.conditional_variance), "volatility")

    def variance_forecast(self):
        """Forecast the variance quantity itself, with paths only when available.

        Its variance(), if available, is uncertainty about future variance,
        not the conditional return variance in this object's variance().
        """
        if self.target_kind == "realized_variance":
            return ForecastResult(
                **{f.name: getattr(self, f.name) for f in fields(ForecastResult)}
            )
        return ForecastResult(
            self.conditional_variance,
            horizon=self.horizon,
            cutoff=self.cutoff,
            model_identity=self.model_identity,
            metadata={**self.metadata, "quantity": "conditional_variance"},
            mean_values=self.conditional_variance
            if self.conditional_variance_paths is None
            else None,
            paths=self.conditional_variance_paths,
        )

    def select(self, indices, *, horizon=None):
        # Use the base selection checks before slicing our additional axes.
        base = ForecastResult(
            **{f.name: getattr(self, f.name) for f in fields(ForecastResult)}
        )
        selected = base.select(indices, horizon=horizon)
        ix = np.asarray(indices)
        return type(self)(
            **{f.name: getattr(selected, f.name) for f in fields(ForecastResult)},
            conditional_variance=self.conditional_variance[ix],
            conditional_variance_paths=None
            if self.conditional_variance_paths is None
            else self.conditional_variance_paths[:, ix],
            target_kind=self.target_kind,
        )

    def __init__(
        self,
        point: ArrayLike,
        horizon: ForecastHorizon | None = None,
        mean_values: ArrayLike | None = None,
        variance_values: ArrayLike | None = None,
        quantile_levels: ArrayLike | None = None,
        quantile_values: ArrayLike | None = None,
        paths: ArrayLike | None = None,
        joint_covariance: ArrayLike | None = None,
        gaussian: bool = False,
        law: Distribution | None = None,
        marginal_laws: Sequence[Distribution] | None = None,
        model_identity: str = "user",
        cutoff: float | int | np.datetime64 | None = None,
        metadata: dict[str, object] = cast("dict", _FACTORY),
        conditional_variance: ArrayLike | None = None,
        conditional_variance_paths: ArrayLike | None = None,
        target_kind: str = "return",
    ) -> None:
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "mean_values", mean_values)
        object.__setattr__(self, "variance_values", variance_values)
        object.__setattr__(self, "quantile_levels", quantile_levels)
        object.__setattr__(self, "quantile_values", quantile_values)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "joint_covariance", joint_covariance)
        object.__setattr__(self, "gaussian", gaussian)
        object.__setattr__(self, "law", law)
        object.__setattr__(self, "marginal_laws", marginal_laws)
        object.__setattr__(self, "model_identity", model_identity)
        object.__setattr__(self, "cutoff", cutoff)
        if metadata is _FACTORY:
            metadata = __class__.__dataclass_fields__["metadata"].default_factory()
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "conditional_variance", conditional_variance)
        object.__setattr__(
            self, "conditional_variance_paths", conditional_variance_paths
        )
        object.__setattr__(self, "target_kind", target_kind)
        self.__post_init__()
