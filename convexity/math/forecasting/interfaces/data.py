"""Time-first series, explicitly available covariates, and relative horizons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import DateLike, FloatArray, IntArray

if TYPE_CHECKING:
    from .. import ForecastHorizon


from dataclasses import dataclass
from typing import cast

import numpy as np

from ..._typing import IndexArray
from .. import _validation as v


@dataclass(frozen=True)
class Covariates:
    """Training covariates. Dynamic arrays align row-for-row with the target.

    known_future is available at forecast time; observed_past is available only
    through the cutoff. static is a numeric vector constant over this series.
    """

    known_future: FloatArray | None = cast("FloatArray | None", None)
    observed_past: FloatArray | None = cast("FloatArray | None", None)
    static: FloatArray | None = cast("FloatArray | None", None)

    def __post_init__(self) -> None:
        for name in ("known_future", "observed_past"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, v.matrix(getattr(self, name), name))
        if self.static is not None:
            object.__setattr__(self, "static", v.array(self.static, "static", ndim=1))

    def __reduce__(self):
        return type(self), (self.known_future, self.observed_past, self.static)

    def slice(self, start, stop) -> Covariates:
        return Covariates(
            None if self.known_future is None else self.known_future[start:stop],
            None if self.observed_past is None else self.observed_past[start:stop],
            self.static,
        )

    def __init__(
        self,
        known_future: ArrayLike | None = None,
        observed_past: ArrayLike | None = None,
        static: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "known_future", known_future)
        object.__setattr__(self, "observed_past", observed_past)
        object.__setattr__(self, "static", static)
        self.__post_init__()


@dataclass(frozen=True)
class TimeSeries:
    """One series with values (n,k), numeric/datetime times, and optional NaNs.

    A vector becomes (n,1). No sorting, resampling, or imputation is implicit.
    Names and covariate columns have positional identity.
    """

    values: FloatArray
    timestamps: IndexArray = cast("IndexArray", None)
    covariates: Covariates | None = None
    names: tuple[str, ...] = cast("tuple[str, ...]", None)

    def __post_init__(self) -> None:
        values = v.matrix(self.values, "values", missing=True)
        times = np.arange(len(values)) if self.timestamps is None else self.timestamps
        times = v.index(times, len(values))
        names = (
            tuple(f"y{i}" for i in range(values.shape[1]))
            if self.names is None
            else tuple(self.names)
        )
        if (
            len(names) != values.shape[1]
            or len(set(names)) != len(names)
            or any(not isinstance(x, str) or not x for x in names)
        ):
            raise ValueError("names must contain one unique nonempty string per target")
        cov = self.covariates
        if cov is not None:
            if not isinstance(cov, Covariates):
                raise TypeError("covariates must be Covariates")
            for x in (cov.known_future, cov.observed_past):
                if x is not None and len(x) != len(values):
                    raise ValueError("training covariates must align with target rows")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "names", names)

    def __reduce__(self):
        return type(self), (self.values, self.timestamps, self.covariates, self.names)

    def __len__(self):
        return len(self.values)

    @property
    def dimension(self):
        return self.values.shape[1]

    @property
    def cutoff(self):
        return self.timestamps[-1]

    @property
    def frequency(self):
        if len(self) < 2:
            return None
        delta = np.diff(self.timestamps)
        equal = (
            np.all(delta == delta[0])
            if delta.dtype.kind == "m"
            else np.allclose(delta, delta[0], rtol=1e-10, atol=0)
        )
        return delta[0] if equal else None

    def slice(self, start, stop) -> TimeSeries:
        return TimeSeries(
            self.values[start:stop],
            self.timestamps[start:stop],
            None if self.covariates is None else self.covariates.slice(start, stop),
            self.names,
        )

    def __init__(
        self,
        values: ArrayLike,
        timestamps: ArrayLike | None = None,
        covariates: Covariates | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "names", names)
        self.__post_init__()


@dataclass(frozen=True, init=False)
class ForecastHorizon:
    """Positive increasing relative row offsets; integer H means 1,...,H.

    Explicit timestamps label requested rows. They do not change a discrete
    model into an elapsed-time model. Irregular histories require these labels.
    """

    steps: IntArray
    timestamps: IndexArray | None
    calendar: object

    def __init__(
        self,
        steps: int | ArrayLike,
        timestamps: ArrayLike | None = None,
        calendar: object = None,
    ) -> None:
        if isinstance(steps, (int, np.integer)) and not isinstance(
            steps, (bool, np.bool_)
        ):
            steps = np.arange(1, v.integer(steps, "horizon", 1) + 1)
        raw = np.asarray(steps)
        if (
            raw.ndim != 1
            or not raw.size
            or raw.dtype.kind not in "iu"
            or np.any(raw <= 0)
            or np.any(raw[1:] <= raw[:-1])
            or np.any(raw > np.iinfo(np.int64).max)
        ):
            raise ValueError(
                "horizon steps must be strictly increasing positive integers"
            )
        out = np.array(raw, dtype=np.int64)
        out.setflags(write=False)
        object.__setattr__(self, "steps", out)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(
            self,
            "timestamps",
            None if timestamps is None else v.index(timestamps, len(out)),
        )

    def __reduce__(self):
        return type(self), (self.steps, self.timestamps, self.calendar)

    def __len__(self):
        return len(self.steps)

    @property
    def maximum(self) -> int:
        return int(self.steps[-1])

    def align(self, history, *, check_frequency: bool = True) -> ForecastHorizon:
        times = self.timestamps
        if times is None:
            frequency = history.frequency
            if frequency is None:
                raise ValueError(
                    "irregular or single-row histories require explicit future timestamps"
                )
            times = history.cutoff + frequency * self.steps
        if (np.asarray(times).dtype.kind == "M") != (
            history.timestamps.dtype.kind == "M"
        ):
            raise ValueError("future and history timestamps must have compatible types")
        if times[0] <= history.cutoff:
            raise ValueError("forecast timestamps must follow the training cutoff")
        if check_frequency and self.calendar is None and history.frequency is not None:
            expected = history.cutoff + history.frequency * self.steps
            equal = (
                np.array_equal(times, expected)
                if expected.dtype.kind == "M"
                else np.allclose(times, expected, rtol=1e-10, atol=0)
            )
            if not equal:
                raise ValueError(
                    "future timestamps must match the relative steps and training frequency"
                )
        return ForecastHorizon(self.steps, times, self.calendar)


def as_series(data: TimeSeries | ArrayLike, covariates: Covariates | None = None):
    if isinstance(data, TimeSeries):
        if covariates is not None:
            raise ValueError("supply covariates in TimeSeries or fit, not both")
        return data
    return TimeSeries(data, covariates=covariates)


def prepare(
    data: TimeSeries | ArrayLike,
    covariates: Covariates | None = None,
    *,
    missing: str = "raise",
):
    data = as_series(data, covariates)
    v.choice(missing, ("raise", "ffill", "native"), "missing")
    if np.isnan(data.values).any():
        if missing == "raise":
            raise ValueError("missing targets require an explicit missing policy")
        if missing == "ffill":
            y = data.values.copy()
            if np.isnan(y[0]).any():
                raise ValueError(
                    "forward filling requires an observed initial value in every target"
                )
            for t in range(1, len(y)):
                y[t] = np.where(np.isnan(y[t]), y[t - 1], y[t])
            data = TimeSeries(y, data.timestamps, data.covariates, data.names)
        elif np.any(np.all(np.isnan(data.values), axis=0)):
            raise ValueError("every target needs at least one observation")
    return data


def reject_covariates(
    data: TimeSeries,
    *,
    known: bool = False,
    observed: bool = False,
    static: bool = False,
):
    cov = data.covariates
    if cov is not None:
        for name, allowed in (
            ("known_future", known),
            ("observed_past", observed),
            ("static", static),
        ):
            if getattr(cov, name) is not None and not allowed:
                raise ValueError(f"this model does not support {name} covariates")


def calendar_horizon(
    cutoff: DateLike | np.datetime64,
    steps: int | ArrayLike | ForecastHorizon,
    *,
    frequency: str = "B",
    holidays: Sequence[DateLike | np.datetime64] = (),
) -> ForecastHorizon:
    """Explicit calendar offsets (business days, month ends, etc.) via pandas.

    Holidays are supported for business-day frequency. No exchange calendar is
    assumed: supply its holiday dates explicitly.
    """
    import pandas as pd

    h = steps if isinstance(steps, ForecastHorizon) else ForecastHorizon(steps)
    offset = (
        pd.offsets.CustomBusinessDay(holidays=list(holidays))
        if frequency == "B"
        else pd.tseries.frequencies.to_offset(frequency)
    )
    if holidays and frequency != "B":
        raise ValueError("holidays apply only to business-day frequency")
    start = pd.Timestamp(cutoff)
    dates = np.array([(start + int(step) * offset).to_datetime64() for step in h.steps])
    return ForecastHorizon(h.steps, dates, calendar=str(offset))
