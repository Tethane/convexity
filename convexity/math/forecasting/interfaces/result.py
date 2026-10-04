"""Checked forecasts distinguish point estimates, marginals, and joint laws."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray

if TYPE_CHECKING:
    from ...distributions import Distribution
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from ..._typing import _FACTORY
from .. import _validation as v
from .data import ForecastHorizon


@dataclass(frozen=True)
class ForecastResult:
    """Forecast axes are always (horizon,target); paths are (draw,horizon,target).

    joint_covariance uses time-major flattened coordinates. gaussian=True
    declares Gaussian marginals only; joint_covariance additionally declares a
    joint Gaussian law. Paths define an equally weighted empirical joint law.
    A supplied distribution has event_shape (horizon * targets,). Scalar
    marginal_laws may accompany it to provide exact marginal quantiles. Their
    finite moments are checked; the caller must ensure equality of the full laws. Without
    a supplied joint law, marginal_laws declare an independent product.
    """

    point: FloatArray
    horizon: ForecastHorizon = cast("ForecastHorizon", None)
    mean_values: FloatArray | None = cast("FloatArray | None", None)
    variance_values: FloatArray | None = cast("FloatArray | None", None)
    quantile_levels: FloatArray | None = cast("FloatArray | None", None)
    quantile_values: FloatArray | None = cast("FloatArray | None", None)
    paths: FloatArray | None = cast("FloatArray | None", None)
    joint_covariance: FloatArray | None = cast("FloatArray | None", None)
    gaussian: bool = False
    law: Distribution | None = field(default=None, repr=False)
    marginal_laws: tuple[Distribution, ...] | None = field(default=None, repr=False)
    model_identity: str = "user"
    cutoff: float | int | np.datetime64 | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        point = v.matrix(self.point, "point")
        object.__setattr__(self, "point", point)
        h = ForecastHorizon(len(point)) if self.horizon is None else self.horizon
        if not isinstance(h, ForecastHorizon) or len(h) != len(point):
            raise ValueError("horizon must align with point rows")
        object.__setattr__(self, "horizon", h)
        v.boolean(self.gaussian, "gaussian")
        for name in ("mean_values", "variance_values"):
            raw = getattr(self, name)
            if raw is not None:
                out = v.matrix(raw, name)
                if out.shape != point.shape or (
                    name == "variance_values" and np.any(out < 0)
                ):
                    raise ValueError(
                        f"{name} must align with point and variances must be nonnegative"
                    )
                object.__setattr__(self, name, out)
        if (self.quantile_levels is None) != (self.quantile_values is None):
            raise ValueError("quantile values and levels must be supplied together")
        if self.quantile_levels is not None:
            levels = v.levels(self.quantile_levels)
            values = v.array(self.quantile_values, "quantiles", ndim=3)
            if values.shape != point.shape + (len(levels),) or np.any(
                np.diff(values, axis=-1) < 0
            ):
                raise ValueError("quantiles must have shape (h,k,q) and cannot cross")
            object.__setattr__(self, "quantile_levels", levels)
            object.__setattr__(self, "quantile_values", values)
        if (
            self.paths is not None
            or self.law is not None
            or self.marginal_laws is not None
        ) and (self.mean_values is not None or self.variance_values is not None):
            raise ValueError(
                "moments are derived from the supplied paths/law; do not supply separate moments"
            )
        if self.paths is not None:
            paths = v.array(self.paths, "paths", ndim=3)
            if paths.shape[1:] != point.shape:
                raise ValueError("paths must have shape (draw,horizon,target)")
            object.__setattr__(self, "paths", paths)
        if self.joint_covariance is not None:
            if not self.gaussian:
                raise ValueError(
                    "joint_covariance currently requires an explicit Gaussian law"
                )
            cov = v.covariance(self.joint_covariance, point.size)
            variance = np.diag(cov).reshape(point.shape)
            if self.variance_values is not None and not np.allclose(
                variance, self.variance_values, rtol=1e-10, atol=0
            ):
                raise ValueError(
                    "joint covariance diagonal must match marginal variances"
                )
            object.__setattr__(self, "joint_covariance", cov)
            object.__setattr__(self, "variance_values", v.array(variance, "variance"))
        if self.gaussian and (self.mean_values is None or self.variance_values is None):
            raise ValueError("Gaussian forecasts require explicit means and variances")
        if (
            sum(
                (
                    self.paths is not None,
                    self.gaussian,
                    self.law is not None or self.marginal_laws is not None,
                )
            )
            > 1
        ):
            raise ValueError(
                "supply one probability representation: paths, Gaussian, or law"
            )
        if self.marginal_laws is not None:
            from ...distributions import Distribution, independent

            laws = tuple(self.marginal_laws)
            if len(laws) != point.size or any(
                not isinstance(x, Distribution) or x.event_shape for x in laws
            ):
                raise ValueError(
                    "marginal_laws must contain one scalar Distribution per coordinate"
                )
            object.__setattr__(self, "marginal_laws", laws)
            if self.law is None:
                object.__setattr__(self, "law", independent(laws))
        if self.law is not None:
            from ...distributions import Distribution

            if not isinstance(self.law, Distribution) or self.law.event_shape != (
                point.size,
            ):
                raise ValueError(
                    "law must be a Convexity Distribution over flattened forecast coordinates"
                )
        if self.law is not None and self.marginal_laws is not None:
            for operation in ("mean", "variance"):
                if operation in self.law.capabilities and all(
                    operation in law.capabilities for law in self.marginal_laws
                ):
                    joint = np.asarray(getattr(self.law, operation)())
                    marginal = np.array(
                        [float(getattr(law, operation)()) for law in self.marginal_laws]
                    )
                    if not np.allclose(
                        joint, marginal, rtol=1e-10, atol=0, equal_nan=True
                    ):
                        raise ValueError(
                            "marginal laws must agree with joint law moments"
                        )
        if not isinstance(self.model_identity, str) or not self.model_identity:
            raise ValueError("model_identity must be a nonempty string")
        object.__setattr__(self, "metadata", deepcopy(self.metadata))

    @classmethod
    def from_independent(cls, laws, shape: tuple[int, int], **kwargs):
        laws = tuple(laws)
        point = np.array([float(law.mean()) for law in laws]).reshape(shape)
        return cls(point, marginal_laws=laws, **kwargs)

    @classmethod
    def from_paths(cls, paths: ArrayLike, **kwargs):
        x = v.array(paths, "paths", ndim=3)
        return cls(np.mean(x, axis=0), paths=x, **kwargs)

    @property
    def shape(self):
        return self.point.shape

    @property
    def _invertible_marginals(self):
        return self.marginal_laws is not None and all(
            "quantile" in law.capabilities for law in self.marginal_laws
        )

    @property
    def capabilities(self):
        caps = {"point"}
        if (
            self.mean_values is not None
            or self.paths is not None
            or (self.law is not None and "mean" in self.law.capabilities)
        ):
            caps.add("mean")
        if (
            self.variance_values is not None
            or self.paths is not None
            or (self.law is not None and "variance" in self.law.capabilities)
        ):
            caps.add("variance")
        if (
            self.gaussian
            or self.paths is not None
            or self.quantile_values is not None
            or self._invertible_marginals
        ):
            caps |= {"quantile", "interval"}
        if (
            self.gaussian
            or self.paths is not None
            or (self.law is not None and "marginal" in self.law.capabilities)
        ):
            caps.add("marginal")
        if self.marginal_laws is not None:
            caps.add("marginal")
        if self.joint_covariance is not None or self.paths is not None:
            caps |= {"sample", "distribution", "log_density"}
        elif self.law is not None:
            caps.add("distribution")
            caps |= {"sample", "log_density"} & self.law.capabilities
        if (
            self.joint_covariance is not None
            or self.paths is not None
            or (self.law is not None and "covariance" in self.law.capabilities)
        ):
            caps |= {"covariance", "correlation"}
        if self.paths is not None:
            caps.add("probability")
        return frozenset(caps)

    def mean(self) -> FloatArray:
        """Return expectations with shape (horizon, target); point predictions alone do not supply means."""
        if self.mean_values is not None:
            return self.mean_values
        if self.paths is not None:
            return v.array(self.paths.mean(axis=0), "mean")
        if self.law is not None:
            return v.array(self.law.mean().reshape(self.shape), "mean")
        raise NotImplementedError("this forecast supplies no mean")

    def variance(self) -> FloatArray:
        """Return marginal variances with shape (horizon, target), in squared target units."""
        if self.variance_values is not None:
            return self.variance_values
        if self.paths is not None:
            return v.array(self.paths.var(axis=0), "variance")
        if self.law is not None:
            return v.array(self.law.variance().reshape(self.shape), "variance")
        raise NotImplementedError("this forecast supplies no variance")

    def quantile(self, probability: float) -> FloatArray:
        """Return one marginal quantile per (horizon, target); unsupplied levels may be unavailable."""
        p = v.probability(probability)
        if self.gaussian:
            from scipy.special import ndtri

            return v.array(
                self.mean() + np.sqrt(self.variance()) * ndtri(p), "quantile"
            )
        if self.paths is not None:
            # Inverse empirical CDF, consistent with the empirical marginal law.
            return v.array(
                np.quantile(self.paths, p, axis=0, method="inverted_cdf"), "quantile"
            )
        if self._invertible_marginals:
            assert self.marginal_laws is not None
            return v.array(
                np.array(
                    [float(law.quantile(p)) for law in self.marginal_laws]
                ).reshape(self.shape),
                "quantile",
            )
        if self.quantile_levels is not None:
            assert self.quantile_values is not None
            where = np.flatnonzero(
                np.isclose(self.quantile_levels, p, atol=1e-12, rtol=0)
            )
            if len(where):
                return v.array(self.quantile_values[..., where[0]], "quantile")
            raise NotImplementedError(
                "only explicitly supplied quantile levels are available"
            )
        raise NotImplementedError("this forecast supplies no marginal quantiles")

    def interval(self, probability: float = 0.95) -> FloatArray:
        """Return marginal lower/upper bounds with shape (horizon, target, 2); not a simultaneous band."""
        p = v.probability(probability)
        return v.array(
            np.stack((self.quantile((1 - p) / 2), self.quantile((1 + p) / 2)), axis=-1),
            "interval",
        )

    def covariance(self) -> FloatArray:
        """Return (horizon * target, horizon * target) covariance in time-major order."""
        if self.joint_covariance is not None:
            return self.joint_covariance
        if self.paths is not None:
            flat = self.paths.reshape(len(self.paths), -1)
            centered = flat - flat.mean(axis=0)
            return v.array(centered.T @ centered / len(flat), "covariance")
        if self.law is not None:
            return self.law.covariance()
        raise NotImplementedError("marginal forecasts do not define joint covariance")

    def correlation(self) -> FloatArray:
        cov = self.covariance()
        scale = np.sqrt(np.diag(cov))
        if np.any(scale == 0):
            raise ValueError("correlation is undefined for zero-variance coordinates")
        return v.array(cov / np.outer(scale, scale), "correlation")

    def distribution(self) -> Distribution:
        """Return the joint law of the time-major flattened path; marginal intervals alone cannot define it."""
        from ... import distributions as dist

        if self.law is not None:
            return self.law
        if self.paths is not None:
            return dist.empirical(self.paths.reshape(len(self.paths), -1))
        if self.joint_covariance is not None:
            return dist.Distribution(
                _GaussianAdapter(self.mean().ravel(), self.joint_covariance)
            )
        raise NotImplementedError("this forecast does not define a joint path law")

    def marginal(self, horizon_index: int, target: int = 0) -> Distribution:
        """Return the law of a zero-based horizon row and target coordinate."""
        from ... import distributions as dist

        h = v.integer(horizon_index, "horizon_index")
        k = v.integer(target, "target")
        if h >= self.shape[0] or k >= self.shape[1]:
            raise ValueError("marginal coordinate out of range")
        if self.marginal_laws is not None:
            return self.marginal_laws[h * self.shape[1] + k]
        if self.paths is not None:
            return dist.empirical(self.paths[:, h, k])
        if self.gaussian:
            mean, variance = self.mean()[h, k], self.variance()[h, k]
            return (
                dist.normal(mean, np.sqrt(variance))
                if variance > 0
                else dist.empirical([mean])
            )
        if self.law is not None:
            return self.law.marginal([h * self.shape[1] + k])
        raise NotImplementedError("quantile levels alone do not define a marginal law")

    def sample(self, size: int, *, rng: np.random.Generator) -> FloatArray:
        """Draw complete paths with shape (size, horizon, target), retaining supplied dependence."""
        n = v.integer(size, "size")
        v.generator(rng)
        return v.array(
            self.distribution().sample(n, rng=rng).reshape((n,) + self.shape),
            "paths",
            empty=True,
        )

    def log_density(self, values: ArrayLike) -> FloatArray:
        """Evaluate joint log density for values ending in (horizon, target); preserve leading batch axes."""
        x = v.array(values, "values")
        if x.shape[-2:] != self.shape:
            raise ValueError("values must end in (horizon,target)")
        return self.distribution().log_density(
            x.reshape(x.shape[:-2] + (self.point.size,))
        )

    def probability(self, event: Callable[[FloatArray], bool | np.bool_]) -> float:
        """Return empirical mass of a predicate on complete paths, each shaped (horizon, target)."""
        if self.paths is None:
            raise NotImplementedError(
                "predicate probability is exact only for empirical path forecasts"
            )
        if not callable(event):
            raise TypeError("event must be callable")
        decisions = [event(path) for path in self.paths]
        if any(not isinstance(x, (bool, np.bool_)) for x in decisions):
            raise ValueError("event must return one boolean per complete path")
        return float(np.mean(np.asarray(decisions, dtype=bool)))

    def select(
        self, indices: ArrayLike, *, horizon: int | ForecastHorizon | None = None
    ) -> ForecastResult:
        """Select rows, retaining cross-time dependence and coordinate order."""
        ix = np.asarray(indices)
        if (
            ix.ndim != 1
            or not ix.size
            or ix.dtype.kind not in "iu"
            or np.any(ix < 0)
            or np.any(ix >= len(self.point))
            or np.any(np.diff(ix.astype(np.int64)) <= 0)
        ):
            raise ValueError("indices must increase strictly within forecast rows")
        coords = (ix[:, None] * self.shape[1] + np.arange(self.shape[1])).ravel()
        old: ForecastHorizon = self.horizon
        new_h = horizon or ForecastHorizon(
            old.steps[ix],
            None if old.timestamps is None else old.timestamps[ix],
            old.calendar,
        )
        if np.array_equal(ix, np.arange(len(self.point))):
            return replace(self, horizon=new_h)
        return replace(
            self,
            point=self.point[ix],
            horizon=new_h,
            mean_values=None if self.mean_values is None else self.mean_values[ix],
            variance_values=None
            if self.variance_values is None
            else self.variance_values[ix],
            quantile_values=None
            if self.quantile_values is None
            else self.quantile_values[ix],
            paths=None if self.paths is None else self.paths[:, ix],
            joint_covariance=None
            if self.joint_covariance is None
            else self.joint_covariance[np.ix_(coords, coords)],
            law=None if self.law is None else self.law.marginal(coords),
            marginal_laws=None
            if self.marginal_laws is None
            else tuple(self.marginal_laws[i] for i in coords),
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
        self.__post_init__()


@dataclass(frozen=True)
class _GaussianAdapter:
    mean_values: FloatArray
    covariance_values: FloatArray
    measure: Literal["continuous"] = "continuous"
    capabilities = frozenset(
        {"mean", "variance", "covariance", "sample", "log_density"}
    )

    @property
    def event_shape(self):
        return self.mean_values.shape

    def evaluate(self, operation, *args, **kwargs):
        if operation == "mean":
            return self.mean_values
        if operation == "variance":
            return np.diag(self.covariance_values)
        if operation == "covariance":
            return self.covariance_values
        if operation == "sample":
            from ..._backends.gaussian import sample

            return sample(
                self.mean_values, self.covariance_values, args[0], kwargs["rng"]
            )
        from ..._backends.gaussian import logpdf

        return logpdf(args[0], self.mean_values, self.covariance_values)

    def __init__(
        self,
        mean_values: ArrayLike,
        covariance_values: ArrayLike,
        measure: Literal["continuous"] = "continuous",
    ) -> None:
        object.__setattr__(self, "mean_values", mean_values)
        object.__setattr__(self, "covariance_values", covariance_values)
        object.__setattr__(self, "measure", measure)
