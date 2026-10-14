"""SciPy owns densities, quantiles, random variates and scalar integration."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg, stats
from scipy.special import poch
from ..._backends.gaussian import sample as gaussian_sample, logpdf as gaussian_logpdf

from .. import _validation as v
from ..interfaces import Distribution, Measure

SCALAR_CAPABILITIES = frozenset(
    {
        "log_density",
        "sample",
        "cdf",
        "survival",
        "quantile",
        "support",
        "mean",
        "variance",
        "moment",
        "expect",
    }
)


@dataclass(frozen=True)
class ScalarAdapter:
    law: Any
    measure: Measure = "continuous"
    event_shape: tuple[int, ...] = ()
    capabilities: frozenset[str] = SCALAR_CAPABILITIES
    support_bounds: tuple[float, float] | None = None

    def evaluate(self, operation, *args, **kwargs) -> Any:
        if operation == "log_density":
            method = self.law.logpmf if self.measure == "discrete" else self.law.logpdf
            return method(args[0])
        if operation == "sample":
            return np.asarray(
                self.law.rvs(size=args[0], random_state=kwargs["rng"])
            ).reshape(args[0])
        if operation == "cdf":
            return self.law.cdf(args[0])
        if operation == "survival":
            return self.law.sf(args[0])
        if operation == "quantile":
            p = args[0]
            lower, upper = self.evaluate("support")
            # SciPy discrete ppf(0) is one step below support. Our endpoints are
            # support limits for both continuous and discrete laws.
            return np.where(p == 0, lower, np.where(p == 1, upper, self.law.ppf(p)))
        if operation == "variance":
            return self.law.var()
        if operation == "support":
            return (
                self.support_bounds
                if self.support_bounds is not None
                else self.law.support()
            )
        if operation == "expect":
            # Bounds/conditional flags would change the meaning of this method.
            if {"lb", "ub", "conditional"} & kwargs.keys():
                raise ValueError(
                    "expect integrates the full law; bounds/conditional are not supported"
                )

            def callback(x):
                points = np.asarray(x)
                return np.array(
                    [
                        v.scalar(args[0](float(point)), "expectation callback")
                        for point in points.flat
                    ]
                ).reshape(points.shape)

            return self.law.expect(callback, **kwargs)
        return getattr(self.law, operation)(*args, **kwargs)


def scalar_law(name, **parameters):
    generators = {
        "normal": stats.norm,
        "student_t": stats.t,
        "lognormal": stats.lognorm,
        "bernoulli": stats.bernoulli,
        "binomial": stats.binom,
        "poisson": stats.poisson,
    }
    measure: Measure = (
        "discrete" if name in {"bernoulli", "binomial", "poisson"} else "continuous"
    )
    bounds = None
    if name in {"bernoulli", "binomial"} and parameters["p"] in (0, 1):
        point = float(parameters.get("n", 1) * parameters["p"])
        bounds = (point, point)
    if name == "poisson" and parameters["mu"] == 0:
        bounds = (0.0, 0.0)
    return Distribution(
        ScalarAdapter(generators[name](**parameters), measure, support_bounds=bounds)
    )


@dataclass(frozen=True)
class EllipticalAdapter:
    location: Any
    matrix: Any
    df: float | None = None
    measure: Measure = "continuous"
    capabilities: frozenset[str] = frozenset(
        {
            "log_density",
            "sample",
            "cdf",
            "mean",
            "variance",
            "covariance",
            "moment",
            "marginal",
            "condition",
        }
    )

    @property
    def event_shape(self):
        return self.location.shape

    def _law(self, rng=None) -> Any:
        if self.df is None:
            generator: Any = stats.multivariate_normal
            return generator(
                self.location,
                stats.Covariance.from_cholesky(np.linalg.cholesky(self.matrix)),
                seed=rng,
            )
        # Some SciPy stubs incorrectly restrict df to int; the library accepts real df.
        generator: Any = stats.multivariate_t
        return generator(self.location, self.matrix, self.df, seed=rng)

    def evaluate(self, operation, *args, **kwargs) -> Any:
        d = self.event_shape[0]
        if operation in {"log_density", "cdf"}:
            points = args[0]
            shape = points.shape[:-1]
            if points.size == 0:
                return np.empty(shape)
            flat = points.reshape(-1, d)
            if operation == "log_density":
                # Infinite coordinates are outside any positive-definite density.
                finite = np.all(np.isfinite(flat), axis=-1)
                result = np.full(len(flat), -np.inf)
                if self.df is None:
                    result[finite] = gaussian_logpdf(
                        flat[finite], self.location, self.matrix
                    )
                elif np.any(finite):
                    units = np.sqrt(np.diag(self.matrix))
                    corr = self.matrix / units[:, None] / units[None, :]
                    root = np.linalg.cholesky(corr)
                    z = (flat[finite] - self.location) / units
                    white = linalg.solve_triangular(root, z.T, lower=True).T
                    log_volume = np.log(units).sum() + np.log(np.diag(root)).sum()
                    # Integer dimension permits a stable gamma-ratio product.
                    # Avoid subtracting enormous log-gammas in the normal limit.
                    half_df = self.df / 2
                    offset = 0.5 if d % 2 else 0.0
                    normalization = np.log1p(
                        (np.arange(d // 2) + offset) / half_df
                    ).sum()
                    if d % 2:
                        normalization += np.log(poch(half_df, 0.5) / np.sqrt(half_df))
                    result[finite] = (
                        normalization
                        - d / 2 * np.log(2 * np.pi)
                        - log_volume
                        - (self.df + d)
                        / 2
                        * np.log1p(np.sum(white**2, axis=1) / self.df)
                    )
                return result.reshape(shape)
            rng = v.generator(kwargs.get("rng"))
            result = np.empty(len(flat))
            finite = np.all(np.isfinite(flat), axis=-1)
            if np.any(finite):
                if d == 1:
                    z = (flat[finite, 0] - self.location[0]) / np.sqrt(
                        self.matrix[0, 0]
                    )
                    result[finite] = (
                        stats.norm.cdf(z)
                        if self.df is None
                        else stats.t.cdf(z, self.df)
                    )
                elif self.df is None:
                    result[finite] = self._law(rng).cdf(flat[finite])
                else:
                    units = np.sqrt(np.diag(self.matrix))
                    corr = self.matrix / units[:, None] / units[None, :]
                    generator: Any = stats.multivariate_t
                    law = generator(shape=corr, df=self.df)
                    result[finite] = law.cdf(
                        (flat[finite] - self.location) / units, random_state=rng
                    )
            for i in np.flatnonzero(~finite):
                point = flat[i]
                if np.any(np.isneginf(point)):
                    result[i] = 0.0
                elif np.all(np.isposinf(point)):
                    result[i] = 1.0
                else:
                    keep = np.flatnonzero(np.isfinite(point))
                    result[i] = self.evaluate("marginal", keep).cdf(
                        point[keep], rng=rng
                    )
            return result.reshape(shape)
        if operation == "sample":
            shape = args[0]
            if 0 in shape:
                return np.empty(shape + self.event_shape)
            rng = kwargs["rng"]
            noise = gaussian_sample(np.zeros(d), self.matrix, shape, rng)
            if self.df is not None:
                noise = (
                    noise
                    / np.sqrt(rng.chisquare(self.df, size=shape) / self.df)[..., None]
                )
            return self.location + noise
        if operation == "mean":
            return (
                self.location if self.df is None or self.df > 1 else np.full(d, np.nan)
            )
        if operation in {"variance", "covariance"}:
            if self.df is None:
                covariance = self.matrix
            elif self.df > 2:
                covariance = self.matrix * (self.df / (self.df - 2))
            else:
                # Off-diagonal infinite covariances have no canonical sign/value.
                covariance = np.full((d, d), np.nan)
                if self.df > 1:
                    np.fill_diagonal(covariance, np.inf)
            return np.diag(covariance) if operation == "variance" else covariance
        if operation == "moment":
            generator = stats.norm if self.df is None else stats.t
            params = {} if self.df is None else {"df": self.df}
            return generator(
                loc=self.location, scale=np.sqrt(np.diag(self.matrix)), **params
            ).moment(args[0])
        if operation == "marginal":
            keep = args[0]
            location = v.array(self.location[keep], "marginal location")
            matrix = v.array(self.matrix[np.ix_(keep, keep)], "marginal shape")
            return Distribution(EllipticalAdapter(location, matrix, self.df))
        if operation == "condition":
            observed, values = args
            remaining = np.setdiff1d(np.arange(d), observed)
            aa = self.matrix[np.ix_(remaining, remaining)]
            ab = self.matrix[np.ix_(remaining, observed)]
            bb = self.matrix[np.ix_(observed, observed)]
            delta = values - self.location[observed]
            factor = linalg.cho_factor(bb)
            solved_delta = linalg.cho_solve(factor, delta)
            location = self.location[remaining] + ab @ solved_delta
            matrix = aa - ab @ linalg.cho_solve(factor, ab.T)
            df = self.df
            if df is not None:
                matrix = matrix * ((df + delta @ solved_delta) / (df + len(observed)))
                df += len(observed)
            location, matrix = v.location_matrix(location, matrix, "conditional shape")
            return Distribution(EllipticalAdapter(location, matrix, df))
        raise NotImplementedError(operation)
