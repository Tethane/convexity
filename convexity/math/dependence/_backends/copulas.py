"""statsmodels copulas; SciPy supplies reproducible elliptical CDFs."""

from dataclasses import dataclass
from inspect import signature
from typing import Any

import numpy as np

from .. import _validation as v


@dataclass(frozen=True)
class Adapter:
    family: str
    dimension: int
    correlation: Any = None
    df: float | None = None
    theta: float | None = None
    capabilities: frozenset[str] = frozenset(
        {"log_density", "cdf", "sample", "tail_dependence"}
    )

    def model(self):
        from statsmodels.distributions.copula.api import (
            ClaytonCopula,
            GaussianCopula,
            GumbelCopula,
            StudentTCopula,
        )

        if self.family == "gaussian":
            return GaussianCopula(corr=self.correlation, k_dim=self.dimension)
        if self.family == "student_t":
            return StudentTCopula(
                corr=self.correlation, df=self.df, k_dim=self.dimension
            )
        if self.family == "clayton":
            return ClaytonCopula(theta=self.theta, k_dim=2)
        return GumbelCopula(theta=self.theta, k_dim=2)

    def evaluate(self, operation, *args, **kwargs):
        from scipy import stats

        independent = self.family == "independence" or (
            self.family == "gumbel" and self.theta == 1
        )
        if operation == "tail_dependence":
            if independent or self.family == "gaussian":
                return (0.0, 0.0)
            if self.family == "student_t":
                rho, df = self.correlation[args[0], args[1]], self.df
                # Use the analytic pair formula; older statsmodels versions had
                # a precedence error in dependence_tail.
                assert df is not None
                value = 2 * stats.t.cdf(
                    -np.sqrt((df + 1) * (1 - rho) / (1 + rho)), df + 1
                )
                return (value, value)
            assert self.theta is not None
            return (
                (2 ** (-1 / self.theta), 0.0)
                if self.family == "clayton"
                else (0.0, 2 - 2 ** (1 / self.theta))
            )
        if operation == "sample":
            shape = args[0]
            n = int(np.prod(shape, dtype=int))
            if n == 0:
                return np.empty(shape + (self.dimension,))
            if independent:
                return kwargs["rng"].uniform(size=shape + (self.dimension,))
            model = self.model()
            key = "rng" if "rng" in signature(model.rvs).parameters else "random_state"
            return np.asarray(model.rvs(nobs=n, **{key: kwargs["rng"]})).reshape(
                shape + (self.dimension,)
            )
        u = args[0]
        shape = u.shape[:-1]
        flat = u.reshape(-1, self.dimension)
        if operation == "log_density":
            if independent or not len(flat):
                return np.zeros(shape)
            return np.asarray(self.model().logpdf(flat)).reshape(shape)
        if independent:
            return np.prod(u, axis=-1)
        if self.family in ("gaussian", "student_t"):
            return elliptical_cdf(
                flat, self.correlation, self.df, v.generator(kwargs["rng"])
            ).reshape(shape)
        result = np.empty(len(flat))
        zeros = np.any(flat == 0, axis=1)
        ones = np.any(flat == 1, axis=1)
        result[zeros] = 0
        result[ones & ~zeros] = np.min(flat[ones & ~zeros], axis=1)
        active = ~zeros & ~ones
        if np.any(active):
            result[active] = self.model().cdf(flat[active])
        return result.reshape(shape)


def elliptical_cdf(uniforms, correlation, df, rng):
    from scipy import stats

    result = np.empty(len(uniforms))
    # Boundary faces reduce to the appropriate marginal law, avoiding inf-inf
    # arithmetic in backend quadrature. One active coordinate has exact CDF u.
    for index, u in enumerate(uniforms):
        if np.any(u == 0):
            result[index] = 0.0
            continue
        keep = np.flatnonzero(u < 1)
        if len(keep) < 2:
            result[index] = u[keep[0]] if len(keep) else 1.0
            continue
        matrix = correlation[np.ix_(keep, keep)]
        if df is None:
            model = stats.multivariate_normal(cov=matrix, seed=rng)
            result[index] = model.cdf(stats.norm.ppf(u[keep]))
        else:
            generator: Any = stats.multivariate_t
            model = generator(shape=matrix, df=df)
            result[index] = model.cdf(stats.t.ppf(u[keep], df), random_state=rng)
    return result


def fit_parameter(family, x, df):
    # Delegate tau inversion to statsmodels on each pair. Pair calls work
    # consistently with both 0.14 and newer multidimensional implementations.
    d = x.shape[1]
    adapter = Adapter(family, 2, correlation=np.eye(2), df=df, theta=2.0)
    model = adapter.model()
    if family in ("clayton", "gumbel"):
        return float(model.fit_corr_param(x))
    result = np.eye(d)
    for i in range(d):
        for j in range(i):
            result[i, j] = result[j, i] = model.fit_corr_param(x[:, [i, j]])
    return result
