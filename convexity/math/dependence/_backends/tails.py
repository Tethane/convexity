"""SciPy densities and optimizers, with explicit convergence propagation."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..interfaces.base import Diagnostics
from ..interfaces.tails import StudentTResult, TailFit


def frozen_tail(family, xi, location, scale) -> Any:
    from scipy import stats

    if family == "gpd":
        return stats.genpareto(xi, loc=location, scale=scale)
    return stats.genextreme(-xi, loc=location, scale=scale)


@dataclass(frozen=True)
class TailAdapter:
    """The family-2 public adapter protocol supplies validation and composition."""

    family: str
    xi: float
    location: float
    scale: float
    measure: Literal["continuous"] = "continuous"
    event_shape: tuple[int, ...] = ()
    capabilities: frozenset[str] = frozenset(
        {
            "log_density",
            "cdf",
            "survival",
            "quantile",
            "sample",
            "support",
            "mean",
            "variance",
            "moment",
        }
    )

    def evaluate(self, operation: str, *args, **kwargs):
        law = frozen_tail(self.family, self.xi, self.location, self.scale)
        if operation == "sample":
            return np.asarray(
                law.rvs(size=args[0], random_state=kwargs["rng"])
            ).reshape(args[0])
        methods = {
            "log_density": "logpdf",
            "survival": "sf",
            "quantile": "ppf",
            "variance": "var",
        }
        return getattr(law, methods.get(operation, operation))(*args)


def fit_tail(x, *, family, maxiter, tolerance):
    from scipy import optimize, stats

    report = None

    def optimizer(function, initial, args=(), disp=0):
        nonlocal report
        report = optimize.minimize(
            function,
            initial,
            args=args,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": tolerance, "fatol": tolerance},
        )
        return report.x

    generator = stats.genpareto if family == "gpd" else stats.genextreme
    options = {"floc": 0.0} if family == "gpd" else {}
    shape, location, scale = generator.fit(x, optimizer=optimizer, **options)
    xi = shape if family == "gpd" else -shape
    values = frozen_tail(family, xi, location, scale).logpdf(x)
    if not np.isfinite(values).all():
        raise ValueError(
            "extreme-value fit has nonfinite likelihood or observations outside fitted support"
        )
    assert report is not None
    return TailFit(
        family,
        float(xi),
        float(location),
        float(scale),
        len(x),
        float(np.sum(values)),
        Diagnostics(
            backend=f"scipy.stats.{generator.name}.fit / Nelder-Mead",
            converged=bool(report.success),
            message=str(report.message),
            iterations=int(report.nit),
        ),
    )


def fit_student_t(x, *, df, estimate_df, df_bounds, maxiter, tolerance):
    from scipy import linalg, optimize, stats

    d = x.shape[1]
    initial_cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=0))
    if np.linalg.matrix_rank(initial_cov, hermitian=True) < d:
        raise ValueError("Student-t fitting requires full column rank after centering")
    initial_shape = initial_cov * ((df - 2) / df if df > 2 else 1.0)
    root = linalg.cholesky(initial_shape, lower=True)
    rows, columns = np.tril_indices(d)
    diagonal = rows == columns
    packed = root[rows, columns]
    packed[diagonal] = np.log(packed[diagonal])
    initial = (
        np.r_[x.mean(axis=0), packed, np.log(df)]
        if estimate_df
        else np.r_[x.mean(axis=0), packed]
    )

    def unpack(theta):
        chol = np.zeros((d, d))
        parameters = theta[d : d + len(rows)].copy()
        parameters[diagonal] = np.exp(parameters[diagonal])
        chol[rows, columns] = parameters
        degrees = np.exp(theta[-1]) if estimate_df else df
        return theta[:d], chol @ chol.T, degrees

    generator: Any = stats.multivariate_t

    def objective(theta):
        with np.errstate(
            over="ignore", invalid="ignore", divide="ignore", under="ignore"
        ):
            loc, shape, degrees = unpack(theta)
            if not np.isfinite(shape).all() or not np.isfinite(degrees):
                return np.inf
            try:
                # The unfrozen logpdf permits a numerically singular shape.
                # A frozen law can enforce full-dimensional support explicitly.
                value = -np.sum(
                    generator(
                        loc=loc, shape=shape, df=degrees, allow_singular=False
                    ).logpdf(x)
                )
            except ValueError, np.linalg.LinAlgError:
                return np.inf
            return float(value) if np.isfinite(value) else np.inf

    bounds = [(None, None)] * len(initial)
    if estimate_df:
        bounds[-1] = tuple(np.log(df_bounds))
    if not np.isfinite(objective(initial)):
        raise ValueError(
            "initial Student-t shape is numerically singular; rescale the supplied coordinates"
        )
    report = optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": tolerance, "maxls": 40},
    )
    location, matrix, degrees = unpack(report.x)
    message = str(report.message)
    if estimate_df and (
        np.isclose(degrees, df_bounds[0]) or np.isclose(degrees, df_bounds[1])
    ):
        message += "; fitted df reached a configured bound"
    return StudentTResult(
        location,
        matrix,
        degrees,
        nobs=len(x),
        log_likelihood=-objective(report.x),
        diagnostics=Diagnostics(
            backend="scipy.stats.multivariate_t / L-BFGS-B",
            converged=bool(report.success),
            message=message,
            iterations=int(report.nit),
        ),
    )
