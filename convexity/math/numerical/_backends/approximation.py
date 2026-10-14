import numpy as np
from typing import Any, cast
from scipy import interpolate
from scipy.spatial import Delaunay
from ..interfaces.base import Diagnostics
from ..interfaces.interpolation import Interpolant
from ..interfaces.approximation import MultivariateInterpolant, PolynomialResult


def spline(problem, *, degree, boundary):
    x, y = np.asarray(problem.x), np.asarray(problem.y)
    if x.size < degree + 1:
        raise ValueError("spline requires at least degree+1 samples")
    fitted = interpolate.make_interp_spline(x, y, k=degree, bc_type=boundary, axis=0)
    return Interpolant(
        fitted, (float(x[0]), float(x[-1])), y.shape[1:], problem.extrapolate
    )


def multivariate(problem, *, method):
    if problem.layout == "regular":
        if method not in ("linear", "nearest", "cubic"):
            raise ValueError("regular interpolation supports linear, nearest, or cubic")
        if method == "cubic" and any(len(x) < 4 for x in problem.points):
            raise ValueError(
                "cubic grid interpolation needs at least four points on every axis"
            )
        fitted = interpolate.RegularGridInterpolator(
            problem.points,
            problem.values,
            method=method,
            bounds_error=problem.extrapolation == "raise",
            fill_value=cast(
                Any, None if problem.extrapolation == "extrapolate" else np.nan
            ),
        )

        def evaluate_regular(points):
            return fitted(points)

        return MultivariateInterpolant(
            evaluate_regular,
            problem.dimension,
            problem.output_shape,
            problem.extrapolation,
        )
    else:
        if method not in ("linear", "nearest", "cubic"):
            raise ValueError(
                "scattered interpolation supports linear, nearest, or 2-D cubic"
            )
        if problem.extrapolation == "extrapolate" and method != "nearest":
            raise ValueError(
                "scattered extrapolation is supported only by nearest interpolation"
            )
        hull = Delaunay(problem.points)
        if method == "nearest":
            fitted = interpolate.NearestNDInterpolator(problem.points, problem.values)
        elif method == "linear":
            fitted = interpolate.LinearNDInterpolator(hull, problem.values)
        else:
            if problem.dimension != 2:
                raise ValueError(
                    "scattered cubic interpolation requires two dimensions"
                )
            fitted = interpolate.CloughTocher2DInterpolator(hull, problem.values)

        def evaluate(points):
            outside = hull.find_simplex(points) < 0
            if problem.extrapolation == "raise" and np.any(outside):
                raise ValueError("query is outside the sample convex hull")
            values = np.asarray(fitted(points))
            if problem.extrapolation == "nan":
                values[outside] = np.nan
            return values

    return MultivariateInterpolant(
        evaluate, problem.dimension, problem.output_shape, problem.extrapolation
    )


def polynomial(problem, *, basis):
    x, y = np.asarray(problem.x), np.asarray(problem.y)
    constructor = (
        np.polynomial.Polynomial if basis == "polynomial" else np.polynomial.Chebyshev
    )
    fitted, info = constructor.fit(x, y, problem.degree, full=True)
    rank = int(info[1])
    # Conversion affects only reported coefficients; evaluation retains the scaled fit.
    coefficients = fitted.convert(kind=np.polynomial.Polynomial).coef
    coefficients = np.pad(coefficients, (0, problem.degree + 1 - len(coefficients)))
    return PolynomialResult(
        Interpolant(
            fitted, (float(x[0]), float(x[-1])), extrapolate=problem.extrapolate
        ),
        coefficients,
        rank,
        Diagnostics(
            rank == problem.degree + 1,
            ""
            if rank == problem.degree + 1
            else "rank-deficient polynomial approximation",
            residual_norm=float(np.linalg.norm(fitted(x) - y)),
        ),
    )
