"""Private SciPy interpolation adapter."""

import numpy as np
from scipy import interpolate

from ..interfaces import Interpolant, InterpolationProblem


def fit(problem: InterpolationProblem, *, method: str) -> Interpolant:
    x, y = np.asarray(problem.x), np.asarray(problem.y)
    if method == "linear":
        fitted = interpolate.make_interp_spline(x, y, k=1, axis=0)
    elif method == "cubic":
        fitted = interpolate.CubicSpline(x, y, axis=0, extrapolate=problem.extrapolate)
    else:
        fitted = interpolate.PchipInterpolator(
            x, y, axis=0, extrapolate=problem.extrapolate
        )
    return Interpolant(
        fitted, (float(x[0]), float(x[-1])), y.shape[1:], problem.extrapolate
    )
