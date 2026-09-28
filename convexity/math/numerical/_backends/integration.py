import warnings
from typing import Any, cast
import numpy as np
from scipy import integrate, special
from .. import _validation as v
from ..interfaces.base import ArrayResult, Diagnostics
from ..interfaces.integration import QuadratureRule


def box_integral(problem, *, atol, rtol, limit):
    def function(*coordinates):
        return v.scalar(problem.function(np.asarray(coordinates)), "integrand output")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", integrate.IntegrationWarning)
        value, error, info = cast(tuple[Any, Any, Any], integrate.nquad(function, problem.bounds,
                                             opts={"epsabs": atol, "epsrel": rtol, "limit": limit}, full_output=True))
    failures = [str(w.message) for w in caught if issubclass(w.category, integrate.IntegrationWarning)]
    for warning in caught:
        if not issubclass(warning.category, integrate.IntegrationWarning):
            warnings.warn(warning.message, warning.category, stacklevel=2)
    # nquad full_output can suppress QUADPACK messages. Independently enforce
    # the reported (maximum nested absolute) error criterion.
    success = not failures and error <= max(atol, rtol*abs(value))
    return ArrayResult(value, Diagnostics(success, "; ".join(failures) or ("" if success else "nested quadrature error exceeds tolerance"),
                                          evaluations=int(info["neval"]), error_estimate=error))


def rule(problem):
    if problem.family == "legendre":
        x, w = special.roots_legendre(problem.n)
        half = (problem.b-problem.a)/2
        return QuadratureRule((problem.a+problem.b)/2+half*x, half*w)
    if problem.family == "hermite":
        x, w = special.roots_hermite(problem.n)
        return QuadratureRule(x, w, "exp(-x**2)")
    x, w = special.roots_laguerre(problem.n)
    return QuadratureRule(x, w, "exp(-x)")


def gaussian(problem, *, n):
    from ..interfaces.integration import QuadratureRuleProblem
    if not np.isfinite(problem.a) or not np.isfinite(problem.b) or problem.a == problem.b:
        if problem.a == problem.b:
            return ArrayResult(0., Diagnostics(evaluations=0))
        raise ValueError("Gaussian quadrature requires finite bounds")
    points = rule(QuadratureRuleProblem(n, a=min(problem.a, problem.b), b=max(problem.a, problem.b)))
    values = np.array([v.scalar(problem.function(x), "integrand output") for x in points.nodes])
    return ArrayResult(np.sign(problem.b-problem.a) * np.dot(points.weights, values), Diagnostics(evaluations=n))
