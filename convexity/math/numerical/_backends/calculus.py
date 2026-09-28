"""SciPy quadrature and adaptive finite-difference adapters."""

import numpy as np
from scipy import differentiate, integrate

from .. import _validation as v
from ..interfaces import ArrayResult, Diagnostics, DifferentiationProblem, QuadratureProblem


def quadrature(problem: QuadratureProblem, *, atol: float, rtol: float,
               limit: int) -> ArrayResult:
    def evaluate(x):
        return v.scalar(problem.function(x), "integrand output")

    raw = integrate.quad(evaluate, problem.a, problem.b, epsabs=atol, epsrel=rtol,
                         limit=limit, full_output=True)
    value, error, info = raw[:3]
    success = len(raw) == 3
    return ArrayResult(value, Diagnostics(converged=success, error_estimate=error,
                                         evaluations=int(info["neval"]),
                                         message="" if success else str(raw[3])))


def finite_difference(problem: DifferentiationProblem, *, atol: float, rtol: float,
                      order: int, maxiter: int, initial_step: float,
                      direction: int) -> ArrayResult:
    evaluations = 0
    output_shape = None

    def evaluate(point):
        nonlocal evaluations, output_shape
        evaluations += 1
        output = v.array(problem.function(point), "function output", real=True,
                         ndim=1 if problem.kind == "jacobian" else 0)
        if output_shape is not None and output.shape != output_shape:
            raise ValueError("function output shape changed during differentiation")
        output_shape = output.shape
        return output

    # Adapt ordinary one-point callbacks to SciPy's vectorized sampling protocol.
    # SciPy owns the step selection, finite-difference weights, and convergence.
    def vectorized(points):
        return np.apply_along_axis(evaluate, 0, points)

    tolerances = {"atol": atol, "rtol": rtol}
    if problem.kind == "derivative":
        raw = differentiate.derivative(np.vectorize(evaluate, otypes=[float]), problem.x,
                                       step_direction=direction, tolerances=tolerances,
                                       order=order, maxiter=maxiter, initial_step=initial_step)
    elif problem.kind == "hessian":
        raw = differentiate.hessian(vectorized, problem.x, tolerances=tolerances,
                                    order=order, maxiter=maxiter, initial_step=initial_step)
    else:
        raw = differentiate.jacobian(vectorized, problem.x, tolerances=tolerances,
                                     order=order, maxiter=maxiter, initial_step=initial_step)
    value = raw.ddf if problem.kind == "hessian" else raw.df
    success = bool(np.all(raw.success))
    nit = getattr(raw, "nit", None)
    message = "" if success else f"Finite differences did not converge (status={np.unique(raw.status).tolist()})"
    return ArrayResult(value, Diagnostics(converged=success, message=message,
                                         error_estimate=raw.error, evaluations=evaluations,
                                         iterations=None if nit is None else int(np.max(nit))))
