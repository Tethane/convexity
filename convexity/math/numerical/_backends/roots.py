"""Scalar and vector root adapters to SciPy optimize (no optimization models)."""

import numpy as np
from scipy import optimize

from .. import _validation as v
from ..interfaces import ArrayResult, Diagnostics, RootSystem, ScalarRootProblem


def scalar_root(
    problem: ScalarRootProblem, *, method: str, xtol: float, rtol: float, maxiter: int
) -> ArrayResult:
    if method in ("brentq", "bisect", "toms748") and problem.bracket is None:
        raise ValueError(f"{method} requires a bracket")
    if method in ("newton", "secant") and problem.x0 is None:
        raise ValueError(f"{method} requires x0")
    if method == "newton" and problem.derivative is None:
        raise ValueError("newton requires a derivative callable")
    if method == "secant" and (problem.x1 is None or problem.x0 == problem.x1):
        raise ValueError("secant requires distinct x0 and x1")
    evaluations = 0

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        return v.scalar(problem.function(x), "function output")

    def derivative(x):
        assert problem.derivative is not None
        return v.scalar(problem.derivative(x), "derivative output")

    # Handle endpoint roots explicitly: some SciPy versions leave the iteration
    # count undefined on this fast path. These checks also give a clear bracket error.
    if method in ("brentq", "bisect", "toms748"):
        assert problem.bracket is not None
        left, right = problem.bracket
        fl, fr = evaluate(left), evaluate(right)
        if fl == 0 or fr == 0:
            return ArrayResult(
                left if fl == 0 else right,
                Diagnostics(iterations=0, evaluations=evaluations, residual_norm=0.0),
            )
        if np.signbit(fl) == np.signbit(fr):
            raise ValueError("function must change sign across the bracket")
    raw = optimize.root_scalar(
        evaluate,
        bracket=problem.bracket,
        x0=problem.x0,
        x1=problem.x1,
        fprime=derivative if method == "newton" else None,
        method=method,
        xtol=xtol,
        rtol=rtol,
        maxiter=maxiter,
    )
    residual = abs(evaluate(raw.root))
    return ArrayResult(
        raw.root,
        Diagnostics(
            converged=bool(raw.converged),
            message=str(raw.flag),
            iterations=int(raw.iterations),
            evaluations=evaluations,
            residual_norm=residual,
        ),
    )


def system_root(
    problem: RootSystem,
    *,
    xtol: float,
    residual_tol: float,
    max_evaluations: int,
    method: str = "hybr",
    maxiter: int = 100,
) -> ArrayResult:
    shape = np.shape(problem.x0)
    evaluations = 0

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        output = v.array(problem.function(x), "function output", ndim=1, real=True)
        if output.shape != shape:
            raise ValueError("root function must return shape (n,), matching x0")
        return output

    def jacobian(x):
        assert problem.jacobian is not None
        output = v.array(problem.jacobian(x), "Jacobian", ndim=2, real=True)
        if output.shape != shape + shape:
            raise ValueError("Jacobian must have shape (n,n)")
        return output

    if method != "hybr" and problem.jacobian is not None:
        raise ValueError("explicit Jacobians are supported by the hybrid solver only")
    options = (
        {"xtol": xtol, "maxfev": max_evaluations}
        if method == "hybr"
        else {"xatol": xtol, "fatol": residual_tol, "maxiter": maxiter}
    )
    raw = optimize.root(
        evaluate,
        problem.x0,
        jac=jacobian if problem.jacobian is not None else None,
        method=method,
        options=options,
    )
    residual = float(np.linalg.norm(raw.fun))
    success = bool(raw.success and residual <= residual_tol)
    message = str(raw.message)
    if raw.success and not success:
        message = (
            f"Step tolerance reached but residual {residual:g} exceeds {residual_tol:g}"
        )
    return ArrayResult(
        raw.x,
        Diagnostics(
            converged=success,
            message=message,
            evaluations=evaluations,
            residual_norm=residual,
            iterations=getattr(raw, "nit", None),
        ),
    )
