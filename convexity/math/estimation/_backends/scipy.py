"""SciPy solvers; solver inverse-Hessian approximations are never used as SEs."""

from typing import Any

import numpy as np
from scipy import linalg, optimize

from ..interfaces import Diagnostics, Estimate, LinearConstraint


def inverse_information(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)) or np.linalg.norm(
        matrix - matrix.T, ord=np.inf
    ) > 1e-6 * max(1.0, np.linalg.norm(matrix, ord=np.inf)):
        raise ValueError("information matrix is not finite and symmetric")
    matrix = (matrix + matrix.T) / 2
    if np.linalg.matrix_rank(matrix) < len(matrix):
        raise ValueError("information matrix is rank deficient")
    factor = linalg.cho_factor(matrix)
    return linalg.cho_solve(factor, np.eye(len(matrix)))


def curvature_covariance(problem, theta):
    if problem.constraints:
        return None, "Curvature uncertainty is unavailable with general constraints."
    if problem.space.at_boundary(theta):
        return (
            None,
            "Curvature uncertainty is unavailable at an active parameter bound.",
        )
    # Reuse the completed numerical family only when differentiation is needed.
    from ...numerical import ConvergenceError, FiniteDifference, hessian

    try:
        if problem.hessian is not None:
            matrix = problem.second_derivative(theta)
        else:
            distance = min(
                np.min(theta - problem.space.lower), np.min(problem.space.upper - theta)
            )
            step = min(0.1, float(distance) / 4)
            matrix = hessian(
                problem.evaluate,
                theta,
                tool=FiniteDifference(initial_step=step, atol=1e-6, rtol=1e-4),
            ).value
        return inverse_information(
            matrix
        ), "Local inverse curvature; regular interior identifiable model assumed."
    except (ValueError, linalg.LinAlgError, ConvergenceError) as error:
        return None, f"Curvature uncertainty unavailable: {error}"


def minimize(problem, *, method, tolerance, maxiter):
    bounded = np.any(np.isfinite(problem.space.lower)) or np.any(
        np.isfinite(problem.space.upper)
    )
    selected = method
    if selected == "auto":
        selected = "SLSQP" if problem.constraints or bounded else "L-BFGS-B"
    if selected == "BFGS" and bounded:
        raise ValueError("BFGS does not support parameter bounds")
    if problem.constraints and selected not in ("SLSQP", "trust-constr"):
        raise ValueError("general constraints require SLSQP or trust-constr")
    if selected == "Powell" and problem.gradient is not None:
        raise ValueError("Powell does not use a gradient")
    constraints = []
    # SciPy's scalar-default annotations omit its documented vector bounds.
    native_linear_constraint: Any = optimize.LinearConstraint
    for constraint in problem.constraints:
        if isinstance(constraint, LinearConstraint):
            constraints.append(
                native_linear_constraint(
                    constraint.matrix, constraint.lower, constraint.upper
                )
            )
        else:
            constraints.append(
                optimize.NonlinearConstraint(
                    constraint.evaluate,
                    constraint.lower,
                    constraint.upper,
                    jac=constraint.derivative
                    if constraint.jacobian is not None
                    else "2-point",
                )
            )
    if not np.isfinite(problem.evaluate(problem.space.initial)):
        raise ValueError("initial objective must be finite")
    nonfinite_trials = 0

    def evaluate(theta):
        nonlocal nonfinite_trials
        value = problem.evaluate(theta)
        nonfinite_trials += int(not np.isfinite(value))
        return value

    kwargs = {}
    if selected == "trust-constr" and problem.hessian is not None:
        kwargs["hess"] = problem.second_derivative
    raw = optimize.minimize(
        evaluate,
        problem.space.initial,
        method=selected,
        jac=problem.derivative
        if problem.gradient is not None
        else None
        if selected == "Powell"
        else "3-point",
        bounds=optimize.Bounds(problem.space.lower, problem.space.upper)
        if bounded
        else None,
        constraints=constraints,
        tol=tolerance,
        options={"maxiter": maxiter},
        **kwargs,
    )
    violation = problem.violation(raw.x)
    success = bool(raw.success and np.isfinite(raw.fun) and violation <= 1e-6)
    message = str(raw.message)
    if raw.success and not success:
        message += "; returned a nonfinite objective or infeasible parameters"
    if success and selected == "L-BFGS-B" and nonfinite_trials:
        # L-BFGS-B can report relative-function convergence without moving after
        # an infinite trial. Check its projected gradient before accepting that.
        gradient = np.asarray(raw.jac).copy()
        eps = 1e-8 * (1 + np.abs(raw.x))
        gradient[(raw.x - problem.space.lower <= eps) & (gradient > 0)] = 0
        gradient[(problem.space.upper - raw.x <= eps) & (gradient < 0)] = 0
        if not np.all(np.isfinite(gradient)) or np.max(np.abs(gradient)) > max(
            1e-5, np.sqrt(tolerance)
        ):
            success = False
            message = "L-BFGS-B stopped after nonfinite trials with a nonstationary projected gradient; try SLSQP or change coordinates."
    covariance = None
    note = "Covariance was not requested for this objective."
    if problem.curvature is not None:
        if success:
            covariance, note = curvature_covariance(problem, raw.x)
        else:
            note = "Covariance is unavailable for a failed fit."
    return Estimate(
        raw.x,
        names=problem.space.names,
        objective=raw.fun,
        covariance=covariance,
        covariance_kind=problem.curvature if covariance is not None else None,
        nobs=problem.nobs,
        diagnostics=Diagnostics(
            converged=success,
            message=message,
            backend=f"scipy.optimize.minimize:{selected}",
            iterations=getattr(raw, "nit", None),
            evaluations=raw.nfev,
            constraint_violation=violation,
            uncertainty_note=note,
        ),
    )


def least_squares(
    problem, *, method, loss, f_scale, tolerance, max_evaluations, absolute_sigma
):
    initial_residuals = problem.evaluate(problem.space.initial)
    shape = initial_residuals.shape

    def residuals(theta):
        value = problem.evaluate(theta)
        if value.shape != shape:
            raise ValueError("residual shape changed during fitting")
        # SciPy 1.15 robust losses rescale residuals/Jacobians in place. Hand
        # the solver its own writable buffers; public callback data stays owned.
        return value.copy()

    def jacobian(theta):
        return problem.derivative(theta).copy()

    if method == "lm" and (
        loss != "linear"
        or np.any(np.isfinite(problem.space.lower))
        or np.any(np.isfinite(problem.space.upper))
    ):
        raise ValueError("lm requires unbounded parameters and linear loss")
    # SciPy's inferred annotations omit its documented callable Jacobian.
    native_least_squares: Any = optimize.least_squares
    raw = native_least_squares(
        residuals,
        problem.space.initial,
        jac=jacobian if problem.jacobian is not None else "2-point",
        bounds=(problem.space.lower, problem.space.upper),
        method=method,
        loss=loss,
        f_scale=f_scale,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=max_evaluations,
    )
    rank = int(np.linalg.matrix_rank(raw.jac))
    n, p = raw.jac.shape
    covariance = None
    df = None
    note = "Local linearization with independent Gaussian residuals."
    if not raw.success:
        note = "Covariance is unavailable for a failed fit."
    elif loss != "linear":
        note = "Robust residual losses require a separate sandwich or bootstrap uncertainty calculation."
    elif problem.space.at_boundary(raw.x):
        note = "Least-squares covariance is unavailable at an active parameter bound."
    elif rank < p or (not absolute_sigma and n <= p):
        note = "Covariance needs full column rank and positive residual degrees of freedom for estimated noise."
    else:
        try:
            covariance = inverse_information(raw.jac.T @ raw.jac)
            if not absolute_sigma:
                df = float(n - p)
                covariance *= float(np.dot(raw.fun, raw.fun)) / df
        except (ValueError, linalg.LinAlgError) as error:
            note = f"Least-squares covariance unavailable: {error}"
            df = None
    return Estimate(
        raw.x,
        names=problem.space.names,
        objective=raw.cost,
        nobs=n,
        covariance=covariance,
        covariance_kind="least_squares" if covariance is not None else None,
        df=df,
        residuals=raw.fun,
        diagnostics=Diagnostics(
            converged=bool(raw.success),
            message=str(raw.message),
            backend=f"scipy.optimize.least_squares:{method}",
            evaluations=raw.nfev,
            rank=rank,
            constraint_violation=problem.space.violation(raw.x),
            uncertainty_note=note,
        ),
    )
