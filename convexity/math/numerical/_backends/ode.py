"""SciPy adaptive ODE/BVP solvers and continuous interpolation."""

import numpy as np
from scipy import integrate
from .. import _validation as v
from ..interfaces.base import Diagnostics
from ..interfaces.ode import ODEResult, DenseSolution, EventResult, BoundaryValueResult


def solve_ivp(problem, *, method, atol, rtol, max_step):
    n = np.size(problem.y0)
    evaluations = 0

    def function(t, y):
        nonlocal evaluations
        evaluations += 1
        output = v.array(
            problem.function(t, v.array(y, "state", real=True)),
            "ODE output",
            ndim=1,
            real=True,
        )
        if output.shape != (n,):
            raise ValueError("ODE function must return shape (n,)")
        return output

    def jacobian(t, y):
        output = v.matrix(
            problem.jacobian(t, v.array(y, "state", real=True)), "ODE Jacobian"
        )
        if output.shape != (n, n) or np.iscomplexobj(output):
            raise ValueError("ODE Jacobian must be real with shape (n,n)")
        return output

    def event_wrapper(event):
        def evaluate(t, y):
            return v.scalar(
                event.function(t, v.array(y, "state", real=True)), "event output"
            )

        evaluate.terminal = event.terminal
        evaluate.direction = event.direction
        return evaluate

    options = {}
    if problem.jacobian is not None:
        if method not in ("Radau", "BDF", "LSODA"):
            raise ValueError("Jacobian callbacks apply to stiff solvers only")
        options["jac"] = jacobian
    raw = integrate.solve_ivp(
        function,
        problem.t_span,
        problem.y0,
        t_eval=problem.t_eval,
        method=method,
        atol=atol,
        rtol=rtol,
        max_step=max_step,
        dense_output=True,
        events=[event_wrapper(e) for e in problem.events] or None,
        **options,
    )
    times = np.asarray(raw.t)
    values = np.asarray(raw.y).reshape(n, times.size)
    endpoints = sorted((problem.t_span[0], float(raw.sol.ts[-1])))
    domain = (float(endpoints[0]), float(endpoints[1]))
    events = tuple(
        EventResult(t, np.asarray(y).reshape(-1, n))
        for t, y in zip(raw.t_events or [], raw.y_events or [])
    )
    return ODEResult(
        times,
        values,
        DenseSolution(raw.sol, domain, n),
        Diagnostics(
            converged=bool(raw.success), message=raw.message, evaluations=evaluations
        ),
        events,
        raw.status == 1,
    )


def solve_bvp(problem, *, tolerance, boundary_tolerance, max_nodes):
    n = np.shape(problem.y)[0]
    evaluations = 0

    def function(x, y):
        nonlocal evaluations
        evaluations += 1
        output = v.array(
            problem.function(v.array(x, "mesh"), v.array(y, "states")),
            "BVP output",
            ndim=2,
            real=True,
        )
        if output.shape != y.shape:
            raise ValueError("BVP function must preserve state/mesh shape")
        return output

    def boundary(left, right):
        output = v.array(
            problem.boundary(
                v.array(left, "left state"), v.array(right, "right state")
            ),
            "boundary residual",
            ndim=1,
            real=True,
        )
        if output.shape != (n,):
            raise ValueError("boundary residual must have shape (n,)")
        return output

    raw = integrate.solve_bvp(
        function,
        boundary,
        problem.x,
        problem.y,
        tol=tolerance,
        bc_tol=boundary_tolerance,
        max_nodes=max_nodes,
    )
    residual = float(np.linalg.norm(boundary(raw.y[:, 0], raw.y[:, -1])))
    return BoundaryValueResult(
        raw.x,
        raw.y,
        DenseSolution(raw.sol, (float(raw.x[0]), float(raw.x[-1])), n),
        raw.rms_residuals,
        Diagnostics(
            bool(raw.success),
            raw.message,
            iterations=int(raw.niter),
            evaluations=evaluations,
            residual_norm=residual,
        ),
    )
