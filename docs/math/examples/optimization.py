"""array QP, conic budget, integer LP and callback optimization."""

import numpy as np
from convexity.math import optimization as opt

problem = opt.QuadraticProgram(
    [-1.0, -2.0],
    quadratic=np.eye(2),
    equality=[[1.0, 1.0]],
    target=[1.0],
    lower_bounds=0.0,
)
result = opt.solve(problem)
print("QP:", result.solution, result.objective, dict(result.duals))
cone = opt.SecondOrderCone(np.eye(2), [0.0, 0.0], [0.0, 0.0], 1.0)
print("Conic:", opt.solve(opt.QuadraticProgram([-3.0, -4.0], cones=(cone,))).solution)
integer = opt.QuadraticProgram(
    [-3.0, -2.0],
    inequality=[[2.0, 1.0]],
    upper=[3.0],
    lower_bounds=0,
    upper_bounds=[1, 3],
    domains=["binary", "integer"],
)
print("Integer LP:", opt.solve(integer, opt.ConvexSolver("SCIPY")).solution)
print(
    "Nonlinear:",
    opt.solve(
        opt.NonlinearProblem(lambda x: (x[0] - 2) ** 2, [0.0], upper_bounds=1.0)
    ).solution,
)
