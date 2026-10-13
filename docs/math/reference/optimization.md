# `convexity.math.optimization`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Optimization mathematics on supplied arrays and models.

## `SecondOrderCone`

```python
SecondOrderCone(matrix: 'ArrayLike', offset: 'ArrayLike', linear: 'ArrayLike', bound: 'float' = 0.0) -> 'None'
```

||matrix @ x + offset||_2 <= linear @ x + bound.

## `QuadraticProgram`

```python
QuadraticProgram(linear: 'ArrayLike', quadratic: 'ArrayLike | None' = None, inequality: 'ArrayLike | None' = None, upper: 'ArrayLike | None' = None, equality: 'ArrayLike | None' = None, target: 'ArrayLike | None' = None, lower_bounds: 'ArrayLike' = -inf, upper_bounds: 'ArrayLike' = inf, domains: "Sequence[Literal['continuous', 'integer', 'binary']] | None" = None, cones: 'tuple[SecondOrderCone, ...]' = ()) -> 'None'
```

Minimize .5*x.T@quadratic@x + linear@x with affine/SOC constraints.

Matrices have (constraint,variable) axes. Domains are continuous, integer
or binary per variable. Bounds default to free variables, not nonnegative.

## `NonlinearProblem`

```python
NonlinearProblem(objective: 'Callable[[FloatArray], float]', initial: 'ArrayLike', lower_bounds: 'ArrayLike' = -inf, upper_bounds: 'ArrayLike' = inf, inequalities: 'Callable[[FloatArray], ArrayLike] | None' = None, equalities: 'Callable[[FloatArray], ArrayLike] | None' = None, gradient: 'Callable[[FloatArray], ArrayLike] | None' = None) -> 'None'
```

Minimize objective(x), with inequalities(x)<=0 and equalities(x)==0.

## `SolveResult`

```python
SolveResult(solution: 'ArrayLike | None', objective: 'float | None', status: 'str', solver: 'str', max_violation: 'float | None' = None, duals: 'Mapping[str, ArrayLike]' = <factory>, gap: 'float | None' = None, iterations: 'int | None' = None, message: 'str' = '') -> 'None'
```

Optimization solution and termination diagnostics; success is not a global-optimality certificate for nonconvex problems.

## `Solver`

Solve a built-in or custom optimization problem, retaining its type and returning SolveResult.

### `Solver.solve`

```python
solve(self, problem: 'ProblemT') -> 'SolveResult'
```

## `OptimizationError`

```python
OptimizationError(result: 'SolveResult') -> 'None'
```

Raised when optimization cannot supply an acceptable solution; inspect retained status and diagnostics.

## `ConvexSolver`

```python
ConvexSolver(solver: 'str | None' = None, options: 'Mapping[str, object]' = <factory>, tolerance: 'float' = 1e-06) -> None
```

Solve a declared convex problem with the selected provider solver and options.

### `ConvexSolver.solve`

```python
solve(self, problem)
```

## `NonlinearSolver`

```python
NonlinearSolver(tolerance: 'float' = 1e-09, maxiter: 'int' = 1000) -> None
```

Numerically optimize a nonlinear objective from a supplied initial point, with explicit constraints and tolerances.

### `NonlinearSolver.solve`

```python
solve(self, problem)
```

## `solve`

```python
solve(problem: 'ProblemT', solver: 'Solver[ProblemT] | None' = None, *, check_success: 'bool' = True) -> 'SolveResult'
```

Solve a QuadraticProgram or NonlinearProblem using a built-in or custom Solver.

Return SolveResult with status, feasibility and optional primal solution.
check_success=True raises on failed solves; a failed result may have solution=None.

## `enumerate_minimum`

```python
enumerate_minimum(objective: 'Callable[[FloatArray], float]', domains: 'Sequence[ArrayLike]', *, feasible: 'Callable[[FloatArray], bool] | None' = None, max_candidates: 'int' = 100000) -> 'SolveResult'
```

Exact Cartesian finite search; first candidate wins ties; bounded upfront.
