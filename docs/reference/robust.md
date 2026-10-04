# `convexity.math.robust`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Robust mathematics on supplied arrays and models.

## `UncertaintySet`

Set of admissible uncertain coordinates; its support operation uses the supplied direction and units.

### `UncertaintySet.support`

```python
support(self, direction: 'ArrayLike') -> 'float'
```

## `Box`

```python
Box(center: 'ArrayLike', radius: 'ArrayLike') -> 'None'
```

Coordinatewise uncertainty set defined by a center and nonnegative coordinate radii in the same units as the uncertain vector.

### `Box.support`

```python
support(self, direction: 'ArrayLike') -> 'float'
```

## `Ellipsoid`

```python
Ellipsoid(center: 'ArrayLike', shape: 'ArrayLike', radius: 'float' = 1.0) -> 'None'
```

center + shape @ u, ||u||_2 <= radius; shape need not be square.

### `Ellipsoid.support`

```python
support(self, direction: 'ArrayLike') -> 'float'
```

## `minimax`

```python
minimax(loss_matrix: 'ArrayLike', offsets: 'ArrayLike | None' = None, *, lower_bounds: 'float' = 0.0, upper_bounds: 'float' = 1.0, budget: 'float' = 1.0, solver: 'str | None' = None) -> 'SolveResult'
```

Minimize max_s(A_s @ x + b_s), with sum(x)=budget.

## `mean_cvar`

```python
mean_cvar(returns: 'ArrayLike', *, weights: 'ArrayLike | None' = None, aversion: 'float' = 1.0, level: 'float' = 0.95, lower_bounds: 'float' = 0.0, upper_bounds: 'float' = 1.0, budget: 'float' = 1.0, solver: 'str | None' = None) -> 'SolveResult'
```

Minimize -E[return@x] + aversion*ES(-return@x).

## `robust_allocation`

```python
robust_allocation(mean: 'ArrayLike', covariance: 'ArrayLike', ambiguity: 'Ellipsoid', *, aversion: 'float' = 1.0, lower_bounds: 'float' = 0.0, upper_bounds: 'float' = 1.0, budget: 'float' = 1.0, solver: 'str | None' = None) -> 'SolveResult'
```

Minimize -mean@x + rho||shape.T@x|| + aversion/2*x.T@cov@x.

Ambiguity center must equal the supplied mean (an explicit consistency check).

## `gaussian_chance_constraint`

```python
gaussian_chance_constraint(mean: 'ArrayLike', covariance: 'ArrayLike', bound: 'float', *, probability: 'float' = 0.95) -> 'SecondOrderCone'
```

Return SOC for P[a.T@x <= bound]>=probability, Gaussian a; p>=.5.

## `kl_worst_case`

```python
kl_worst_case(losses: 'ArrayLike', radius: 'float', *, weights: 'ArrayLike | None' = None, solver: 'str | None' = None) -> 'SolveResult'
```

Maximize q@loss over KL(q||p)<=radius on supplied finite support.

Returns SolveResult with solution=q and objective=worst expected loss.
Zero nominal mass excludes that outcome from the ambiguity set.

## `chance_probability`

```python
chance_probability(losses: 'ArrayLike', bound: 'float' = 0.0, *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return normalized scenario mass for losses <= bound; this is a finite-law probability.
