# `convexity.math.stopping`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: stopping.

## `PolicyEvaluation`

```python
PolicyEvaluation(value: 'float', standard_error: 'float', exercise_indices: 'ArrayLike', discounted_payoffs: 'ArrayLike') -> 'None'
```

Stopping-policy values on supplied paths, with standard error, exercise indices and discounted payoffs. Independent evaluation paths are the caller's responsibility.

### `PolicyEvaluation.items`

```python
items(self)
```

### `PolicyEvaluation.keys`

```python
keys(self)
```

## `StoppingProblem`

```python
StoppingProblem(rewards: 'ArrayLike', transitions: 'ArrayLike', discounts: 'ArrayLike', exercise: 'ArrayLike | None' = None) -> 'None'
```

Exercise rewards and state paths for a stopping decision under the supplied discount convention.

## `StoppingResult`

```python
StoppingResult(values: 'ArrayLike', continuation: 'ArrayLike', exercise_region: 'ArrayLike') -> 'None'
```

Estimated stopping values and policy information; inspect training/evaluation semantics before pricing.

## `StoppingSolver`

Solve the supplied stopping problem and retain policy/value semantics in StoppingResult.

### `StoppingSolver.solve`

```python
solve(self, problem: 'StoppingProblem') -> 'StoppingResult'
```

## `BackwardInduction`

```python
BackwardInduction() -> None
```

Solve the supplied finite stopping problem backward over its exercise times.

### `BackwardInduction.solve`

```python
solve(self, problem)
```

## `ExercisePolicy`

```python
ExercisePolicy(regressions: 'tuple[ContinuationRegressor | None, ...]', exercise: 'tuple[bool, ...]', state_dimension: 'int') -> None
```

Fitted continuation regressions; evaluate on independent Q paths.

Validation output is a feasible-policy lower estimate, not a dual upper
bound. Standard error assumes IID validation paths and fixed learned policy.

### `ExercisePolicy.evaluate`

```python
evaluate(self, states, rewards, discounts)
```

## `least_squares_policy`

```python
least_squares_policy(states: 'ArrayLike', rewards: 'ArrayLike', discounts: 'ArrayLike', *, degree: 'int' = 2, exercise: 'ArrayLike | None' = None) -> 'ExercisePolicy'
```

Longstaff–Schwartz fit on Q paths; use independent paths for valuation.

Nonnegative exercise rewards are required. State includes all information
needed to make the contract Markov (e.g. running average for Asian claims).

## `ContinuationRegressor`

Predict continuation values from state rows; one scalar value per row.

### `ContinuationRegressor.predict`

```python
predict(self, states: 'ArrayLike', /) -> 'ArrayLike'
```
