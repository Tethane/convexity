# `convexity.math.policies`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Policies mathematics on supplied arrays and models.

## `PolicyState`

```python
PolicyState(deadline: 'float', mode: 'str' = 'waiting', last_time: 'float | None' = None, reason: 'str' = 'initialized') -> None
```

Policy lifecycle state: deadline, mode, last update time and reason. It is distinct from market/model state.

## `ActionDecision`

```python
ActionDecision(action: 'int', value: 'float', holding_value: 'float | None', advantage: 'float | None', reason: 'str') -> None
```

Policy action with accompanying decision metadata; it does not execute an external trade.

## `Policy`

Map supplied state and context to an ActionDecision without executing external actions.

### `Policy.update`

```python
update(self, state: 'PolicyState', probability: 'float', time: 'float', **evidence) -> 'PolicyState'
```

## `Hysteresis`

```python
Hysteresis(entry: 'float' = 0.9, exit: 'float' = 0.6, convergence_gap: 'float' = 0.0) -> None
```

Policy switching with separate entry/exit thresholds to avoid boundary chattering.

### `Hysteresis.update`

```python
update(self, state: 'PolicyState', probability: 'float', time: 'float', *, gap: 'float | None' = None, adverse: 'bool' = False, mandatory: 'bool' = False, scheduled: 'bool' = True)
```

## `no_action`

```python
no_action(values: 'ArrayLike', *, hold: 'int' = 0, feasible: 'ArrayLike | None' = None, switching: 'float' = 0.0, mandatory: 'int | None' = None) -> 'ActionDecision'
```

Maximize action values less switching cost. Hold wins exact ties.

`mandatory` is an explicit feasible action index and overrides the band.
If hold is infeasible, choose the first maximizing feasible alternative.
