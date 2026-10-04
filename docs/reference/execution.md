# `convexity.math.execution`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Execution mathematics on supplied arrays and models.

## `LiquidationResult`

```python
LiquidationResult(trades: 'ArrayLike', inventory: 'ArrayLike', expected_cost: 'float', inventory_penalty: 'float', objective: 'float', solver: 'str', max_violation: 'float') -> 'None'
```

Inventory/trade schedule and modeled liquidation objective under supplied impact/risk assumptions.

## `LogisticFit`

```python
LogisticFit(coefficients: 'ArrayLike', covariance: 'ArrayLike | None', log_likelihood: 'float', iterations: 'int') -> 'None'
```

Fitted logistic response parameters and diagnostics for the declared explanatory coordinates.

### `LogisticFit.probability`

```python
probability(self, features: 'ArrayLike') -> 'FloatArray'
```

## `CostModel`

Evaluate modeled execution costs for the supplied trade/state coordinates.

## `FillLaw`

Conditional execution-fill law; distinguish fill probability from price and cost expectations.

### `FillLaw.sample`

```python
sample(self, quantities: 'ArrayLike', *, draws: 'int', rng: 'np.random.Generator') -> 'FloatArray'
```

## `QuadraticCost`

```python
QuadraticCost(spread: 'ArrayLike', impact: 'ArrayLike') -> 'None'
```

Cash cost spread@abs(trade) + trade.T@impact@trade, in trade units.

## `BinomialFills`

```python
BinomialFills(probability: 'ArrayLike') -> 'None'
```

Conditionally independent counts per leg, with a supplied probability.

### `BinomialFills.sample`

```python
sample(self, quantities: 'ArrayLike', *, draws: 'int', rng: 'np.random.Generator')
```

## `assignment`

```python
assignment(short_quantities: 'ArrayLike', probabilities: 'ArrayLike', *, draws: 'int', rng: 'np.random.Generator') -> 'IntArray'
```

Positive counts of short contracts; no claim of independence across time.

## `fit_assignment`

```python
fit_assignment(features: 'ArrayLike', assigned: 'ArrayLike', trials: 'ArrayLike', *, lower_bounds: 'ArrayLike' = -20.0, upper_bounds: 'ArrayLike' = 20.0) -> 'LogisticFit'
```

Bounded binomial logistic MLE. Include an intercept column explicitly.

## `liquidate`

```python
liquidate(initial: 'ArrayLike', intervals: 'ArrayLike', *, spread: 'ArrayLike' = 0.0, impact: 'ArrayLike', covariance: 'ArrayLike', aversion: 'float' = 0.0, maximum_rate: 'ArrayLike' = inf, monotone: 'bool' = True, solver: 'str | None' = None) -> 'LiquidationResult'
```

Signed inventory, trades reduce it; terminal inventory exactly zero.

Cost = sum(spread*abs(trade) + trade'impact trade/dt).
Risk preference = aversion*sum(dt*end_inventory'covariance end_inventory).
