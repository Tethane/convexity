# `convexity.math.risk`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Risk mathematics on supplied arrays and models.

## `LossScenarios`

```python
LossScenarios(losses: 'ArrayLike', weights: 'ArrayLike | None' = None) -> 'None'
```

Scalar losses (scenario,), positive means adverse; normalized masses.

## `RiskContributions`

```python
RiskContributions(value: 'float', marginal: 'ArrayLike', component: 'ArrayLike') -> 'None'
```

Risk allocated across the declared coordinates; the allocation convention depends on the chosen risk functional.

## `RiskMeasure`

Evaluate risk of supplied outcomes under the family's loss/return sign convention.

## `variance`

```python
variance(losses: 'ArrayLike | LossScenarios', *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return the population variance of scalar loss scenarios with normalized masses.

Weights are probabilities, not frequency degrees-of-freedom corrections.

## `downside_moment`

```python
downside_moment(returns: 'ArrayLike', *, target: 'float' = 0.0, order: 'float' = 2.0, weights: 'ArrayLike | None' = None) -> 'float'
```

E[max(target-return,0)**order], without conditioning on shortfall.

## `value_at_risk`

```python
value_at_risk(losses: 'ArrayLike | LossScenarios', level: 'float' = 0.95, *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return the lower inverse-CDF loss quantile at level in (0,1).

Losses are positive for adverse outcomes; finite scenario weights are normalized.

## `expected_shortfall`

```python
expected_shortfall(losses: 'ArrayLike | LossScenarios', level: 'float' = 0.95, *, weights: 'ArrayLike | None' = None) -> 'float'
```

Mean of the worst 1-level mass, splitting the atom at VaR exactly.

## `volatility_contributions`

```python
volatility_contributions(holdings: 'ArrayLike', covariance: 'ArrayLike') -> 'RiskContributions'
```

Return marginal and component portfolio volatility from holdings and covariance.

Covariance uses squared per-period return units; components sum to volatility.

## `es_contributions`

```python
es_contributions(holdings: 'ArrayLike', unit_losses: 'ArrayLike | LossScenarios', level: 'float' = 0.95, *, weights: 'ArrayLike | None' = None) -> 'RiskContributions'
```

Euler ES subgradient; tied tail mass is distributed proportionally.

## `drawdown`

```python
drawdown(wealth: 'ArrayLike', *, relative: 'bool' = True) -> 'FloatArray'
```

Running peak minus wealth on the final time axis. Include initial wealth.

## `maximum_drawdown`

```python
maximum_drawdown(wealth: 'ArrayLike', *, relative: 'bool' = True) -> 'FloatArray'
```

Return the maximum peak-to-trough drawdown along the last wealth axis.

relative=True reports a fraction of running peak; False reports wealth units.

## `breach_probability`

```python
breach_probability(paths: 'ArrayLike | PathScenarios', barrier: 'ArrayLike', *, direction: 'str' = 'below', weights: 'ArrayLike | None' = None) -> 'float'
```

Inclusive breach on supplied grid only; no between-grid interpolation.

## `stress`

```python
stress(values: 'ArrayLike', *, shift: 'ArrayLike' = 0.0, scale: 'ArrayLike' = 1.0, transform: 'Callable[[FloatArray], ArrayLike] | None' = None) -> 'FloatArray'
```

Affine stress or supplied shape-preserving map; no probability inferred.

## `scenario_margin`

```python
scenario_margin(losses: 'ArrayLike | LossScenarios', *, floor: 'float' = 0.0, addon: 'float' = 0.0, multiplier: 'float' = 1.0) -> 'float'
```

max(floor, multiplier*max(0,worst supplied loss)) + addon.
