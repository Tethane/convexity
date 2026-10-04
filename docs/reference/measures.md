# `convexity.math.measures`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: measures.

## `Numeraire`

Positive value process used as a unit of account for a declared pricing measure.

### `Numeraire.value`

```python
value(self, time: 'float') -> 'ArrayLike'
```

## `MeasureModel`

```python
MeasureModel(model: 'ModelT', measure: 'str', numeraire: 'Numeraire | None' = None) -> None
```

Model plus explicitly declared probability measure and optional numeraire; a tag is not a measure transformation.

### `MeasureModel.value`

```python
value(self, cashflow, maturity: 'float', *, expectation)
```

N(0) E^Q[cashflow/N(T)], expectation supplied by the joint law.

## `MoneyMarketNumeraire`

```python
MoneyMarketNumeraire(rate: 'float') -> None
```

Deterministic money-market numeraire exp(rate * time) for a constant continuously compounded rate.

### `MoneyMarketNumeraire.value`

```python
value(self, time: 'float')
```

## `CurveNumeraire`

```python
CurveNumeraire(curve: 'DiscountCurve') -> None
```

Deterministic bank account implied by a discount curve.

### `CurveNumeraire.value`

```python
value(self, time: 'float')
```

## `DiffusionPair`

```python
DiffusionPair(physical: 'MeasureModel[GeometricBrownianMotion | CIR | OrnsteinUhlenbeck | Heston]', pricing: 'MeasureModel[GeometricBrownianMotion | CIR | OrnsteinUhlenbeck | Heston]') -> None
```

Compatible named P/Q diffusions; dW_Q = dW_P + lambda dt.

Compatibility is checked at construction for supported model families, and
the drift-span condition is checked at every market_price_of_risk call.
This does not by itself establish Novikov/integrability conditions.

### `DiffusionPair.market_price_of_risk`

```python
market_price_of_risk(self, time: 'float', state)
```

## `gbm_pair`

```python
gbm_pair(mu: 'float', rate: 'float', dividend: 'float', sigma: 'float') -> 'DiffusionPair'
```

Construct explicitly supplied physical and pricing GBM laws with a money-market numeraire.

mu, rate, dividend are annual drifts; sigma is annual volatility. Q drift is
rate-dividend. This constructor records assumptions, it does not estimate a change of measure.

## `girsanov_log_density`

```python
girsanov_log_density(brownian_increments: 'ArrayLike', risk: 'ArrayLike', dt: 'ArrayLike') -> 'FloatArray'
```

Euler stochastic exponential log(dQ/dP), (draw,step,noise).

risk broadcasts to increments; it must be predictable (evaluated BEFORE
each increment). Constant risk is exact; state-dependent risk is discretized.

## `reweight`

```python
reweight(scenarios: 'PathScenarios', log_likelihood: 'ArrayLike') -> 'PathScenarios'
```

Self-normalized importance law; no IID error estimate is asserted.

## `change_numeraire_weights`

```python
change_numeraire_weights(old_numeraire_terminal: 'ArrayLike', new_numeraire_terminal: 'ArrayLike', old_initial: 'float', new_initial: 'float') -> 'FloatArray'
```

dQ_new/dQ_old = (N_new(T)/N_old(T))/(N_new(0)/N_old(0)).
