# `convexity.math.utility`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Utility mathematics on supplied arrays and models.

## `Utility`

Wealth utility extension with explicit domain and preference parameters.

### `Utility.inverse`

```python
inverse(self, utility: 'ArrayLike') -> 'FloatArray'
```

## `LinearUtility`

```python
LinearUtility() -> None
```

Affine utility of supplied wealth; coordinate units and any normalization are caller choices.

### `LinearUtility.inverse`

```python
inverse(self, utility: 'ArrayLike') -> 'FloatArray'
```

## `LogUtility`

```python
LogUtility() -> None
```

Logarithmic wealth utility on its positive domain.

### `LogUtility.inverse`

```python
inverse(self, utility: 'ArrayLike') -> 'FloatArray'
```

## `ExponentialUtility`

```python
ExponentialUtility(aversion: 'float' = 1.0) -> None
```

Exponential utility with constant absolute risk aversion in reciprocal wealth units.

### `ExponentialUtility.inverse`

```python
inverse(self, utility: 'ArrayLike') -> 'FloatArray'
```

## `PowerUtility`

```python
PowerUtility(aversion: 'float' = 2.0) -> None
```

CRRA: (w**(1-gamma)-1)/(1-gamma), log at gamma=1; w>0.

### `PowerUtility.inverse`

```python
inverse(self, utility: 'ArrayLike')
```

## `SmoothUtility`

```python
SmoothUtility(aversion: 'float' = 0.5, scale: 'float' = 1.0) -> None
```

Options sheet O5: w - aversion*sqrt(scale**2+w**2).

### `SmoothUtility.inverse`

```python
inverse(self, utility: 'ArrayLike') -> 'FloatArray'
```

## `expected_utility`

```python
expected_utility(wealth: 'ArrayLike', utility: 'Utility', *, weights: 'ArrayLike | None' = None) -> 'float'
```

Average utility(wealth) under normalized nonnegative scenario masses.

Utility domain errors remain explicit; weights describe a finite probability law.

## `certainty_equivalent`

```python
certainty_equivalent(wealth: 'ArrayLike', utility: 'Utility', *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return inverse_utility(E[utility(wealth)]) in wealth units.

The supplied Utility must implement an inverse on the attainable utility domain.

## `mean_variance`

```python
mean_variance(gains: 'ArrayLike', aversion: 'float' = 1.0, *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return E[gain] - aversion*Var(gain)/2 under scenario masses.

Aversion has inverse gain units; this is a preference criterion, not expected utility.

## `mean_cvar`

```python
mean_cvar(gains: 'ArrayLike', aversion: 'float' = 1.0, level: 'float' = 0.95, *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return E[gain] - aversion*CVaR(-gain) at the supplied confidence level.

Gains are positive for favorable outcomes; aversion is dimensionless.

## `preference_penalty`

```python
preference_penalty(trade: 'ArrayLike', *, turnover: 'float' = 0.0, switching: 'float' = 0.0, scale: 'ArrayLike' = 1.0) -> 'float'
```

Return turnover and fixed switching penalties for a trade vector.

Scale broadcasts to trade coordinates and must be positive; coefficients
set the penalty units and must be nonnegative.
