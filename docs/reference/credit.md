# `convexity.math.credit`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: credit.

## `CDSResult`

```python
CDSResult(value: 'float', fair_spread: 'float', premium_leg: 'float', protection_leg: 'float', spread_sensitivity: 'float', accrual_rebate: 'float' = 0.0) -> None
```

Credit-default-swap valuation under explicit premium, default and recovery assumptions.

### `CDSResult.items`

```python
items(self)
```

### `CDSResult.keys`

```python
keys(self)
```

## `HazardEstimate`

```python
HazardEstimate(hazard: 'float', events: 'int', exposure: 'float', standard_error: 'float | None', measure: 'str' = 'P') -> None
```

Estimated default intensity/survival information from the supplied observations.

### `HazardEstimate.items`

```python
items(self)
```

### `HazardEstimate.keys`

```python
keys(self)
```

## `StructuralCreditResult`

```python
StructuralCreditResult(equity: 'float', debt: 'float', q_default_probability: 'float', credit_spread: 'float', p_default_probability: 'float | None' = None) -> None
```

Structural default-model values and probability quantities under the declared model assumptions.

### `StructuralCreditResult.items`

```python
items(self)
```

### `StructuralCreditResult.keys`

```python
keys(self)
```

## `DefaultLaw`

Default-time probability law with survival/default semantics declared by its operations.

### `DefaultLaw.default_probability`

```python
default_probability(self, start, end) -> 'float'
```

### `DefaultLaw.hazard`

```python
hazard(self, maturity: 'float') -> 'float'
```

### `DefaultLaw.survival`

```python
survival(self, maturity: 'float') -> 'float'
```

## `HazardCurve`

```python
HazardCurve(structure: 'NativeDefaultCurve', measure: 'str' = 'Q', _calibration_source: 'object | None' = None) -> None
```

Term structure of default intensity/survival with explicit reference date and day-count convention.

### `HazardCurve.default_probability`

```python
default_probability(self, start, end)
```

### `HazardCurve.density`

```python
density(self, maturity: 'float | DateLike | NativeDate')
```

### `HazardCurve.hazard`

```python
hazard(self, maturity: 'float | DateLike | NativeDate')
```

### `HazardCurve.sample`

```python
sample(self, draws: 'int', *, horizon, rng: 'np.random.Generator')
```

Default times up to horizon; +inf means survival beyond horizon.

### `HazardCurve.survival`

```python
survival(self, maturity: 'float | DateLike | NativeDate')
```

## `flat_hazard`

```python
flat_hazard(reference_date: 'DateLike | NativeDate', hazard: 'float', *, measure: 'str' = 'Q', convention: 'str' = 'actual365') -> 'HazardCurve'
```

Construct constant default intensity per year under an explicit P or Q label.

Survival at time t is exp(-hazard*t); the measure label does not transform a model.

## `survival_curve`

```python
survival_curve(dates: 'Sequence[DateLike | NativeDate]', probabilities: 'ArrayLike', *, measure: 'str' = 'Q', convention: 'str' = 'actual365', extrapolate: 'bool' = False) -> 'HazardCurve'
```

Interpolate nonincreasing survival probabilities at increasing dates.

Probabilities are unconditional survival from the reference date; P/Q labeling
is explicit and extrapolation must be enabled deliberately.

## `bootstrap_cds`

```python
bootstrap_cds(reference_date: 'DateLike | NativeDate', helpers: 'Sequence[NativeDefaultHelper]', *, convention: 'str' = 'actual365', extrapolate: 'bool' = False) -> 'HazardCurve'
```

Bootstrap a Q default curve from native CDS calibration helpers.

Helpers own spreads, recoveries, schedules and discount curves; the returned
curve retains repricing diagnostics. Extrapolation is explicit.

## `cds_price`

```python
cds_price(schedule: 'NativeSchedule', spread: 'float', discount: 'YieldCurve', default: 'HazardCurve', *, notional: 'float' = 1.0, recovery: 'float' = 0.4, buyer: 'bool' = True, protection_start: 'DateLike | NativeDate | None' = None, trade_date: 'DateLike | NativeDate | None' = None, cash_settlement_days: 'int' = 3) -> 'CDSResult'
```

Value scheduled CDS protection and premium legs under Q discount/default curves.

Spread is annual decimal premium, recovery a fraction, notional a currency amount.
Protection/trade dates and cash settlement lag are explicit.

## `risky_cashflows`

```python
risky_cashflows(times: 'ArrayLike', amounts: 'ArrayLike', discount: 'YieldCurve', default: 'HazardCurve', *, recovery: 'float' = 0.0, recovery_notional: 'float' = 1.0, integration_points: 'ArrayLike | None' = None) -> 'float'
```

Recovery of par paid at default through final payment, continuous integral.

Float times use each curve's year convention; matching conventions required.

## `estimate_hazard`

```python
estimate_hazard(exposure_times: 'ArrayLike', defaults: 'ArrayLike') -> 'HazardEstimate'
```

Constant P intensity MLE with right-censoring; times are at-risk durations.

## `MigrationModel`

```python
MigrationModel(generator: 'ArrayLike', measure: 'str' = 'P') -> 'None'
```

Continuous-time rating transitions with absorbing final default state.

### `MigrationModel.fit`

```python
fit(transition_counts, exposures, *, measure: 'str' = 'P')
```

### `MigrationModel.transition`

```python
transition(self, time: 'float')
```

## `merton_structural`

```python
merton_structural(asset_value: 'float', debt: 'float', asset_volatility: 'float', maturity: 'float', rate: 'float', *, physical_drift: 'float | None' = None) -> 'StructuralCreditResult'
```

Merton terminal default model; Q debt valuation and optional P probability.

This is terminal insolvency, not a first-passage default-time intensity model.

## `cds_quote`

```python
cds_quote(spread: 'float', tenor: 'str', discount: 'YieldCurve', *, recovery: 'float' = 0.4, settlement_days: 'int' = 0, calendar: 'str' = 'none', adjustment: 'str' = 'unadjusted', frequency: 'str' = 'quarterly', convention: 'str' = 'actual360') -> 'NativeDefaultHelper'
```

Par CDS running spread for bootstrap, using forward schedule generation.
