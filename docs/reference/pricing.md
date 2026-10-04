# `convexity.math.pricing`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: pricing.

## `PricingProblem`

```python
PricingProblem(contract: 'ContractT', model: 'ModelT', valuation_date: 'DateLike | NativeDate') -> 'None'
```

Contract, pricing model and valuation date. Generic types retain custom models without requiring native conversion.

## `PricingResult`

```python
PricingResult(value: 'float', greeks: 'Mapping[str, float]' = <factory>, error_estimate: 'float | None' = None, engine: 'str' = '', currency: 'str' = 'USD') -> None
```

Valuation with named Greeks and engine information; absent Greeks are not silently replaced by zeros.

## `PricingEngine`

Price a PricingProblem into an owned PricingResult; custom engines need no native provider inheritance.

### `PricingEngine.price`

```python
price(self, problem: 'PricingProblem') -> 'PricingResult'
```

## `price`

```python
price(problem: 'PricingProblem', engine: 'PricingEngine') -> 'PricingResult'
```

Evaluate a PricingProblem using a built-in or custom PricingEngine.

Return an owned PricingResult whose currency matches the contract. The model
must explicitly declare Q; physical forecasts do not imply pricing probabilities.

## `VanillaEngine`

```python
VanillaEngine(method: 'str' = 'analytic', steps: 'int | np.integer' = 300, grid: 'int | np.integer' = 300, samples: 'int | np.integer' = 32768, seed: 'int | np.integer' = 42, dividends: 'tuple[CashFlow, ...]' = ()) -> None
```

Named built-in vanilla pricing method with explicit numerical settings and supported model restrictions.

### `VanillaEngine.price`

```python
price(self, problem)
```

## `QuantLibEngine`

```python
QuantLibEngine(factory: 'Callable[[NativePricingModel], NativePricingEngine]', name: 'str' = 'QuantLib') -> None
```

Extension adapter for native instruments and explicit engine factories.

factory(model) creates a QuantLib engine. The caller owns instrument/engine
compatibility; evaluation and all result extraction remain date-scoped.

### `QuantLibEngine.price`

```python
price(self, problem)
```

## `forward_option`

```python
forward_option(forward: 'float', strike: 'float', volatility: 'float', maturity: 'float', *, discount: 'float' = 1.0, kind: 'str' = 'call', model: 'str' = 'black76', displacement: 'float' = 0.0) -> 'float'
```

Black-76/normal prices; normal volatility has price/sqrt(year) units.

## `asian_price`

```python
asian_price(model: 'BlackScholesModel', valuation_date: 'DateLike | NativeDate', strike: 'float', fixing_dates: 'Sequence[DateLike | NativeDate]', *, kind: 'str' = 'call', average: 'str' = 'arithmetic', samples: 'int' = 32768, seed: 'int' = 42, past_fixings: 'int' = 0, running_accumulator: 'float | None' = None) -> 'PricingResult'
```

Discrete European average-price option; running sum/product is explicit.

## `basket_price`

```python
basket_price(models: 'Sequence[BlackScholesModel]', correlation: 'ArrayLike', weights: 'ArrayLike', valuation_date: 'DateLike | NativeDate', expiry: 'DateLike | NativeDate', strike: 'float', *, kind: 'str' = 'call', samples: 'int' = 32768, seed: 'int' = 42) -> 'PricingResult'
```

European weighted arithmetic basket; correlated Q GBM processes.

## `barrier_price`

```python
barrier_price(model: 'BlackScholesModel', valuation_date: 'DateLike | NativeDate', expiry: 'DateLike | NativeDate', strike: 'float', barrier: 'float', *, barrier_type: 'str' = 'down_out', rebate: 'float' = 0.0, kind: 'str' = 'call') -> 'PricingResult'
```

Price a continuously monitored Black-Scholes barrier option under Q.

Expiry/valuation are calendar dates; strike, barrier and rebate use spot-price units.
barrier_type selects down_out, down_in, up_out or up_in.

## `scenario_price`

```python
scenario_price(scenarios: 'PathScenarios', payoff: 'Callable[[FloatArray], ArrayLike]', measure_model: 'MeasureModel[ModelT]', *, currency: 'str' = 'USD') -> 'PricingResult'
```

Value a scalar terminal/path payoff on explicitly Q joint scenarios.

The payoff consumes a single (time[,state]) path. This adapter requires a
deterministic numeraire; use joint state/numeraire payoffs for stochastic
discounting. It preserves the simulation family's sampling-error semantics.

## `BlackScholesModel`

```python
BlackScholesModel(spot: 'float', discount: 'YieldCurve', dividend: 'YieldCurve', volatility: 'float | NativeVolatility') -> None
```

Risk-neutral lognormal model inputs for vanilla pricing, with discount/dividend curves and volatility.

### `BlackScholesModel.from_measure_pair`

```python
from_measure_pair(pair, spot: 'float', reference_date)
```

Compose an explicitly supplied GBM Q model with QuantLib curves.

### `BlackScholesModel.to_quantlib`

```python
to_quantlib(self)
```

## `HestonModel`

```python
HestonModel(spot: 'float', discount: 'YieldCurve', dividend: 'YieldCurve', variance: 'float', kappa: 'float', theta: 'float', sigma: 'float', rho: 'float') -> None
```

Risk-neutral stochastic-variance pricing specification with explicit variance-process parameters.

### `HestonModel.to_quantlib`

```python
to_quantlib(self)
```

## `PricingModel`

A supplied pricing-measure model; each engine checks its required capabilities.

## `NativePricingModel`

Explicit native-engine extension; ordinary custom engines need only PricingModel.

### `NativePricingModel.to_quantlib`

```python
to_quantlib(self) -> 'object'
```
