# `convexity.math.rates`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: rates.

## `RatePaths`

```python
RatePaths(values: 'ArrayLike', weights: 'ArrayLike', times: 'ArrayLike', measure: 'str' = 'terminal_forward') -> 'None'
```

Simulated rate coordinates and time grid, with model-dependent state meanings.

### `RatePaths.items`

```python
items(self)
```

### `RatePaths.keys`

```python
keys(self)
```

## `InflationDynamics`

```python
InflationDynamics(volatility: 'float', measure: 'str' = 'P', drift: 'float | None' = None, nominal: 'YieldCurve | None' = None, real: 'YieldCurve | None' = None) -> None
```

P uses a supplied expected proportional CPI drift. Q uses D_real/D_nominal.

Curves are deterministic; stochastic rate/inflation correlation and index
publication lags are not implicit. Use the contract's observation dates and
existing CPI fixings when combining this law with inflation instruments.

### `InflationDynamics.growth`

```python
growth(self, time: 'float')
```

### `InflationDynamics.option`

```python
option(self, index, strike: 'float', maturity: 'float', *, kind: 'str' = 'call')
```

European option on the CPI level in nominal currency at maturity.

### `InflationDynamics.sample`

```python
sample(self, index, times, *, draws: 'int', rng: 'np.random.Generator') -> 'PathScenarios'
```

### `InflationDynamics.transition`

```python
transition(self, index, start, end)
```

Exact scalar CPI law conditional on the supplied starting index.

## `RateModel`

Interest-rate model extension with explicitly interpreted state and time coordinates.

### `RateModel.discount_bond`

```python
discount_bond(self, time: float, maturity: float, state)
```

## `ShortRateModel`

```python
ShortRateModel(kind: 'str', parameters: 'Mapping[str, float]', curve: 'YieldCurve | None' = None, measure: 'str' = 'Q') -> None
```

Short-rate dynamics and parameters; time and rate units must agree with the supplied curve convention.

### `ShortRateModel.discount_bond`

```python
discount_bond(self, time: 'float', maturity: 'float', state)
```

### `ShortRateModel.process`

```python
process(self)
```

Existing exact transition interface for time-homogeneous P or Q laws.

### `ShortRateModel.to_quantlib`

```python
to_quantlib(self)
```

## `gaussian_hjm`

```python
gaussian_hjm(curve: 'YieldCurve', step_dates: 'Sequence[DateLike | NativeDate]', volatilities: 'ArrayLike', reversions: 'ArrayLike', *, terminal_horizon: 'float' = 60.0) -> 'NativeGSR'
```

QuantLib GSR: one-factor Gaussian HJM with piecewise volatility/reversion.

Native GSR state uses its terminal forward measure; do not label simulated
native states as bank-account-measure short rates.

## `market_model_paths`

```python
market_model_paths(model: 'NativeMarketModel', *, draws: 'int', seed: 'int' = 42) -> 'RatePaths'
```

QuantLib predictor-corrector forward-rate paths under terminal measure.

model is a native QuantLib MarketModel (e.g. AbcdVol); output axes are
(draw,evolution step,forward rate). Numeraire weights are returned explicitly.

## `EmpiricalCurveForecaster`

```python
EmpiricalCurveForecaster(forecaster: 'Forecaster', tenors: 'ArrayLike') -> None
```

Compose an existing multivariate forecaster with tenors; P only.

### `EmpiricalCurveForecaster.fit`

```python
fit(self, series: 'TimeSeries | ArrayLike', *, covariates=None)
```

## `inflation_curve`

```python
inflation_curve(reference_date: 'DateLike | NativeDate', dates: 'Sequence[DateLike | NativeDate]', rates: 'ArrayLike', *, frequency: 'str | int | None' = None, convention: 'str' = 'actual365', year_on_year: 'bool' = False) -> 'NativeInflationCurve'
```

Construct a native zero or year-on-year inflation curve from dated annual rates.

Reference date, observation frequency and day-count convention define its clock.

## `inflation_index`

```python
inflation_index(curve: 'NativeInflationCurve', *, year_on_year: 'bool' = False, fixings: 'Mapping[FixingDateT, float] | None' = None) -> 'NativeInflationIndex'
```

Inflation index; supplied histories are applied only during Convexity valuation.

## `rate_option_price`

```python
rate_option_price(instrument: 'NativeInstrument', curve: 'YieldCurve', *, kind: 'str | None' = None, method: 'str' = 'black', volatility: 'float | None' = None, model: 'ShortRateModel | None' = None, steps: 'int' = 300, displacement: 'float' = 0.0) -> 'PricingResult'
```

Price a supported native rate option under an explicit discount curve.

method selects Black, normal, tree or model-based valuation; volatility units
follow that method. Instrument/model compatibility is checked by the engine.

## `inflation_swap_price`

```python
inflation_swap_price(instrument: 'NativeInflationSwap', discount: 'YieldCurve') -> 'PricingResult'
```

Value a native inflation swap using an explicit nominal discount curve.

The instrument must already carry its inflation index and forecast curve.

## `rate_index`

```python
rate_index(curve: 'YieldCurve', *, tenor: 'str' = '6M', currency: 'str' = 'USD', fixing_days: 'int' = 0, calendar: 'str' = 'none', adjustment: 'str' = 'unadjusted', convention: 'str' = 'actual360', fixings: 'Mapping[FixingDateT, float] | None' = None) -> 'NativeIborIndex'
```

Forecast index with optional {date: rate} history, scoped during valuation.

## `cap_floor`

```python
cap_floor(schedule: 'NativeSchedule', index: 'NativeIborIndex', strike: 'float', *, kind: 'str' = 'cap', notional: 'float' = 100.0, spread: 'float' = 0.0) -> 'NativeCapFloor'
```

European cap/floor on the supplied index schedule; rates are decimal.

## `interest_rate_swap`

```python
interest_rate_swap(fixed_schedule: 'NativeSchedule', floating_schedule: 'NativeSchedule', index: 'NativeIborIndex', fixed_rate: 'float', *, notional: 'float' = 100.0, payer: 'bool' = True, spread: 'float' = 0.0, fixed_convention: 'str' = '30_360', floating_convention: 'str' = 'actual360') -> 'NativeSwap'
```

Fixed/IBOR swap; payer=True pays fixed and receives index plus spread.

Schedules come from contracts.schedule. Rates are decimal; notional and
swap_price leg values are in the index currency. Forecasting uses the
index curve; discounting is supplied separately to swap_price.

## `swaption`

```python
swaption(swap: 'NativeSwap', exercise_dates: 'Sequence[DateLike | NativeDate]', *, style: 'str' = 'european') -> 'NativeSwaption'
```

Physically settled swaption with European or Bermudan exercise.

The first exercise is on/before swap start; later Bermudan exercise dates
must precede maturity. Exercise dates are an ordered sequence of dates.

## `swap_price`

```python
swap_price(swap: 'NativeSwap', curve: 'YieldCurve') -> 'SwapResult'
```

Return SwapResult with signed leg NPVs, decimal fair_rate and currency.

## `inflation_swap`

```python
inflation_swap(start: 'DateLike | NativeDate', end: 'DateLike | NativeDate', index: 'NativeInflationIndex', fixed_rate: 'float', *, notional: 'float' = 100.0, payer: 'bool' = True, calendar: 'str' = 'none', adjustment: 'str' = 'unadjusted', convention: 'str' = 'actual365', observation_lag_months: 'int' = 3, interpolation: 'str' = 'flat') -> 'NativeInflationSwap'
```

Zero-coupon CPI swap; payer=True pays fixed growth and receives inflation.

Dates accept ISO strings or Python dates. fixed_rate is an annual decimal
growth rate, and notional is in index currency. Supply required historical
CPI observations through inflation_index(..., fixings={date: level}).

## `forward_market_model`

```python
forward_market_model(rate_times: 'ArrayLike', forwards: 'ArrayLike', *, volatility: 'float' = 0.2, factors: 'int | None' = None, displacement: 'float' = 0.0, long_run_correlation: 'float' = 0.5, correlation_decay: 'float' = 0.2) -> 'NativeMarketModel'
```

Lognormal forward-rate model with constant vol and exponential correlation.

rate_times are N+1 reset/payment times, forwards have N entries. Paths use
the terminal bond measure. Displacements allow negative unshifted forwards.

## `gaussian_bond_price`

```python
gaussian_bond_price(model: 'NativeGSR', maturity: 'float', *, time: 'float' = 0.0, state: 'float' = 0.0) -> 'float'
```

GSR conditional zero-bond price; state is the standardized Gaussian factor.

## `SwapResult`

```python
SwapResult(value: 'float', fair_rate: 'float', fixed_leg: 'float', floating_leg: 'float', currency: 'str' = 'USD') -> None
```

Swap valuation with leg/rate information under the specified curve and schedule conventions.

## `HullWhiteProcess`

```python
HullWhiteProcess(model: 'ShortRateModel') -> None
```

Gaussian mean-reverting short-rate process tied to an initial term structure.

### `HullWhiteProcess.diffusion`

```python
diffusion(self, time: 'float', state)
```

### `HullWhiteProcess.drift`

```python
drift(self, time: 'float', state)
```

### `HullWhiteProcess.generator`

```python
generator(self, time: 'float', state, gradient: 'ArrayLike | Callable[[FloatArray], ArrayLike]', hessian: 'ArrayLike | Callable[[FloatArray], ArrayLike]')
```

### `HullWhiteProcess.sample`

```python
sample(self, initial: 'ArrayLike', times, *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1)
```

### `HullWhiteProcess.transition`

```python
transition(self, state, dt: 'float', *, time: 'float' = 0.0, method: 'str' = 'exact', substeps: 'int' = 1)
```
