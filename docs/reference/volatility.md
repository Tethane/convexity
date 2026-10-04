# `convexity.math.volatility`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Conditional variance forecasting and realized-variance models (family 9).

Models use the existing forecasting and distribution contracts. Heavy numerical
dependencies, including arch and statsmodels, are imported only when used.

## `ARCHModelForecaster`

```python
ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'Mapping[str, object]' = <factory>) -> None
```

Conditional-volatility specification with explicit mean, innovation and variance dynamics.

### `ARCHModelForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `ARCHForecaster`

```python
ARCHForecaster(p: 'int' = 1, **kwargs: 'Unpack[ArchforecasterOptions]') -> 'ARCHModelForecaster'
```

Construct the ARCH forecasting specification.

ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: o, mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options.

Keyword options:

```python
o: int
mean: str
lags: int
innovation: str
method: str
simulations: int
decay: float
rescale: bool
check_convergence: bool
fit_options: Mapping[str, object]
```

## `GARCHForecaster`

```python
GARCHForecaster(p: 'int' = 1, q: 'int' = 1, **kwargs: 'Unpack[GarchforecasterOptions]') -> 'ARCHModelForecaster'
```

Construct the GARCH forecasting specification.

ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: o, mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options.

Keyword options:

```python
o: int
mean: str
lags: int
innovation: str
method: str
simulations: int
decay: float
rescale: bool
check_convergence: bool
fit_options: Mapping[str, object]
```

## `GJRGARCHForecaster`

```python
GJRGARCHForecaster(p: 'int' = 1, o: 'int' = 1, q: 'int' = 1, **kwargs: 'Unpack[GjrgarchforecasterOptions]') -> 'ARCHModelForecaster'
```

Construct the GJRGARCH forecasting specification.

ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options.

Keyword options:

```python
mean: str
lags: int
innovation: str
method: str
simulations: int
decay: float
rescale: bool
check_convergence: bool
fit_options: Mapping[str, object]
```

## `EGARCHForecaster`

```python
EGARCHForecaster(p: 'int' = 1, o: 'int' = 1, q: 'int' = 1, **kwargs: 'Unpack[EgarchforecasterOptions]') -> 'ARCHModelForecaster'
```

Construct the EGARCH forecasting specification.

ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options.

Keyword options:

```python
mean: str
lags: int
innovation: str
method: str
simulations: int
decay: float
rescale: bool
check_convergence: bool
fit_options: Mapping[str, object]
```

## `ConstantVarianceForecaster`

```python
ConstantVarianceForecaster(**kwargs: 'Unpack[ConstantvarianceforecasterOptions]') -> 'ARCHModelForecaster'
```

Construct the ConstantVariance forecasting specification.

ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: o, mean, lags, innovation, method, simulations, decay, rescale, check_convergence, fit_options.

Keyword options:

```python
o: int
mean: str
lags: int
innovation: str
method: str
simulations: int
decay: float
rescale: bool
check_convergence: bool
fit_options: Mapping[str, object]
```

## `EWMAForecaster`

```python
EWMAForecaster(decay: 'float' = 0.94, **kwargs: 'Unpack[EwmaforecasterOptions]') -> 'ARCHModelForecaster'
```

Construct the EWMA forecasting specification.

ARCHModelForecaster(model: 'str' = 'garch', p: 'int' = 1, o: 'int' = 0, q: 'int' = 1, mean: 'str' = 'constant', lags: 'int' = 0, innovation: 'str' = 'normal', method: 'str' = 'auto', simulations: 'int' = 5000, decay: 'float' = 0.94, rescale: 'bool' = True, check_convergence: 'bool' = True, fit_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: o, mean, lags, innovation, method, simulations, rescale, check_convergence, fit_options.

Keyword options:

```python
o: int
mean: str
lags: int
innovation: str
method: str
simulations: int
rescale: bool
check_convergence: bool
fit_options: Mapping[str, object]
```

## `HARForecaster`

```python
HARForecaster(lags: 'tuple[int, ...]' = (1, 5, 22), transform: 'str' = 'log', method: 'str' = 'analytic', simulations: 'int' = 5000, covariance: 'str' = 'HC0') -> None
```

OLS on lagged window means of variance or log variance.

log models assume homoskedastic Gaussian log innovations for uncertainty
and Jensen-corrected means. level models supply point forecasts only and
reject negative predictions. Lags are counts of observations, not days.

### `HARForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `innovation_distribution`

```python
innovation_distribution(name: 'str' = 'normal', parameters: 'ArrayLike' = ()) -> 'Distribution'
```

Normal: (); Student-t: (nu,); skew Student-t: (eta,lambda); GED: (nu,).

Every law has mean zero and variance one, unlike an unscaled Student-t.
Shape-parameter constraints are those of the arch distribution backend.

## `realized_variance`

```python
realized_variance(returns: 'ArrayLike', *, axis: 'int' = -1) -> 'FloatArray'
```

Sum squared intraperiod returns along axis, without demeaning or annualizing.

For (period,intraperiod) returns the default returns one variance per period.
Sampling intervals/session boundaries are caller-supplied; NaNs are rejected.

## `realized_volatility`

```python
realized_volatility(returns: 'ArrayLike', *, axis: 'int' = -1) -> 'FloatArray'
```

Square root of realized_variance in the supplied return units.

## `ConvergenceError`

```python
ConvergenceError(result: 'FittedForecaster') -> 'None'
```

Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics.

## `Covariates`

```python
Covariates(known_future: 'ArrayLike | None' = None, observed_past: 'ArrayLike | None' = None, static: 'ArrayLike | None' = None) -> 'None'
```

Training covariates. Dynamic arrays align row-for-row with the target.

known_future is available at forecast time; observed_past is available only
through the cutoff. static is a numeric vector constant over this series.

### `Covariates.slice`

```python
slice(self, start, stop) -> 'Covariates'
```

## `Diagnostics`

```python
Diagnostics(backend: 'str' = 'user', converged: 'bool' = True, message: 'str' = '', iterations: 'int | None' = None, warnings: 'tuple[str, ...]' = (), residuals: 'ArrayLike | None' = None, loss_history: 'ArrayLike | None' = None, validation_history: 'ArrayLike | None' = None) -> 'None'
```

Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable.

## `FittedForecaster`

```python
FittedForecaster(adapter: 'ForecastAdapter', history: 'TimeSeries', specification: 'Forecaster | None' = None, diagnostics: 'Diagnostics' = <factory>, parameters: 'dict[str, object]' = <factory>, model_identity: 'str' = 'user', random_state: 'int | None' = None, artifact_identity: 'str' = <factory>, dependency_versions: 'dict[str, str]' = <factory>) -> None
```

Fitted forecasting adapter with retained history, specification and diagnostics. Future covariates cover the maximum requested step.

### `FittedForecaster.forecast`

```python
forecast(self, horizon: 'int | ForecastHorizon | ArrayLike', *, covariates: 'Covariates | ArrayLike | None' = None, rng: 'np.random.Generator | None' = None)
```

### `FittedForecaster.predict`

```python
predict(self, horizon: 'int | ForecastHorizon | ArrayLike', *, covariates: 'Covariates | ArrayLike | None' = None, rng: 'np.random.Generator | None' = None)
```

### `FittedForecaster.save`

```python
save(self, path)
```

Save a Python pickle artifact; load only files from trusted sources.

### `FittedForecaster.update`

```python
update(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None, mode: 'str' = 'refit')
```

Append observations and refit, incrementally filter, or warm-start.

## `ForecastAdapter`

Produce a ForecastResult for requested future steps; advertise only uncertainty actually provided by the model.

### `ForecastAdapter.forecast`

```python
forecast(self, steps: 'int', covariates: 'FloatArray | None', *, rng: 'np.random.Generator | None' = None) -> 'ForecastResult'
```

Return dense steps 1..steps, using only supplied future-known data.

## `Forecaster`

Fit supplied time-series observations into a reusable fitted forecasting contract.

### `Forecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> "'FittedForecaster'"
```

## `ForecastHorizon`

```python
ForecastHorizon(steps: 'int | ArrayLike', timestamps: 'ArrayLike | None' = None, calendar: 'object' = None) -> 'None'
```

Positive increasing relative row offsets; integer H means 1,...,H.

Explicit timestamps label requested rows. They do not change a discrete
model into an elapsed-time model. Irregular histories require these labels.

### `ForecastHorizon.align`

```python
align(self, history, *, check_frequency: 'bool' = True) -> 'ForecastHorizon'
```

## `ForecastResult`

```python
ForecastResult(point: 'ArrayLike', horizon: 'ForecastHorizon | None' = None, mean_values: 'ArrayLike | None' = None, variance_values: 'ArrayLike | None' = None, quantile_levels: 'ArrayLike | None' = None, quantile_values: 'ArrayLike | None' = None, paths: 'ArrayLike | None' = None, joint_covariance: 'ArrayLike | None' = None, gaussian: 'bool' = False, law: 'Distribution | None' = None, marginal_laws: 'Sequence[Distribution] | None' = None, model_identity: 'str' = 'user', cutoff: 'float | int | np.datetime64 | None' = None, metadata: 'dict[str, object]' = <factory>) -> 'None'
```

Forecast axes are always (horizon,target); paths are (draw,horizon,target).

joint_covariance uses time-major flattened coordinates. gaussian=True
declares Gaussian marginals only; joint_covariance additionally declares a
joint Gaussian law. Paths define an equally weighted empirical joint law.
A supplied distribution has event_shape (horizon * targets,). Scalar
marginal_laws may accompany it to provide exact marginal quantiles. Their
finite moments are checked; the caller must ensure equality of the full laws. Without
a supplied joint law, marginal_laws declare an independent product.

### `ForecastResult.correlation`

```python
correlation(self) -> 'FloatArray'
```

### `ForecastResult.covariance`

```python
covariance(self) -> 'FloatArray'
```

Return (horizon * target, horizon * target) covariance in time-major order.

### `ForecastResult.distribution`

```python
distribution(self) -> 'Distribution'
```

Return the joint law of the time-major flattened path; marginal intervals alone cannot define it.

### `ForecastResult.from_independent`

```python
from_independent(laws, shape: 'tuple[int, int]', **kwargs)
```

### `ForecastResult.from_paths`

```python
from_paths(paths: 'ArrayLike', **kwargs)
```

### `ForecastResult.interval`

```python
interval(self, probability: 'float' = 0.95) -> 'FloatArray'
```

Return marginal lower/upper bounds with shape (horizon, target, 2); not a simultaneous band.

### `ForecastResult.log_density`

```python
log_density(self, values: 'ArrayLike') -> 'FloatArray'
```

Evaluate joint log density for values ending in (horizon, target); preserve leading batch axes.

### `ForecastResult.marginal`

```python
marginal(self, horizon_index: 'int', target: 'int' = 0) -> 'Distribution'
```

Return the law of a zero-based horizon row and target coordinate.

### `ForecastResult.mean`

```python
mean(self) -> 'FloatArray'
```

Return expectations with shape (horizon, target); point predictions alone do not supply means.

### `ForecastResult.probability`

```python
probability(self, event: 'Callable[[FloatArray], bool | np.bool_]') -> 'float'
```

Return empirical mass of a predicate on complete paths, each shaped (horizon, target).

### `ForecastResult.quantile`

```python
quantile(self, probability: 'float') -> 'FloatArray'
```

Return one marginal quantile per (horizon, target); unsupplied levels may be unavailable.

### `ForecastResult.sample`

```python
sample(self, size: 'int', *, rng: 'np.random.Generator') -> 'FloatArray'
```

Draw complete paths with shape (size, horizon, target), retaining supplied dependence.

### `ForecastResult.select`

```python
select(self, indices: 'ArrayLike', *, horizon: 'int | ForecastHorizon | None' = None) -> 'ForecastResult'
```

Select rows, retaining cross-time dependence and coordinate order.

### `ForecastResult.variance`

```python
variance(self) -> 'FloatArray'
```

Return marginal variances with shape (horizon, target), in squared target units.

## `TimeSeries`

```python
TimeSeries(values: 'ArrayLike', timestamps: 'ArrayLike | None' = None, covariates: 'Covariates | None' = None, names: 'Sequence[str] | None' = None) -> 'None'
```

One series with values (n,k), numeric/datetime times, and optional NaNs.

A vector becomes (n,1). No sorting, resampling, or imputation is implicit.
Names and covariate columns have positional identity.

### `TimeSeries.slice`

```python
slice(self, start, stop) -> 'TimeSeries'
```

## `fit`

```python
fit(data: 'TimeSeries | ArrayLike', forecaster: 'Forecaster', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

Fit a Forecaster to TimeSeries or time-major values (time,target).

Return FittedForecaster with history, diagnostics and checked forecast methods.
Covariates distinguish known future information from observed-past inputs.
Point predictions alone do not create a distribution or temporal dependence.

## `load_forecaster`

```python
load_forecaster(path: 'str | PathLike[str]') -> 'FittedForecaster'
```

Restore a trusted pickle in a compatible Python/dependency environment.

## `VolatilityForecastResult`

```python
VolatilityForecastResult(point: 'ArrayLike', horizon: 'ForecastHorizon | None' = None, mean_values: 'ArrayLike | None' = None, variance_values: 'ArrayLike | None' = None, quantile_levels: 'ArrayLike | None' = None, quantile_values: 'ArrayLike | None' = None, paths: 'ArrayLike | None' = None, joint_covariance: 'ArrayLike | None' = None, gaussian: 'bool' = False, law: 'Distribution | None' = None, marginal_laws: 'Sequence[Distribution] | None' = None, model_identity: 'str' = 'user', cutoff: 'float | int | np.datetime64 | None' = None, metadata: 'dict[str, object]' = <factory>, conditional_variance: 'ArrayLike | None' = None, conditional_variance_paths: 'ArrayLike | None' = None, target_kind: 'str' = 'return') -> 'None'
```

Target forecasts plus E[future innovation variance] (or E[realized variance]).

All variance quantities use squared input-return units per observation.
conditional_variance_paths, when present, are (draw,horizon,target) and
retain the same draw order as target paths. They are not squared returns.

### `VolatilityForecastResult.correlation`

```python
correlation(self) -> 'FloatArray'
```

### `VolatilityForecastResult.covariance`

```python
covariance(self) -> 'FloatArray'
```

Return (horizon * target, horizon * target) covariance in time-major order.

### `VolatilityForecastResult.distribution`

```python
distribution(self) -> 'Distribution'
```

Return the joint law of the time-major flattened path; marginal intervals alone cannot define it.

### `VolatilityForecastResult.from_independent`

```python
from_independent(laws, shape: 'tuple[int, int]', **kwargs)
```

### `VolatilityForecastResult.from_paths`

```python
from_paths(paths: 'ArrayLike', **kwargs)
```

### `VolatilityForecastResult.interval`

```python
interval(self, probability: 'float' = 0.95) -> 'FloatArray'
```

Return marginal lower/upper bounds with shape (horizon, target, 2); not a simultaneous band.

### `VolatilityForecastResult.log_density`

```python
log_density(self, values: 'ArrayLike') -> 'FloatArray'
```

Evaluate joint log density for values ending in (horizon, target); preserve leading batch axes.

### `VolatilityForecastResult.marginal`

```python
marginal(self, horizon_index: 'int', target: 'int' = 0) -> 'Distribution'
```

Return the law of a zero-based horizon row and target coordinate.

### `VolatilityForecastResult.mean`

```python
mean(self) -> 'FloatArray'
```

Return expectations with shape (horizon, target); point predictions alone do not supply means.

### `VolatilityForecastResult.probability`

```python
probability(self, event: 'Callable[[FloatArray], bool | np.bool_]') -> 'float'
```

Return empirical mass of a predicate on complete paths, each shaped (horizon, target).

### `VolatilityForecastResult.quantile`

```python
quantile(self, probability: 'float') -> 'FloatArray'
```

Return one marginal quantile per (horizon, target); unsupplied levels may be unavailable.

### `VolatilityForecastResult.sample`

```python
sample(self, size: 'int', *, rng: 'np.random.Generator') -> 'FloatArray'
```

Draw complete paths with shape (size, horizon, target), retaining supplied dependence.

### `VolatilityForecastResult.select`

```python
select(self, indices, *, horizon=None)
```

Select rows, retaining cross-time dependence and coordinate order.

### `VolatilityForecastResult.variance`

```python
variance(self) -> 'FloatArray'
```

Return marginal variances with shape (horizon, target), in squared target units.

### `VolatilityForecastResult.variance_forecast`

```python
variance_forecast(self)
```

Forecast the variance quantity itself, with paths only when available.

Its variance(), if available, is uncertainty about future variance,
not the conditional return variance in this object's variance().

### `VolatilityForecastResult.volatility`

```python
volatility(self, *, periods_per_year: 'float' = 1.0) -> 'FloatArray'
```

sqrt(periods_per_year * E[variance]); not E[sqrt(variance)].

## `VarianceBacktestResult`

```python
VarianceBacktestResult(records: 'tuple[ForecastRecord, ...]', model_identity: 'str') -> None
```

Held-out variance forecasts with proxy/score semantics retained; a proxy is not observed latent variance.

### `VarianceBacktestResult.calibration`

```python
calibration(self, levels=(0.1, 0.5, 0.9))
```

### `VarianceBacktestResult.conditional_errors`

```python
conditional_errors(self, labels)
```

Group origin-level errors by supplied regime labels, without refitting.

### `VarianceBacktestResult.score`

```python
score(self, metric: 'str' = 'qlike', *, by_horizon: 'Literal[False]' = False, **kwargs: 'object') -> 'float'
score(self, metric: 'str' = 'qlike', *, by_horizon: 'Literal[True]', **kwargs: 'object') -> 'FloatArray'
score(self, metric: 'str' = 'qlike', *, by_horizon: 'bool', **kwargs: 'object') -> 'float | FloatArray'
```

Pool origins; by_horizon=True retains (horizon,target) score axes.

## `qlike`

```python
qlike(actual: 'ArrayLike', predicted: 'ArrayLike', *, axis: 'int | tuple[int, ...] | None' = None) -> 'np.float64 | FloatArray'
```

Gaussian quasi-likelihood loss log(predicted) + actual/predicted.

actual is a nonnegative variance proxy; predicted must be positive. This
version permits zero realized variance and omits forecast-independent terms.
It is not constrained to be nonnegative. Lower scores are better.

## `variance_backtest`

```python
variance_backtest(model: 'Forecaster', data: 'TimeSeries | ArrayLike', *, variance_proxy: 'TimeSeries | ArrayLike | None' = None, **kwargs: 'Unpack[BacktestOptions]') -> 'VarianceBacktestResult'
```

Refit on data prefixes; score conditional variance against a supplied proxy.

For return models supply a proxy aligned with every input row (for example
independently measured realized variance or explicitly chosen squared
demeaned returns). HAR uses its realized-variance targets by default.
Proxies enter scoring only, never fitting. TimeSeries proxies must have
exactly matching timestamps. Horizons/origins/options are those of backtest.

Keyword options:

```python
horizon: int | ForecastHorizon | ArrayLike
initial_window: int | None
step: int
window: int | None
origins: ArrayLike | None
seed: int
```

## `volatility_diagnostics`

```python
volatility_diagnostics(standardized_residuals: 'ArrayLike', *, lags: 'int' = 10, ddof: 'int' = 0) -> 'VolatilityDiagnostics'
```

Ljung-Box on z and z**2, plus Engle ARCH LM on supplied residuals.

Pass finite post-initialization standardized residuals. Interior NaNs are
rejected rather than silently collapsing observation gaps. ddof is the
caller's fitted-parameter correction for the LM statistic.

## `FittedVolatilityForecaster`

```python
FittedVolatilityForecaster(adapter: 'ForecastAdapter', history: 'TimeSeries', specification: 'Forecaster | None' = None, diagnostics: 'Diagnostics' = <factory>, parameters: 'dict[str, object]' = <factory>, model_identity: 'str' = 'user', random_state: 'int | None' = None, artifact_identity: 'str' = <factory>, dependency_versions: 'dict[str, str]' = <factory>) -> None
```

Fitted conditional-variance model with retained return history and forecast capabilities.

### `FittedVolatilityForecaster.forecast`

```python
forecast(self, horizon, *, covariates: 'Covariates | ArrayLike | None' = None, rng: 'np.random.Generator | None' = None) -> 'VolatilityForecastResult'
```

### `FittedVolatilityForecaster.predict`

```python
predict(self, horizon: 'int | ForecastHorizon | ArrayLike', *, covariates: 'Covariates | ArrayLike | None' = None, rng: 'np.random.Generator | None' = None)
```

### `FittedVolatilityForecaster.save`

```python
save(self, path)
```

Save a Python pickle artifact; load only files from trusted sources.

### `FittedVolatilityForecaster.update`

```python
update(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None, mode: 'str' = 'refit')
```

Append observations and refit, incrementally filter, or warm-start.
