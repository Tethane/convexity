# `convexity.math.forecasting`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Generic forecasting, explicit uncertainty, and causal temporal evaluation.

Only NumPy loads on family import; optional/model dependencies load on use.

## `load_panel_forecaster`

```python
load_panel_forecaster(path: 'str | PathLike[str]') -> 'GlobalFittedForecaster'
```

Load a trusted shared-panel artifact saved by GlobalFittedForecaster.

## `load_portable`

```python
load_portable(path: 'str | PathLike[str]', *, device: 'str' = 'cpu') -> 'FittedForecaster'
```

Reconstruct numerical state with current compatible implementations.

The schema is independent of Python pickle and backend object layouts.
Mathematical parameter/architecture definitions must remain compatible;
arbitrary future third-party API changes are not silently accepted.

## `save_portable`

```python
save_portable(fitted: 'FittedForecaster', path: 'str | PathLike[str]') -> 'None'
```

Write portable numerical state for built-in engines or registered codecs.

Supported built-ins: baselines, OU, linear lag regression, state-space
classical models, all native neural heads/architectures, DDPM, flow, VAE,
autoregressive and score models. No optimization runs when they are loaded.

## `register_artifact_codec`

```python
register_artifact_codec(name: 'str', adapter_type: 'type[ForecastAdapter]', encode: 'Callable[[FittedForecaster], Mapping[str, object]]', decode: 'Callable[[Mapping[str, object], TimeSeries], FittedForecaster]') -> 'None'
```

Register trusted local codec functions; artifact files cannot import code.

## `ConditionalGenerativeForecaster`

```python
ConditionalGenerativeForecaster(model: 'str' = 'flow', context_length: 'int' = 12, max_horizon: 'int' = 3, hidden_size: 'int' = 64, latent_size: 'int' = 8, layers: 'int' = 4, epochs: 'int' = 100, batch_size: 'int' = 32, learning_rate: 'float' = 0.001, samples: 'int' = 500, seed: 'int' = 0, device: 'str' = 'cpu', missing: 'str' = 'raise', kl_weight: 'float' = 1.0, noise_min: 'float' = 0.01, noise_max: 'float' = 5.0, noise_levels: 'int' = 20, langevin_steps: 'int' = 10, langevin_rate: 'float' = 2e-05) -> None
```

Train an actual conditional flow, VAE, autoregressive law, or score model.

### `ConditionalGenerativeForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `FlowForecaster`

```python
FlowForecaster(**kwargs: 'Unpack[FlowforecasterOptions]') -> 'ConditionalGenerativeForecaster'
```

Construct the Flow forecasting specification.

Train an actual conditional flow, VAE, autoregressive law, or score model.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, hidden_size, latent_size, layers, epochs, batch_size, learning_rate, samples, seed, device, missing, kl_weight, noise_min, noise_max, noise_levels, langevin_steps, langevin_rate.

Keyword options:

```python
context_length: int
max_horizon: int
hidden_size: int
latent_size: int
layers: int
epochs: int
batch_size: int
learning_rate: float
samples: int
seed: int
device: str
missing: str
kl_weight: float
noise_min: float
noise_max: float
noise_levels: int
langevin_steps: int
langevin_rate: float
```

## `VAEForecaster`

```python
VAEForecaster(**kwargs: 'Unpack[VaeforecasterOptions]') -> 'ConditionalGenerativeForecaster'
```

Construct the VAE forecasting specification.

Train an actual conditional flow, VAE, autoregressive law, or score model.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, hidden_size, latent_size, layers, epochs, batch_size, learning_rate, samples, seed, device, missing, kl_weight, noise_min, noise_max, noise_levels, langevin_steps, langevin_rate.

Keyword options:

```python
context_length: int
max_horizon: int
hidden_size: int
latent_size: int
layers: int
epochs: int
batch_size: int
learning_rate: float
samples: int
seed: int
device: str
missing: str
kl_weight: float
noise_min: float
noise_max: float
noise_levels: int
langevin_steps: int
langevin_rate: float
```

## `AutoregressiveForecaster`

```python
AutoregressiveForecaster(**kwargs: 'Unpack[AutoregressiveforecasterOptions]') -> 'ConditionalGenerativeForecaster'
```

Construct the Autoregressive forecasting specification.

Train an actual conditional flow, VAE, autoregressive law, or score model.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, hidden_size, latent_size, layers, epochs, batch_size, learning_rate, samples, seed, device, missing, kl_weight, noise_min, noise_max, noise_levels, langevin_steps, langevin_rate.

Keyword options:

```python
context_length: int
max_horizon: int
hidden_size: int
latent_size: int
layers: int
epochs: int
batch_size: int
learning_rate: float
samples: int
seed: int
device: str
missing: str
kl_weight: float
noise_min: float
noise_max: float
noise_levels: int
langevin_steps: int
langevin_rate: float
```

## `ScoreForecaster`

```python
ScoreForecaster(**kwargs: 'Unpack[ScoreforecasterOptions]') -> 'ConditionalGenerativeForecaster'
```

Construct the Score forecasting specification.

Train an actual conditional flow, VAE, autoregressive law, or score model.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, hidden_size, latent_size, layers, epochs, batch_size, learning_rate, samples, seed, device, missing, kl_weight, noise_min, noise_max, noise_levels, langevin_steps, langevin_rate.

Keyword options:

```python
context_length: int
max_horizon: int
hidden_size: int
latent_size: int
layers: int
epochs: int
batch_size: int
learning_rate: float
samples: int
seed: int
device: str
missing: str
kl_weight: float
noise_min: float
noise_max: float
noise_levels: int
langevin_steps: int
langevin_rate: float
```

## `Panel`

```python
Panel(series: 'Mapping[str, TimeSeries | ArrayLike]') -> 'None'
```

Separate histories; windows never cross series boundaries.

## `NeuralForecastForecaster`

```python
NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'Mapping[str, object]' = <factory>) -> None
```

Wrap a supplied neural forecasting specification in the checked forecast result contract.

### `NeuralForecastForecaster.fit`

```python
fit(self, data: 'Panel', *, covariates: 'None' = None) -> 'GlobalFittedForecaster'
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

### `NeuralForecastForecaster.fit_panel`

```python
fit_panel(self, panel)
```

## `GlobalFittedForecaster`

```python
GlobalFittedForecaster(engine: 'NeuralForecast', panel: 'Panel', specification: 'NeuralForecastForecaster', schema: 'dict[str, int]', coordinates: 'tuple[tuple[str, str, int], ...]', artifact_identity: 'str' = <factory>, dependency_versions: 'dict[str, str]' = <factory>) -> None
```

Fitted shared forecasting model for named series; outputs retain each series identity and horizon.

### `GlobalFittedForecaster.forecast`

```python
forecast(self, horizon: 'int | ForecastHorizon | ArrayLike | None' = None, *, covariates: 'Mapping[str, ArrayLike] | None' = None)
```

### `GlobalFittedForecaster.save`

```python
save(self, path)
```

Persist a trusted Python artifact in a compatible backend environment.

### `GlobalFittedForecaster.update`

```python
update(self, panel, *, mode: 'str' = 'refit')
```

## `TFTForecaster`

```python
TFTForecaster(**kwargs: 'Unpack[TftforecasterOptions]') -> 'NeuralForecastForecaster'
```

Construct the TFT forecasting specification.

NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options.

Keyword options:

```python
context_length: int
max_horizon: int
max_steps: int
validation_size: int
seed: int
device: str
quantiles: tuple[float, ...]
model_options: Mapping[str, object]
```

## `PatchTSTForecaster`

```python
PatchTSTForecaster(**kwargs: 'Unpack[PatchtstforecasterOptions]') -> 'NeuralForecastForecaster'
```

Construct the PatchTST forecasting specification.

NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options.

Keyword options:

```python
context_length: int
max_horizon: int
max_steps: int
validation_size: int
seed: int
device: str
quantiles: tuple[float, ...]
model_options: Mapping[str, object]
```

## `NBEATSForecaster`

```python
NBEATSForecaster(**kwargs: 'Unpack[NbeatsforecasterOptions]') -> 'NeuralForecastForecaster'
```

Construct the NBEATS forecasting specification.

NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options.

Keyword options:

```python
context_length: int
max_horizon: int
max_steps: int
validation_size: int
seed: int
device: str
quantiles: tuple[float, ...]
model_options: Mapping[str, object]
```

## `NHiTSForecaster`

```python
NHiTSForecaster(**kwargs: 'Unpack[NhitsforecasterOptions]') -> 'NeuralForecastForecaster'
```

Construct the NHiTS forecasting specification.

NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options.

Keyword options:

```python
context_length: int
max_horizon: int
max_steps: int
validation_size: int
seed: int
device: str
quantiles: tuple[float, ...]
model_options: Mapping[str, object]
```

## `EncoderDecoderForecaster`

```python
EncoderDecoderForecaster(**kwargs: 'Unpack[EncoderdecoderforecasterOptions]') -> 'NeuralForecastForecaster'
```

Construct the EncoderDecoder forecasting specification.

NeuralForecastForecaster(model: 'str' = 'TFT', context_length: 'int' = 24, max_horizon: 'int' = 6, max_steps: 'int' = 100, validation_size: 'int' = 0, seed: 'int' = 0, device: 'str' = 'cpu', quantiles: 'tuple[float, ...]' = (), model_options: 'dict' = <factory>)
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: context_length, max_horizon, max_steps, validation_size, seed, device, quantiles, model_options.

Keyword options:

```python
context_length: int
max_horizon: int
max_steps: int
validation_size: int
seed: int
device: str
quantiles: tuple[float, ...]
model_options: Mapping[str, object]
```

## `OUForecaster`

```python
OUForecaster(time_unit: 'str' = 'D', maxiter: 'int' = 1000) -> None
```

Independent scalar OU components; parameters use the selected time unit.

### `OUForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `BootstrapForecaster`

```python
BootstrapForecaster(forecaster: 'Forecaster', replications: 'int' = 30, samples: 'int' = 1000, seed: 'int' = 0) -> None
```

Parametric bootstrap of fitted state-space histories, followed by refits.

Simulated histories preserve the fitted model's transition dynamics and
covariates. Parameter draws are re-filtered against the actual history before
forecasting. This approximates sampling uncertainty, not a Bayesian posterior.

### `BootstrapForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `Checkpoint`

```python
Checkpoint(repository: 'str', revision: 'str', directory: 'str') -> None
```

Resolved model repository, immutable revision and local artifact directory.

## `download_checkpoint`

```python
download_checkpoint(repository: 'str', *, revision: 'str | None' = None, cache_dir: 'str | PathLike[str] | None' = None, local_files_only: 'bool' = False) -> 'Checkpoint'
```

Resolve a Hub revision into an immutable cached local checkpoint.

No remote Python code is executed. The resolved commit is retained; offline
reloads use the same directory rather than resolving a moving branch again.

## `load_pretrained`

```python
load_pretrained(provider: "Literal['chronos', 'chronos_bolt', 'chronos2', 'timesfm3', 'timesfm2.5']", *, repository: 'str | None' = None, revision: 'str | None' = None, cache_dir: 'str | PathLike[str] | None' = None, local_files_only: 'bool' = False, device: 'str' = 'cpu', **options: 'object') -> 'ChronosForecaster | Chronos2Forecaster | TimesFMForecaster | TimesFM25Forecaster'
```

Load a concrete provider model and wrap it in the forecasting contract.

## `Chronos2Forecaster`

```python
Chronos2Forecaster(pipeline: 'Chronos2Pipeline', quantiles: 'tuple[float, ...]' = (0.1, 0.5, 0.9), checkpoint: 'str' = 'supplied-chronos2', finetune_steps: 'int' = 0, finetune_horizon: 'int' = 6, finetune_options: 'Mapping[str, object]' = <factory>) -> None
```

Pretrained Chronos-2 specification with explicit marginal quantile levels and optional fine-tuning controls.

### `Chronos2Forecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `TimesFM25Forecaster`

```python
TimesFM25Forecaster(model: 'TimesFM_2p5_200M_torch', checkpoint: 'str' = 'supplied-timesfm2.5') -> None
```

Pretrained TimesFM 2.5 specification; its quantile head supplies marginal forecasts, not a joint path law.

### `TimesFM25Forecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `calendar_horizon`

```python
calendar_horizon(cutoff: 'DateLike | np.datetime64', steps: 'int | ArrayLike | ForecastHorizon', *, frequency: 'str' = 'B', holidays: 'Sequence[DateLike | np.datetime64]' = ()) -> 'ForecastHorizon'
```

Explicit calendar offsets (business days, month ends, etc.) via pandas.

Holidays are supported for business-day frequency. No exchange calendar is
assumed: supply its holiday dates explicitly.

## `covariate_sensitivity`

```python
covariate_sensitivity(fitted: 'FittedForecaster', horizon: 'int | ForecastHorizon | ArrayLike', covariates: 'ArrayLike', *, delta: 'float' = 0.0001) -> 'FloatArray'
```

Central differences of point forecasts per known-future input coordinate.

Returns (forecast_horizon,targets,future_rows,covariate_columns). This is
local numerical sensitivity, not causal attribution.

## `residual_diagnostics`

```python
residual_diagnostics(residuals: 'ArrayLike', *, lags: 'int' = 10) -> 'dict[str, FloatArray]'
```

Per-target mean, RMSE, ACF and Ljung-Box; finite residual rows only.

Leading state-space initialization NaNs can be removed explicitly by the
caller. Interior missing observations are not compressed into adjacent time.

## `BacktestResult`

```python
BacktestResult(records: 'tuple[ForecastRecord, ...]', model_identity: 'str') -> None
```

Chronological held-out forecast records with scalar or per-horizon scoring.

### `BacktestResult.calibration`

```python
calibration(self, levels=(0.1, 0.5, 0.9))
```

### `BacktestResult.conditional_errors`

```python
conditional_errors(self, labels)
```

Group origin-level errors by supplied regime labels, without refitting.

### `BacktestResult.score`

```python
score(self, metric: 'str' = 'mae', *, by_horizon: 'Literal[False]' = False, seasonality: 'int' = 1, probability: 'float' = 0.9) -> 'float'
score(self, metric: 'str' = 'mae', *, by_horizon: 'Literal[True]', seasonality: 'int' = 1, probability: 'float' = 0.9) -> 'FloatArray'
score(self, metric: 'str' = 'mae', *, by_horizon: 'bool', seasonality: 'int' = 1, probability: 'float' = 0.9) -> 'float | FloatArray'
```

Pool origins; by_horizon=True retains (horizon,target) score axes.

## `ForecastRecord`

```python
ForecastRecord(origin: 'int', training_start: 'int', training: 'TimeSeries', actual: 'FloatArray', forecast: 'ForecastResult') -> None
```

One backtest origin, training window, held-out targets and forecast with aligned horizon coordinates.

## `SelectionResult`

```python
SelectionResult(best_name: 'str', best_model: 'Forecaster', scores: 'dict[str, float]', backtests: 'dict[str, BacktestResult]', fitted: 'FittedForecaster | None') -> None
```

Model-selection scores, named backtests, selected specification and optional refit.

### `SelectionResult.best_by_horizon`

```python
best_by_horizon(self, metric='mae')
```

## `backtest`

```python
backtest(model: 'Forecaster', data: 'TimeSeries | ArrayLike', *, horizon: 'int | ForecastHorizon | ArrayLike' = 1, initial_window: 'int | None' = None, step: 'int' = 1, window: 'int | None' = None, origins: 'ArrayLike | None' = None, seed: 'int' = 0) -> 'BacktestResult'
```

Refit at every origin; expose future rows only for known-future covariates.

An origin is the exclusive end of its training slice. Direct/multi-horizon
estimators construct all supervised labels inside that slice. Explicit
origins are strictly increasing positions, never timestamps.

## `compare`

```python
compare(models: 'Mapping[str, Forecaster]', data: 'TimeSeries | ArrayLike', *, include_baselines: 'bool' = True, seasonality: 'int' = 1, **kwargs: 'Unpack[BacktestOptions]') -> 'dict[str, BacktestResult]'
```

Return named backtests including last/drift/mean/seasonal references.

Keyword options:

```python
horizon: int | ForecastHorizon | ArrayLike
initial_window: int | None
step: int
window: int | None
origins: ArrayLike | None
seed: int
```

## `grid_search`

```python
grid_search(factory: 'Callable[..., Forecaster]', parameter_grid: 'Mapping[str, Sequence[object]]', data: 'TimeSeries | ArrayLike', **kwargs: 'Unpack[SelectionOptions]') -> 'SelectionResult'
```

Evaluate a finite Cartesian parameter grid using temporal backtests.

Keyword options:

```python
horizon: int | ForecastHorizon | ArrayLike
initial_window: int | None
step: int
window: int | None
origins: ArrayLike | None
seed: int
metric: str
refit: bool
```

## `select_model`

```python
select_model(models: 'Mapping[str, Forecaster]', data: 'TimeSeries | ArrayLike', *, metric: 'str' = 'mae', refit: 'bool' = True, **kwargs: 'Unpack[BacktestOptions]') -> 'SelectionResult'
```

Temporal validation selection. Scores are selection scores, not test estimates.

Keyword options:

```python
horizon: int | ForecastHorizon | ArrayLike
initial_window: int | None
step: int
window: int | None
origins: ArrayLike | None
seed: int
```

## `DiffusionForecaster`

```python
DiffusionForecaster(context_length: 'int' = 12, max_horizon: 'int' = 3, hidden_size: 'int' = 64, diffusion_steps: 'int' = 50, epochs: 'int' = 100, batch_size: 'int' = 32, learning_rate: 'float' = 0.001, samples: 'int' = 500, seed: 'int' = 0, device: 'str' = 'cpu', missing: 'str' = 'raise') -> None
```

Small conditional denoising network; a working research baseline.

Gaussian forward noise, velocity prediction and the DDPM posterior-variance
reverse process generate joint paths. Velocity avoids dividing network
errors by near-zero terminal signal. Parameters/checkpoints are stored in
the fitted model; a finite training budget does not establish calibration.

### `DiffusionForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `ChronosForecaster`

```python
ChronosForecaster(pipeline: 'ChronosPipeline | ChronosBoltPipeline', output: 'str' = 'quantiles', quantiles: 'tuple[float, ...]' = (0.1, 0.5, 0.9), samples: 'int' = 100, seed: 'int' = 0, checkpoint: 'str' = 'supplied-chronos-checkpoint') -> None
```

Original Chronos/Chronos-Bolt tensor APIs, with explicit output semantics.

output=quantiles calls predict_quantiles. output=samples requires the
original Chronos sampled-path API, not Bolt/Chronos-2 quantile tensors.

### `ChronosForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `FoundationForecaster`

```python
FoundationForecaster(predictor: 'ForecastFunction', identity: 'str' = 'pretrained', missing: 'str' = 'raise') -> None
```

Wrap predictor(history, steps, future_known, rng=...) -> ForecastResult.

The supplied engine owns preprocessing/checkpoints. fit snapshots context;
it does not imply fine-tuning. No weights are downloaded by Convexity.

### `FoundationForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `GenerativeForecaster`

```python
GenerativeForecaster(sampler: 'PathSampler | None' = None, trainer: 'Callable[[TimeSeries], PathSampler] | None' = None, samples: 'int' = 1000, identity: 'str' = 'generative', missing: 'str' = 'raise') -> None
```

Wrap a trained diffusion/flow/VAE/autoregressive conditional path sampler.

sampler(history, steps, future_known, size=..., rng=...) returns (size,H,k).
An optional trainer(history) returns a fitted sampler during fit.

### `GenerativeForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `SktimeForecaster`

```python
SktimeForecaster(estimator: 'BaseForecaster', max_horizon: 'int' = 1, quantiles: 'tuple[float, ...]' = ()) -> None
```

Explicit sktime interoperability wrapper retaining checked forecast axes and supplied uncertainty semantics.

### `SktimeForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `TimesFMForecaster`

```python
TimesFMForecaster(model: 'TimesFM3Forecaster', checkpoint: 'str' = 'supplied-timesfm3-checkpoint') -> None
```

TimesFM 3 predict_batch API with multivariate and dynamic covariates.

### `TimesFMForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `BaselineForecaster`

```python
BaselineForecaster(strategy: 'str' = 'last', period: 'int' = 1, window: 'int' = 5, missing: 'str' = 'raise') -> None
```

Reusable baseline specification; method and seasonality determine its deterministic point forecast.

### `BaselineForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `DriftForecaster`

```python
DriftForecaster(period: 'int' = 1, window: 'int' = 5, missing: 'str' = 'raise') -> None
```

Extrapolate the mean historical change from the last observation; point forecasts alone declare no distribution.

### `DriftForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `MovingMeanForecaster`

```python
MovingMeanForecaster(period: 'int' = 1, window: 'int' = 5, missing: 'str' = 'raise') -> None
```

Forecast the recent-window mean for each target; window is counted in observations.

### `MovingMeanForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `NaiveForecaster`

```python
NaiveForecaster(period: 'int' = 1, window: 'int' = 5, missing: 'str' = 'raise') -> None
```

Repeat the most recent observation for each target and requested future step.

### `NaiveForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `SeasonalMeanForecaster`

```python
SeasonalMeanForecaster(period: 'int' = 1, window: 'int' = 5, missing: 'str' = 'raise') -> None
```

Forecast historical means at matching seasonal positions, with an explicit integer period.

### `SeasonalMeanForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `SeasonalNaiveForecaster`

```python
SeasonalNaiveForecaster(period: 'int' = 1, window: 'int' = 5, missing: 'str' = 'raise') -> None
```

Repeat the most recent observed seasonal cycle using an explicit integer period.

### `SeasonalNaiveForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `ARForecaster`

```python
ARForecaster(lags: 'int' = 1, **kwargs: 'Unpack[ArforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the AR forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: seasonal_order, trend, model_options, fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
seasonal_order: tuple[int, int, int, int]
trend: str | None
model_options: Mapping[str, object] | None
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `ARIMAForecaster`

```python
ARIMAForecaster(order: 'tuple[int, ...]' = (1, 0, 0), *, seasonal_order: 'tuple[int, int, int, int]' = (0, 0, 0, 0), trend: 'str | None' = None, model_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[ArimaforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the ARIMA forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `ARMAForecaster`

```python
ARMAForecaster(order: 'tuple[int, ...]' = (1, 1), **kwargs: 'Unpack[ArmaforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the ARMA forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: seasonal_order, trend, model_options, fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
seasonal_order: tuple[int, int, int, int]
trend: str | None
model_options: Mapping[str, object] | None
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `DynamicFactorForecaster`

```python
DynamicFactorForecaster(k_factors: 'int' = 1, factor_order: 'int' = 1, *, model_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[DynamicfactorforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the DynamicFactor forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `ETSForecaster`

```python
ETSForecaster(*, error: 'str' = 'add', trend: 'str | None' = None, seasonal: 'str | None' = None, seasonal_periods: 'int | None' = None, model_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[EtsforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the ETS forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `ExponentialSmoothingForecaster`

```python
ExponentialSmoothingForecaster(*, trend: 'str | None' = None, seasonal: 'str | None' = None, seasonal_periods: 'int | None' = None, damped_trend: 'bool' = False, model_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[ExponentialsmoothingforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the ExponentialSmoothing forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `HoltForecaster`

```python
HoltForecaster(**kwargs: 'Unpack[HoltforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the Holt forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: seasonal, seasonal_periods, damped_trend, model_options, fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
seasonal: str | None
seasonal_periods: int | None
damped_trend: bool
model_options: Mapping[str, object] | None
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `HoltWintersForecaster`

```python
HoltWintersForecaster(period: 'int' = 12, *, seasonal: 'str' = 'add', **kwargs: 'Unpack[HoltwintersforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the HoltWinters forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: damped_trend, model_options, fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
damped_trend: bool
model_options: Mapping[str, object] | None
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `SARIMAForecaster`

```python
SARIMAForecaster(order: 'tuple[int, ...]' = (1, 0, 0), *, seasonal_order: 'tuple[int, int, int, int]' = (1, 0, 0, 12), **kwargs: 'Unpack[SarimaforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the SARIMA forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: trend, model_options, fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
trend: str | None
model_options: Mapping[str, object] | None
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `SARIMAXForecaster`

```python
SARIMAXForecaster(order: 'tuple[int, ...]' = (1, 0, 0), *, seasonal_order: 'tuple[int, int, int, int]' = (0, 0, 0, 0), trend: 'str | None' = None, model_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[SarimaxforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the SARIMAX forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `StatsmodelsForecaster`

```python
StatsmodelsForecaster(model: 'str' = 'arima', model_options: 'Mapping[str, object]' = <factory>, fit_options: 'Mapping[str, object]' = <factory>, missing: 'str' = 'raise', check_convergence: 'bool' = True, samples: 'int' = 1000, seed: 'int' = 0) -> None
```

Finite catalog with native model/fit options passed to statsmodels.

AR, ARMA and SARIMA are configurations of ARIMA/SARIMAX. Holt and
Holt-Winters are configurations of exponential smoothing.

### `StatsmodelsForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `StructuralForecaster`

```python
StructuralForecaster(level: 'str' = 'local level', *, model_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[StructuralforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the Structural forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `VARForecaster`

```python
VARForecaster(lags: 'int' = 1, *, fit_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[VarforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the VAR forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: model_options, missing, check_convergence, samples, seed.

Keyword options:

```python
model_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `VARMAXForecaster`

```python
VARMAXForecaster(order: 'tuple[int, ...]' = (1, 0), *, model_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[VarmaxforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the VARMAX forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `VECMForecaster`

```python
VECMForecaster(*, k_ar_diff: 'int' = 1, coint_rank: 'int' = 1, model_options: 'Mapping[str, object] | None' = None, **kwargs: 'Unpack[VecmforecasterOptions]') -> 'StatsmodelsForecaster'
```

Construct the VECM forecasting specification.

Finite catalog with native model/fit options passed to statsmodels.
Fit with forecasting.fit(TimeSeries(values), specification). Forecast rows
are horizons and columns are targets; variances use squared target units.
Optional settings: fit_options, missing, check_convergence, samples, seed.

Keyword options:

```python
fit_options: Mapping[str, object]
missing: str
check_convergence: bool
samples: int
seed: int
```

## `EnsembleForecaster`

```python
EnsembleForecaster(forecasters: 'tuple[Forecaster, ...]', weights: 'ArrayLike | Callable[[TimeSeries], ArrayLike] | None' = None, method: 'str' = 'mean', samples: 'int' = 1000) -> 'None'
```

Combine constituent forecasts with normalized weights and explicit uncertainty-combination semantics.

### `EnsembleForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `StackingForecaster`

```python
StackingForecaster(forecasters: 'tuple[Forecaster, ...]', horizon: 'int' = 1, initial_window: 'int | None' = None, step: 'int' = 1, window: 'int | None' = None) -> None
```

Learn nonnegative, sum-to-one weights from temporal out-of-fold predictions.

Validation origins see only their own historical prefixes. A full-data
refit follows weight selection; keep separate test data for final scoring.

### `StackingForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `ensemble`

```python
ensemble(forecasters: 'Sequence[Forecaster]', *, weights: 'ArrayLike | Callable[[TimeSeries], ArrayLike] | None' = None, method: 'str' = 'mean', samples: 'int' = 1000) -> 'EnsembleForecaster'
```

Construct an EnsembleForecaster from compatible forecasting specifications.

method=mean pools points, median takes coordinatewise medians, mixture pools
complete path laws. Weights are nonnegative masses or a history-to-weights callback.

## `SequenceSchema`

```python
SequenceSchema(context_length: 'int', horizon: 'int', targets: 'int', known: 'int', observed: 'int', static: 'int', channels: 'int') -> None
```

Ordered lag/covariate feature layout for converting a time series into supervised rows.

## `TorchForecaster`

```python
TorchForecaster(module_factory: 'Callable[[SequenceSchema], Module] | None' = None, architecture: 'str' = 'mlp', context_length: 'int' = 12, max_horizon: 'int' = 1, hidden_size: 'int' = 32, head: 'str' = 'point', quantiles: 'tuple[float, ...]' = (0.1, 0.5, 0.9), epochs: 'int' = 100, batch_size: 'int' = 32, learning_rate: 'float' = 0.001, validation_size: 'int' = 0, patience: 'int' = 10, seed: 'int' = 0, device: 'str' = 'cpu', missing: 'str' = 'raise', loss: 'Callable[[Tensor, Tensor, Tensor], Tensor] | None' = None, decoder: 'ForecastDecoder | None' = None, output_channels: 'int' = 1, mixture_components: 'int' = 3, category_values: 'tuple[float, ...]' = (0.0, 1.0), discretize: 'bool' = False) -> None
```

A module factory receives SequenceSchema and returns module(batch).

batch contains history, mask, past_known, past_observed, future_known, static.
Output shape is (batch,horizon,targets,channels). Custom heads supply loss
and decoder callbacks; see the forecasting guide for their contracts.

### `TorchForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None)
```

## `RegressionForecaster`

```python
RegressionForecaster(estimator: 'RegressionEstimator | Learner | None' = None, lags: 'tuple[int, ...]' = (1, 2, 3), rolling_windows: 'tuple[int, ...]' = (), strategy: 'str' = 'recursive', max_horizon: 'int' = 1, missing: 'str' = 'raise') -> None
```

Fit a regression learner to explicit lag features; forecast strategy controls multi-step recursion.

### `RegressionForecaster.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, covariates: 'Covariates | None' = None) -> 'FittedForecaster'
```

## `ConvergenceError`

```python
ConvergenceError(result: 'FittedForecaster') -> 'None'
```

Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics.

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

## `ForecastFunction`

Produce a checked result from history and H future known-covariate rows.

## `PathSampler`

Generate joint paths (size,H,target); rng is owned by the caller.

## `RegressionEstimator`

Fit a supervised numeric predictor; fit and predict receive sample rows.

### `RegressionEstimator.fit`

```python
fit(self, features: 'FloatArray', response: 'FloatArray', /) -> 'object'
```

### `RegressionEstimator.predict`

```python
predict(self, features: 'FloatArray', /) -> 'ArrayLike'
```

## `ForecastDecoder`

Decode a neural head without inventing uncertainty absent from the head.
