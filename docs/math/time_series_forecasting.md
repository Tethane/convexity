# Time-series forecasting

Family 8 lives in `convexity.math.forecasting`. It provides one forecasting
contract, a finite working model catalog, temporal evaluation, and fitted
artifacts. NumPy loads on import. Classical, regression, neural and optional
interoperability dependencies load when used.

```python
import numpy as np
from convexity.math import forecasting as ts

rng = np.random.default_rng(42)
y = np.cumsum(rng.normal(size=200))
fitted = ts.ARIMAForecaster(order=(1, 1, 0)).fit(y)
forecast = fitted.forecast(horizon=10)
forecast.mean()                 # (10, 1)
forecast.interval(.9)           # (10, 1, 2), marginal equal-tail intervals

results = ts.compare(
    {"ARIMA": ts.ARIMAForecaster(order=(1, 1, 0))},
    y, horizon=5, initial_window=120, step=20,
)
print({name: result.score("rmse") for name, result in results.items()})
```

The [generated-series script](examples/forecasting.py) and
[worked notebook](notebooks/forecasting.ipynb) require no external data or
pretrained checkpoint downloads.

## Working catalog and acceptance boundary

| Specification | Implemented behavior | Uncertainty returned |
|---|---|---|
| `NaiveForecaster`, `DriftForecaster` | Last value; endpoint slope extrapolation | Point/mean estimates |
| `SeasonalNaiveForecaster`, `MovingMeanForecaster`, `SeasonalMeanForecaster` | Last season; trailing window mean; historical phase means | Point/mean estimates |
| `ARForecaster`, `ARMAForecaster`, `ARIMAForecaster`, `SARIMAForecaster` | statsmodels ARIMA, including seasonal orders and exogenous regressors | Joint Gaussian paths conditional on fitted parameters |
| `SARIMAXForecaster` | statsmodels SARIMAX with model/fit options | Joint Gaussian paths conditional on fitted parameters |
| `ExponentialSmoothingForecaster`, `HoltForecaster`, `HoltWintersForecaster`, `ETSForecaster` | statsmodels exponential smoothing and ETS | Simulated innovation paths, including multiplicative variants |
| `VARForecaster`, `VECMForecaster` | Vector autoregression/error correction | Joint Gaussian paths from fitted innovation covariance and moving-average responses |
| `VARMAXForecaster`, `StructuralForecaster`, `DynamicFactorForecaster` | statsmodels multivariate/structural state-space fitting | Joint Gaussian paths conditional on fitted parameters |
| `RegressionForecaster` | Cloned sklearn-compatible or Convexity learners; lags, rolling mean/std, recursive/direct/multi-output strategies | Point/mean estimates |
| `TorchForecaster` | MLP, temporal CNN, RNN, GRU, LSTM, Transformer; user module factories | Point, Gaussian, Student-t, lognormal, Gaussian mixture, categorical, quantile, or user-decoded law |
| `DiffusionForecaster` | Conditional DDPM velocity-prediction training and reverse diffusion | Empirical joint paths |
| `GenerativeForecaster` | Supplied conditional sampler, or a trainer returning one | Empirical joint paths |
| `FoundationForecaster` | Supplied pretrained predictor returning a checked result | Predictor's declared outputs |
| `ChronosForecaster` | Original Chronos sample API; original/Bolt tensor quantile API | Samples **or** quantiles/point (mean only for original Chronos) |
| `TimesFMForecaster` | TimesFM 3 `predict_batch`, multivariate targets and dynamic covariates | Point forecast and nine deciles |
| `SktimeForecaster` | Optional cloned sktime estimators, relative horizons, point/quantile APIs | Point and optionally requested quantiles |
| `EnsembleForecaster`, `ensemble` | Mean, median, fixed/cutoff-dependent weights, mixtures of full paths | Point pooling or empirical probabilistic mixture |
| `StackingForecaster` | Temporal out-of-fold predictions; nonnegative sum-to-one least-squares weights; full-data refit | Weighted point/mean estimates |

The expanded [completion checklist](forecasting_completion.md) also includes:

| Specification | Implementation | Output |
|---|---|---|
| `TFTForecaster`, `PatchTSTForecaster`, `NBEATSForecaster`, `NHiTSForecaster`, `EncoderDecoderForecaster` | Actual NeuralForecast TFT, PatchTST, NBEATS, NHITS and encoder-decoder RNN architectures | Points or monotonically rearranged marginal quantiles |
| `FlowForecaster` | Conditional RealNVP affine couplings with exact likelihood training | Joint sampled paths |
| `VAEForecaster` | Conditional sequence VAE with Gaussian recognition/decoder and ELBO training | Joint sampled paths |
| `AutoregressiveForecaster` | GRU-cell Gaussian emissions over time-major coordinates, teacher-forcing training | Joint sampled paths |
| `ScoreForecaster` | Noise-conditioned denoising score matching and annealed Langevin sampling | Joint sampled paths |
| `Chronos2Forecaster`, `load_pretrained` | Real cached Chronos, Bolt, Chronos-2, TimesFM 3 and 2.5 checkpoints; Chronos-2 fine-tuning | Provider quantiles; original Chronos also supplies paths |
| `OUForecaster` | Exact Ornstein–Uhlenbeck transition likelihood with actual elapsed gaps | Joint Gaussian paths; independent target components |
| `BootstrapForecaster` | Parametric-bootstrap refits of state-space parameters, re-filtered on actual history | Innovation and approximate parameter uncertainty |
| `Panel`, `GlobalFittedForecaster` | Shared specialist model across distinct series and target channels | One aligned result per series |

Built-in generative networks are trainable research models. Small smoke-test
training budgets establish executable behavior, not forecasting accuracy or
calibration on a particular dataset.

## Series, horizons, timestamps and available information

All target/result arrays use **time first, target second**. Input `(n,)` becomes
`(n,1)`; output never squeezes away the target axis. One `TimeSeries` represents
one univariate or multivariate history. `Panel({id: series})` represents separate histories.

| Object | Shape/meaning |
|---|---|
| `TimeSeries.values` | `(n,k)`, real numeric values; missing targets can be NaN |
| `TimeSeries.timestamps` | `(n,)`, strictly increasing numeric or NumPy datetime64 labels; default `arange(n)` |
| `Covariates.known_future` | Training `(n,p)` values of regressors known at forecast time |
| `Covariates.observed_past` | Training `(n,q)` values available only through the cutoff |
| `Covariates.static` | Numeric `(s,)` vector fixed across this series |
| `ForecastHorizon(H)` | Relative row offsets `1,...,H` |
| `ForecastHorizon([1,3,6])` | Strictly increasing positive requested offsets |
| Forecast-time `covariates` | `(max_horizon,p)` covering **all** intervening rows |
| `result.point`, `mean()`, `variance()`, `quantile(p)` | `(requested_horizons,k)` |
| `result.interval(p)` | `(requested_horizons,k,2)`, lower then upper |
| `result.paths`, `result.sample(n,rng=...)` | `(draws,requested_horizons,k)` |
| `result.covariance()` | `(h*k,h*k)` in time-major flattened order |

Construct `TimeSeries(values, timestamps, covariates, names)` to name targets and
attach times. Alternatively, pass arrays and `fit(..., covariates=Covariates(...))`.
All public numeric data/results are copied and read-only. Columns retain their
positional identities; no alignment by pandas column names is implicit.

Regular numeric/datetime frequency is inferred from timestamp differences.
Sparse forecasts retain their requested offsets and aligned timestamps. An
irregular or single-row history requires `ForecastHorizon(steps, timestamps=...)`.
For a regular history, explicit forecast labels must match the relative steps
and fitted frequency. `calendar_horizon(cutoff, H, frequency='B', holidays=...)`
constructs business-day labels; `frequency='ME'` constructs month ends. Calendar
horizons explicitly override fixed elapsed spacing. No resampling is implicit.
Most models operate in observation order. `OUForecaster(time_unit='D')` instead
fits exact transitions using timestamp gaps and forecasts against supplied future
times. Numeric timestamps use their original units; datetime inputs support
seconds, minutes, hours or days.

At forecast time, only known-future arrays are accepted. Observed-past and static
information live in the fitted history. Covariate widths and row counts must
match training; unsupported covariates raise instead of being ignored. A
caller declaring a feature "known future" is responsible for its actual
availability at each forecast origin.

Missing policy defaults to `raise`. `ffill` carries values forward causally and
rejects a missing first observation in any target. It never backfills or
interpolates from future targets. `native` is available for the state-space
adapters (ARIMA/SARIMAX, VARMAX, structural and dynamic factor). Neural
`missing='mask'` supplies an observation mask and excludes missing labels from
losses. Covariates must be finite. No model silently drops time rows. Backtest
held-out target rows must be observed, and MASE requires finite training values.

## Forecasts as mathematical objects

`ForecastResult.capabilities` states what is actually available. Unsupported
operations raise `NotImplementedError`. Construction supports:

- `ForecastResult(point, mean_values=...)` for point/mean estimates.
- `mean_values`, `variance_values`, `gaussian=True` for Gaussian marginals.
- `joint_covariance` with `gaussian=True` for an explicitly joint Gaussian law.
- `quantile_levels`, `quantile_values` for only the supplied marginal quantiles.
- `ForecastResult.from_independent(scalar_laws, (h,k))` for explicit independent coordinates.
- `ForecastResult.from_paths(paths)` for an equally weighted empirical joint law.
- `ForecastResult(point, law=distribution)` for a Convexity distribution with
  vector event shape `(h*k,)`; its capabilities determine available moments.

A supplied joint `law` can be accompanied by `marginal_laws`, one scalar law
per flattened coordinate, to expose exact marginal quantiles and intervals.
These must describe the marginals of the joint law; the caller is responsible
for their mathematical consistency. The joint law retains its dependence for
sampling, covariance, density, and horizon selection. Without a joint `law`,
`marginal_laws` declares an independent product.

Point forecasts receive no invented variance. Gaussian marginal intervals do
not supply a joint law. Separate quantiles also do not define a continuous
density or joint dependence. Quantile levels must increase inside `(0,1)` and
quantiles cannot cross. Only supplied levels are queried; there is no implicit
tail extrapolation/interpolation. Intervals are equal-tail **pointwise**
intervals, not simultaneous path bands.

`marginal(horizon_index, target=0)` returns a Convexity distribution when one
exists. For a user-supplied vector law its marginal retains a one-coordinate
vector event. `distribution()` returns a law over flattened path coordinates;
`sample()` restores `(draw,h,k)`. A joint Gaussian supports singular covariance
without adding jitter. Its log density follows SciPy's density-on-support
convention; deterministic means have log density zero at the mean and minus
infinity elsewhere. Gaussian covariance matrices tolerate roundoff-scale
negative eigenvalues only.

Paths define a finite empirical law: means, population variances/covariance and
inverse-ECDF quantiles are exact for that finite law. `sample()` resamples whole
paths, preserving temporal and cross-target dependence. `probability(predicate)`
averages one boolean predicate per complete stored path. It is an empirical
probability estimate for the generating model. Empirical log density is log
**mass**, not a smooth density estimate. Correlation rejects zero-variance
coordinates because their correlation is undefined.

Classical uncertainty conditions on fitted parameters. VAR/VECM uses innovation
covariance and moving-average propagation. State-space models use forecast-state
covariance and transition/design matrices, preserving cross-time and cross-target
errors. Smoothing/ETS uses the backend simulation equations with fitted residual
innovation scale and explicit local random draws. Their `point` is the backend
forecast, while moments are computed from the empirical paths.

`BootstrapForecaster(base, replications=30)` additionally simulates histories
recursively through the fitted state-space transitions, conditioned on an initial
observed prefix and the original covariates. It refits every simulated history.
Each parameter vector then re-filters the **actual** history before generating
future paths. This is an approximate parametric-bootstrap mixture, not a Bayesian
posterior. It requires a linear state-space backend and finite training values.
Replications and sample count control approximation error.
Forecasts carry cutoff, target names, identity and training metadata.

## Regression and neural learning

`RegressionForecaster(estimator, lags=(1,2,3), rolling_windows=(5,),
strategy='direct', max_horizon=5)` constructs features at an exclusive origin.
Lag one is the last observed row; rolling means and population standard
deviations end at that origin. Direct targets are `y[origin+h-1]`, and the
known-future feature row is exactly that target row. Static features and lagged
observed-past covariates are supported. Recursive regression rejects
observed-past covariates because their later lag values would be unavailable.

Recursive models fit one-step targets and feed predictions back as target lags.
Direct models fit one estimator per lead. Multi-output models fit one estimator
on flattened horizon/target coordinates. The default estimator is sklearn
`Ridge(alpha=1)`. Supply any compatible estimator, including tree ensembles or
installed XGBoost/LightGBM regressors. Estimators supporting only scalar targets
need sklearn `MultiOutputRegressor` for multivariate/multi-output fitting.
Convexity mean learners can also supply the fit/predict interface. Classification
or quantile learners do not become mean forecasters implicitly.

Neural training uses float32 Adam, causal context windows, minibatches, masks,
training-prefix target normalization, optional contiguous validation holdout,
early stopping and restoration of the best validation state. Overlapping
training/validation **labels** are excluded; validation context may use prior
observations available at its own origin. Target normalization never sees the
validation suffix. Covariates are not automatically scaled. CPU is the default;
`device='cuda'` is supported. The epoch budget is not an optimizer-convergence
claim. PyTorch RNG state is restored after built-in training; seeds control
initialization and batching, with usual backend/device reproducibility limits.

`TorchForecaster(module_factory=...)` calls `factory(SequenceSchema)` to construct
a fresh `torch.nn.Module`. The schema describes context length, horizon, targets,
known/observed/static widths and output channels. `module(batch)` receives:

```text
history, mask:       (batch,context,targets)  # normalized; missing entries zero
past_known:          (batch,context,known)
past_observed:       (batch,context,observed)
future_known:        (batch,horizon,known)
static:              (batch,static)
output:              (batch,horizon,targets,channels)
```

Point heads have one channel and squared loss. Gaussian heads have location and
raw scale; softplus scale gives a **conditionally independent** Gaussian law
across forecast coordinates. Quantile heads have one channel per configured
level, use ordered softplus increments and pinball loss. Their point forecast
is the median when fitted, otherwise the supplied level nearest 0.5.

Built-in `head='student_t'` learns location, positive scale and degrees of freedom
above two; `head='lognormal'` models log targets and requires positive observations.
`head='mixture'` learns `mixture_components` Gaussian weights, means and scales.
`head='categorical'` uses explicit ordered `category_values`; observed targets must
match that support unless `discretize=True`, which assigns the nearest support
value. These heads return exact scalar Convexity laws and an explicitly independent
joint law in original target units. Lognormal normalization occurs in log space.

For other distributions use `head='custom'`, `output_channels=...`, and supply
`loss(raw_output, normalized_target, observation_mask) -> scalar torch loss`
and `decoder(raw_output, location, scale, rng=...) -> ForecastResult` in original
target units. The decoder runs without gradients. It can return arbitrary
Convexity laws or sampled paths. Serialization of callbacks requires importable
Python definitions rather than local lambdas.

Direct/multi-output/neural/diffusion forecasts reject leads beyond their fitted
maximum horizon. Multi-output, neural and diffusion models using known-future
covariates require the full trained horizon in each call because all those
future inputs enter the prediction jointly. Sparse selections are supported
when their maximum reaches that full horizon.

## Dedicated architectures and shared panel training

```python
panel = ts.Panel({'a': np.sin(np.arange(80.) / 6),
                  'b': np.cos(np.arange(100.) / 6)})
model = ts.NHiTSForecaster(context_length=12, max_horizon=4, max_steps=100)
global_fit = model.fit_panel(panel)
forecasts = global_fit.forecast(4)  # dict: a and b have their own cutoffs
```

TFT, PatchTST, N-BEATS, N-HiTS and encoder-decoder wrappers instantiate the named
NeuralForecast architectures. `model_options` exposes backend architecture and
trainer settings without overriding data/horizon contracts. A shared network
trains on windows within each history. `global_fit.save(path)` / `load_panel_forecaster(path)` preserve a trusted
compatible-environment panel artifact with its identity, data digest and backend
versions. Target channels also form separate
training series; this is shared parameter training, not a joint multivariate
innovation law. Panel histories may differ in length but must share target names
and covariate widths. Each retains its own timestamps and cutoff.

TFT/N-HiTS/encoder-decoder support known, past and static covariates; architecture
restrictions are enforced by the backend. Future covariates are a mapping from
series IDs to full trained-horizon arrays. Unknown IDs are rejected. Use explicit
`ForecastHorizon` labels for irregular histories. `quantiles=(.1,.5,.9)` trains
pinball loss; predicted marginal levels undergo monotone rearrangement to remove
crossings, recorded in result metadata. No joint law is inferred from them.

## Pretrained and generative engines

```python
model = ts.load_pretrained('chronos2', cache_dir='.cache/forecasting/hub')
forecast = model.fit(y).forecast(4)
print(forecast.quantile(.1))
```

`download_checkpoint(repository, revision=..., local_files_only=...)` resolves a
Hugging Face checkpoint into an immutable cached directory and records its commit.
`load_pretrained` supports `chronos`, `chronos_bolt`, `chronos2`, `timesfm3` and
`timesfm2.5`. Pass a pinned revision for repeatable downloads or
`local_files_only=True` for offline use. No repository Python code is loaded.
CPU is the default device; set `device='cuda'` when appropriate.

Original Chronos and Bolt accept univariate histories without covariates. Original
Chronos supports `output='samples'`; Bolt exposes marginal quantiles and its median
point prediction. The adapter restricts original trajectories to the checkpoint's
native horizon. Chronos-2 supports multivariate targets and known/past covariates.
`Chronos2Forecaster(..., finetune_steps=100, finetune_horizon=4,
finetune_options={...})` trains a copied checkpoint through the provider's native
fine-tuning API. TimesFM 3 supports multivariate targets and dynamic covariates;
TimesFM 2.5 forecasts each target separately. The loader enables TimesFM 2.5's
continuous quantile head and native quantile-crossing correction by default.
Point outputs are not relabeled as means when the provider returns a median.

[Real-checkpoint verification](examples/forecasting_pretrained.py) downloads
and executes all five providers, including Chronos-2 training and multivariate
covariate cases. Reports retain exact revisions under `.cache/forecasting`.
These are integration checks, not pretrained-model quality benchmarks.

`FoundationForecaster(predictor, identity=...)` remains the user extension point:
`predictor(history, steps, future_known, rng=...) -> ForecastResult`.
`GenerativeForecaster` accepts
`sampler(history, steps, future_known, size=..., rng=...) -> (size,steps,k)` or
`trainer(history)` returning a sampler. These contracts coexist with the concrete
implemented models; no architecture is represented solely by an empty callback.

`DiffusionForecaster` trains a conditional DDPM with a cosine schedule and
velocity prediction. The reverse posterior mean reconstructs the clean path
without dividing network errors by near-zero terminal signal. Existing saved
epsilon-prediction adapters retain their original sampling interpretation.
`FlowForecaster`, `VAEForecaster`, `AutoregressiveForecaster` and `ScoreForecaster`
train conditional path generators as described in the catalog. All use causal
context/target windows and incorporate supported known, past and static inputs.
Forecasting requires an explicit NumPy generator. Returned paths define empirical
moments, quantiles and event probabilities; their log density is empirical mass,
not the underlying continuous generator's likelihood. Flows train exact continuous
likelihood internally; VAEs optimize a variational bound; score models use denoising
score matching with finite-step Langevin approximation.

All five native generators have a held-out periodic-cycle recovery test in
addition to execution and shape checks. That controlled test establishes a
limited learning capability; it does not establish calibrated uncertainty or
forecasting skill on a user's dataset. In particular, finite-budget score and
diffusion fits can retain excess variance even on deterministic data. See the
[cross-family audit](mathematical_audit.md) for measured results and limitations.

Install `convexity[neural]` for native neural/generative models,
`convexity[forecasting]` for sktime, and `convexity[forecasting-advanced]` for
NeuralForecast and real pretrained providers. Optional SDKs load only on use.

## Evaluation, ensembles, diagnostics and selection

`backtest` refits at every expanding/rolling origin. Origins are exclusive
integer training ends; `window` bounds the training prefix and `step` spaces
origins. All target-derived features, label windows, normalization and fitting
occur inside that prefix. Only declared known-future covariates are exposed
beyond it. Sparse horizons select the appropriate held-out target rows.

`compare({name: model}, data)` includes last-value, drift and moving-mean
references automatically, plus seasonal references with `seasonality>1`.
Baselines receive target history only. `BacktestResult` retains origins,
training slices, realized targets and forecast objects. `score(...,
by_horizon=True)` returns `(h,k)` scores pooled over origins. RMSE takes the
square root **after** pooling squared errors.

Metrics include MAE, RMSE, sMAPE (0–200, both-zero is zero), training-scaled MASE,
pinball loss, marginal negative log score, closed-form Gaussian/exact empirical
CRPS, interval coverage, interval width (sharpness), quantile calibration, and
an empirical multivariate/path energy score. MASE rejects zero naive-error
scales. Energy score preserves complete-path geometry and costs quadratic time
in the number of draws. Scores require the corresponding forecast capabilities;
unsupported probability outputs are never fabricated. Calibration reports
nominal levels versus observed proportions; it does not claim uncertainty
bounds for dependent/overlapping forecast origins.

`residual_diagnostics` supplies mean, RMSE, ACF and Ljung-Box values;
`conditional_errors(labels)` summarizes errors by supplied origin regimes.
Horizon-specific scores show degradation with lead time.
`covariate_sensitivity` computes local finite differences of point predictions
with respect to known-future inputs. These are numerical sensitivities, not
causal effects. Results are arrays suitable for plotting, including the notebook.

Mean/median ensembles pool point predictions. Mixtures choose a **single model
per entire path**, preserving that model's path dependence. Every chosen member
must supply a joint law. A callable `weights(history)` can implement dynamic or
regime-dependent weights using information at the cutoff. Refit updates
recompute those weights. `StackingForecaster` learns simplex weights with
SciPy SLSQP on temporal out-of-fold predictions, then refits component models.

`select_model` and `grid_search` perform finite temporal validation search.
`SelectionResult.best_by_horizon()` identifies winners by lead/target;
`refit=True` returns the winner fitted on all supplied data. These are selection
scores, not an unbiased final test estimate: reserve separate outer test data.
Coverage and sharpness alone are rejected as selection objectives. Baseline and
ensemble selection use the same public forecaster contracts.

## Artifacts, updates and extension contracts

`Forecaster.fit` returns `FittedForecaster`, which checks forecast shapes,
relative horizon alignment and future covariate availability.
`ForecastAdapter.forecast(steps, covariates, rng=...)` returns a dense
`ForecastResult(steps,k)`. No inheritance or registration is necessary.

Single-series fitted objects retain history/cutoff, schema, specification,
parameters, diagnostics, seed, backend versions, data digest and artifact identity.
`update(new_data)` returns a full refit. `mode='incremental'` appends observations
to state-space filtering with fixed parameters. Native Torch and specialist models
support `mode='warm_start'`, retaining learned weights while starting a fresh
optimizer; Torch retains its original normalization. Updates return independent
models. Panel updates require every fitted ID and preserve each series boundary.
Static values and covariate widths must match; irregular updates need timestamps.

`fitted.save(path)` / `load_forecaster(path)` retain a versioned trusted Python
pickle for compatible environments, including picklable provider objects and
custom callbacks. For portable numerical state, use:

```python
ts.save_portable(fitted, 'forecast.cvf')
restored = ts.load_portable('forecast.cvf', device='cpu')
```

The portable v1 format is a ZIP containing `manifest.json` and `arrays.npz`.
The manifest declares engine, numerical specification, shape/schema, data digest,
identities and versions. Arrays load with `allow_pickle=False`; constructors come
from an explicit registry. Loading restores numerical state without optimization.
Built-in codecs cover baselines, OU, state-space classical engines, affine sklearn
lag regression (Ridge/LinearRegression/Lasso/ElasticNet), all native Torch
architectures/heads, DDPM, flow, VAE, autoregressive and score models. Neural state
uses named weight arrays; regression preserves its estimator/refit configuration.

Unsupported engines require their native checkpoint or a trusted explicit
`register_artifact_codec` encoder/decoder. This includes external specialist and
foundation engines, custom module factories and arbitrary regressors. Unknown
format versions are rejected; v1 has no prior schema to migrate. Numerical
architecture semantics still need compatible implementations. Portability does
not promise support for arbitrary future third-party API changes.

Classical optimizer failures raise `ConvergenceError` with `.result` retained
unless `check_convergence=False` is requested. Backend fit warnings remain in
diagnostics. Fixed neural training budgets do not assert mathematical optimality.

## Verification

```bash
conda activate quant  # or another environment with the required dependencies
python -m pytest -q tests/math/forecasting
python -m pyright --pythonpath "$(command -v python)" convexity/math/forecasting tests/math/forecasting
python docs/math/examples/forecasting.py
```

The final conda `quant` run passed 980 repository tests, including 109 forecasting
tests; Pyright, examples, notebook execution and actual checkpoint integrations
also passed. Verification includes analytic AR and VAR joint covariance, irregular OU parameter
recovery, neural likelihood/decoded-law agreement, RealNVP inverse/Jacobian
identities, all dedicated architectures, shared panels, native updates, portable
round trips, leakage controls, evaluation and CPU/CUDA. The completion record
lists final suite results and real-checkpoint revisions. Run the advanced script
with `python docs/math/examples/forecasting_advanced.py`; run actual checkpoint integrations
separately with `python docs/math/examples/forecasting_pretrained.py`.

Backend references: [statsmodels time series](https://www.statsmodels.org/stable/tsa.html),
[scikit-learn regression](https://scikit-learn.org/stable/supervised_learning.html),
[PyTorch modules](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html),
[sktime forecasters](https://www.sktime.net/docs/api-reference/sktimeforecastingbasebaseforecaster/),
[Chronos](https://github.com/amazon-science/chronos-forecasting),
[TimesFM](https://github.com/google-research/timesfm),
[DDPM](https://arxiv.org/abs/2006.11239),
[improved diffusion noise schedules](https://arxiv.org/abs/2102.09672),
[NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/models.html),
[RealNVP](https://arxiv.org/abs/1605.08803),
[variational autoencoders](https://arxiv.org/abs/1312.6114),
[score matching and Langevin generation](https://arxiv.org/abs/1907.05600).
