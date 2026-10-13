# Volatility forecasting

Family 9 is implemented in `convexity.math.volatility`. It models conditional
return variance and supplied realized variance. Implied-volatility surfaces,
multivariate dynamic covariance and latent stochastic-volatility inference belong
to other roadmap families.

## Completed catalog

| Public specification | Numerical implementation | Forecasts |
|---|---|---|
| `ConstantVarianceForecaster` | `arch.univariate.ConstantVariance` | Constant conditional variance baseline |
| `EWMAForecaster(decay=.94)` | `arch.univariate.EWMAVariance` | Fixed-decay variance recursion; flat expected future variance |
| `ARCHForecaster(p=1)` | `arch` squared-innovation recursion with no variance lags | ARCH(p) |
| `GARCHForecaster(p=1, q=1)` | `arch.univariate.GARCH` | Symmetric GARCH(p,q) |
| `GJRGARCHForecaster(p=1, o=1, q=1)` | `arch` GARCH with negative-shock terms | Asymmetric GJR-GARCH(p,o,q) |
| `EGARCHForecaster(p=1, o=1, q=1)` | `arch.univariate.EGARCH` | Log-variance dynamics with magnitude and signed-shock terms |
| `HARForecaster(lags=(1,5,22), transform='level')` | statsmodels OLS | Recursive heterogeneous autoregression of realized variance |
| `HARForecaster(lags=(1,5,22), transform='log')` | statsmodels OLS and Gaussian log-innovation propagation | Correlated lognormal law, exact marginal intervals, and joint paths |

All six ARCH-family specifications support normal, standardized Student-t,
standardized skew Student-t and generalized-error innovations. They support
zero, constant and autoregressive conditional means. Known-future regressors
enter constant or AR means. `ARCHModelForecaster` is the common explicit
specification; the named factories select its model and orders.

## Inputs, contracts and units

```python
import numpy as np
from convexity.math import volatility as vol

returns = np.random.default_rng(4).normal(size=1000)
fitted = vol.GARCHForecaster(innovation='student_t').fit(returns)
forecast = fitted.forecast([1, 5, 20])
print(forecast.conditional_variance)
print(forecast.volatility())
print(fitted.forecast(1).interval(.95))
```

Inputs use the existing `TimeSeries`, `Covariates`, `ForecastHorizon`, `Forecaster`
and `ForecastAdapter` contracts. Arrays are `(n,)` or `(n,1)`; these are univariate
models. Multitarget arrays, NaNs and infinities are rejected. No imputation,
sorting, return construction or resampling occurs. Supply returns to ARCH-family
models and **realized variance**, rather than its square root, to HAR models.
For observed realized volatility `s`, supply `s**2`.

An integer horizon means leads `1..H`; an increasing vector selects sparse
leads. All intermediate recursion steps are computed before selection. Forecast
arrays are `(horizon,1)`; path arrays are `(draw,horizon,1)`. Irregular timestamps
require explicit future labels, following family 8. These models evolve per row;
timestamp labels do not change their recursions into elapsed-time dynamics.
HAR windows `(1,5,22)` mean observations, with no implicit trading calendar.

`Covariates(known_future=x)` supplies `(n,features)` training inputs. Forecast
covariates must cover **every** lead through the largest requested horizon.
Observed-past and static covariates are rejected. ARCH `mean='zero'` also rejects
known-future regressors; use `mean='constant'` or `mean='ar', lags=p` instead.
Regressors enter the return mean in ARCH models and the chosen transformed
realized-variance regression in HAR. They do not directly enter the ARCH variance
equation. Future covariate uncertainty is not integrated out.

Returns retain the user's scale: decimal returns produce decimal-return-squared
variances; percentage returns produce percentage-point-squared variances.
`rescale=True` permits `arch` to rescale internally for optimization. Means,
residuals, forecasts and conditional volatility are converted back to input
units. `parameters['coefficients']` and optimizer options such as starting values
use backend scaled units; `parameters['scale']` records the factor.
`parameters['loglikelihood']` includes the Jacobian correction to input units.
No forecast is annualized implicitly. `forecast.volatility(periods_per_year=252)`
computes `sqrt(252 * conditional_variance)`; this is a unit convention, not a
forecast of the variance of an annual cumulative return.

## Return uncertainty versus variance uncertainty

`FittedVolatilityForecaster` subclasses the existing `FittedForecaster`.
`VolatilityForecastResult` subclasses `ForecastResult`, preserving generic
forecasting, evaluation, horizon selection and user adapters.

For ARCH-family results:

- `point` is the conditional **return mean**.
- `conditional_variance` is the forecast innovation variance `E_t[h_(t+j)]`,
  computed analytically or averaged over simulated variance paths.
- `variance()` is the return forecast-error variance. AR mean dynamics make it
  larger than innovation variance beyond the first lead. For path forecasts it
  is the empirical variance of the return draws.
- `volatility()` is `sqrt(E_t[h])`, not `E_t[sqrt(h)]`.
- `paths` contains simulated returns; `conditional_variance_paths` contains the
  matching variance states in the same draw order. Squared returns are not
  interchangeable with these states.
- `variance_forecast()` returns a regular `ForecastResult` whose point target is
  conditional variance. Its intervals describe uncertainty about future
  variance states, when paths exist. Its `variance()` therefore means variance
  **of variance**, not return variance.

For HAR, `target_kind='realized_variance'`: the forecast target and
`conditional_variance` are realized variance itself. Its `variance()` describes
uncertainty about realized variance, and `variance_forecast()` retains its
available uncertainty outputs. For all models, `target_kind` makes this target
distinction inspectable. Result arrays are copied and read-only.

## Forecast methods and innovation distributions

`method='auto'` uses analytic forecasts when supported and simulation otherwise.
An analytic **one-step** ARCH-family forecast returns the fitted standardized
innovation law shifted by the return mean and scaled by conditional volatility.
It supports density, CDF, quantiles, intervals and sampling through Convexity's
`Distribution` contract.

Analytic multistep ARCH/GARCH and symmetric-innovation GJR forecasts expose means
and variances. They do not assert a joint Gaussian law: future stochastic
variances induce mixtures even with normal innovations. For GARCH(1,1),

```text
h[t+1] = omega + alpha * epsilon[t]**2 + beta * h[t]
E_t[h[t+j]] = omega + (alpha + beta) * E_t[h[t+j-1]], j > 1
```

Symmetric GJR adds `gamma * epsilon[t]**2 * I(epsilon[t]<0)` at the first step
and replaces persistence by `alpha + gamma/2 + beta` at later steps.

EGARCH needs simulation beyond one lead. The `arch` analytic GJR recursion uses
a half second moment for negative innovations, which is unsuitable for skewed
innovations. Convexity also selects simulation for asymmetric GARCH with
`innovation='skew_student_t'` beyond one lead. Explicit `method='analytic'`
raises for these cases instead of returning the inappropriate approximation.

`method='simulation'` draws from the fitted innovation distribution;
`method='bootstrap'` resamples in-sample standardized residuals using `arch`'s
bootstrap implementation and its minimum-sample/horizon requirements.
Both require an explicit `numpy.random.Generator`; `simulations=5000` controls
the path count. `auto` also requires a generator when it selects simulation.
Equal fresh seeds reproduce equal draws, without mutating NumPy's global RNG.

```python
model = vol.EGARCHForecaster(simulations=10000).fit(returns)
out = model.forecast(10, rng=np.random.default_rng(7))
return_intervals = out.interval(.9)
variance_intervals = out.variance_forecast().interval(.9)
```

Path outputs define an empirical joint law. Quantiles and event probabilities
retain complete-path dependence; log density is empirical mass, not the
continuous underlying likelihood. `point` retains the model mean, while
`mean()` and `variance()` on a path result are empirical moments. Forecasts
condition on fitted parameters. Finite Monte Carlo averages do not establish
existence of population moments; in particular EGARCH with heavy-tailed
Student-t innovations can lack finite multistep variance expectations.

`innovation_distribution(name, parameters)` exposes the same standardized
laws independently of forecasting:

| Name | Parameters in order | Mean / variance |
|---|---|---|
| `normal` | `()` | 0 / 1 |
| `student_t` | `(nu,)` | 0 / 1 |
| `skew_student_t` | `(eta, lambda)` | 0 / 1 |
| `ged` | `(nu,)` | 0 / 1 |

Shape bounds are checked against `arch` constraints; skewness must be strictly
inside `(-1,1)` to exclude singular endpoint formulas. A standardized Student-t
uses the ordinary Student-t scale `sqrt((nu-2)/nu)`. The skew parameter changes
shape without changing zero mean or unit variance.

## Realized measures and HAR

`realized_variance(intraperiod_returns, axis=-1)` sums squared returns within
each supplied period. For `(period,intraperiod)` input it returns `(period,)`.
`realized_volatility` takes its square root. Session boundaries and sampling
frequency are supplied by the caller. These helpers neither demean nor correct
microstructure noise or omitted overnight returns.

HAR fits an intercept and causal trailing window means; the response at row `t`
uses only rows before `t` for these features. Optional known covariates use row
`t`. All future window means update recursively. Regression design must have
full column rank and positive residual degrees of freedom.

Level HAR uses means of **variance** and returns point estimates. Negative
predictions raise an error; no silent clipping or Gaussian law with negative
variance support is introduced. Log HAR uses means of **log variance**, which
differs from the log of average variance. Input variance must be strictly
positive. Gaussian homoskedastic log innovations are an explicit model
assumption. For log forecast mean `m` and variance `s2`, its original-unit mean is
`exp(m+s2/2)`, not `exp(m)`. Multistep log covariance is propagated through the
full HAR recursion before transforming back.

Analytic log HAR supplies the correlated lognormal path law, including its mean,
variance, cross-horizon covariance, exact marginal quantiles and intervals,
joint samples, and log density. Its median is `exp(m)`; its mean includes the
log-variance correction above. For log covariance `C`, original-unit covariance
is `outer(mean, mean) * expm1(C)`. Scalar marginal laws provide exact quantiles
without replacing this joint dependence with an independent product. A zero
innovation variance produces a deterministic discrete law.
`method='simulation'` samples the same joint Gaussian log dynamics and
exponentiates complete paths, yielding an empirical approximation to that law.
Simulation is unavailable for level HAR.
Parameter covariance uses statsmodels `covariance='HC0'` by default (also
`nonrobust`, `HC1`, `HC2`, `HC3`); this coefficient covariance is not added to
predictive uncertainty. Innovation variance is residual SSE divided by residual
degrees of freedom. HAR fit residuals are in the chosen transform's units.

## Evaluation, diagnostics, updates and extension

Generic `forecasting.backtest` evaluates each model's original target: returns
for ARCH models, realized variance for HAR. `variance_backtest` reuses the same
causal refits, horizons, origins, rolling windows and future-covariate rules,
but scores `variance_forecast()` outputs. For return models supply an explicit
`variance_proxy` aligned with every input row. A `TimeSeries` proxy must have
identical timestamps. Proxies enter evaluation only. HAR uses its observed
realized-variance targets by default.

```python
scores = vol.variance_backtest(
    vol.GARCHForecaster(mean='zero'), returns,
    variance_proxy=returns**2,  # explicit noisy proxy under a zero mean
    horizon=[1, 5], initial_window=700, step=50,
)
print(scores.score('qlike', by_horizon=True))
```

`qlike(actual, predicted)` is `log(predicted) + actual/predicted`, omitting terms
independent of the forecast. Actual variance may be zero; forecasts must be
positive. Scores need not be nonnegative; lower is better. MAE and RMSE are also
available. Distribution scores require actual distributional outputs. MASE is
rejected on variance backtests because their retained model-input histories can
be in return units rather than variance units.

`volatility_diagnostics(z, lags=10, ddof=0)` gives Ljung-Box tests on supplied
standardized residuals and their squares, plus Engle's ARCH LM statistic and
p-value. Remove only known initialization rows before passing AR residuals;
interior missing observations are rejected. These are diagnostic tests, not
guarantees that volatility is correctly specified.

Fitted models retain history, cutoff, schema, specification, parameter values,
dependency versions, training digest and artifact identity. `update(new_data)`
returns a complete independent refit; incremental and warm-start modes are not
implemented for these engines. Optimizer failure raises the shared
`ConvergenceError` with `.result` retained, unless `check_convergence=False`.
Backend fit warnings remain in diagnostics.

`fitted.save(path)` and `forecasting.load_forecaster(path)` round-trip trusted
Python artifacts in compatible environments. The numerical portable-artifact
registry remains available for custom codecs; these engines currently use
trusted Python artifacts, not a new portable schema.

User models implement the existing `Forecaster.fit` and
`ForecastAdapter.forecast` protocols. An adapter returns a dense
`VolatilityForecastResult` with `conditional_variance` and optional aligned
variance paths. Wrap it in `FittedVolatilityForecaster`, or the existing
`FittedForecaster`, to obtain horizon validation, metadata and sparse selection.
No registration or changes to Convexity internals are needed.

## Examples and verification

- [Return models, innovation laws, simulation, bootstrap and evaluation](examples/volatility.py).
- [Generated intraperiod returns, level/log HAR and known covariates](examples/volatility_har.py).
- [Worked notebook](notebooks/volatility.ipynb).

```bash
conda activate quant
python -m pytest -q tests/math/volatility
python -m pytest -q
python -m pyright --pythonpath "$(command -v python)"
python docs/math/examples/volatility.py
python docs/math/examples/volatility_har.py
```

Tests cover parameter recovery, GARCH closed forms, every named backend,
pathwise ARCH/GJR/EGARCH/EWMA recursions, analytic versus Monte Carlo moments,
innovation normalization and unit variance, scaling, AR error propagation,
HAR regression and Jensen corrections, causal proxy evaluation, explicit
covariates, missing/invalid data, convergence, sparse horizons, extensibility,
artifact restoration and lazy imports. `arch>=8.0` is a core dependency but
loads only when an ARCH model or innovation law is used.

Verified on 2026-09-29 in conda `quant` with `arch` 8.0.0:

- **1,043 repository tests passed**, including **63 volatility tests** and all
  24 combinations of the six return models and four innovation laws.
- Repository-wide Pyright and final changed-file type checks passed without
  errors or warnings.
- Both volatility example scripts and the original notebook executed. The
  expanded [inspection notebook](notebooks/volatility.ipynb) now retains its
  executed plots and tables; see the [notebook guide](notebooks/README.md).
- Editable package installation and `pip check` passed; documentation links and
  whitespace checks passed. Remaining test warnings are backend notices and
  deprecations, not test failures.

Backend references: [arch forecasting](https://bashtage.github.io/arch/univariate/forecasting.html),
[volatility processes](https://bashtage.github.io/arch/univariate/volatility.html),
[innovation distributions](https://bashtage.github.io/arch/univariate/distribution.html),
[statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html).
