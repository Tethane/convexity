# Forecasting completion checklist

This is the expanded family-8 acceptance list requested after the first release.
All items below are implemented and verified. Family 8 is complete for this list;
family 9 (volatility forecasting) is now also [implemented](volatility_forecasting.md).

The [2026-09-30 audit](mathematical_audit.md) supersedes any interpretation of
this checklist as a predictive-quality guarantee. It found a diffusion recovery
failure missed by the original tests, corrected it, and added accuracy checks
for the five native generators. Dedicated and pretrained architectures still
have primarily integration evidence, not broad accuracy/calibration benchmarks.

- [x] Dedicated TFT, PatchTST, N-BEATS, N-HiTS and encoder-decoder models.
- [x] Conditional flow, VAE, score-based and autoregressive path models.
- [x] Student-t, lognormal, mixture and categorical/discretized neural heads.
- [x] Real-checkpoint Chronos, Chronos-Bolt, Chronos-2 and TimesFM integration tests.
- [x] Checkpoint download/cache management and foundation fine-tuning.
- [x] Joint state-space path laws and smoothing/ETS predictive paths.
- [x] Explicit parameter-uncertainty forecasts.
- [x] Incremental statistical updates and neural warm-start updates.
- [x] Versioned portable numerical artifacts and round-trip verification.
- [x] Elapsed-time forecasting and calendar-aware horizon construction.
- [x] Panel data and shared/global forecasting across separate series.
- [x] Updated catalog, examples, mathematical tests, full regression suite and type checks.

Acceptance means executable, tested implementations of the named capabilities,
not aliases to callbacks. Arbitrary third-party code and unknown future library
versions cannot be guaranteed portable; the documented numerical artifact schema
and supported migrations must make the supported boundary explicit.

## Original verification (2026-09-29)

- **980 repository tests passed**, including **109 forecasting tests** and CUDA.
- Repository Pyright validation passed; final changed-file checks passed.
- Basic and advanced scripts and the forecasting notebook executed successfully.
- All five real pretrained checkpoint integrations passed, including Chronos-2 fine-tuning.
- Conda `quant`: all 170 installed package requirements are compatible.
- Optional forecasting engines remain unloaded until used.

Verified on 2026-09-29 using Python 3.14.7 in conda `quant`. Warnings are
backend notices (optimizer diagnostics, Lightning data-loader advice and
third-party deprecations); there were no test failures.

## Implementations and checks

| Checklist area | Public implementation | Verification |
|---|---|---|
| Dedicated architectures | TFT, PatchTST, N-BEATS, N-HiTS, encoder-decoder via NeuralForecast | Every named architecture trains and forecasts with actual backend modules |
| Generative forecasting | Conditional RealNVP, sequence VAE, Gaussian autoregressive model, denoising score model | Seeded path generation, flow inverse/log-Jacobian identities, portable weight restoration |
| Probability heads | Student-t, lognormal, Gaussian mixtures, categorical/nearest-support discretization | Likelihood agrees with returned scalar law; support, quantiles, sampling and warm starts checked |
| Foundation checkpoints | Managed immutable cache, Chronos/Bolt/Chronos-2, TimesFM 3/2.5 | Actual pretrained inference, multivariate covariates, original Chronos paths, one-step Chronos-2 fine-tuning |
| Classical uncertainty | Joint state-space covariance, ETS/smoothing simulation | Analytic AR/VAR covariance and reproducible innovation paths |
| Parameter uncertainty | Parametric-bootstrap histories, refits, actual-history re-filtering | Draws preserve AR dynamics and condition on the actual observations |
| Updates | State-space incremental filtering; native and specialist neural warm starts | Parameters/weights retained appropriately; original fitted models unchanged |
| Artifacts | Versioned JSON + NumPy numerical state, explicit codecs, trusted panel artifacts | Prediction/variance/identity round trips; version rejection; no RNG mutation during neural restoration |
| Time and panels | Exact irregular-time OU, business/month-end horizons, shared panel models | OU parameter recovery, holidays, explicit irregular labels, separate series windows and cutoffs |

Core examples: [basic](examples/forecasting.py),
[advanced](examples/forecasting_advanced.py),
[worked notebook](notebooks/forecasting.ipynb).
The [guide](time_series_forecasting.md) documents array contracts, supported
covariates, training assumptions, uncertainty meaning and artifact boundaries.

## Actual checkpoint verification

Executed with `docs/math/examples/forecasting_pretrained.py` in conda `quant`. The cached
report is `.cache/forecasting/pretrained-verification.json`. Downloads and
fine-tuned weights remain outside source control. The exercised revisions are:

| Provider | Repository | Revision |
|---|---|---|
| chronos | `amazon/chronos-t5-tiny` | `29d808298f1a62493e7b9a5e08529d0d930fa189` |
| chronos_bolt | `amazon/chronos-bolt-tiny` | `a0e552de83495b5c28c14c71c374f3e33280b340` |
| chronos2 | `autogluon/chronos-2-small` | `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` |
| timesfm3 | `google/timesfm-3.0-pytorch` | `43046b85ec22d584a13f8098c2ed39c889e129c2` |
| timesfm2.5 | `google/timesfm-2.5-200m-pytorch` | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` |

These runs validate integration and training execution. They do not establish
forecasting quality or calibration for user data. NeuralForecast 3.2.2,
chronos-forecasting 2.3.2, TimesFM 3.0.2 and PyTorch 2.14.0+cu130 were used.

## Reproducible verification commands

```bash
conda activate quant
python -m pytest -q
python -m pyright --pythonpath "$(command -v python)"
python docs/math/examples/forecasting.py
python docs/math/examples/forecasting_advanced.py
python docs/math/examples/forecasting_pretrained.py
```

Portable v1 numerical state covers the documented built-in engines. External
engines use their native checkpoints, compatible trusted artifacts, or registered
codecs. Unknown schema versions are rejected rather than silently migrated.
General Bayesian posterior inference remains family 11; the family-8 parameter
uncertainty implementation is an explicit parametric bootstrap.
