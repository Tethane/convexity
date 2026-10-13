# Simulation and joint path distributions

Family 15 is implemented in `convexity.math.simulation`. It composes the existing
distribution, process, rough-process and forecasting contracts with explicit
joint scenarios, random drivers and integration-error estimates. NumPy/SciPy
supply the core calculations. Existing QuantLib and sdeint integrations remain
optional through `convexity[processes]`.

## Completed bounded catalog

| Public API | Capability |
|---|---|
| `PathDistribution(times,state_shape,law)` | Path-shaped facade over an actual flattened joint `Distribution`; sampling, moments, joint covariance and supported finite-law operations |
| `PathScenarios(times,values,weights)` | Finite weighted joint paths, exact empirical expectations/probabilities/pushforwards, reweighting, entire-row resampling and weighted forecast conversion |
| `gaussian_innovations` | Correlated Gaussian innovations, including singular positive-semidefinite covariance |
| `BrownianIncrements`, `brownian_increments` | Irregular-grid independent Wiener coordinates; IID MC, antithetic MC, scrambled Sobol and Halton; coupled grid coarsening |
| `simulate` | Sequential process-kernel composition: exact Brownian/OU/GBM/CIR/jump laws, existing Euler/SRI2 and QuantLib Heston QE |
| `integrate` | Euler for scalar/vector Itô coefficients; Milstein for a scalar state and one Wiener driver with supplied analytic diffusion derivative |
| `geometric_brownian_paths` | Exact grid simulation of correlated multi-asset GBM with rectangular loading matrices |
| `gaussian_path_distribution` | Joint Gaussian law of supplied affine transitions, including initial uncertainty and singular deterministic initial states |
| `PathSampler`, `sample_paths` | Structural user sampler contract and checked dispatch |
| `estimate`, `estimate_scenarios`, `MonteCarloEstimate` | Means and appropriate IID/pair standard errors; explicit absence of error estimates for unknown dependence or unequal weights |
| `control_variates` | Fixed-coefficient controls with supplied known expectations |
| `qmc_expectation` | Standard-normal integration with independent randomized nets and errors across scramble means |
| `simulate_observations`, `SimulatedObservations` | Conditional observation laws on latent scenarios; retained joint state/observation dependence |

Acceptance covers this catalog, extension callbacks, analytic cross-time and
cross-asset covariance identities, strong convergence on coupled drivers,
variance reduction, reproducibility, validation, optional backend execution,
and the [executable generated-data example](examples/simulation.py).
General vector Milstein/Levy areas, adaptive SDE solvers, Brownian bridges,
multilevel MC, automatic importance proposals, pricing/measure changes and
inference from observations are outside this catalog.

## Shapes, time, ownership and meaning

- Times are finite, strictly increasing numbers in one caller-selected unit;
  nonzero origins and grids containing only the initial time are supported.
- Scenario values are `(draw,time)` for scalar states or `(draw,time,state)` for
  vector states. The initial state is included. All assets and times in a row
  belong to **one joint draw**.
- `PathDistribution.event_shape` is `(time,*state_shape)`. Its underlying
  `Distribution` uses flattened time-major `(time*state_dimension,)` coordinates.
  `covariance()` returns a matrix in that order; `marginal(index)` retains a vector
  event, including a one-coordinate vector for scalar states.
- Brownian values are `(draw,time-1,noise_dimension)` with covariance `dt*I`.
  Correlation belongs in model loadings. `gaussian_innovations` separately accepts
  a covariance matrix; passing correlated noise to a model that already includes
  a correlation loading would change its dynamics.
- Numeric inputs are copied into read-only arrays. Random operations require an
  explicit `numpy.random.Generator`; callbacks must not mutate inputs and must
  retain stable parameters. No global seeds, calendar conventions or probability
  measures are inferred.

`PathScenarios` is always an **empirical finite law**, even when `exact=True`
records an exact source sampler. It does not gain a continuous density by having
many rows. `PathDistribution` preserves its supplied law's capabilities; analytic
Gaussian callbacks do not acquire arbitrary exact expectations or predicate
probabilities. Singular Gaussian densities use the underlying law's subspace
convention. Separate marginal intervals cannot construct a joint path law.

Weights are finite nonnegative relative masses with positive total mass. They
are normalized safely; zero-mass rows do not contribute to empirical support or
expectation callbacks. `reweight` multiplies supplied relative likelihoods and
clears the sampling-design claim. It does not establish a financial measure
change. `effective_size` is `1/sum(w**2)`, a weight-concentration diagnostic, not
an independent sample count. `resample` samples entire paths from the empirical
law; its IID claim is conditional on that finite law. It cannot quantify the
original model approximation error.

`to_forecast()` retains weighted joint dependence through `ForecastResult.law`.
Every supplied time becomes a forecast row, including the initial time. Slice
the scenarios before constructing them if only future rows are desired; pass an
explicit `ForecastHorizon` for forecasting time semantics. Rough paths can be
wrapped via `PathScenarios.from_process_paths(result.paths())` where that result
exposes `paths()`, or by supplying the rough model's `ProcessPaths` directly.

## Schemes and extension points

`simulate` accepts the family-13 structural `StochasticProcess.transition`
contract, so custom exact kernels and model-specific schemes require no registry.
Its `method` and `substeps` are passed through; a Heston simulation must explicitly
request `method='qe'`. Its provenance records actual kernel backend/exactness.
`sample_paths` instead calls a user object's `sample(initial,times,draws=...,rng=...)`
and requires `PathScenarios` aligned with the requested grid and draw count.

`integrate` accepts the family-13 `ItoProcess` drift/diffusion interface and a
`BrownianIncrements` object. Euler uses `x+b*dt+B*dW`. Scalar Milstein adds
`0.5*sigma*d_sigma_dx*(dW**2-dt)`, with
`diffusion_derivative(time,state)` explicitly supplied. Coefficients are evaluated
at the left endpoint. No automatic clipping or positivity fix is applied; use
exact CIR or a model-specific scheme where Euler does not respect the domain.
Both generic schemes are labeled approximate even in special cases where Euler
is exact. Generate a fine grid and use `increments.coarsen(indices)` to preserve
the same Brownian realization in convergence and common-random-number comparisons.

`gaussian_path_distribution` takes `coefficients(time,dt) -> (F,b,Q)` for affine
Gaussian transitions with independent step innovations. For vector states these
have shapes `(d,d),(d,),(d,d)`; scalar states use scalars. Existing vector OU and
linear diffusions provide `model.coefficients(dt)`. The supplied callback determines
whether these are exact continuous-process transitions or a discretized model.

`simulate_observations` calls `observation(time,state)` for a `Distribution` or
`Transition` with a fixed scalar/vector event shape, then samples independently
conditional on the latent path at every time, including the first. Observation
serial dependence requires an augmented latent process. Random observation noise
does not preserve a QMC net, so only an existing IID row label is propagated.

## Randomized integration and variance reduction

Sobol draw counts must be powers of two; no padding, thinning or skipping is
performed. Halton permits any positive count. Each complete Brownian path uses
one point with `intervals*noise_dimension` coordinates. All nets are scrambled;
a fresh seed is consumed from the supplied generator per call, allowing saved
generator-state replay. Inverse-normal endpoints are clipped at floating-point
precision. These choices follow [SciPy's QMC guidance](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html)
and [Sobol balance requirements](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).

Antithetic MC requires an even draw count and stores all base rows followed by
their negatives. It is intentionally separate from the scrambled-net options.
`estimate(...,sampling='antithetic')` computes standard errors from pair averages,
not individual dependent rows. `estimate` assumes the caller's sampling label is
correct and that the integrand has finite variance. With fewer than two independent
units, unknown dependence, a single QMC net or unequal scenario weights, the
standard error is `None`. It is never inferred from path dispersion alone.

`qmc_expectation` evaluates a vectorized `function(z)` on `(draw,dimension)`
standard normals and expects `(draw,...)` outputs. It reports standard errors
across at least two independently scrambled net means. A small reported error
does not measure SDE discretization bias. `control_variates` uses known control
expectations and coefficients fixed beforehand or fitted on an independent pilot;
same-sample fitted controls are not automatically treated as unbiased estimates.

```bash
python docs/math/examples/simulation.py
python -m pytest -q tests/math/simulation
```
