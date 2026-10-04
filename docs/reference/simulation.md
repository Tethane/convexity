# `convexity.math.simulation`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Simulation and joint path distributions (family 15).

## `PathDistribution`

```python
PathDistribution(times: 'ArrayLike', state_shape: 'tuple[int, ...]', law: 'Distribution') -> 'None'
```

A Distribution on time-major flattened coordinates, with explicit times.

One event is (time,*state_shape). Density is with respect to the underlying
law's measure; singular Gaussian laws retain their supported-subspace density.
This wrapper never manufactures temporal dependence from marginal intervals.

### `PathDistribution.covariance`

```python
covariance(self)
```

Time-major (time*dimension,time*dimension) joint covariance.

### `PathDistribution.expect`

```python
expect(self, function)
```

### `PathDistribution.log_density`

```python
log_density(self, paths: 'ArrayLike')
```

### `PathDistribution.map`

```python
map(self, function)
```

### `PathDistribution.marginal`

```python
marginal(self, index: 'int')
```

Vector law at one grid index (one coordinate for scalar states).

### `PathDistribution.mean`

```python
mean(self)
```

### `PathDistribution.probability`

```python
probability(self, event)
```

### `PathDistribution.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator') -> 'FloatArray'
```

### `PathDistribution.variance`

```python
variance(self)
```

## `PathScenarios`

```python
PathScenarios(times: 'ArrayLike', values: 'ArrayLike', weights: 'ArrayLike | None' = None, sampling: 'str' = 'unspecified', method: 'str' = 'supplied', backend: 'str' = 'numpy', exact: 'bool' = False) -> 'None'
```

Finite joint law: values (scenario,time[,state]), normalized masses.

`exact` describes the source sampler, never claims that this finite law is
the continuous model law. `sampling` describes row dependence before any
reweighting. No Monte Carlo error is inferred from weights or effective size.

### `PathScenarios.covariance`

```python
covariance(self)
```

### `PathScenarios.distribution`

```python
distribution(self) -> 'PathDistribution'
```

### `PathScenarios.expect`

```python
expect(self, function)
```

### `PathScenarios.from_process_paths`

```python
from_process_paths(paths, *, weights: 'ArrayLike | None' = None, sampling: 'str' = 'unspecified')
```

### `PathScenarios.map`

```python
map(self, function)
```

### `PathScenarios.mean`

```python
mean(self)
```

### `PathScenarios.probability`

```python
probability(self, event)
```

### `PathScenarios.resample`

```python
resample(self, draws: 'int', *, rng: 'np.random.Generator') -> 'PathScenarios'
```

IID resampling of entire rows preserves within-path dependence.

### `PathScenarios.reweight`

```python
reweight(self, likelihood: 'ArrayLike') -> 'PathScenarios'
```

Multiply masses by supplied finite nonnegative relative likelihoods.

### `PathScenarios.to_forecast`

```python
to_forecast(self, *, horizon=None, model_identity: 'str' = 'simulation')
```

All supplied times become forecast rows; caller chooses whether to omit t0.

## `BrownianIncrements`

```python
BrownianIncrements(times: 'ArrayLike', values: 'ArrayLike', sampling: 'str' = 'unspecified') -> 'None'
```

Independent Wiener coordinates, values (draw,interval,noise_dimension).

Covariance within each row is dt*I. Correlation belongs in model loadings.
Supplied arrays are trusted as driver realizations, not statistically tested.
Antithetic order is all base rows followed by their negatives.

### `BrownianIncrements.coarsen`

```python
coarsen(self, indices) -> 'BrownianIncrements'
```

Sum adjacent increments; retain both endpoints and all driver coupling.

## `PathSampler`

Structural extension: return a checked PathScenarios from sample().

### `PathSampler.sample`

```python
sample(self, initial: 'ArrayLike', times: 'ArrayLike', *, draws: 'int', rng: 'np.random.Generator') -> 'PathScenarios'
```

## `MonteCarloEstimate`

```python
MonteCarloEstimate(value: 'ArrayLike', standard_error: 'ArrayLike | None', independent_units: 'int', method: 'str') -> 'None'
```

Monte Carlo estimate with sampling uncertainty and retained simulation diagnostics.

## `gaussian_innovations`

```python
gaussian_innovations(draws: 'int', covariance: 'ArrayLike', *, rng: 'np.random.Generator', method: 'str' = 'mc', antithetic: 'bool' = False) -> 'FloatArray'
```

N(0,covariance) rows; singular PSD covariance is supported.

Sobol uses one full-dimensional point per row. Antithetic MC returns base
rows then their negatives. A covariance, not a correlation, is supplied.

## `brownian_increments`

```python
brownian_increments(times: 'ArrayLike', *, draws: 'int' = 1, dimension: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'mc', antithetic: 'bool' = False) -> 'BrownianIncrements'
```

Independent-coordinate Wiener increments on an irregular time grid.

For QMC each complete path is one point of dimension intervals*dimension,
with time-major coordinates. Time has the same units as model coefficients.

## `simulate`

```python
simulate(process: 'StochasticProcess', initial: 'ArrayLike', times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'exact', substeps: 'int' = 1) -> 'PathScenarios'
```

Compose family-13 transition kernels, including CIR, jumps, QE and SRI2.

Exactness/backend labels come from the kernels. This sampler is IID MC;
use explicit Brownian increments for QMC, coupling or antithetic schemes.

## `sample_paths`

```python
sample_paths(sampler: 'PathSampler', initial: 'ArrayLike', times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator') -> 'PathScenarios'
```

Checked dispatch for a user PathSampler; no registry or inheritance.

## `integrate`

```python
integrate(process: 'ItoProcess', initial: 'ArrayLike', increments: 'BrownianIncrements', *, method: 'str' = 'euler', diffusion_derivative: 'Callable[[float, FloatArray], ArrayLike] | None' = None) -> 'PathScenarios'
```

Ito Euler or scalar one-driver Milstein on supplied increments.

Milstein requires the analytic scalar derivative d sigma(t,x)/dx. Vector
Milstein, Levy areas, state clipping and automatic domain repair are excluded.
Refinement is explicit: generate a fine driver and coarsen it for comparisons.

## `geometric_brownian_paths`

```python
geometric_brownian_paths(initial: 'ArrayLike', drift: 'ArrayLike', loading: 'ArrayLike', increments: 'BrownianIncrements') -> 'PathScenarios'
```

Exact vector GBM on a grid, with correlated log returns via loading B.

dS_i/S_i = drift_i dt + sum_j B_ij dW_j. Initial (d,), drift (d,),
loading (d,m); zero initial levels stay zero. No probability measure inferred.

## `gaussian_path_distribution`

```python
gaussian_path_distribution(initial: 'ArrayLike', times: 'ArrayLike', coefficients: 'Callable[[float, float], tuple[ArrayLike, ArrayLike, ArrayLike]]', *, initial_covariance: 'ArrayLike | None' = None) -> 'PathDistribution'
```

Law of X[j+1]=F X[j]+b+epsilon, epsilon~N(0,Q), independent by step.

coefficients(time,dt) returns (F,b,Q). Scalar initial uses scalar F,b,Q;
vector initial uses (d,d),(d,),(d,d). `initial` is the initial mean;
omitted initial_covariance means a deterministic initial condition.
The callback must describe exact transitions to claim an exact process law.

## `estimate`

```python
estimate(values: 'ArrayLike', *, sampling: 'str' = 'iid') -> 'MonteCarloEstimate'
```

Average (draw,...) values; errors require independent units, finite variance.

Antithetic halves are paired before estimating error. A single QMC net or
unspecified dependence supplies a mean and no standard error. With one
independent unit the standard error is also unavailable (None), never zero.

## `estimate_scenarios`

```python
estimate_scenarios(scenarios: 'PathScenarios', function: 'Callable[[FloatArray], ArrayLike]') -> 'MonteCarloEstimate'
```

Evaluate one path at a time; unequal weights have no automatic MC error.

## `control_variates`

```python
control_variates(values: 'ArrayLike', controls: 'ArrayLike', expected: 'ArrayLike', coefficients: 'ArrayLike') -> 'FloatArray'
```

Y - (C-E[C]) beta for a scalar response and (draw,control) controls.

Coefficients must be fixed in advance or fitted on an independent pilot.
This function does not fit on the estimation sample or assume known means.
Evaluate the adjusted rows with estimate(), retaining the sampling design.

## `qmc_expectation`

```python
qmc_expectation(function: 'Callable[[FloatArray], ArrayLike]', dimension: 'int', *, draws: 'int', replicates: 'int', rng: 'np.random.Generator', method: 'str' = 'sobol') -> 'MonteCarloEstimate'
```

Integrate a vectorized function of standard-normal rows using fresh nets.

function receives (draw,dimension), returns (draw,...). Standard error is
computed across independent scramble means, never across points in a net.
Transform the normal rows inside the callback for another target law.

## `SimulatedObservations`

```python
SimulatedObservations(latent: 'PathScenarios', observations: 'PathScenarios') -> None
```

Observed values generated from a latent simulation, preserving observation-time alignment.

### `SimulatedObservations.joint`

```python
joint(self) -> 'PathScenarios'
```

One vector state per time, latent coordinates then observations.

## `simulate_observations`

```python
simulate_observations(latent: 'PathScenarios', observation: 'Callable[[float, FloatArray], Distribution | Transition]', *, rng: 'np.random.Generator') -> 'SimulatedObservations'
```

observation(time,state) returns Distribution or Transition.

Draws are conditionally independent across grid times and rows given the
entire latent path. Correlated observation noise must instead be included
in an augmented latent process. The first grid point is observed as well.
