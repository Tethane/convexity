# `convexity.math.state_space`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

State-space models, filtering and smoothing (family 10).

TimeSeries inputs and Distribution outputs compose with existing families.
statsmodels, FilterPy, particles and SciPy are loaded only when needed.

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

## `DistributionObservation`

```python
DistributionObservation(state_dimension: 'int', dimension: 'int', law: 'Callable[[FloatArray, TimeStep], Distribution]') -> None
```

Observation law supplied as a state/time-dependent Distribution callback.

### `DistributionObservation.distribution`

```python
distribution(self, state: 'ArrayLike', step: 'TimeStep')
```

### `DistributionObservation.log_density`

```python
log_density(self, value, states: 'ArrayLike', step: 'TimeStep')
```

### `DistributionObservation.sample`

```python
sample(self, states: 'ArrayLike', step: 'TimeStep', *, rng: 'np.random.Generator')
```

## `DistributionTransition`

```python
DistributionTransition(dimension: 'int', law: 'Callable[[FloatArray, TimeStep], Distribution]') -> None
```

Any conditional Convexity vector law; scalar states still use event (1,).

### `DistributionTransition.distribution`

```python
distribution(self, state: 'ArrayLike', step: 'TimeStep')
```

### `DistributionTransition.log_density`

```python
log_density(self, values: 'ArrayLike', states: 'ArrayLike', step: 'TimeStep')
```

### `DistributionTransition.sample`

```python
sample(self, states: 'ArrayLike', step: 'TimeStep', *, rng: 'np.random.Generator')
```

## `ExtendedKalmanFilter`

```python
ExtendedKalmanFilter() -> None
```

Gaussian state approximation using local derivatives of nonlinear transition and observation maps.

### `ExtendedKalmanFilter.filter`

```python
filter(self, model, observations: 'TimeSeries | ArrayLike')
```

### `ExtendedKalmanFilter.smooth`

```python
smooth(self, model, observations: 'TimeSeries | ArrayLike')
```

## `GaussianObservation`

```python
GaussianObservation(state_dimension: 'int', dimension: 'int', function: 'Callable[[FloatArray, TimeStep], ArrayLike]', noise: 'ArrayLike | Callable[[TimeStep], ArrayLike]', derivative: 'Callable[[FloatArray, TimeStep], ArrayLike] | None' = None) -> None
```

Additive Gaussian observations; function(state, step), optional derivative.

### `GaussianObservation.covariance`

```python
covariance(self, step: 'TimeStep')
```

### `GaussianObservation.distribution`

```python
distribution(self, state, step)
```

### `GaussianObservation.jacobian`

```python
jacobian(self, state: 'ArrayLike', step: 'TimeStep') -> 'FloatArray'
```

### `GaussianObservation.log_density`

```python
log_density(self, value, states: 'ArrayLike', step)
```

### `GaussianObservation.mean`

```python
mean(self, state: 'ArrayLike', step: 'TimeStep') -> 'FloatArray'
```

### `GaussianObservation.sample`

```python
sample(self, states: 'ArrayLike', step, *, rng: 'np.random.Generator')
```

## `GaussianTransition`

```python
GaussianTransition(dimension: 'int', function: 'Callable[[FloatArray, TimeStep], ArrayLike]', noise: 'ArrayLike | Callable[[TimeStep], ArrayLike]', derivative: 'Callable[[FloatArray, TimeStep], ArrayLike] | None' = None) -> None
```

Additive Gaussian dynamics; function(state, step), optional derivative.

### `GaussianTransition.covariance`

```python
covariance(self, step: 'TimeStep')
```

### `GaussianTransition.distribution`

```python
distribution(self, state, step)
```

### `GaussianTransition.jacobian`

```python
jacobian(self, state: 'ArrayLike', step: 'TimeStep') -> 'FloatArray'
```

### `GaussianTransition.log_density`

```python
log_density(self, values: 'ArrayLike', states: 'ArrayLike', step)
```

### `GaussianTransition.mean`

```python
mean(self, state: 'ArrayLike', step: 'TimeStep') -> 'FloatArray'
```

### `GaussianTransition.sample`

```python
sample(self, states: 'ArrayLike', step, *, rng: 'np.random.Generator')
```

## `KalmanFilter`

```python
KalmanFilter() -> None
```

Exact Gaussian filtering for a compatible linear Gaussian state-space model.

### `KalmanFilter.filter`

```python
filter(self, model, observations: 'TimeSeries | ArrayLike')
```

### `KalmanFilter.smooth`

```python
smooth(self, model, observations: 'TimeSeries | ArrayLike')
```

## `LinearGaussianObservation`

```python
LinearGaussianObservation(matrix: 'ArrayLike | Callable[[TimeStep], ArrayLike]', noise: 'ArrayLike | Callable[[TimeStep], ArrayLike]', offset: 'ArrayLike | Callable[[TimeStep], ArrayLike] | None' = None, state_dimension: 'int | None' = None, dimension: 'int | None' = None) -> 'None'
```

y[t] = matrix(step) @ x[t] + offset(step) + N(0,noise(step)).

### `LinearGaussianObservation.coefficients`

```python
coefficients(self, step: 'TimeStep')
```

### `LinearGaussianObservation.covariance`

```python
covariance(self, step: 'TimeStep')
```

### `LinearGaussianObservation.distribution`

```python
distribution(self, state, step)
```

### `LinearGaussianObservation.jacobian`

```python
jacobian(self, state, step: 'TimeStep')
```

### `LinearGaussianObservation.log_density`

```python
log_density(self, value, states: 'ArrayLike', step)
```

### `LinearGaussianObservation.mean`

```python
mean(self, state: 'ArrayLike', step: 'TimeStep')
```

### `LinearGaussianObservation.sample`

```python
sample(self, states: 'ArrayLike', step, *, rng: 'np.random.Generator')
```

## `LinearGaussianTransition`

```python
LinearGaussianTransition(matrix: 'ArrayLike | Callable[[TimeStep], ArrayLike]', noise: 'ArrayLike | Callable[[TimeStep], ArrayLike]', offset: 'ArrayLike | Callable[[TimeStep], ArrayLike] | None' = None, dimension: 'int | None' = None) -> 'None'
```

x[t] = matrix(step) @ x[t-1] + offset(step) + N(0,noise(step)).

### `LinearGaussianTransition.coefficients`

```python
coefficients(self, step: 'TimeStep')
```

### `LinearGaussianTransition.covariance`

```python
covariance(self, step: 'TimeStep')
```

### `LinearGaussianTransition.distribution`

```python
distribution(self, state, step)
```

### `LinearGaussianTransition.jacobian`

```python
jacobian(self, state, step: 'TimeStep')
```

### `LinearGaussianTransition.log_density`

```python
log_density(self, values: 'ArrayLike', states: 'ArrayLike', step)
```

### `LinearGaussianTransition.mean`

```python
mean(self, state: 'ArrayLike', step: 'TimeStep')
```

### `LinearGaussianTransition.sample`

```python
sample(self, states: 'ArrayLike', step, *, rng: 'np.random.Generator')
```

## `ParticleFilter`

```python
ParticleFilter(particles: 'int' = 1000, resampling: 'str' = 'systematic', ess_threshold: 'float' = 0.5) -> None
```

Monte Carlo state filtering with explicit particle count, resampling controls and caller RNG.

### `ParticleFilter.filter`

```python
filter(self, model, observations: 'TimeSeries | ArrayLike', *, rng: 'np.random.Generator')
```

### `ParticleFilter.smooth`

```python
smooth(self, model, observations: 'TimeSeries | ArrayLike', *, rng: 'np.random.Generator', paths: 'int' = 100, method: 'str' = 'backward')
```

Filter then sample smoothing paths; backward FFBS or ancestral tracing.

### `ParticleFilter.smooth_result`

```python
smooth_result(self, result, *, rng: 'np.random.Generator', paths: 'int' = 100, method: 'str' = 'backward')
```

Reuse a completed particle filter without resampling its forward pass.

## `UnscentedKalmanFilter`

```python
UnscentedKalmanFilter(alpha: 'float' = 0.5, beta: 'float' = 2.0, kappa: 'float' = 0.0) -> None
```

Gaussian state approximation using a configured sigma-point rule for nonlinear maps.

### `UnscentedKalmanFilter.filter`

```python
filter(self, model, observations: 'TimeSeries | ArrayLike')
```

### `UnscentedKalmanFilter.smooth`

```python
smooth(self, model, observations: 'TimeSeries | ArrayLike')
```

## `continuous_linear_transition`

```python
continuous_linear_transition(drift: 'ArrayLike', diffusion_covariance: 'ArrayLike', *, forcing: 'ArrayLike | None' = None) -> 'LinearGaussianTransition'
```

Exact transition of dx=(A x + b)dt + L dW, diffusion_covariance=L L.T.

SciPy's block matrix exponential integrates both drift and covariance,
including singular drift matrices. Noise is a rate per unit elapsed time.

## `linear_gaussian`

```python
linear_gaussian(initial: 'GaussianState | Distribution', transition: 'ArrayLike | Callable[[TimeStep], ArrayLike]', observation: 'ArrayLike | Callable[[TimeStep], ArrayLike]', process_covariance: 'ArrayLike | Callable[[TimeStep], ArrayLike]', observation_covariance: 'ArrayLike | Callable[[TimeStep], ArrayLike]', *, transition_offset: 'ArrayLike | Callable[[TimeStep], ArrayLike] | None' = None, observation_offset: 'ArrayLike | Callable[[TimeStep], ArrayLike] | None' = None, time_mode: 'str' = 'discrete', time_unit: 'np.timedelta64 | None' = None, state_names: 'Sequence[str] | None' = None, observation_dimension: 'int | None' = None) -> 'StateSpaceModel'
```

Convenience composition; callable matrices require explicit dimensions.

## `local_level`

```python
local_level(process_variance: 'float', observation_variance: 'float', *, initial: 'GaussianState | None' = None, time_mode: 'str' = 'discrete', time_unit: 'np.timedelta64 | None' = None) -> 'StateSpaceModel'
```

Random walk observed in Gaussian noise; elapsed process variance is a rate.

## `local_linear_trend`

```python
local_linear_trend(acceleration_variance: 'float', observation_variance: 'float', *, initial: 'GaussianState | None' = None, time_unit: 'np.timedelta64 | None' = None) -> 'StateSpaceModel'
```

Integrated Brownian velocity: Q=q*[[dt^3/3,dt^2/2],[dt^2/2,dt]].

Always elapsed-time. The two states are position and velocity; numeric
timestamps supply units, datetime timestamps require an explicit time_unit.

## `simulate`

```python
simulate(model: 'StateSpaceModel', timestamps: 'ArrayLike', *, rng: 'np.random.Generator') -> 'SimulationResult'
```

One joint path with known initial time; no burn-in or implicit transition.

## `FilterResult`

```python
FilterResult(observations: 'TimeSeries', filtered: 'StateSequence', log_likelihoods: 'ArrayLike', backend: 'str', method: 'str', predicted: 'StateSequence | None' = None, smoothed: 'StateSequence | None' = None, observation_mean: 'ArrayLike | None' = None, observation_covariance: 'ArrayLike | None' = None, paths: 'ArrayLike | None' = None, effective_sample_size: 'ArrayLike | None' = None, resampled: 'ArrayLike | None' = None, state_names: 'tuple[str, ...] | None' = None, approximation: 'str' = 'exact Gaussian', _history: 'ParticleHistory | None' = None) -> 'None'
```

Filtered state laws and observations, with likelihood/diagnostics and optional smoothing information.

### `FilterResult.path_distribution`

```python
path_distribution(self)
```

## `GaussianState`

```python
GaussianState(mean: 'ArrayLike', covariance: 'ArrayLike') -> 'None'
```

A vector latent state with a positive-semidefinite covariance.

Singular Gaussians are supported. Their density uses the SciPy convention
on the affine support; a fully deterministic state is an empirical atom.

### `GaussianState.distribution`

```python
distribution(self) -> 'Distribution'
```

## `ObservationModel`

State-conditioned observation mechanism with an explicit observed dimension and time-step convention.

### `ObservationModel.log_density`

```python
log_density(self, value: 'ArrayLike', states: 'ArrayLike', step: 'TimeStep') -> 'FloatArray'
```

One log likelihood per state, marginalizing NaN observation coordinates.

### `ObservationModel.sample`

```python
sample(self, states: 'ArrayLike', step: 'TimeStep', *, rng: 'np.random.Generator') -> 'FloatArray'
```

## `SimulationResult`

```python
SimulationResult(observations: 'TimeSeries', states: 'ArrayLike') -> 'None'
```

Simulated latent states and observations aligned to the declared state-space time coordinates.

## `StateEstimator`

Estimator whose additional arguments are retained by ``filter_states``.

### `StateEstimator.filter`

```python
filter(self, model: 'StateSpaceModel', observations: 'TimeSeries | ArrayLike', *args: 'P.args', **kwargs: 'P.kwargs') -> 'FilterResult'
```

## `StateSequence`

```python
StateSequence(mean: 'ArrayLike', covariance: 'ArrayLike', particles: 'ArrayLike | None' = None, weights: 'ArrayLike | None' = None) -> 'None'
```

Time-indexed marginals, never an implicitly independent joint path law.

Means (n,d), covariances (n,d,d); optional particles (n,N,d), weights (n,N).
Particle moments are computed from those exact normalized masses.

### `StateSequence.distribution`

```python
distribution(self, index: 'int') -> 'Distribution'
```

### `StateSequence.from_particles`

```python
from_particles(particles: 'ArrayLike', weights: 'ArrayLike')
```

## `StateSpaceModel`

```python
StateSpaceModel(initial: 'GaussianState | Distribution', transition: 'TransitionModel', observation: 'ObservationModel', time_mode: 'str' = 'discrete', time_unit: 'np.timedelta64 | None' = None, state_names: 'tuple[str, ...] | None' = None) -> None
```

Initial vector law and independent Markov transition/observation kernels.

Gaussian algorithms additionally require the built-in Gaussian kernels (or
structural equivalents with mean, covariance, and, for EKF, jacobian).
Covariates may be captured explicitly by time-indexed kernel callbacks.

### `StateSpaceModel.initial_distribution`

```python
initial_distribution(self)
```

### `StateSpaceModel.steps`

```python
steps(self, series: 'TimeSeries')
```

## `TimeStep`

```python
TimeStep(index: 'int', time: 'float | int | np.datetime64', dt: 'float') -> None
```

Destination row, timestamp, and elapsed duration from the previous row.

The initial law lives at row zero, where dt=0 and no transition is applied.
Discrete models use dt=1 thereafter. Numeric elapsed units are user-defined.

## `TransitionDensityModel`

Optional density capability required by backward particle smoothing.

### `TransitionDensityModel.log_density`

```python
log_density(self, values: 'ArrayLike', states: 'ArrayLike', step: 'TimeStep') -> 'FloatArray'
```

### `TransitionDensityModel.sample`

```python
sample(self, states: 'ArrayLike', step: 'TimeStep', *, rng: 'np.random.Generator') -> 'FloatArray'
```

## `TransitionModel`

Latent-state transition mechanism with explicit state dimension and time-step convention.

### `TransitionModel.sample`

```python
sample(self, states: 'ArrayLike', step: 'TimeStep', *, rng: 'np.random.Generator') -> 'FloatArray'
```

## `filter_states`

```python
filter_states(model: 'StateSpaceModel', observations: 'TimeSeries | ArrayLike', estimator: 'StateEstimator[P]', *args: 'P.args', **kwargs: 'P.kwargs') -> 'FilterResult'
```

Run a built-in or user estimator and check the shared result contract.

## `ParticleDegeneracyError`

```python
ParticleDegeneracyError
```

The supplied observation assigns zero likelihood to every particle.
