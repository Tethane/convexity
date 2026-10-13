# State-space models, filtering and smoothing

Family 10 is implemented in `convexity.math.state_space`. It uses the existing
`forecasting.TimeSeries` input and `distributions.Distribution` output contracts.
This family estimates latent states conditional on supplied observations;
parameters are supplied explicitly, or estimated with the existing estimation
family using the innovation likelihood.

## Completed catalog

| Capability | Public entry points | Implementation and scope |
|---|---|---|
| Transition and observation contracts | `TransitionModel`, `ObservationModel`, `TimeStep`, `StateSpaceModel` | Structural, vector-valued kernels; no inheritance or registry required |
| Latent states and uncertainty | `GaussianState`, `StateSequence`, `FilterResult` | Owned, read-only means, full covariance matrices, Gaussian or weighted empirical marginal laws |
| Linear Gaussian dynamics | `LinearGaussianTransition`, `LinearGaussianObservation`, `linear_gaussian` | Constant or time-dependent matrices, offsets, and covariances; scalar and vector states/observations |
| Exact elapsed linear dynamics | `continuous_linear_transition` | SciPy block matrix exponential for drift, forcing, and integrated diffusion covariance |
| Local models | `local_level`, `local_linear_trend` | Random walk; elapsed-time integrated Brownian velocity with position/velocity states |
| Kalman filtering and smoothing | `KalmanFilter.filter`, `.smooth` | statsmodels known-prior linear state-space filter and fixed-interval smoother |
| Extended filtering and smoothing | `ExtendedKalmanFilter.filter`, `.smooth` | FilterPy extended filter, supplied transition/observation Jacobians, first-order RTS smoother |
| Unscented filtering and smoothing | `UnscentedKalmanFilter.filter`, `.smooth` | FilterPy scaled Merwe sigma points, additive-noise UKF and unscented RTS smoother |
| Nonlinear additive Gaussian models | `GaussianTransition`, `GaussianObservation` | Checked callbacks, time-dependent additive covariance; optional derivatives for EKF |
| Arbitrary conditional laws | `DistributionTransition`, `DistributionObservation` | Compose existing Convexity distributions, including discrete and heavy-tailed laws |
| Particle filtering | `ParticleFilter.filter` | `particles.SMC` bootstrap filter, log weights, likelihood estimates, ESS and four resampling schemes |
| Particle smoothing | `ParticleFilter.smooth`, `.smooth_result` | Backend forward-filter/backward simulation (FFBS) or ancestral trajectory tracing; reuse retained forward history |
| Missing observations | All filters | NaNs omit only the corresponding measurement coordinates; fully missing rows perform prediction only |
| Irregular elapsed times | Models declaring `time_mode='elapsed'` | Destination timestamps and actual durations passed to every kernel; explicit units for datetimes |
| Generated paths | `simulate`, `SimulationResult` | A joint latent/observed realization through the same kernels, with an explicit NumPy generator |
| User estimators | `StateEstimator`, `filter_states` | Checked result dimensions and preservation of observations/timestamps |
| Parameter estimation composition | `FilterResult.log_likelihood` | A scalar objective for `estimation.minimize_estimate`; generated-data example included |

The numerical implementations build on the
[statsmodels state-space framework](https://www.statsmodels.org/stable/statespace.html),
[FilterPy](https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html),
and [`particles` SMC and smoothing](https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/_autosummary/particles.smoothing.html).
Convexity supplies validation, shape/time conventions, missing-coordinate
selection, distribution composition, and the nonlinear RTS backward identity.

## Installation and execution

The linear backend uses the standard project dependencies. Nonlinear and
particle inference use an optional extra:

```bash
conda run -n quant python -m pip install -e '.[dev,state-space,notebooks]'
conda run -n quant python docs/math/examples/state_space.py
conda run -n quant python docs/math/examples/state_space_nonlinear.py
conda run -n quant python docs/math/examples/state_space_estimation.py
conda run -n quant python -m pytest -q tests/math/state_space
```

The extra pins `particles` to upstream commit
[`3388c4bd30ab9acc5a27c970089230f93610b627`](https://github.com/nchopin/particles/tree/3388c4bd30ab9acc5a27c970089230f93610b627).
The PyPI 0.4 release requires NumPy <2, whereas this upstream revision removes
that upper bound. The pin lets the catalog run in the existing Python 3.14,
NumPy 2 `quant` environment without downgrading its numerical stack. Installation
requires Git and access to GitHub; the installed backends run offline. NumPy,
SciPy and `particles` dependencies remain governed by their normal metadata.

Importing `state_space` does not import SciPy, statsmodels, FilterPy, `particles`,
scikit-learn, or PyTorch. Missing optional backends produce an installation hint
only when their algorithms are called.

## Shapes, ownership, and initial timing

Observations are `(n,k)`; a one-dimensional input vector becomes `(n,1)` through
`TimeSeries`. States always retain their vector axis, including `d=1`:

| Quantity | Shape / interpretation |
|---|---|
| `GaussianState.mean`, `.covariance` | `(d,)`, `(d,d)` |
| `StateSequence.mean`, `.covariance` | `(n,d)`, `(n,d,d)` |
| `StateSequence.particles`, `.weights` | `(n,N,d)`, `(n,N)`; normalized separately at each time |
| `FilterResult.predicted` | Prior at each row, conditional on earlier observations |
| `FilterResult.filtered` | Posterior at each row, conditional on observations through that row |
| `FilterResult.smoothed` | Posterior at each row, conditional on the complete supplied series |
| `observation_mean`, `observation_covariance` | Gaussian predictive observation moments `(n,k)`, `(n,k,k)` |
| `log_likelihoods` | `(n,)` innovation log densities, including normalization constants |
| `paths` | Particle smoothing paths `(draw,n,d)` |

The initial law is **at the first observation timestamp**, before assimilating
that row. There is no transition before row zero. `TimeStep` contains the
destination `index`, destination `time`, and incoming `dt`; row zero has `dt=0`.
This convention avoids an extra process-noise contribution at initialization.
For `t>0`, linear models mean

\[
x_t=F_t x_{t-1}+b_t+\eta_t,\quad \eta_t\sim N(0,Q_t),\qquad
y_t=H_t x_t+c_t+\epsilon_t,\quad \epsilon_t\sim N(0,R_t).
\]

Noise sources are independent across time and between state and observation
equations. A nonlinear Gaussian kernel replaces the affine mean with
`function(state, step)`; noise remains additive and independent of the state.
Arrays supplied as constant parameters and arrays in public results are copied
and read-only. Callbacks must be deterministic, must not mutate their arguments,
and must retain stable captured parameters. Built-in kernels validate callback
outputs whenever evaluated.

An initial `GaussianState` explicitly declares the Gaussian prior required by
Kalman/EKF/UKF. A generic `Distribution` prior is supported by particles and
simulation. Merely possessing a mean and covariance does not make an arbitrary
law a Gaussian prior. All generic state and observation laws have vector event
shape `(d,)` or `(k,)`: use `dist.independent([scalar_law])` for a one-coordinate
law.

`StateSequence.distribution(t)` returns a **marginal** Convexity distribution.
Gaussian covariance can be semidefinite, including deterministic coordinates.
Singular Gaussian log densities follow SciPy's affine-support convention; a
fully deterministic state is returned as an empirical atom. Observed Gaussian
innovation covariances must be positive definite. Singular observed innovations
are rejected rather than silently assigning a likelihood to inconsistent data.
There is no automatic jitter or diffuse-prior substitution.

Marginal state covariances do not define a joint path law. Only retained particle
smoothing paths expose `path_distribution()`, with time-major flattened event
coordinates. Its samples are paths from that finite empirical law; independently
sampling filtered marginals is not trajectory smoothing.

## Missing values and time

Use NaN for missing observations; infinity is rejected. Entirely missing rows,
columns, and even series are permitted. Gaussian filters select the observed
coordinates and the corresponding covariance submatrix, preserving correlation
between retained coordinates. A fully missing row contributes zero log likelihood.
Prediction still advances the latent state through the elapsed interval. No
imputation, sorting, deletion, or regular-grid resampling occurs.

`DistributionObservation` uses the law's `marginal` capability for partially
missing vectors. If that capability is absent, it raises `NotImplementedError`;
it never substitutes conditioning on an imputed value. Custom observation kernels
must implement the same marginal likelihood convention. Particle filtering
skips the likelihood callback entirely for a fully missing row.

`time_mode='discrete'` requires regularly spaced timestamps and uses `dt=1`
between rows. Timestamp labels may be numeric or datetime. An elapsed model
uses actual numeric differences; datetime observations require a positive fixed
`np.timedelta64` `time_unit`. For example, `time_unit=np.timedelta64(1,'D')`
makes `dt` a number of days. Covariance rates, drift rates, and velocities must
use that same unit. Setting elapsed mode forwards the durations; a constant
user-supplied transition matrix still remains constant. It does not automatically
become a continuous-time model.

`continuous_linear_transition(A, D, forcing=b)` discretizes
`dx=(A x+b)dt+L dW`, where `D=L L.T`, by SciPy block exponentials:

\[
F(\Delta)=e^{A\Delta},\quad b_\Delta=\int_0^\Delta e^{Au}b\,du,\quad
Q(\Delta)=\int_0^\Delta e^{Au}D e^{A^T u}\,du.
\]

It also handles singular `A`; long intervals use semigroup composition to avoid
overflow in the auxiliary block for stable dynamics. `local_level` uses `Q=q*dt`;
`local_linear_trend` integrates Brownian velocity exactly, with
`Q=q*[[dt**3/3,dt**2/2],[dt**2/2,dt]]`. These support irregular observation gaps
without inventing intermediate measurements.

`TimeSeries.covariates` are rejected by these inference adapters. Exogenous
inputs can be incorporated explicitly into time-indexed transition or observation
offsets/functions. The model and timestamps control their alignment.

## Basic use

```python
import numpy as np
from convexity.math import state_space as ss

model = ss.local_linear_trend(.03, .5)
generated = ss.simulate(model, [0., .2, 1.1, 3.], rng=np.random.default_rng(7))
y = generated.observations.values.copy()
y[1] = np.nan
result = ss.KalmanFilter().smooth(model, ss.TimeSeries(y, generated.observations.timestamps))
print(result.filtered.mean)        # causal state estimates
print(result.smoothed.mean)        # uses all supplied observations
print(result.log_likelihood)
posterior = result.filtered.distribution(3)
print(posterior.sample(5, rng=np.random.default_rng(8)))
```

`linear_gaussian` provides a compact constructor. Explicit transition and
observation objects also permit composition. Constant matrices infer dimensions;
callable matrices need declared dimensions (the convenience constructor infers
the state dimension from the prior and accepts `observation_dimension`). Matrix,
offset, and noise callbacks receive `step` alone. Nonlinear functions and their
Jacobians receive `(state, step)`.

## Nonlinear approximations and particles

EKF uses supplied Jacobians at the filtered state for prediction and at the
predicted state for measurement assimilation. It does not silently finite-difference
a missing derivative. UKF uses scaled sigma points (`alpha=.5`, `beta=2`,
`kappa=0` by default). The adapter regenerates observation sigma points from the
**complete** predicted covariance, including process noise. This is required for
the additive-noise convention and exact reduction to Kalman filtering on a
linear model. Both smoothers retain forward predicted means and state cross
covariances for the nonlinear RTS backward recursion. Their distributions and
likelihoods are Gaussian approximations, recorded in `result.approximation`.
Arbitrary non-additive noise, angular/manifold residuals, and iterated EKF are
outside this bounded catalog.

Particles use the model's initial and transition laws as bootstrap proposals.
The supported resampling schemes are `systematic`, `stratified`, `multinomial`,
and `residual`. Resampling occurs before propagation when the previous posterior
ESS divided by `N` is below `ess_threshold`. `resampled[t]` records that decision,
and `effective_sample_size[t]` describes the posterior weights after row `t`.
Threshold zero disables adaptive resampling. No Gaussian replacement for the
weighted particles is implied.

The log likelihood is the log of the SMC normalizing-constant estimate, with
contributions accounting for whether resampling occurred. It is stochastic and
is not itself an unbiased log-likelihood estimator. All-zero particle likelihoods
raise `ParticleDegeneracyError`; malformed/nonfinite weights are rejected.

```python
particle_filter = ss.ParticleFilter(particles=2000, ess_threshold=.5)
filtered = particle_filter.filter(model, generated.observations,
                                 rng=np.random.default_rng(9))
smoothed = particle_filter.smooth_result(filtered, paths=100, method='backward',
                                       rng=np.random.default_rng(10))
print(smoothed.paths.shape)  # (100, 4, 2)
```

Backward FFBS samples from the discrete smoothing approximation using transition
densities; its cost is `O(time * particles * paths)`. Gaussian transition densities
require positive-definite process covariance. Ancestral tracing supports sampled
transitions without evaluable densities, including degenerate dynamics, but
suffers path coalescence over long histories. Both are explicitly selectable.
Particle filters retain full history (`O(time * particles * state_dimension)`)
so `.smooth_result` can reuse the forward pass. A result contains private backend
history and is not a portable serialized artifact.

Stochastic entry points require an explicit `np.random.Generator`. Model sampling
uses that generator directly. The `particles` backend uses NumPy's legacy global
random API for resampling and backward draws; the adapter supplies a generated
seed, serializes its backend calls, and restores the prior global state even on
failure. Calls are reproducible and do not advance the caller's legacy global
stream. Applications that concurrently use that global API in unrelated threads
should isolate particle runs in separate processes; the adapter's lock does not
control external threads.

## Extensibility and parameter estimation

A transition kernel declares `dimension`, implements
`sample(states, step, *, rng) -> (N,d)`, and optionally supplies
`log_density(values, states, step) -> (N,)` for backward smoothing
(`TransitionDensityModel` describes this extra capability). `values` can
be one next-state vector or one vector per preceding state. An observation kernel
declares `state_dimension` and `dimension`, with `sample -> (N,k)` and
`log_density(observation, states, step) -> (N,)`. Its observation is one vector
that may contain NaNs. `DistributionTransition` and `DistributionObservation`
adapt `law(state, step) -> Distribution`; user vectorized kernels avoid creating
one distribution object per particle when performance matters.

Gaussian structural kernels additionally expose `mean(state, step)` and
`covariance(step)`. EKF needs `jacobian(state, step)`; linear inference needs
`coefficients(step) -> (matrix, offset)`. Declaring these is a promise of the
corresponding additive Gaussian model, not automatic recognition of arbitrary
probability kernels.

An estimator can implement `.filter(model, observations)` and return
`FilterResult`; `filter_states` checks alignment and observation preservation.
Smoothing methods return the same result with `smoothed` populated, so forward
and backward estimates are available together.

For parameter fitting, minimize the negative **complete innovation likelihood**
with `estimation.minimize_estimate`. Positive covariance parameters can be
represented in log coordinates. The
[estimation example](examples/state_space_estimation.py) recovers process and
measurement variances from a generated local-level series. Do not treat the
innovation contributions as independent identically distributed raw observations
or fit each state row separately. Parameter posteriors and joint state/parameter
inference are available in [family 11](bayesian_inference.md). EM and particle
MCMC rejuvenation are outside the completed catalogs.

## Verification and worked examples

The [tests](../../tests/math/state_space) check:

- Dense joint-Gaussian conditioning for every filtered prefix and the full
  smoother, including correlated multivariate observations, offsets, irregular
  gaps, and partial/complete missing rows; exact evidence and terminal identities.
- Scalar initial-prior timing, singular latent covariances, causal prefixes,
  probability-law composition, and explicit rejection of singular innovations.
- Exact OU transitions and semigroup identities, integrated-velocity covariance,
  omitted-observation gap equivalence, and numeric/datetime unit equivalence.
- EKF/UKF reduction to Kalman filtering and smoothing, squared-observation
  Gaussian moments for UKF, and generated nonlinear state recovery.
- Particle likelihoods, moments, ESS, all four resampling methods, finite-state
  exact enumeration, backward and ancestral paths, missing likelihood
  marginalization, reproducibility, RNG restoration, and explicit degeneracy.
- User-written kernels and estimators, input/result ownership, shape and
  covariance validation, lazy imports, and public exports.

The [linear example](examples/state_space.py) estimates position and velocity
through irregular noisy observations and missing stretches. The
[nonlinear example](examples/state_space_nonlinear.py) demonstrates EKF/UKF
and a latent process observed through Poisson counts. The
[worked notebook](notebooks/state_space.ipynb) plots the latent states,
observations, filtering/smoothing means, and uncertainty bands from generated
data. No market data or external service is needed to execute the examples.
