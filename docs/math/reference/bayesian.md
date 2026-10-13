# `convexity.math.bayesian`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Bayesian inference and parameter uncertainty (family 11).

Conjugate inference uses NumPy/SciPy; MCMC and VI use optional PyMC, and joint
sequential inference composes the state-space particles backend. Heavy libraries
are imported only when needed. Stochastic operations require an explicit RNG.

## `BetaBinomial`

```python
BetaBinomial(alpha: 'float' = 1.0, beta: 'float' = 1.0) -> None
```

Beta prior on a success probability; update by binomial counts and predict future counts.

### `BetaBinomial.predictive`

```python
predictive(self, trials: 'int' = 1)
```

Beta-binomial count law, integrating probability uncertainty.

### `BetaBinomial.update`

```python
update(self, successes, trials: 'ArrayLike' = 1)
```

Independent binomial rows; scalar trials broadcasts to each row.

## `GammaPoisson`

```python
GammaPoisson(shape: 'float' = 1.0, rate: 'float' = 1.0) -> None
```

Gamma prior on a Poisson rate with explicit exposure; update counts and predict future observations.

### `GammaPoisson.predictive`

```python
predictive(self, exposure: 'float' = 1.0)
```

### `GammaPoisson.update`

```python
update(self, counts, exposure: 'ArrayLike' = 1.0)
```

## `GaussianLinearRegression`

```python
GaussianLinearRegression(mean: 'ArrayLike', covariance: 'ArrayLike', observation_variance: 'float' = 1.0) -> 'None'
```

Gaussian coefficient prior, known scalar noise variance, explicit design.

No intercept is inserted. predictive returns a joint law sharing the same
coefficient uncertainty across all query rows.

### `GaussianLinearRegression.predictive`

```python
predictive(self, features: 'ArrayLike', *, include_noise: 'bool' = True)
```

### `GaussianLinearRegression.update`

```python
update(self, features: 'ArrayLike', response: 'ArrayLike')
```

## `JointParticleFilter`

```python
JointParticleFilter(particles: 'int' = 2000, ess_threshold: 'float' = 0.5, resampling: 'str' = 'systematic') -> None
```

Bootstrap SMC on (theta,x), with theta constant along each path.

No parameter jitter or artificial dynamics. Static parameter diversity can
collapse over long series: inspect ESS and surviving distinct parameters.
This bounded adapter is not SMC² or particle MCMC rejuvenation.

### `JointParticleFilter.filter`

```python
filter(self, model: 'StateParameterModel', observations: 'TimeSeries | ArrayLike', *, rng: 'np.random.Generator')
```

### `JointParticleFilter.smooth`

```python
smooth(self, model: 'StateParameterModel', observations: 'TimeSeries | ArrayLike', *, paths: 'int' = 100, rng: 'np.random.Generator') -> 'JointFilterResult'
```

## `MCMC`

```python
MCMC(draws: 'int' = 1000, tune: 'int' = 1000, chains: 'int' = 4, cores: 'int' = 1, method: 'str' = 'auto', target_accept: 'float' = 0.9, progressbar: 'bool' = False) -> None
```

PyMC auto steps (usually NUTS), explicit NUTS, or random-walk Metropolis.

A numerical BayesianProblem uses Metropolis and needs a finite initial point.
PyMCModel uses native symbolic gradients, constraints and compound steps.

### `MCMC.fit`

```python
fit(self, model, *, rng: 'np.random.Generator')
```

## `NormalInverseGamma`

```python
NormalInverseGamma(mean: 'float' = 0.0, precision: 'float' = 1.0, shape: 'float' = 2.0, scale: 'float' = 1.0) -> None
```

sigma² ~ IG(shape,scale); mu | sigma² ~ N(mean,sigma²/precision).

### `NormalInverseGamma.predictive`

```python
predictive(self)
```

### `NormalInverseGamma.update`

```python
update(self, observations: 'ArrayLike')
```

## `NormalMean`

```python
NormalMean(mean: 'float' = 0.0, variance: 'float' = 1.0, observation_variance: 'float' = 1.0) -> None
```

Unknown Gaussian mean, with known independent observation variance.

### `NormalMean.predictive`

```python
predictive(self)
```

### `NormalMean.update`

```python
update(self, observations: 'ArrayLike')
```

## `VariationalInference`

```python
VariationalInference(iterations: 'int' = 20000, draws: 'int' = 2000, method: 'str' = 'advi', learning_rate: 'float' = 0.01, progressbar: 'bool' = False) -> None
```

PyMC mean-field or full-rank ADVI; loss history is not a convergence proof.

### `VariationalInference.fit`

```python
fit(self, model, *, rng: 'np.random.Generator')
```

## `beta`

```python
beta(alpha: 'float' = 1.0, beta: 'float' = 1.0) -> 'Distribution'
```

Beta law on a probability; alpha and beta are positive shapes.

## `gamma`

```python
gamma(shape: 'float', rate: 'float' = 1.0) -> 'Distribution'
```

Gamma law parameterized by rate, not scale.

## `inverse_gamma`

```python
inverse_gamma(shape: 'float', scale: 'float' = 1.0) -> 'Distribution'
```

Return the scalar inverse-gamma law with positive shape and scale.

Density is proportional to x**(-shape-1)*exp(-scale/x) for x>0.
Moments exist only above their corresponding shape thresholds.

## `posterior_predictive`

```python
posterior_predictive(posterior: 'PosteriorSamples', conditional: 'Callable[[Mapping[str, FloatArray | np.float64]], Distribution]', *, draws: 'int' = 1000, rng: 'np.random.Generator') -> 'Distribution'
posterior_predictive(posterior: 'Distribution', conditional: 'Callable[[FloatArray | np.float64], Distribution]', *, draws: 'int' = 1000, rng: 'np.random.Generator') -> 'Distribution'
```

Return a finite joint predictive Distribution.

posterior is a Distribution or PosteriorSamples. conditional(theta) returns
one scalar/vector Distribution; for samples theta is a name->event mapping.
Each draw uses one joint parameter draw and one conditional observation draw.

## `BayesianProblem`

```python
BayesianProblem(prior: 'Distribution', likelihood: 'Callable[[FloatArray], float] | Likelihood', names: 'tuple[str, ...] | None' = None, initial: 'ArrayLike | None' = None) -> 'None'
```

Vector parameter prior and a complete log-likelihood callback.

likelihood may also be an estimation.Likelihood. Constants must be retained
for evidence calculations. Coordinates and Jacobians are the caller's choice.

### `BayesianProblem.log_likelihood`

```python
log_likelihood(self, theta: 'ArrayLike') -> 'float'
```

### `BayesianProblem.log_posterior`

```python
log_posterior(self, theta: 'ArrayLike')
```

### `BayesianProblem.point`

```python
point(self, theta: 'ArrayLike')
```

## `InferenceDiagnostics`

```python
InferenceDiagnostics(method: 'str', backend: 'str' = 'user', rhat: 'Mapping[str, ArrayLike] | None' = None, ess_bulk: 'Mapping[str, ArrayLike] | None' = None, ess_tail: 'Mapping[str, ArrayLike] | None' = None, divergences: 'int | None' = None, loss: 'ArrayLike | None' = None, note: 'str' = '') -> 'None'
```

Inference status and available sampling diagnostics; absence of a metric is not evidence of convergence.

## `InferenceMethod`

Infer a posterior from a built-in or custom model, preserving the model type and caller RNG contract.

### `InferenceMethod.fit`

```python
fit(self, model: 'ModelT', *, rng: 'np.random.Generator') -> 'PosteriorSamples'
```

## `JointFilterResult`

```python
JointFilterResult(filtering: 'FilterResult', parameter_names: 'tuple[str, ...]') -> None
```

Augmented filter coordinates are (parameters, states), in that order.

### `JointFilterResult.joint_posterior`

```python
joint_posterior(self, index, *, smoothed: 'bool' = False)
```

### `JointFilterResult.parameter_posterior`

```python
parameter_posterior(self, index)
```

### `JointFilterResult.state_posterior`

```python
state_posterior(self, index, *, smoothed: 'bool' = False)
```

## `PosteriorSamples`

```python
PosteriorSamples(samples: 'Mapping[str, ArrayLike]', diagnostics: 'InferenceDiagnostics' = <factory>, _model: 'Model | None' = None, _trace: 'InferenceData | None' = None) -> 'None'
```

samples[name] has shape (chain,draw,*event); warmup is excluded.

MCMC draws remain correlated. VI uses one draw axis, not a Markov chain.
distribution() is a finite empirical law, not an estimated continuous PDF.

### `PosteriorSamples.distribution`

```python
distribution(self, names: 'Sequence[str] | None' = None)
```

### `PosteriorSamples.interval`

```python
interval(self, name, level: 'float' = 0.95) -> 'FloatArray'
```

### `PosteriorSamples.mean`

```python
mean(self, name) -> 'FloatArray'
```

### `PosteriorSamples.posterior_predictive`

```python
posterior_predictive(self, *, rng: 'np.random.Generator', names: 'Sequence[str] | None' = None, model=None)
```

PyMC forward draws, optionally in a compatible fresh prediction model.

## `PyMCModel`

```python
PyMCModel(builder: 'Callable[[], Model]') -> None
```

builder() returns a fresh pymc.Model with named priors and observations.

Keep captured inputs stable. Native symbolic models support automatic NUTS,
discrete compound steps, VI, and joint latent state/parameter posteriors.

## `StateParameterModel`

```python
StateParameterModel(prior: 'Distribution', model_factory: 'Callable[[FloatArray], StateSpaceModel]', reference: 'ArrayLike', parameter_names: 'tuple[str, ...] | None' = None) -> 'None'
```

A proper vector parameter prior and theta -> StateSpaceModel factory.

reference is used only to check dimensions/time conventions, never as a
plug-in estimate. Each parameter draw receives its own conditional state prior.

### `StateParameterModel.conditional_model`

```python
conditional_model(self, theta: 'ArrayLike')
```

## `infer`

```python
infer(model: 'ModelT', method: 'InferenceMethod[ModelT]', *, rng: 'np.random.Generator') -> 'PosteriorSamples'
```

Run built-in or user inference and validate the shared sample result.
