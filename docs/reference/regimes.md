# `convexity.math.regimes`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Regimes, catalysts and event models (family 12).

Distribution and TimeSeries composition, statsmodels switching fits, explicit
random generators, and numeric event times in a caller-selected common unit.

## `MarkovChain`

```python
MarkovChain(transition: 'ArrayLike') -> 'None'
```

transition[i,j] = P(S[t+1]=j | S[t]=i); one step is one row.

### `MarkovChain.durations`

```python
durations(self) -> 'FloatArray'
```

Expected consecutive observations in each regime, including entry.

### `MarkovChain.fit`

```python
fit(states: 'ArrayLike', *, size: 'int | None' = None, pseudocount: 'float' = 0.0)
```

Observed-state MLE; optional symmetric transition pseudocounts.

### `MarkovChain.matrix`

```python
matrix(self, steps: 'int' = 1) -> 'FloatArray'
```

### `MarkovChain.propagate`

```python
propagate(self, probabilities, steps: 'int' = 1)
```

### `MarkovChain.sample`

```python
sample(self, initial: 'ArrayLike', steps: 'int', *, draws: 'int' = 1, rng: 'np.random.Generator')
```

Return (draw,steps+1), including the initial state at column zero.

### `MarkovChain.stationary`

```python
stationary(self)
```

Unique invariant law, including periodic chains; reject nonuniqueness.

## `HiddenMarkovModel`

```python
HiddenMarkovModel(chain: 'MarkovChain', emissions: 'Sequence[Distribution]', initial: 'ArrayLike') -> 'None'
```

Discrete hidden-state model with explicit transition probabilities and state-dependent observation laws.

### `HiddenMarkovModel.decode`

```python
decode(self, data)
```

Viterbi joint maximum-probability path, not marginal argmax labels.

### `HiddenMarkovModel.filter`

```python
filter(self, data) -> 'RegimeResult'
```

### `HiddenMarkovModel.posterior_paths`

```python
posterior_paths(self, data, *, draws: 'int' = 1, rng: 'np.random.Generator')
```

Exact forward-filter/backward draws of S[0:n] given all observations.

### `HiddenMarkovModel.predictive`

```python
predictive(self, probabilities, *, steps: 'int' = 1)
```

### `HiddenMarkovModel.sample`

```python
sample(self, observations: 'int', *, draws: 'int' = 1, rng: 'np.random.Generator') -> 'RegimePaths'
```

### `HiddenMarkovModel.smooth`

```python
smooth(self, data) -> 'RegimeResult'
```

## `SwitchingDynamics`

```python
SwitchingDynamics(chain: 'MarkovChain', law: 'Callable[[int, FloatArray, int], Distribution]') -> None
```

First draw S[t], then X[t] ~ law(S[t], X[t-1], t), for t=1,... .

The callback returns an existing Distribution, allowing non-Gaussian and
nonlinear dynamics. Initial state and regime are retained in sampled paths.

### `SwitchingDynamics.distribution`

```python
distribution(self, state: 'ArrayLike', regime, *, step: 'int' = 1)
```

### `SwitchingDynamics.sample`

```python
sample(self, initial_state: 'ArrayLike', initial_regime, steps, *, draws: 'int' = 1, rng: 'np.random.Generator') -> 'RegimePaths'
```

## `SwitchingRegression`

```python
SwitchingRegression(regimes: 'int' = 2, trend: 'str' = 'c', switching_trend: 'bool' = True, switching_exog: 'bool' = True, switching_variance: 'bool' = False, maxiter: 'int' = 200, em_iter: 'int' = 10) -> None
```

Regression whose parameters depend on a latent Markov regime, with explicit state order.

### `SwitchingRegression.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, exog=None, initial: 'ArrayLike | None' = None, start_params=None)
```

Fit deterministic statsmodels MLE; no random optimizer restarts.

initial is passed to statsmodels initialize_known with its native
presample convention; inference.predicted[0] reports the first scored
prior. The default uses stationary initialization.

## `SwitchingAutoregression`

```python
SwitchingAutoregression(regimes: 'int' = 2, trend: 'str' = 'c', switching_trend: 'bool' = True, switching_exog: 'bool' = True, switching_variance: 'bool' = False, maxiter: 'int' = 200, em_iter: 'int' = 10, order: 'int' = 1, switching_ar: 'bool' = True) -> None
```

statsmodels centered AR: y[t]-x[t]β[S[t]] = Σφ[j,S[t]](y[t-j]-x[t-j]β[S[t-j]])+ε[t].

### `SwitchingAutoregression.fit`

```python
fit(self, data: 'TimeSeries | ArrayLike', *, exog=None, initial: 'ArrayLike | None' = None, start_params=None)
```

Fit deterministic statsmodels MLE; no random optimizer restarts.

initial is passed to statsmodels initialize_known with its native
presample convention; inference.predicted[0] reports the first scored
prior. The default uses stationary initialization.

## `FittedSwitching`

```python
FittedSwitching(specification: 'SwitchingRegression', history: 'TimeSeries', chain: 'MarkovChain', inference: 'RegimeResult', parameters: 'ArrayLike', parameter_names: 'tuple[str, ...]', coefficients: 'ArrayLike', variances: 'ArrayLike', autoregressive: 'ArrayLike', converged: 'bool', aic: 'float', bic: 'float', _design: 'ArrayLike', _joint: 'ArrayLike', exog_columns: 'int' = 0) -> 'None'
```

Owned fit snapshot. Parameter uncertainty is not included in predictions.

### `FittedSwitching.marginal`

```python
marginal(self, steps: 'int' = 1, *, exog=None)
```

Exact regression marginal at the requested step; AR uses joint sampling.

### `FittedSwitching.predict`

```python
predict(self, horizon, *, draws: 'int' = 1000, exog=None, rng: 'np.random.Generator')
```

Existing ForecastResult with empirical joint paths; sparse horizons allowed.

### `FittedSwitching.sample`

```python
sample(self, steps, *, draws: 'int' = 1000, exog=None, rng: 'np.random.Generator') -> 'RegimePaths'
```

Joint future draws conditional on observed data and fitted parameters.

## `ConstantIntensity`

```python
ConstantIntensity(rate: 'float') -> None
```

Homogeneous event intensity per unit of the supplied time coordinate.

### `ConstantIntensity.integrated`

```python
integrated(self, start: 'float', end: 'float') -> 'float'
```

### `ConstantIntensity.intensity`

```python
intensity(self, time: 'float')
```

## `PiecewiseConstantIntensity`

```python
PiecewiseConstantIntensity(breaks: 'ArrayLike', rates: 'ArrayLike') -> 'None'
```

rates[i] applies between breaks[i-1] and breaks[i], extending to ±inf.

At a breakpoint the right-hand rate is used. Rates have len(breaks)+1 rows.

### `PiecewiseConstantIntensity.integrated`

```python
integrated(self, start: 'float', end: 'float') -> 'float'
```

### `PiecewiseConstantIntensity.intensity`

```python
intensity(self, time: 'float') -> 'float'
```

## `PoissonProcess`

```python
PoissonProcess(rate: 'Intensity') -> None
```

Independent increments with a deterministic Intensity implementation.

### `PoissonProcess.count_distribution`

```python
count_distribution(self, start: 'float', end: 'float')
```

### `PoissonProcess.event_probability`

```python
event_probability(self, start: 'float', end: 'float') -> 'float'
```

### `PoissonProcess.fit`

```python
fit(times: 'ArrayLike', start: 'float', end: 'float')
```

Homogeneous Poisson MLE, including windows with no events.

### `PoissonProcess.integrated`

```python
integrated(self, start: 'float', end: 'float') -> 'float'
```

### `PoissonProcess.log_likelihood`

```python
log_likelihood(self, times: 'ArrayLike', start: 'float', end: 'float') -> 'float'
```

### `PoissonProcess.sample`

```python
sample(self, start: 'float', end: 'float', *, rng: 'np.random.Generator')
```

Poisson count then inverse integrated-intensity ordered times.

## `ExponentialHawkes`

```python
ExponentialHawkes(baseline: 'float', excitation: 'float', decay: 'float') -> None
```

λ(t|H) = baseline + excitation * sum(exp(-decay*(t-ti))), ti < t.

excitation/decay < 1 is required. Supplied history is conditioned on;
omitted history means an empty prehistory, not stationary initialization.

### `ExponentialHawkes.integrated`

```python
integrated(self, start: 'float', end: 'float', history: 'ArrayLike' = ()) -> 'float'
```

Compensator along a supplied complete history, not an expected count.

### `ExponentialHawkes.intensity`

```python
intensity(self, time: 'float', history: 'ArrayLike' = ()) -> 'float'
```

### `ExponentialHawkes.log_likelihood`

```python
log_likelihood(self, times: 'ArrayLike', start: 'float', end: 'float', *, history: 'ArrayLike' = ())
```

### `ExponentialHawkes.sample`

```python
sample(self, start: 'float', end: 'float', *, rng: 'np.random.Generator', history: 'ArrayLike' = (), max_events: 'int' = 1000000)
```

Exact thinning with a decaying intensity upper bound.

## `ScheduledEvent`

```python
ScheduledEvent(time: 'float', surprise: 'Distribution', impact: 'Callable[[FloatArray], Distribution] | None' = None, name: 'str' = 'event') -> None
```

Known event time, surprise law, and optional conditional impact law.

impact(surprise) returns a Distribution; without it the mark is the surprise.
Surprise values are supplied deviations from a caller's reference expectation.

### `ScheduledEvent.impact_distribution`

```python
impact_distribution(self, *, draws: 'int' = 1000, rng: 'np.random.Generator')
```

Exact surprise law for identity impact; otherwise an empirical mixture.

### `ScheduledEvent.sample`

```python
sample(self, *, draws: 'int' = 1, rng: 'np.random.Generator')
```

Return (surprises, impacts), each with a leading draw axis.

## `EventSchedule`

```python
EventSchedule(events: 'Sequence[ScheduledEvent]', event_shape: 'tuple[int, ...]' = ()) -> None
```

Independent scheduled events, with additive scalar/vector marks.

Simultaneous events are allowed. Dependence within a surprise/impact vector
is preserved; dependence across separate events must be modeled explicitly.

### `EventSchedule.sample`

```python
sample(self, start: 'float', end: 'float', *, rng: 'np.random.Generator') -> 'MarkedEvents'
```

## `MarkedPointProcess`

```python
MarkedPointProcess(process: 'EventProcess', marks: 'Distribution') -> None
```

Event times from any EventProcess, IID independent jump-size law.

Totals are returned as samples or an empirical law. A compound Poisson sum
generally has an atom at zero plus a continuous part, so it is not disguised
as a continuous Distribution with a PDF.

### `MarkedPointProcess.aggregate`

```python
aggregate(self, start: 'float', end: 'float', *, draws: 'int' = 1000, rng: 'np.random.Generator')
```

### `MarkedPointProcess.moments`

```python
moments(self, start: 'float', end: 'float')
```

Exact compound-Poisson (mean,covariance); marks must have moments.

### `MarkedPointProcess.sample`

```python
sample(self, start: 'float', end: 'float', *, rng: 'np.random.Generator') -> 'MarkedEvents'
```

## `EventProcess`

Draw sorted, strictly increasing event times on (start,end].

### `EventProcess.sample`

```python
sample(self, start: 'float', end: 'float', *, rng: 'np.random.Generator') -> 'ArrayLike'
```

## `Intensity`

Deterministic rate per time unit and its exact/numerical integral.

### `Intensity.integrated`

```python
integrated(self, start: 'float', end: 'float') -> 'float'
```

### `Intensity.intensity`

```python
intensity(self, time: 'float') -> 'float'
```

## `RegimeInference`

Infer latent-regime probabilities under the supplied model, retaining explicit regime order.

### `RegimeInference.filter`

```python
filter(self, data: 'TimeSeries | ArrayLike') -> 'RegimeResult'
```

## `infer_regimes`

```python
infer_regimes(data: 'TimeSeries | ArrayLike', model: 'RegimeInference') -> 'RegimeResult'
```

Filter observations through a RegimeInference implementation.

Rows are time; return state probabilities and predictive log likelihoods.
Smoothed probabilities, when available, condition on the complete sample.

## `sample_events`

```python
sample_events(process: 'EventProcess', start: 'float', end: 'float', *, rng: 'np.random.Generator') -> 'FloatArray'
```

Sample ordered event times in (start,end] with an explicit NumPy Generator.

Return finite times in the same units as the process intensity; seeded generators
make stochastic calls reproducible.

## `MarkedEvents`

```python
MarkedEvents(times: 'ArrayLike', marks: 'ArrayLike', start: 'float', end: 'float') -> 'None'
```

A realization on (start,end], with scalar/vector marks at ordered times.

## `RegimePaths`

```python
RegimePaths(regimes: 'ArrayLike', values: 'ArrayLike') -> 'None'
```

Joint draws: regimes (draw,time), values (draw,time,*event_shape).

### `RegimePaths.distribution`

```python
distribution(self)
```

## `RegimeResult`

```python
RegimeResult(predicted: 'ArrayLike', filtered: 'ArrayLike', log_likelihoods: 'ArrayLike', smoothed: 'ArrayLike | None' = None, pair_probabilities: 'ArrayLike | None' = None, timestamps: 'ArrayLike | None' = None) -> 'None'
```

Rows are observation times, columns are integer regime labels.

predicted uses observations strictly before the row; filtered includes that
row; smoothed uses the entire supplied sample. Labels have no imposed order.

### `RegimeResult.distribution`

```python
distribution(self, index, *, kind: 'str' = 'filtered')
```
