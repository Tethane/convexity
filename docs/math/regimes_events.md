# Regimes, catalysts and event models

Family 12 is implemented in `convexity.math.regimes`. It composes the existing
`Distribution`, `TimeSeries` and `ForecastResult` contracts, uses statsmodels for
switching-model estimation, and keeps event-arrival and jump-size laws separate.
All examples use generated or supplied inputs; no calendar or data acquisition
service is assumed. Importing the family loads neither SciPy nor statsmodels.

## Completed catalog

| Capability | Public entry points | Bounded scope |
|---|---|---|
| Finite Markov chains | `MarkovChain` | Row-stochastic transitions, powers, propagation, unique stationary law, durations, observed-label estimation and joint paths |
| Hidden regimes | `HiddenMarkovModel` | Exact log-space filtering/smoothing, pair probabilities, Viterbi decoding, backward posterior path sampling, missing coordinates and supplied scalar/vector emission laws |
| Switching dynamics | `SwitchingDynamics` | Arbitrary conditional scalar/vector distributions after drawing the entering regime |
| Estimated Gaussian switching models | `SwitchingRegression`, `SwitchingAutoregression`, `FittedSwitching` | statsmodels MLE, Hamilton filtering/Kim smoothing, fixed transition probabilities, common or switching trend/exogenous coefficients/variance/AR coefficients, joint future simulations |
| Scheduled catalysts | `ScheduledEvent`, `EventSchedule` | Known event times, supplied surprise distributions, conditional impact laws and additive event marks |
| Deterministic event rates | `ConstantIntensity`, `PiecewiseConstantIntensity`, `PoissonProcess` | Counts, event probabilities, exact integrated rates, event-time simulation, likelihood and homogeneous-rate MLE |
| Self-exciting event rates | `ExponentialHawkes` | Univariate subcritical exponential Hawkes intensity, compensator, conditional likelihood and exact thinning |
| Jump sizes and marked events | `MarkedPointProcess`, `MarkedEvents` | Existing scalar/vector distribution laws as IID marks, sampled compound totals and exact compound-Poisson moments |
| Extension contracts | `RegimeInference`, `Intensity`, `EventProcess`, `infer_regimes`, `sample_events` | Structural user implementations with checked public results |

This catalog does not include continuous-time regime chains, arbitrary-emission
HMM parameter learning, time-varying transition regression, Hawkes parameter
estimation, multivariate Hawkes networks, or SDE integration. Supplied HMM laws
may come from existing estimation/distribution tools. Gaussian switching
estimation belongs to the statsmodels adapters above. SDEs and jump diffusions now belong to the [family-13 process catalog](stochastic_processes.md).

## Run the examples and verification

Only standard project dependencies are needed:

```bash
python -m pip install -e '.[dev]'
python docs/math/examples/regimes.py
python docs/math/examples/events.py
python -m pytest -q tests/math/regimes
```

The [worked notebook](notebooks/regimes.ipynb) retains plots of generated
regimes, causal/smoothed inference, dependent forecasts, surprise impacts and
self-exciting event arrivals. It additionally requires `.[notebooks]`.

## Markov and hidden-regime conventions

```python
import numpy as np
from convexity.math import distributions as d, regimes as r

chain = r.MarkovChain([[.95, .05], [.1, .9]])
model = r.HiddenMarkovModel(chain, [d.normal(-1), d.normal(2)], [.5, .5])
result = model.smooth([-.8, np.nan, 1.2, 2.1])
print(result.filtered, result.smoothed, result.log_likelihood)
paths = model.posterior_paths([-.8, np.nan, 1.2, 2.1],
                              draws=1000, rng=np.random.default_rng(12))
```

`transition[i,j] = P(S[t+1]=j | S[t]=i)`. Probability vectors must already
sum to one; only roundoff is normalized. Regime labels are integer indices
`0,...,k-1`. Arrays are copied and read-only. `stationary()` returns the unique
invariant law, including periodic or absorbing chains, and rejects nonunique
laws. `durations()` is `1/(1-P[i,i])`, including the entering observation;
absorbing regimes have infinite duration. `fit(states,pseudocount=...)` counts
adjacent observed labels in a single sequence. Unvisited outgoing rows require
positive symmetric pseudocounts; it never silently invents a transition row.

`chain.sample(initial,steps,draws=...,rng=...)` includes the initial regime and
therefore returns `(draw,steps+1)`. HMM `.sample(observations,...)` returns
`RegimePaths` with regimes `(draw,observations)` and values
`(draw,observations,*event_shape)`. For HMMs, the supplied initial regime applies
at observation row zero, **before any transition**. Discrete models advance once
per observation row even when `TimeSeries` timestamps are irregular. Timestamps
label rows and do not automatically imply elapsed-time transition probabilities.

HMM emission laws must have a common scalar/vector shape and reference measure.
Scalar inputs use `(n,)` or a one-target `TimeSeries`; vector emissions use
`(n,d)`. Fully missing rows contribute no evidence. Partially missing vector
observations evaluate the observed-coordinate marginal, preserving within-row
dependence; a law without marginal capability fails explicitly. Covariates are
rejected rather than ignored. Zero likelihood in every reachable regime raises
an error identifying the row. All forward/backward calculations use log space.

`RegimeResult.predicted[t]` conditions on rows before `t`, `filtered[t]` includes
row `t`, and `smoothed[t]` uses the complete input sample. Smoothing is
retrospective. `pair_probabilities[t,i,j]` is the smoothed joint probability of
regimes `i,j` at rows `t,t+1`. Per-row log likelihoods sum to model evidence.
`decode()` maximizes joint path probability; it is not marginal argmax decoding.
`posterior_paths()` samples joint posterior paths by forward filtering and
backward sampling. Independent draws from smoothed marginals would lose temporal
dependence and are not used. `predictive(probabilities,steps=...)` returns the
exact future emission mixture using propagated regime probabilities.

`SwitchingDynamics(chain,law)` calls `law(entering_regime,previous_state,step)`
after each regime transition, with steps starting at one. The callback returns
a compatible `Distribution`; initial states can be scalar or vector. Its
`.sample()` includes the initial state and regime. `.distribution()` integrates
the next-state law over the next regime. Callback inputs are read-only; user
callbacks must keep captured model inputs stable.

## Switching regression and autoregression

The adapters integrate native [MarkovRegression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)
and [MarkovAutoregression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression.html).
Regression has `y[t] = x[t] beta[S[t]] + noise[t]`. The centered autoregression is

```text
y[t] = x[t] beta[S[t]]
       + sum_j phi[j,S[t]] * (y[t-j] - x[t-j] beta[S[t-j]]) + noise[t]
noise[t] | S[t] ~ Normal(0, variance[S[t]])
```

```python
fitted = r.SwitchingRegression(switching_variance=True).fit(values, exog=features)
print(fitted.converged, fitted.parameters, fitted.parameter_names)
forecast = fitted.predict(5, exog=future_features, draws=5000,
                          rng=np.random.default_rng(13))
```

Targets are finite univariate arrays or `TimeSeries`. Supply exogenous arrays
explicitly as `(n,p)` to `.fit()` and `(maximum_future_step,p)` to prediction.
Implicit `TimeSeries` covariates and missing data are rejected. Intercept/time
trend choices are `n`, `c`, `t`, `ct`; time trends count original rows starting at
one and continue in predictions. Booleans control whether trend coefficients,
exogenous coefficients, variances and (for AR) autoregressive coefficients switch.

Estimation delegates to statsmodels MLE with deterministic starting procedures,
EM initialization and no random optimizer restarts. `start_params` uses native
**transformed** parameter order. Optimizer warnings remain visible;
`converged`, `aic`, `bic`, parameter names/values, regime coefficients/variances
and inference results are retained in an owned snapshot. A completed fit does
not guarantee convergence, global optimality or identifiability. Labels can
permute across fits; no regime is automatically named “high” or “low.”

Default initialization is the stationary regime law. An explicit `initial`
vector is passed to statsmodels `initialize_known` using its **native presample
convention**, which differs from the HMM row-zero convention. Inspect
`inference.predicted[0]` for the prior of the first scored row. AR(p) conditions
on the first `p` observed values; inference timestamps therefore start at
original row `p`. They are not padded with invented estimates. Backend
transition matrices are transposed to the public row-stochastic convention.

`.sample(steps,draws=...,exog=...,rng=...)` returns **future-only** `RegimePaths`.
It samples the terminal joint lag-regime posterior before simulating transitions,
centered residual recursion and innovations. The last filtered marginal alone
is insufficient for AR(p) with multiple lags. `.predict()` accepts the existing
`ForecastHorizon`, including sparse steps, and returns an empirical joint
`ForecastResult` using the shared time-alignment rules. Parameter uncertainty is
not included. Forecast path covariance includes uncertainty in regime sequences
and innovations; it is not a product of marginal mixtures. Regression-only
`.marginal(steps,exog=...)` provides an exact scalar Gaussian-mixture law; AR
prediction uses explicit joint simulation.

## Event times, scheduled surprises and impacts

Event times are finite numeric values in a caller-selected common unit. Rates
are counts per that unit; Hawkes decay is inverse time. Convert datetimes to
elapsed values explicitly. Every observation/simulation window is `(start,end]`.
Adjacent windows consequently neither omit nor duplicate a boundary event.
Zero-length windows contain no events. Point-process observations are strictly
increasing; scheduled events may share a time.

A `ScheduledEvent(time,surprise,impact=None,name=...)` has a known occurrence
and a supplied surprise distribution. “Surprise” denotes a deviation from the
caller's reference expectation; the API does not fetch announcements or infer
that expectation. Scalar/vector Gaussian, Student-t, empirical and mixture laws
all reuse family 2. An optional `impact(surprise)` callback returns a conditional
impact distribution, preserving dependence between surprise and impact and
allowing extra conditional noise. `.sample()` returns paired arrays
`(surprises,impacts)` with leading draw axes. Without a callback, impact equals
surprise. `.impact_distribution()` returns that exact identity law or an
explicit empirical approximation obtained by sampling the conditional model.

`EventSchedule(events,event_shape=())` composes independent events, sorts known
times, and returns a `MarkedEvents` realization. Vector impacts require an
explicit `(dimension,)` event shape, also defining the shape of an empty-window
total. Marks at simultaneous events are retained separately. `.total` is their
additive sum. Separate scheduled events are independent; represent dependent
surprises as coordinates of a joint event law rather than implying dependence
between independently supplied laws.

## Event intensities and jump-size distributions

`PoissonProcess(rate)` accepts the structural `Intensity` contract:
`intensity(t)` and `integrated(start,end)`. `ConstantIntensity` and
`PiecewiseConstantIntensity` supply exact integrals. The latter has
`len(breaks)+1` nonnegative rates, extending before the first and after the last
breakpoint. At a breakpoint the right-hand rate applies. An arbitrary user rate
must provide a coherent, finite, nonnegative intensity and integral.

Counts are Poisson with mean equal to integrated intensity; the probability of
at least one event is `-expm1(-integrated_intensity)`. Likelihood for **ordered
event times** is `sum(log(rate(time))) - integrated_intensity`, without a count
factorial. A zero rate at an observed event gives `-inf`. Homogeneous `.fit()`
returns event count divided by observed exposure, including zero-event windows.

Simulation draws a Poisson count then event locations: exact piecewise sampling
for built-in rates, inverse integrated intensity with SciPy Brent root finding
for custom rates. Zero-rate gaps are respected. Counts in disjoint windows are
independent. Custom integrals must be accurate enough for inversion; times that
cannot be distinguished at floating-point resolution cause an explicit error.

`ExponentialHawkes(baseline,excitation,decay)` uses
`baseline + excitation * sum(exp(-decay*(t-ti)))` over past events `ti<t`.
The branching ratio `excitation/decay` must be less than one. The stationary
mean rate is `baseline/(1-excitation/decay)`; empty-history simulations are **not**
stationary initialization. History-conditioned `.sample()` uses exact thinning
and includes history events at the window's start in the initial excitation.
`max_events` is a guard that raises rather than returning a truncated sample.

Hawkes `.integrated(start,end,history)` is the compensator along the supplied
complete history, including events inside the window; it is **not** a forecast
of future counts. `.log_likelihood(times,start,end,history=...)` accepts
prehistory at or before `start` and observed events in the window. Intensities
are predictable left limits: an event does not excite its own intensity. With
zero excitation the process reduces to homogeneous Poisson. The API does not
pretend Hawkes future counts are Poisson.

`MarkedPointProcess(process,marks)` independently samples an existing scalar or
vector `Distribution` at each event time. Thus Gaussian, Student-t, empirical,
mixture and correlated vector jump sizes require no duplicate law framework.
Any user `EventProcess.sample(start,end,rng=...)` can provide the ordered times.
For a Poisson process with integrated intensity `L` and IID mark `J`, exact
aggregate moments are

```text
E[sum J]   = L * E[J]
Cov[sum J] = L * (Cov[J] + E[J] E[J].T)
```

The scalar covariance is variance. Zero exposure gives deterministic zero;
undefined/infinite mark moments otherwise retain family 2's NaN/Inf semantics.
`.aggregate()` returns an empirical distribution over simulated totals. A
continuous-jump compound Poisson law includes an atom at zero, so this API does
not falsely label it as an everywhere-continuous PDF or silently Gaussianize it.
Analytic marked moments are provided only for the Poisson model. To condition a
marked Hawkes simulation on prehistory, supply a user process wrapper that calls
Hawkes `.sample(history=...)` with that fixed history.

All stochastic operations require a `numpy.random.Generator` and do not use or
mutate global random state. Equal fresh seeds reproduce draws in a fixed backend
environment. Callbacks must return the declared shape and valid laws.

## Acceptance checks

Tests compare Markov powers/invariants/durations and observed-label recovery;
HMM filtering, smoothing, adjacent pairs, evidence and Viterbi decoding against
exhaustive finite-path enumeration; posterior joint-path frequencies, missing
coordinate marginals and extreme log likelihoods; generated-data statsmodels
regression and AR recovery; regression likelihood against the independent HMM;
AR(2) forecast means/covariance against joint-regime enumeration; exogenous
prediction and sparse horizons; Poisson count moments and time change; Hawkes
compensators against quadrature, likelihood and finite-horizon count means;
conditional surprise impacts; compound-Poisson moments and the zero-event atom;
plus shape, boundary, ownership, import, reproducibility and extension contracts.
