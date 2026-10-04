# Public contract conventions

The supported development and verification environment is Python 3.14 in Conda
`quant`. Start from `from convexity.math import <family>` and the
[36-family reference](reference/README.md). Each reference page includes public
signatures, class methods, docstrings and expanded convenience-function keyword
schemas. It is generated with `python scripts/build_math_reference.py`; use
`--check` to detect stale pages. The [audit](sdk_adversarial_audit.md) records the
mathematical validation and its limits.

## Inputs, owned results and types

`convexity.math.types` exports the common array/callback vocabulary. `ArrayLike`
accepts lists, tuples and NumPy arrays. Validated real results use `FloatArray`
(`NDArray[np.float64]`); `NumericArray` additionally allows complex128. `IntArray`
and `BoolArray` denote integer and Boolean outputs. `ScalarLike` includes Python
numbers, NumPy real scalars and zero-dimensional real arrays. A scalar validator
still rejects non-scalar arrays. `IndexArray` preserves numeric or datetime
coordinates. These are dtype contracts: Python type checking does **not** prove
rank, dimensions, finiteness, positive definiteness or physical units.

Array-bearing records accept array-like constructor inputs and retain validated
arrays. Records documented as snapshots copy their numerical inputs and expose
read-only arrays; changing the original list does not alter the result. An
explicit constructor preserves this distinction in type hints. Dataclass
`replace` runs the same constructor validation. `<factory>` in a displayed
signature means a fresh default is created for each instance. Frozen records do
not imply that arbitrary user callbacks or native model handles are immutable.

Model-specific `parameters` and free-form `metadata` mappings contain `object`
values because different models and custom adapters store different payloads.
They are not universal coefficient schemas. For a documented numeric key, use
`np.asarray(value, dtype=float)` or an `isinstance` check before numerical use;
ARCH's named coefficient mapping and HAR's coefficient vector are distinct.
Provider option inputs accept `Mapping[str, object]`; native providers validate
advanced settings. The public method signatures and reference expand ordinary
convenience options separately.

Use a result's named fields, not a native provider's result layout. Optional
fields retain `None` when their computation or capability is absent: test for
presence before accessing them. `numerical.Backend.run` overloads describe known
routes; a user-registered route has an `object` result until the caller narrows
it. Convenience functions return the corresponding owned result directly.

```python
from typing import assert_type
from convexity.math import estimation, learning
from convexity.math.types import FloatArray

estimate = estimation.Estimate([1., 2.], names=["level", "slope"])
assert_type(estimate.parameters, FloatArray)
fitted = learning.LinearRegression().fit([[0.], [1.], [2.]], [1., 3., 5.])
assert_type(fitted, learning.FittedModel)
assert_type(fitted.predict_mean([[3.]]), FloatArray)
```

## Axes and mathematical meaning

| Contract | Coordinate convention |
| --- | --- |
| Learning/estimation training data | Rows are observations, columns are features; response shape selects scalar or multiple targets. |
| Distribution | `event_shape` distinguishes a scalar `()` from a vector `(d,)`; leading sample/batch axes are separate. |
| Forecast point, mean, variance | `(horizon, target)`, including one-target forecasts. |
| Forecast sampled paths | `(draw, horizon, target)`; flattened joint-law coordinates are time-major. |
| Forecast covariance | `(horizon * target, horizon * target)`, with the same flattening order. |
| Forecast quantile table | `(horizon, target, level)` with explicitly declared probability levels. |
| Forecast interval | `(horizon, target, 2)`, lower then upper; marginal coverage, not simultaneous path coverage. |
| Scalar process paths | `(draw, time)`; vector process paths add a final state axis. |
| Numerical ODE trajectory | `y=(state, time)`; event records instead use `(event, state)`. |
| State-space filtered means | `(time, state)`; covariance axes are `(time, state, state)`. |
| Covariance | Coordinate order matches the associated vector; entries have products of coordinate units. |

A positive integer forecast horizon requests consecutive future steps; a
`ForecastHorizon` or supported array of increasing positive step offsets can
request sparse steps. Sparse output offsets do not shorten the intervening
process evolution. Known future covariates must cover the maximum step, not
merely the number of requested rows. Feature names, target order, state meanings,
and the initial time point remain part of the relevant family's contract.

A point forecast does not automatically supply an expectation, quantile or
joint law. Inspect `capabilities` before requesting optional operations. A
Gaussian marginal assumption does not supply cross-time dependence; joint
covariance, a joint law or complete sampled paths are required. Unsupported
mathematical operations raise `NotImplementedError`. Invalid domains or shapes
raise `ValueError`/`TypeError`. Numerical methods expose status/diagnostics and,
where available, checked convergence. Successful termination is not a proof of
global optimality or statistical identification.

## Dates, time, units and probability measures

`DateLike` accepts ISO date strings and Python `date`/`datetime` values in the
financial wrappers. Advanced native date inputs remain explicitly annotated.
Financial day-count conventions convert calendar dates to year fractions;
continuous rates and volatilities must match those year units. A state-space
model using datetime timestamps has an explicit `np.timedelta64` time unit for
elapsed-time dynamics. Integer step models use their declared discrete step.
No type alias silently converts annual parameters into daily ones.

Probability levels lie in the documented probability domain, independently of
whether a result stores densities, masses, quantiles or confidence intervals.
Parameter covariance, process innovations, observation noise and predictive
uncertainty are different objects. Fitted models do not automatically integrate
parameter uncertainty. Keep a P-measure fitted process distinct from a Q-measure
pricing law. A measure tag on supplied scenarios is a caller assertion of
provenance; it is not a measure transformation. Consult the family guide before
crossing these boundaries.

## Extension points and provider isolation

Protocols describe structural contracts; users need not inherit from Convexity
classes or edit registries. Public examples and tests exercise custom laws,
forecasters, estimators, processes, solvers, policies and pricing engines.

- `estimation.LogDensityLaw` needs `log_density(observations)`; `likelihood`
  accepts a parameter-to-law factory. An owned distribution satisfies it.
- `forecasting.ForecastFunction` receives history, the maximum requested step,
  future known covariates and an RNG. It returns `ForecastResult`, whose
  capabilities follow the uncertainty actually supplied. Wrap it with
  `FoundationForecaster`.
- `regimes.EventProcess` returns event times as an array-like sequence using the
  supplied RNG; `sample_events` validates the sequence.
- `state_space.StateEstimator` retains estimator-specific arguments through
  `filter_states`, including a required RNG for particle methods.
- Generic `optimization.Solver` and Bayesian inference protocols preserve the
  type of a custom problem/model; unrelated custom implementations need no
  built-in problem representation.
- `stopping.ContinuationRegressor` predicts one continuation value per state row
  for an `ExercisePolicy`; custom regressors need only implement `predict`.
- `pricing.PricingProblem` preserves a custom contract and pricing-model type.
  A custom `PricingModel` needs its measure, not a native conversion method.
  `NativePricingModel` separately describes the QuantLib interoperability path.

```python
import numpy as np
from convexity.math import forecasting
from convexity.math.types import FloatArray

def last_value(history: forecasting.TimeSeries, steps: int,
               future_known: FloatArray | None, *,
               rng: np.random.Generator | None) -> forecasting.ForecastResult:
    return forecasting.ForecastResult(np.repeat(history.values[-1:], steps, axis=0))

callback: forecasting.ForecastFunction = last_value
forecast = forecasting.FoundationForecaster(callback).fit([1., 3.]).forecast([1, 4])
assert forecast.point.shape == (2, 1)
assert "sample" not in forecast.capabilities
```

Ordinary calls use Convexity specifications, arrays and callbacks. Native
provider annotations identify explicit interoperability APIs. NumPy functions
and PyTorch automatic-differentiation callbacks have distinct overloads; a
NumPy conversion inside a tensor callback destroys the gradient graph.
Provider-specific loading/configuration and custom dynamic dispatch may accept
`object` options, with runtime validation. Private adapters still contain
localized `Any` for dynamic provider APIs; there is no blanket claim of strict
static typing of third-party code.

Use `inspect.signature`/`help` or the generated reference for discovery without
loading optional providers. Forward references to optional native types are
intentionally deferred; evaluating every annotation with `get_type_hints`
requires the corresponding names/dependencies to be available.

## Reproducibility and persistence

Pass an explicit `np.random.Generator` where randomness is required. Recreate a
fresh generator with the same seed to reproduce a run; reusing the same instance
advances its state. Retain constructor inputs, coordinate order, time units,
training cutoff, provider versions and immutable checkpoint revision alongside
results. CPU/GPU and upstream library changes can change floating-point or
stochastic behavior.

Use a family's explicit save/load format where supplied. Numerical records can
be reconstructed from their documented fields, but arbitrary callbacks, native
handles and user-defined models have no universal portable serialization.
Pickle is only for trusted artifacts in a compatible environment. Type hints
are neither serialization schemas nor mathematical proofs.
