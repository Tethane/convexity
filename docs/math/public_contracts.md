# Mathematical conventions

## Shapes and results

Inputs generally accept lists, tuples, and NumPy arrays. Real numerical results
use float64. Type hints describe dtypes, not dimensions, finiteness, or units;
constructors and methods validate those at runtime. Snapshot records copy their
arrays and expose read-only values. Arbitrary callbacks and native model handles
are not made immutable by wrapping them in a frozen record.

| Object | Axes |
| --- | --- |
| Learning data | `(observation, feature)`; responses select scalar or multiple targets |
| Distribution | `event_shape` is `()` for a scalar, `(d,)` for a vector; leading draw/batch axes are separate |
| Forecast point/mean/variance | `(horizon, target)`, including one-target forecasts |
| Forecast samples | `(draw, horizon, target)` |
| Forecast covariance | `(horizon * target, horizon * target)`, time-major flattening |
| Forecast quantiles | `(horizon, target, probability_level)` |
| Forecast intervals | `(horizon, target, 2)`, lower then upper |
| Process paths | `(draw, time)`; vector states add a final state axis |
| ODE solution | `(state, time)`; event records use `(event, state)` |
| Filtered states | means `(time, state)`, covariance `(time, state, state)` |

Coordinate order is part of the contract. Covariance entries have products of
coordinate units. A forecast horizon given as an integer requests consecutive
steps; sparse positive offsets still evolve through intervening steps. Known
future covariates must cover the maximum step, not just the requested rows.

Use named result fields rather than a backend's native result layout. An absent
optional output stays `None`. A point forecast is not automatically a mean or a
probability law. Marginal intervals do not establish simultaneous path coverage;
joint covariance or complete sampled paths are needed for cross-time dependence.
Inspect `capabilities` before requesting an optional operation.

## Time, risk, and uncertainty

Annual rates and volatilities must match the model's year fractions/day count.
Datetime state-space models declare an elapsed-time unit. No API automatically
turns an annual parameter into a daily one. Keep parameter uncertainty, process
noise, observation noise, and predictive uncertainty separate; fitting a model
does not automatically integrate all four.

Risk APIs distinguish losses from returns: follow each function's sign and tail
convention. Optimization inputs must use the same sampling horizon for means,
covariances, scenarios, costs, and constraints. Solver success is not a guarantee
of global optimality or reliable estimated moments.

A physical (P) scenario law differs from a pricing (Q) law. A measure label on
supplied samples asserts their provenance; it does not transform the measure.
Native GSR/market-model simulations retain their terminal-forward measure.

QuantLib calls coordinate their shared settings through a lock, which cannot
protect against external code mutating native globals. Missing engine Greeks
remain absent. Tensor automatic differentiation applies to tensor callbacks,
not native QuantLib calls; converting tensors to NumPy breaks the gradient graph.
Sampled surface-arbitrage checks are not global certificates. LSMC estimates a
feasible exercise policy with conditional IID error, not a dual upper bound;
use separate policy-training and valuation paths. Callable carry is unsupported;
promised-cash-flow duration is not option-adjusted effective duration.

## Extension and reproducibility

Protocols allow custom laws, forecasters, estimators, solvers, and pricing models.
Unsupported operations raise `NotImplementedError`; invalid shapes/domains raise
`ValueError` or `TypeError`. Inspect numerical diagnostics rather than accepting a
returned object as proof of convergence or statistical identification.

Pass an explicit NumPy generator where required. A fresh generator with the same
seed reproduces its stream; reusing one advances it. Retain training dates,
coordinate order, units, data provenance, package versions, and checkpoint
revisions. GPU and backend changes can alter floating-point results.

Use explicit save/load formats where provided. Native handles and arbitrary
callbacks have no universal portable format. Load pickle artifacts only from
trusted sources in compatible environments. Optional native annotations are
deferred, so evaluating all type hints may need the corresponding dependencies.
