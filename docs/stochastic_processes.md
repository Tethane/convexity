# Stochastic processes and differential equations

Family 13 is implemented in `convexity.math.processes`. It supplies conditional
transition kernels, Itô drift/diffusion coefficients, spatial generators, exact
laws where available, and explicit numerical schemes. Models take parameters;
they do not estimate them, choose a physical/pricing measure, or acquire data.
Deterministic ODE IVPs/BVPs remain in `convexity.math.numerical`.

## Completed bounded catalog

| Model / interface | Equation or purpose | Transition |
|---|---|---|
| `BrownianMotion` | Standard scalar Wiener process | Exact Gaussian |
| `ArithmeticBrownianMotion(mu,sigma)` | `dX=mu dt+sigma dW` | Exact Gaussian |
| `GeometricBrownianMotion(mu,sigma)` | `dS=mu S dt+sigma S dW` | Exact lognormal, including zero/deterministic limits |
| `OrnsteinUhlenbeck(kappa,theta,sigma)` | `dX=kappa(theta-X)dt+sigma dW` | Exact Gaussian; stationary law for positive kappa |
| `LinearDiffusion(matrix,loading,forcing=None)` | `dX=(A X+b)dt+B dW` | Exact affine Gaussian through existing SciPy block exponentials |
| `VectorOrnsteinUhlenbeck(kappa,theta,loading)` | `dX=K(theta-X)dt+B dW` | Exact vector Gaussian; stationary Lyapunov solution when stable |
| `CIR(kappa,theta,sigma)` | `dX=kappa(theta-X)dt+sigma sqrt(X)dW` | Exact scaled noncentral chi-square; stationary gamma law |
| `Heston(mu,kappa,theta,sigma,rho)` | Correlated spot/variance diffusion | QuantLib quadratic-exponential (QE), explicitly approximate |
| `AdditiveJumpDiffusion(mu,loading,intensity,jumps)` | Constant drift/loading plus IID finite-activity Poisson marks | Exact scalar/vector sampling and finite analytic moments |
| `MertonJumpDiffusion(mu,sigma,intensity,jump_mean,jump_sigma)` | Multiplicative lognormal Poisson jumps | Exact count-conditioned sampling and analytic moments |
| `ItoDiffusion(drift_function,diffusion_function,event_shape=(),noise_dimension=1)` | User-defined time-dependent Itô coefficients | One-step Euler Gaussian, or sdeint Euler/SRI2 integration |
| `StochasticProcess`, `ItoProcess`, `transition`, `diffusion_generator` | Structural user contracts and checked dispatch | No registration or inheritance required |
| `Transition`, `ProcessPaths`, `sample_paths` | Capability declarations and minimal sequential kernel composition | Exactness/backend/scheme retained |

Acceptance comprises these runnable models, input/output contracts, user
extensions, mathematical checks, and generated examples. Infinite-activity Lévy
processes, state-dependent jump intensities, CIR with zero long-run mean,
Stratonovich equations, exact joint Heston sampling, and parameter estimation are
outside this catalog. Fractional/rough dynamics remain family 14; the broader
joint-simulation engine, supplied common drivers, Milstein, quasi-Monte Carlo,
variance reduction and weighted scenarios remain family 15.

## Installation and examples

Core exact models require only standard project dependencies. The optional extra
installs both numerical backends; imports remain lazy and missing dependencies
raise an installation message when requested.

```bash
python -m pip install -e '.[dev,processes]'
python examples/processes.py
python examples/processes_backends.py
python -m pytest -q tests/math/processes
```

The [worked notebook](../notebooks/processes.ipynb) retains generated paths,
conditional densities, coupled OU geometry, CIR boundary behavior, jump tails,
and Heston/custom-SDE outputs. It also requires `.[notebooks]`.

## States, units and transition capabilities

A scalar state has shape `()`; a vector state has shape `(d,)`, including a
one-coordinate vector. Scalar and one-coordinate vector states are distinct.
No implicit broadcasting or parameter batch axes are accepted. Model arrays,
callback state arguments, moments and results are owned/read-only. User callbacks
must keep captured parameters stable. Every stochastic operation requires an
explicit `numpy.random.Generator`; no global random state is modified.

Times are finite numeric values in a caller-selected common unit. Durations are
nonnegative. Drift is state/time and the diffusion loading is state/sqrt(time).
Brownian drivers have independent unit variance rates: a scalar model's loading
has shape `(m,)`, and a vector model's loading has shape `(d,m)`. Instantaneous
covariance is `B B.T`; correlation belongs in `B`. Calendars, time-zone rules and
year fractions are the caller's responsibility.

```python
import numpy as np
from convexity.math import processes as p

model = p.OrnsteinUhlenbeck(kappa=2., theta=.5, sigma=.3)
kernel = model.transition(1., dt=.4, time=0.)
print(kernel.exact, kernel.method, kernel.backend, kernel.capabilities)
print(kernel.mean(), kernel.covariance())
print(kernel.distribution().quantile([.05, .5, .95]))
print(kernel.sample((2, 3), rng=np.random.default_rng(13)))
```

`transition(state,dt,time=0,method='exact',substeps=1)` conditions on exactly one
state. `.sample(size=(),rng=...)` returns `size + event_shape`; zero sample axes
are allowed. `.distribution()` returns the existing family-2 `Distribution`
only when one is supplied. `.mean()` and `.covariance()` are analytic moments,
not fitted Gaussian approximations. Check `capabilities` before assuming these
operations exist. A sample-only kernel raises `NotImplementedError` for a law.
Jump kernels deliberately do not assert a density: a compound law can have both
an atom and a continuous component. Moment capabilities are independent: a finite
jump mean remains available even when the variance is infinite. Divergent jump
moments are not advertised. Explicit moments attached to a `Transition` must
agree with any supplied `Distribution`; covariance checks scale their roundoff
tolerance to the covariance itself, including very small state units.

Zero-duration transitions are identity atoms, consume no randomness when sampled,
and are exact for every supported scheme. Invalid schemes/domains still fail at
zero duration. Deterministic scalar/vector laws are empirical atoms. Singular
vector Gaussian covariance uses family 10's `GaussianState` convention: density
on the affine support, not a full-dimensional Lebesgue density.

`.sample(initial,times,draws=1,rng=...,method=...,substeps=...)` or structural
`sample_paths(model,...)` composes conditional kernels along strictly increasing,
possibly irregular times. Values have shape `(draw,time,*event_shape)` and retain
the initial state at `times[0]`. The result contains joint paths; independent
marginal samples would not preserve temporal covariance. `.marginal(index)` is
explicitly empirical. This small helper is intended for inspection and examples,
not a vectorized high-throughput simulation engine.

`methods` and `backends` retain the distinct labels actually used, in first-use
order. The singular `method`/`backend` fields are that sole label, or `"mixed"`
when multiple kernels were used. `exact` is true only if every transition was
exact. A later exact step cannot erase evidence of earlier numerical steps.

## Exact linear and positive dynamics

`LinearDiffusion.coefficients(dt)` returns `(F,offset,Q)` for
`X[t+dt]=F X[t]+offset+N(0,Q)`. It reuses the existing
`state_space.continuous_linear_transition`: SciPy matrix exponentials integrate
both affine forcing and covariance without inverting `A`. Singular drift,
rectangular diffusion loading and semidefinite covariance are supported.
Long elapsed intervals use scaled semigroup composition. Stationarity requires
a Hurwitz `A`; vector OU equivalently needs eigenvalues of `K` with positive real
parts. The stationary covariance solves `A P+P A.T+B B.T=0`.

Scalar OU accepts `kappa=0`, with the driftless Brownian limit; a unique stationary
law is then unavailable. GBM accepts nonnegative states, with zero absorbing.
`mu` is its proportional expected drift; log drift is `mu-sigma²/2`.
Mean reversion evaluates the initial-state decay and long-run contribution
separately to preserve both at disparate scales. Multiplicative transitions
combine positive initial levels and exponential growth in log coordinates before
exponentiating; the absorbing zero branch is handled without exponentiation.

CIR requires `kappa,theta>0` and `sigma>=0`; it accepts state zero and violations
of `2*kappa*theta >= sigma²`. `.feller_satisfied` reports this condition rather
than rejecting attainable boundaries. For positive sigma and duration, the exact
law has scale `sigma²(1-exp(-kappa*dt))/(4*kappa)`, degrees of freedom
`4*kappa*theta/sigma²`, and noncentrality `x*exp(-kappa*dt)/scale`. A zero sigma
uses the deterministic mean-reversion solution. CIR exposes only the exact
scheme; it never clips Euler samples and calls them exact.

## Numerical schemes and Heston

Continuous Gaussian models and GBM additionally accept `method='euler'` or
`'sri2'`. `ItoDiffusion` requires an explicit numerical method; its default
`'exact'` request is rejected. A single Euler step is the frozen-coefficient
Gaussian `x+b(t,x)dt+B(t,x)sqrt(dt)Z`, with `exact=False`. Euler with multiple
substeps and SRI2 use the actual `sdeint.itoEuler` / `itoSRI2` functions, on an
equally spaced elapsed-time grid from zero to `dt` for each requested interval.
Callbacks still receive absolute `time + elapsed`. This avoids rounding the
integration duration when the absolute time origin is large; floating-point
precision still limits the resolution of absolute callback timestamps.
Each solver receives the caller's generator. SRI2 is the backend's derivative-free strong-order-one Itô
method under its regularity assumptions; Convexity promises no convergence for
arbitrary unstable or nonsmooth callbacks. No adaptive tolerance is implied.
Discretizations do not preserve arbitrary domains: GBM Euler can produce a
negative level, which a subsequent GBM transition rejects. Use its exact law for
positive paths. No automatic clipping or silent scheme fallback occurs.

Heston states are `(spot,variance)`, with positive spot and nonnegative variance.
`kappa,theta,sigma>0`, `rho` lies in `[-1,1]`. Its loading in level coordinates is
`sqrt(v)*[[S,0],[sigma*rho,sigma*sqrt(1-rho²)]]`, so its generator includes the
correct spot/variance cross derivative. `.transition()` and `.sample()` default
to `method='qe'`; the generic dispatch helpers require this explicitly.
The supplied `mu` becomes a local flat drift curve in QuantLib, not a selected
pricing measure. QuantLib receives a fixed reference date and its global
evaluation date is never changed. The homogeneous process is evolved in relative
elapsed time, preserving the user's time unit. `substeps` refines QE. The result
advertises samples only, not exact moments, a PDF, or an exact joint Heston law.

## Jumps and generators

Additive jumps use `dX=mu dt+B dW+J dN` with independent Poisson intensity
`lambda` and IID supplied `Distribution` marks. Drift is **uncompensated**;
mean increment is `(mu+lambda E[J])*dt`, and covariance increment is
`(B B.T+lambda E[J J.T])*dt`. Existing family-12 event laws and marks can supply
inputs, but this exact construction specifically assumes constant intensity,
constant continuous coefficients, and independent marks.

Merton uses log-jumps `J~Normal(jump_mean,jump_sigma)` and the convention

```text
dS/S- = (mu-lambda*k)dt + sigma dW + (exp(J)-1)dN
k = E[exp(J)-1]
```

Thus `E[S[t+dt]|S[t]]=S[t]*exp(mu*dt)`. Conditional on `N=n`, the log increment
is Gaussian with mean `(mu-lambda*k-sigma²/2)dt+n*jump_mean` and variance
`sigma²dt+n*jump_sigma²`. Sampling draws the Poisson count directly; it never
truncates an infinite mixture. Zero diffusion/jump variance and zero intensity
are supported; state zero is absorbing. No physical-to-pricing measure conversion
is performed.

For diffusions, `.generator(time,state,gradient,hessian)` evaluates
`b·grad(f)+0.5*tr(B B.T Hess(f))`. Derivatives can be arrays or callbacks of one
state. This is the **spatial** generator; a time-dependent test function needs
its separate partial time derivative. Derivatives are explicit, not silently
estimated by finite differences.

Jump models additionally require `function=f` and add
`lambda E[f(x+J)-f(x)]` or `lambda E[f(x*exp(J))-f(x)]`. Additive marks use their
existing expectation capability; Merton integrates in probability coordinates
with SciPy quadrature to avoid exponentiating infinite integration coordinates.
Alternatively pass `jump_expectation` as that scalar expectation (without lambda),
for an analytic value or a mark law that cannot integrate. Caller-supplied
expectations and derivatives must agree with `f`; integrability is the caller's
responsibility and quadrature warnings remain visible. The generic
`diffusion_generator` is deliberately only the continuous contribution.

## Extensibility and verification

A custom process needs an `event_shape` and a compatible `transition` method
returning `Transition`; `transition` and `sample_paths` validate its result.
`ItoDiffusion` is the convenient route for user coefficients:

```python
model = p.ItoDiffusion(lambda t,x: -.8*x+.1*np.sin(t), lambda t,x: [.3])
kernel = model.transition(1., 1., method='sri2', substeps=64)
print(kernel.sample(10, rng=np.random.default_rng(131)))
```

Tests cover known conditional moments, scalar/vector OU semigroups and stationary
Lyapunov identities, integrated Brownian covariance, singular/deterministic
limits, CIR quantiles and both Feller regimes, jump compensation/atoms, Heston
instantaneous correlation and generated moments, exact temporal covariance,
Euler strong convergence against GBM on the same driver, the ODE limit, custom
callbacks, ownership/shapes, reproducibility and zero-duration entropy.
Adversarial regressions check long-horizon cancellation, absorbing states,
representable exponential products, large absolute time origins, independent
moment capabilities, covariance scale, conflicting moments, and mixed path metadata.
Optional backend tests skip only when their dependencies are absent.

Numerical references: [sdeint algorithms and generator contract](https://github.com/mattja/sdeint),
[QuantLib Heston process](https://github.com/lballabio/QuantLib/blob/master/ql/processes/hestonprocess.hpp),
[SciPy noncentral chi-square](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html).
