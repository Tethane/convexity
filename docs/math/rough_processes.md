# History-dependent and rough processes

Family 14 is implemented in `convexity.math.rough`. It supplies Gaussian
histories, convolution memory, lognormal forward-variance states, rough Bergomi,
and exponential-kernel lifts with measured approximation error. All calculations
use NumPy/SciPy and existing Convexity process results; no new dependency is
required. These are specialized mathematical implementations, not generic SDE
solvers relabeled as rough models.

## Completed bounded catalog

| Public object | Capability |
|---|---|
| `MemoryKernel` | Structural contract for kernel values, integrals and truncated covariance integrals |
| `FractionalKernel(H)` | Normalized power kernel `sqrt(2H) u^(H-1/2)`, `0<H<1` |
| `ExponentialKernel(weights,rates)` | Finite sum of exponentials, including signed weights and zero/repeated rates |
| `CallableKernel` | User scalar kernel, with optional analytic integral/covariance callbacks |
| `FractionalBrownianMotion(H)` | Standard fBm with stationary increments, exact joint Gaussian grid sampling |
| `VolterraGaussian(kernel)` | Convolution Gaussian process, jointly sampled with its driving Brownian increments |
| `GaussianDriver`, `sample_driver` | Structural user-driver contract and checked sample dispatch |
| `ForwardVarianceCurve` | Positive initial term structure, log-linear interpolation without extrapolation |
| `LognormalVolterraVariance` | Mean-corrected lognormal variance and conditional forward states |
| `ForwardVarianceState` | Owned maturity-grid snapshot with explicit conditioning method |
| `RoughBergomi` | Fractional variance, correlated asset driver and adapted log-Euler spot scheme |
| `VolterraBergomi` | Same dynamics for a supplied kernel, including finite-factor approximations |
| `approximate_fractional`, `kernel_error`, `KernelError` | Positive Laplace quadrature and integrated error/tolerance checks |
| `VolterraPaths`, `BergomiPaths` | Retained joint driver, factor and spot/variance paths, with numerical-method labels |

Acceptance covers this catalog, structural extensions, analytic identities,
generated joint moments, refinement/error checks, invalid inputs, an executable
example and an executed notebook. Rough Heston, general nonlinear stochastic
Volterra equations, arbitrary two-time kernels, rough path signatures, calibration,
pricing engines, FFT/hybrid acceleration, and the general joint-simulation engine
of family 15 are outside this catalog.

```bash
python docs/math/examples/rough.py
python -m pytest -q tests/math/rough
```

The [worked notebook](notebooks/rough.ipynb) retains kernel-error curves,
fractional histories, rough variance/spot paths and conditional forward curves.

## Time, arrays and random state

Times are finite, strictly increasing numbers in one caller-selected unit.
Every path/history grid starts at **zero**. Starting at a later time would silently
discard memory, so it is rejected. A grid containing only zero is supported and
consumes no randomness. Maturity grids for forward states start at or after their
observation time. Paths have shape `(draw,time)`; Brownian increments have shape
`(draw,time-1)`; retained OU factors have shape `(draw,time,factor)`. Forward
states have shape `(draw,maturity)`. Bergomi's `paths.values` has final coordinates
`(spot,variance)` and shape `(draw,time,2)`.

Stochastic calls require an explicit `numpy.random.Generator` and positive integer
`draws`. Numeric parameters/results are copied into read-only arrays. Kernel and
curve callbacks must remain stable; arbitrary callback closures are not copied.
No calendar or pricing measure is inferred. Variance is squared return per unit
time, `mu` is return per unit time, exponential rates are inverse time, and for
the fractional normalization `eta` has units `time^(-H)`.

## Gaussian histories and memory

Standard fBm has covariance

```text
Cov(B_H(t), B_H(s)) = (t^(2H) + s^(2H) - |t-s|^(2H))/2.
```

`VolterraGaussian` instead defines

```text
X(t) = integral_0^t K(t-u) dW(u)
C(t,s; a) = integral_0^a K(t-u) K(s-u) du,  a <= min(t,s).
```

The fractional Volterra process has variance `t^(2H)` but does **not** generally
have stationary increments. It is not interchangeable with standard fBm. Both
reduce to Brownian motion at `H=.5`.

`kernel.covariance(t,s,until=a)` implements the truncated covariance above;
`until=None` means `min(t,s)`. `kernel.integral(lower,upper)` integrates the
kernel over lag, not calendar time. Fractional kernel values at lag zero for
`H<.5` are infinite, while both its required integrals remain finite. Numerical
integration treats the singularity analytically or with weighted quadrature.

```python
import numpy as np
from convexity.math import rough as r

driver = r.VolterraGaussian(r.FractionalKernel(.1))
times = np.array([0., .05, .2, .5, 1.])
joint = driver.sample_joint(times, draws=1000, rng=np.random.default_rng(14))
print(joint.values.shape, joint.brownian_increments.shape)
```

The dense sampler uses the joint covariance of `X(t_i)` and `Delta W_j`, with
cross entries `integral_(t_i-t_(j+1))^(t_i-t_j) K(u)du` for completed intervals
and zero for future intervals. Independent marginal samples or independent
Brownian samples would destroy the required dependence. `.sample()` returns
the shared family-13 `ProcessPaths`; `.sample_joint()` retains both histories.

Dense sampling costs cubic work and quadratic storage in the number of grid
points, plus sample storage. It is a reference method for modest grids. Exact
means exact finite-dimensional Gaussian law up to quadrature and floating-point
arithmetic, not an exact continuous path. Covariance factorization normalizes
coordinates, rejects material indefiniteness and adds no jitter. Correlation
eigenvalues below `100*eps*dimension*max_abs_correlation` are treated as numerical
null directions; nearly singular covariance is resolved only to this precision.
Custom kernel users own local square integrability and integral accuracy;
quadrature warnings remain visible.

## Forward variance and the information being conditioned on

The variance model uses a positive scalar or array-aware callable `xi0`:

```text
v(T) = xi0(T) exp(eta X(T) - eta² C(T,T;T)/2).
```

Thus `E[v(T)]=xi0(T)` for every supported kernel. `ForwardVarianceCurve` provides
a concrete positive interpolator. `eta>=0`; zero gives deterministic variance.
Nonfinite or underflowed output levels fail rather than silently clipping.

With the entire Brownian history through time `t`, the forward state is

```text
xi_t(T) = xi0(T) exp(eta integral_0^t K(T-u)dW(u) - eta² C(T,T;t)/2).
```

`sample_forward(t,maturities,draws=...,rng=...)` samples this state jointly across
maturities. It generates a fresh unconditional joint state; it does not condition
on an existing spot simulation or update one. Its label is
`full-brownian-filtration`.

`forward_from_brownian(history_times,increments,maturities)` instead returns
`E[v(T) | observed Delta W]`. On each history cell let

```text
a_j(T) = integral_(T-t_(j+1))^(T-t_j) K(u)du / Delta t_j.
m(T) = sum_j a_j(T) Delta W_j
q(T) = sum_j a_j(T)^2 Delta t_j
E[v(T) | Delta W] = xi0(T) exp(eta m(T) - eta² q(T)/2).
```

Using the full past variance in place of `q(T)` would bias this conditional
expectation downward. Brownian bridges between the observations remain unknown
and are integrated out. Consequently, even at `T=t`, this state is generally
not equal to the simulated realized variance, which contains additional
information. Recompute from the extended history to assimilate new increments.
The method label is `observed-brownian-increments`.

A maturity snapshot is not itself a sufficient state for exact rough
continuation. For an exponential kernel, retain all factors instead:
`forward_from_factors(t,factors,maturities)` computes the full forward curve of
that finite-factor model. Its method label is `full-factor-state`. This also
recovers realized variance exactly at the current maturity.

## Rough Bergomi and finite-factor dynamics

`RoughBergomi(hurst,eta,rho,xi0,mu)` uses the normalized fractional kernel with
`0<H<=.5`, including the Brownian boundary. `VolterraBergomi(kernel,...)` accepts
other kernels. The model equations and implemented spot step are

```text
dS/S = mu dt + sqrt(v) dB,  B = rho W + sqrt(1-rho²) W_perp
log S_(i+1) = log S_i + (mu-v_i/2) Delta t_i + sqrt(v_i) Delta B_i.
```

`rho` includes both endpoints of `[-1,1]`; initial spot must be positive. The
variance compensator uses the **actual supplied kernel covariance**. Each spot
step uses left endpoint variance, so it is adapted and has the appropriate
conditional mean under the discrete scheme. A continuous-time martingale claim
does not follow for every parameter choice. `BergomiPaths.paths.exact` is always
false: exact Gaussian grid values do not eliminate spot integration error.
`.driver.exact` describes only that Gaussian law. `.spot`, `.variance`, and
`.asset_increments` expose the retained arrays. Refine the time grid to study
discretization error separately from Monte Carlo error.

For `ExponentialKernel`, `method='factors'` samples the lift recursively:

```text
dY_i = -x_i Y_i dt + dW,  X = sum_i w_i Y_i
Cov(epsilon_i,epsilon_j) = (1-exp(-(x_i+x_j)dt))/(x_i+x_j)
Cov(epsilon_i,Delta W) = (1-exp(-x_i dt))/x_i.
```

Zero-rate limits use `dt`. Factors share one Brownian driver; they are not
independent OU processes. The scheme is exact for the exponential kernel, with
per-interval factor covariance factorizations cached for repeated durations.
`kernel.factor_process()` supplies the existing family-13 exact `LinearDiffusion`
for conditional continuation from retained factors. The scalar `X` alone is
generally insufficient for this continuation.

## Finite-factor approximation and error checks

For `0<H<.5`, `approximate_fractional` discretizes the power kernel's positive
Laplace measure over geometric rate bins. Weights are exact bin masses; rates
are bin barycenters. It returns `(ExponentialKernel, KernelError)`. The default
rate range is `[1e-5/T, 1e8/T]`; callers may set both bounds explicitly. `H=.5`
is represented exactly by a single zero-rate factor.

```python
kernel, error = r.approximate_fractional(.15, factors=16, horizon=1., tolerance=.15)
print(error.relative_l2_error, error.squared_error)
lifted = r.VolterraBergomi(kernel, eta=.7, rho=-.7)
paths = lifted.sample(100., times, draws=1000,
                      rng=np.random.default_rng(141), method='factors')
```

`squared_error = integral_0^T (K(u)-K_n(u))²du`. By Itô isometry it is also the
terminal Gaussian-driver mean-square error when both models use the same `W`.
This is not a spot error, price error or calibration error bound.
`relative_l2_error = sqrt(squared_error / integral_0^T K(u)²du)`;
`require(tolerance)` raises if it exceeds the supplied threshold. Optional
`tolerance` in the factory applies that same check, without silently changing
the factor count. `quadrature_error` is SciPy's estimated integration error.

The fractional error integral changes variables to `u=T*z^(1/(2H))` to account
for the entire singular near-zero contribution. Checking only a finite lag grid
would miss this contribution. More factors refine fixed rate bounds, but cannot
remove truncation error: very small `H` can require a substantially wider range.
No universal default accuracy is claimed. Changing horizon or time units requires
rechecking the approximation. General `kernel_error` also accepts user kernels.

## Verification and references

Tests check fractional normalization/scaling, fBm increment stationarity, the
distinction from Volterra covariance, Brownian limits, joint Gaussian moments and
causality, common-noise factors, singular/zero kernels, forward means and log
covariance, conditional bridge integration, the tower property under history
refinement, exact factor continuation, leverage endpoints, the deterministic
GBM limit, time-grid integration refinement, kernel approximation refinement,
an independent incomplete-gamma error identity, near-zero tails, unit scaling,
ownership, reproducibility, zero-time entropy and invalid contracts.

The model conventions follow [Bayer, Friz and Gatheral, Pricing under rough
volatility](https://doi.org/10.1080/14697688.2015.1099717). Positive exponential
lifts are motivated by [Abi Jaber and El Euch, Multi-factor approximation of rough
volatility models](https://arxiv.org/abs/1801.10359). This implementation uses dense
Gaussian reference sampling and explicit Laplace quadrature; it does not claim
to implement the hybrid/variance-reduction algorithms in
[McCrickerd and Pakkanen's research implementation](https://github.com/rmcrkd/rough_bergomi).
