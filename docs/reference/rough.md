# `convexity.math.rough`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

History-dependent and rough processes (family 14).

## `FractionalKernel`

```python
FractionalKernel(hurst: 'float' = 0.1) -> None
```

K(u)=sqrt(2H) u**(H-1/2); its Volterra variance is t**(2H).

### `FractionalKernel.covariance`

```python
covariance(self, t, s, *, until=None)
```

### `FractionalKernel.integral`

```python
integral(self, lower, upper) -> 'float'
```

## `ExponentialKernel`

```python
ExponentialKernel(weights: 'ArrayLike', rates: 'ArrayLike') -> 'None'
```

K(u)=sum_i weights[i] exp(-rates[i]*u), with nonnegative rates.

Signed weights and repeated/zero rates are supported. All factors share W.

### `ExponentialKernel.covariance`

```python
covariance(self, t, s, *, until=None) -> 'float'
```

### `ExponentialKernel.factor_process`

```python
factor_process(self)
```

Family-13 exact OU state for continuation from retained factors.

### `ExponentialKernel.integral`

```python
integral(self, lower, upper) -> 'float'
```

## `CallableKernel`

```python
CallableKernel(function: 'Callable[[float], float]', integral_function: 'Callable[[float, float], float] | None' = None, covariance_function: 'Callable[[float, float, float], float] | None' = None) -> None
```

User kernel with optional exact integral/covariance callbacks.

Scalar callback values must be finite away from zero. The user guarantees
local square integrability; SciPy quadrature warnings remain visible.

### `CallableKernel.covariance`

```python
covariance(self, t, s, *, until=None)
```

### `CallableKernel.integral`

```python
integral(self, lower, upper)
```

## `FractionalBrownianMotion`

```python
FractionalBrownianMotion(hurst: 'float' = 0.1) -> None
```

Standard fBm; stationary increments, Var(B_H(t))=t**(2H).

This is distinct from the Riemann--Liouville fractional Volterra process.

### `FractionalBrownianMotion.covariance`

```python
covariance(self, times: 'ArrayLike') -> 'FloatArray'
```

### `FractionalBrownianMotion.sample`

```python
sample(self, times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator') -> 'ProcessPaths'
```

## `VolterraGaussian`

```python
VolterraGaussian(kernel: 'MemoryKernel') -> None
```

X(t)=integral_0^t K(t-s)dW(s), with its entire grid history.

### `VolterraGaussian.brownian_covariance`

```python
brownian_covariance(self, times: 'ArrayLike') -> 'FloatArray'
```

Cov(X(t_i), W(t_{j+1})-W(t_j)), shape (time,interval).

### `VolterraGaussian.covariance`

```python
covariance(self, times: 'ArrayLike') -> 'FloatArray'
```

### `VolterraGaussian.sample`

```python
sample(self, times, *, draws: 'int' = 1, rng: 'np.random.Generator')
```

### `VolterraGaussian.sample_joint`

```python
sample_joint(self, times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'dense')
```

## `approximate_fractional`

```python
approximate_fractional(hurst: 'float', *, factors: 'int' = 16, horizon: 'float' = 1.0, rate_min: 'float | None' = None, rate_max: 'float | None' = None, tolerance: 'float | None' = None) -> 'tuple[ExponentialKernel, KernelError]'
```

Return (ExponentialKernel, KernelError) from geometric Laplace bins.

Rates are bin barycenters and weights their exact Laplace mass. Increasing
factors refines fixed bounds, but cannot remove their truncation error.
Explicit rate bounds have units 1/time. H=.5 is exactly one zero-rate factor.

## `kernel_error`

```python
kernel_error(target: 'MemoryKernel', approximation: 'MemoryKernel', horizon: 'float') -> 'KernelError'
```

Compute integral_0^T (K-K_n)^2, including the singular near-zero tail.

By Ito isometry this is E[(X(T)-X_n(T))**2] under a common W.
quadrature_error is SciPy's estimate, not a rigorous bound.

## `ForwardVarianceCurve`

```python
ForwardVarianceCurve(times: 'ArrayLike', values: 'ArrayLike') -> 'None'
```

Positive, log-linearly interpolated initial curve; no extrapolation.

## `ForwardVarianceState`

```python
ForwardVarianceState(time: 'float', maturities: 'ArrayLike', values: 'ArrayLike', method: 'str') -> 'None'
```

xi_t(T) on a maturity grid, shape (draw,maturity), with filtration label.

A finite maturity snapshot alone does not permit exact rough continuation.
Retain the underlying Brownian history or all OU factors for future updates.

## `LognormalVolterraVariance`

```python
LognormalVolterraVariance(kernel: 'MemoryKernel', xi0: 'ArrayLike | Callable[[FloatArray], ArrayLike]' = 0.04, eta: 'float' = 1.0) -> None
```

v(t)=xi0(t)*exp(eta*X(t)-eta**2*Var(X(t))/2).

### `LognormalVolterraVariance.evaluate`

```python
evaluate(self, times: 'ArrayLike', driver_values: 'ArrayLike')
```

### `LognormalVolterraVariance.forward_from_brownian`

```python
forward_from_brownian(self, times: 'ArrayLike', increments: 'ArrayLike', maturities) -> 'ForwardVarianceState'
```

E[v(T)|observed Brownian increments], not a full-history plug-in.

The exponent compensator uses the variance explained by those
increments. Unobserved Brownian bridges are integrated out exactly.

### `LognormalVolterraVariance.forward_from_factors`

```python
forward_from_factors(self, time: 'float', factors: 'ArrayLike', maturities) -> 'ForwardVarianceState'
```

Exact full-filtration forward curve of the supplied exponential lift.

### `LognormalVolterraVariance.sample_forward`

```python
sample_forward(self, time: 'float', maturities, *, draws: 'int' = 1, rng: 'np.random.Generator') -> 'ForwardVarianceState'
```

Unconditional draws of the full Brownian-filtration state at time t.

Samples integral_0^t K(T-s)dW(s) jointly across T. This is a new joint
draw, not an update of previously sampled spot/variance paths.

## `RoughBergomi`

```python
RoughBergomi(hurst: 'float' = 0.1, eta: 'float' = 1.0, rho: 'float' = -0.7, xi0: 'ArrayLike | Callable[[FloatArray], ArrayLike]' = 0.04, mu: 'float' = 0.0) -> 'None'
```

Normalized fractional kernel, 0 < H <= .5, including Brownian limit.

### `RoughBergomi.sample`

```python
sample(self, initial: 'float', times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'dense') -> 'BergomiPaths'
```

## `VolterraBergomi`

```python
VolterraBergomi(kernel: 'MemoryKernel', eta: 'float' = 1.0, rho: 'float' = -0.7, xi0: 'ArrayLike | Callable[[FloatArray], ArrayLike]' = 0.04, mu: 'float' = 0.0) -> None
```

dS/S=mu dt+sqrt(v)dB, B=rho W+sqrt(1-rho²)W_perp.

kernel fixes v's Gaussian driver. No pricing measure is selected implicitly.
Exact grid variance sampling does not make the spot discretization exact.

### `VolterraBergomi.sample`

```python
sample(self, initial: 'float', times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator', method: 'str' = 'dense') -> 'BergomiPaths'
```

## `BergomiPaths`

```python
BergomiPaths(paths: 'ProcessPaths', driver: 'VolterraPaths', asset_increments: 'ArrayLike') -> 'None'
```

Spot/variance paths and retained joint volatility/asset innovations.

## `MemoryKernel`

Locally square-integrable convolution kernel, with checked integrals.

covariance(t,s,until) = integral_0^until K(t-u) K(s-u) du.
The default cutoff is min(t,s). Singular K(0) is allowed.

### `MemoryKernel.covariance`

```python
covariance(self, t: 'float', s: 'float', *, until: 'float | None' = None) -> 'float'
```

### `MemoryKernel.integral`

```python
integral(self, lower: 'float', upper: 'float') -> 'float'
```

## `GaussianDriver`

Gaussian driving-noise extension with declared covariance and reproducible sampling behavior.

### `GaussianDriver.covariance`

```python
covariance(self, times: 'ArrayLike') -> 'FloatArray'
```

### `GaussianDriver.sample`

```python
sample(self, times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator') -> 'ProcessPaths'
```

## `sample_driver`

```python
sample_driver(driver: 'GaussianDriver', times: 'ArrayLike', *, draws: 'int' = 1, rng: 'np.random.Generator') -> 'ProcessPaths'
```

Checked structural dispatch for a user-supplied scalar Gaussian driver.

## `VolterraPaths`

```python
VolterraPaths(times: 'ArrayLike', values: 'ArrayLike', brownian_increments: 'ArrayLike', method: 'str', exact: 'bool', factors: 'ArrayLike | None' = None, backend: 'str' = 'numpy/scipy') -> 'None'
```

Joint X(t) and the increments of the same driving Brownian motion.

factors, when present, have shape (draw,time,factor). Exactness refers to
the supplied kernel, never to a different kernel it may approximate.

### `VolterraPaths.paths`

```python
paths(self) -> 'ProcessPaths'
```

## `KernelError`

```python
KernelError(horizon: 'float', squared_error: 'float', target_energy: 'float', quadrature_error: 'float') -> None
```

Integrated squared kernel error equals coupled terminal driver MSE.

### `KernelError.require`

```python
require(self, tolerance: 'float')
```
