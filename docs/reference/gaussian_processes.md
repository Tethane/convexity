# `convexity.math.gaussian_processes`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Kernels and exact Gaussian-process regression, independent of applications.

    from convexity.math import gaussian_processes as gp
    fitted = gp.GaussianProcessRegressor(gp.RBF(), alpha=.01).fit(x, y)
    law = fitted.posterior(query)

Only NumPy loads on family import. SciPy and scikit-learn load on use.

## `Constant`

```python
Constant(value: 'float' = 1.0, *, bounds: 'ArrayLike | str' = (1e-05, 100000.0)) -> 'Kernel'
```

Constant covariance value, not a deterministic mean or a standard deviation.

## `DotProduct`

```python
DotProduct(sigma_0: 'float' = 1.0, *, bounds: 'ArrayLike | str' = (1e-05, 100000.0)) -> 'Kernel'
```

x @ y.T + sigma_0**2. Pure linear covariance uses sigma_0=0, bounds='fixed'.

## `GaussianProcessRegressor`

```python
GaussianProcessRegressor(kernel: 'Kernel | None' = None, alpha: 'object' = 1e-10, normalize_y: 'bool' = False, optimize: 'bool' = True, restarts: 'int' = 0, seed: 'int' = 0, tolerance: 'float' = 1e-08, maxiter: 'int' = 1000, check_convergence: 'bool' = True) -> None
```

Exact scalar-response Gaussian regression using scikit-learn.

alpha is fixed training diagonal variance, in normalized target units when
normalize_y=True. It is never automatically added to future uncertainty.
Kernel hyperparameters are fit by marginal likelihood when optimize=True.

### `GaussianProcessRegressor.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike') -> 'FittedGaussianProcess'
```

## `Matern`

```python
Matern(length_scale: 'ArrayLike' = 1.0, *, nu: 'float' = 1.5, bounds: 'ArrayLike | str' = (1e-05, 100000.0)) -> 'Kernel'
```

Matérn with fixed smoothness nu > 0 (or infinity for RBF).

## `Periodic`

```python
Periodic(length_scale: 'float' = 1.0, periodicity: 'float' = 1.0, *, length_scale_bounds: 'ArrayLike | str' = (1e-05, 100000.0), periodicity_bounds: 'ArrayLike | str' = (1e-05, 100000.0)) -> 'Kernel'
```

Exp-sine-squared periodic covariance for one-dimensional inputs.

The one-dimensional restriction preserves the PSD guarantee; applying
sine to arbitrary Euclidean distances in multiple dimensions need not.

## `RBF`

```python
RBF(length_scale: 'ArrayLike' = 1.0, *, bounds: 'ArrayLike | str' = (1e-05, 100000.0)) -> 'Kernel'
```

Squared exponential; a length-scale vector enables anisotropy.

## `RationalQuadratic`

```python
RationalQuadratic(length_scale: 'float' = 1.0, alpha: 'float' = 1.0, *, length_scale_bounds: 'ArrayLike | str' = (1e-05, 100000.0), alpha_bounds: 'ArrayLike | str' = (1e-05, 100000.0)) -> 'Kernel'
```

Isotropic scale mixture of RBF kernels; alpha is the mixture shape.

## `WhiteNoise`

```python
WhiteNoise(noise_level: 'float' = 1.0, *, bounds: 'ArrayLike | str' = (1e-05, 100000.0)) -> 'Kernel'
```

Independent variance on a Gram diagonal; all cross covariances are zero.

## `gaussian_process_regression`

```python
gaussian_process_regression(features: 'ArrayLike', response: 'ArrayLike', **kwargs: 'Unpack[GaussianProcessRegressionOptions]') -> 'FittedGaussianProcess'
```

Fit GaussianProcessRegressor to features (observation,feature) and aligned response.

Exact scalar-response Gaussian regression using scikit-learn.
Return the checked fitted predictor; no feature scaling is implicit.
Optional settings: kernel, alpha, normalize_y, optimize, restarts, seed, tolerance, maxiter, check_convergence.

Keyword options:

```python
kernel: Kernel | None
alpha: object
normalize_y: bool
optimize: bool
restarts: int
seed: int
tolerance: float
maxiter: int
check_convergence: bool
```

## `ConvergenceError`

```python
ConvergenceError(result: 'FittedGaussianProcess') -> 'None'
```

Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics.

## `Diagnostics`

```python
Diagnostics(converged: 'bool' = True, optimized: 'bool' = False, message: 'str' = '', iterations: 'int' = 0, optimizer_runs: 'int' = 0, warnings: 'tuple[str, ...]' = (), backend: 'str' = 'sklearn.GaussianProcessRegressor') -> None
```

Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable.

## `FittedGaussianProcess`

```python
FittedGaussianProcess(_model: 'GaussianProcessRegressor', diagnostics: 'Diagnostics' = <factory>) -> None
```

A scalar-response GP with private copied backend state.

All queries take nonempty (m,d) features. alpha affects conditioning only;
a WhiteNoise component remains part of the predicted kernel's variance.

### `FittedGaussianProcess.log_marginal_likelihood`

```python
log_marginal_likelihood(self, theta: 'ArrayLike | None' = None, *, eval_gradient: 'Literal[False]' = False) -> 'float'
log_marginal_likelihood(self, theta: 'ArrayLike | None' = None, *, eval_gradient: 'Literal[True]') -> 'tuple[float, NDArray[np.float64]]'
log_marginal_likelihood(self, theta: 'ArrayLike | None' = None, *, eval_gradient: 'bool') -> 'float | tuple[float, NDArray[np.float64]]'
```

Training log marginal likelihood and optional gradient in log coordinates.

Evaluation at another theta never changes the fitted predictions.

### `FittedGaussianProcess.posterior`

```python
posterior(self, features: 'ArrayLike', *, noise_variance: 'ArrayLike' = 0.0) -> 'GaussianPrediction'
```

Joint prediction, optionally adding independent future observation noise.

noise_variance is a scalar or (m,) vector in original target units
squared. It is separate from training alpha and any WhiteNoise kernel.

### `FittedGaussianProcess.predict`

```python
predict(self, features: 'ArrayLike') -> 'FloatArray'
```

Posterior mean (m,), without constructing an (m,m) covariance.

### `FittedGaussianProcess.predict_std`

```python
predict_std(self, features: 'ArrayLike', *, noise_variance: 'ArrayLike' = 0.0) -> 'FloatArray'
```

Marginal standard deviation (m,), without a full query covariance.

## `GaussianPrediction`

```python
GaussianPrediction(mean: 'ArrayLike', covariance: 'ArrayLike') -> 'None'
```

One joint law in query-row order; mean (m,), covariance (m,m).

Roundoff-scale negative eigenvalues are clipped to zero. Materially
indefinite matrices are rejected. No positive jitter is added.

### `GaussianPrediction.interval`

```python
interval(self, probability: 'float' = 0.95) -> 'FloatArray'
```

Pointwise equal-tail intervals (m,2); these are not a simultaneous band.

### `GaussianPrediction.log_density`

```python
log_density(self, values: 'ArrayLike')
```

SciPy Gaussian log density; singular laws use density on their support.

A deterministic law has log density 0 at its mean and -inf elsewhere.

### `GaussianPrediction.marginal`

```python
marginal(self, indices) -> 'GaussianPrediction'
```

### `GaussianPrediction.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator') -> 'FloatArray'
```

Correlated draws with shape size + (m,), using only the supplied RNG.

## `Kernel`

```python
Kernel(_kernel: 'NativeKernel') -> None
```

A copied scikit-learn kernel with checked arrays and immutable updates.

Use the builtin factories or from_sklearn for custom sklearn kernels.
theta and bounds use log coordinates, excluding fixed hyperparameters.

### `Kernel.diag`

```python
diag(self, x)
```

### `Kernel.with_theta`

```python
with_theta(self, theta: 'ArrayLike') -> 'Kernel'
```

Clone at explicit log parameters; never mutate this kernel.

Bounds constrain fitting, not evaluation at an explicit theta.

## `Regressor`

Fit a Gaussian-process prediction contract from supplied feature rows and responses.

### `Regressor.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike') -> "'FittedGaussianProcess'"
```

## `from_sklearn`

```python
from_sklearn(kernel: 'SklearnKernel') -> 'Kernel'
```

Copy a custom sklearn kernel; its implementation owns the PSD claim.
