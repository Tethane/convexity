# Gaussian processes and kernels

Family 7 lives in `convexity.math.gaussian_processes`. It supplies exact,
scalar-response GP regression and joint Gaussian predictions at supplied query
points. It has no dependency on other Convexity math or application families.
NumPy loads on import; SciPy and scikit-learn load when used. No new package
dependencies are required.

```python
import numpy as np
from convexity.math import gaussian_processes as gp

x = np.linspace(-2, 2, 30)[:, None]
y = np.sin(x[:, 0])
kernel = gp.Constant() * gp.RBF(length_scale=.8)
fitted = gp.GaussianProcessRegressor(kernel, alpha=.01).fit(x, y)
query = np.linspace(-2.5, 2.5, 50)[:, None]
latent = fitted.posterior(query)
observed = fitted.posterior(query, noise_variance=.01)
draws = latent.sample(4, rng=np.random.default_rng(42))
```

See the runnable [example](examples/gaussian_processes.py) and
[worked notebook](notebooks/gaussian_processes.ipynb).

## Working catalog

| Tool | Behavior |
|---|---|
| `RBF` | Squared exponential, scalar or per-feature length scales |
| `Matern` | Matérn, scalar or per-feature length scales, fixed `nu > 0`; `nu=inf` gives RBF |
| `Constant` | Constant covariance, including trainable signal amplitude |
| `WhiteNoise` | Independent per-observation variance |
| `DotProduct` | `x @ y.T + sigma_0**2`; pure linear with `sigma_0=0, bounds='fixed'` |
| `RationalQuadratic` | Isotropic RBF scale mixture with fitted length scale and mixture shape |
| `Periodic` | Exp-sine-squared covariance on **one-dimensional inputs** |
| `+`, `*`, `**` | Kernel sum, product and positive integer entrywise powers |
| `from_sklearn` | Copies a custom scikit-learn kernel into the checked facade |
| `GaussianProcessRegressor`, `gaussian_process_regression` | Exact Gaussian regression with optional marginal-likelihood fitting |
| `GaussianPrediction` | Joint mean/covariance, sampling, log density, marginal laws and pointwise intervals |

Kernel factories return `Kernel` objects. Most factories accept `bounds=(lower,
upper)`; `RationalQuadratic` and `Periodic` expose a separate bounds option per
parameter. Bounds are positive, finite and increasing in **physical** units.
Initial values must lie inside them. Use `'fixed'` to exclude a parameter from
optimization. Anisotropic bounds can be one shared pair or a pair per feature.
Zero constant/noise/linear-offset parameters require fixed bounds. Matérn
smoothness `nu` is always fixed, even when length scales are fitted.

`2 * gp.RBF()` scales covariance by two, not standard deviation by two. Numeric
operands are nonnegative, fixed constants and add no fitted parameters. Adding
a constant kernel represents a random constant component, not a deterministic
mean offset. Noninteger kernel powers are excluded because they do not in
general preserve positive semidefiniteness.

## Shapes and ownership

| Operation | Shape |
|---|---|
| Fit features, response | `(n,d)`, `(n,)` |
| Query features | `(m,d)`, including `(1,d)` for one query |
| `kernel(x)`, `kernel(x,y)` | `(n,n)`, `(n,m)` |
| `kernel.diag(x)` | `(n,)` |
| `fitted.predict(query)`, `predict_std(query)` | `(m,)` |
| `fitted.posterior(query)` | `GaussianPrediction` with mean `(m,)`, covariance `(m,m)` |
| `prediction.sample(size, rng=...)` | `size + (m,)`; an integer means `(size,m)` |
| `prediction.interval(probability)` | `(m,2)`, lower then upper |
| `kernel.theta`, `kernel.bounds` | `(p,)`, `(p,2)` in log coordinates, free parameters only |
| `kernel(x, eval_gradient=True)` | Gram matrix and gradient `(n,n,p)` |

Inputs must be finite real numeric arrays. Empty feature/response arrays,
booleans, strings, complex values, NaNs and infinities are rejected. Matrix
responses are deliberately excluded: this first implementation has one scalar
output, not a model of dependence between multiple outputs. Feature identity is
positional; callers select training rows and preserve column order. No feature
scaling, time interpretation, missing-value treatment or data splitting is implicit.

Specifications are frozen and `.fit` returns a new object. Training data, alpha,
kernel parameters and fitted backend state are copied; public arrays are
read-only. Querying never refits. `predict` and `predict_std` avoid allocating the
full query covariance. A full `posterior` also checks positive semidefiniteness
with an eigendecomposition; reserve it for query sets where a joint law is useful.

## Conditioning and noise

With no target normalization, training covariance is
`A = kernel(X) + diag(alpha)` (a scalar alpha broadcasts). Scikit-learn computes
the posterior by Cholesky solves:

```text
mean = kernel(Q, X) @ solve(A, y)
covariance = kernel(Q) - kernel(Q, X) @ solve(A, kernel(X, Q))
```

The default alpha is `1e-10`. Set `alpha=0` for exact interpolation when the
training covariance is positive definite. Repeated training points with zero
noise can be singular and raise `LinAlgError`; the API does not silently raise
alpha or choose jitter for you.

There are three distinct noise settings:

- **`alpha`** is fixed training diagonal variance, scalar or `(n,)`. It affects
  conditioning and is not automatically added to future predictions.
- **`WhiteNoise`** is part of the fitted kernel, with a trainable noise variance
  unless fixed. Its Gram matrix is diagonal even for repeated coordinates;
  explicit cross matrices are zero, including `white(X, X)`. Its variance is
  already included in `posterior` and `predict_std`.
- **`noise_variance`** on a prediction adds independent future observation noise,
  scalar or `(m,)`, in original target units squared. It leaves the mean unchanged.

Thus `posterior` is a latent function law when its kernel describes the latent
function. A kernel containing white noise instead predicts that noisy process.
Do not add the same noise again through `noise_variance`.

With `normalize_y=True`, scikit-learn subtracts the training mean and divides by
the training population standard deviation, using its unit-scale fallback for
constant targets. Kernel parameters **and alpha** then describe
normalized targets. Predictions are returned in original target units, with
covariance multiplied by the squared training scale. Convert known training
noise variance to normalized units yourself, or use `normalize_y=False`.
The mean prior is zero without normalization and the training mean with it.
Normalization statistics are plugged-in estimates, not integrated uncertainty.

## Joint predictive laws

Query rows are the coordinates of one random vector. Off-diagonal covariance
captures dependence between function values, and `.sample` draws jointly using
the caller's `numpy.random.Generator`. Identical noiseless query points therefore
share the same random value. `.marginal(indices)` retains the requested order,
including a one-coordinate vector. `.interval(.95)` returns pointwise equal-tail
intervals; it is not a 95% simultaneous band over the whole query set.

Singular Gaussian laws are supported, including exact interpolation and
deterministic zero-covariance predictions. Construction symmetrizes within a
roundoff tolerance of `100*eps*m*max(1,max(abs(covariance)))`, clips negative
eigenvalues only within that tolerance, and rejects material indefiniteness.
It never adds positive jitter. `.log_density(values)` follows SciPy's singular
Gaussian support/pseudodeterminant convention; this is not a full-dimensional
Lebesgue density for a singular law. An exactly deterministic law has log
density zero at its mean and minus infinity elsewhere. The result class is
local to this family because the distribution family's multivariate Gaussian
currently requires positive-definite covariance.

An explicit prior can be constructed as
`gp.GaussianPrediction(np.zeros(len(query)), kernel(query))`.

## Hyperparameters and diagnostics

By default, `kernel=None` uses a trainable `Constant()*RBF()`. `optimize=True`
maximizes the training log marginal likelihood with SciPy L-BFGS-B through
scikit-learn's optimizer interface. The initial kernel is never mutated.
`optimize=False` retains all supplied values. All-fixed kernels skip optimization.

`restarts=k` adds `k` seeded starts, uniform in finite log bounds, after the
initial start. `seed=0` is the default. `maxiter` defaults to 1000 and `tolerance`
to `1e-8`, passed as both L-BFGS-B `ftol` and `gtol`. Restart search can improve a
local optimum but does not establish a global maximum or integrate parameter
uncertainty. The predictions condition on the selected hyperparameters.

`fitted.kernel` exposes the selected kernel, with `parameter_names`, `theta` and
`bounds` in matching order. `kernel.with_theta(theta)` constructs an independent
kernel at explicit **log** parameters; evaluation may go outside fitting bounds.
`fitted.log_marginal_likelihood(theta=None, eval_gradient=False)` evaluates the
training objective, optionally returning its log-parameter gradient, without
changing the fitted predictions. An invalid covariance at a candidate theta
can have likelihood minus infinity, following scikit-learn.

Diagnostics report the selected optimizer run's convergence, iteration count,
total run count and backend bound warnings. A successful constrained optimum at
a bound is reported in `diagnostics.warnings`; it is not an optimizer failure.
An unsuccessful selected run raises `ConvergenceError`, retaining its fitted
`.result`. `check_convergence=False` returns the failed result for inspection.
Other backend warnings retain their warning category.

Custom kernels enter through `from_sklearn(custom_kernel)`. They must implement
scikit-learn's kernel, cloning and gradient contracts. Convexity copies their
state and checks public matrix/gradient shapes and finiteness; the custom
implementation owns its positive-semidefinite and derivative claims.

This is dense exact GP regression: training takes cubic time and quadratic
storage in the number of observations. Sparse/variational GPs, GP classification,
multioutput kernels and hyperparameter posterior inference are outside this
initial catalog.

## Module verification

All 82 family tests pass with current dependencies and with NumPy 1.26.4,
SciPy 1.15.3 and scikit-learn 1.6.1 (the declared dependency floors). The scoped
pyright check is clean, and the example and notebook execute successfully.

Run only this family's checks when changing this family:

```bash
python -m pytest -q tests/math/gaussian_processes
python -m pyright --pythonpath "$(command -v python)" convexity/math/gaussian_processes
python docs/math/examples/gaussian_processes.py
```

The tests check independent Gaussian conditioning and likelihood formulas,
finite-difference kernel/likelihood gradients, composition, normalization,
training/future noise, duplicate points, singular laws, generated-data fitting,
optimizer failures, sample covariance, ownership, invalid inputs and import
isolation. No unrelated family tests are needed for this implementation.

Backend references:
[scikit-learn Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html),
[GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html),
[SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html),
[SciPy multivariate normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html).
