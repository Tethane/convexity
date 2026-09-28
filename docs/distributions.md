# Probability distributions and conditional laws — family 2

`convexity.math.distributions` is a standalone interface to probability laws.
It consumes supplied mathematical parameters, values and functions. It performs
no data access, fitting, estimation, forecasting, path simulation or financial
modeling. NumPy and SciPy supply numerical algorithms; optional PyTorch laws
retain tensors and gradient graphs.

The [worked problem book](../notebooks/distributions.ipynb) works through
densities, discrete masses, empirical laws, mixtures, transforms and conditioning.
The [script example](../examples/distributions.py) runs without notebooks.

```python
import numpy as np
from convexity.math import distributions as dist

law = dist.normal(loc=2., scale=3.)
law.log_density(1.)
law.quantile([.05, .5, .95])
law.expect(lambda x: x*x)
draws = law.sample((10, 20), rng=np.random.default_rng(42))

joint = dist.multivariate_normal([0., 0.], [[1., .5], [.5, 1.]])
conditional = joint.condition([1], [2.])
assert np.allclose(conditional.mean(), [1.])
assert np.allclose(conditional.covariance(), [[.75]])
```

## Architecture

- `interfaces/`: `DistributionAdapter`, the checked `Distribution` facade, and
  `MonotoneTransform`. User adapters implement a structural protocol; no
  inheritance, registration or whole-backend implementation is required.
- `builtins/`: validated constructors and public composition tools.
- `_backends/`: private SciPy, NumPy and PyTorch adapters and probability algebra.
  Application code should use the public constructors and law methods.
- `tensor`: the optional PyTorch namespace. Importing the family does not import
  PyTorch or other mathematical families.

Laws describe persistent mathematical objects, so operations live on the law
rather than creating a new numerical problem for every density evaluation.
There is no mutable global backend selection. Composition uses the public law
interface. This family imports no other Convexity family.

## Finite catalog

| Constructors | Supported operations |
|---|---|
| `normal`, `student_t`, `lognormal` | Density/log density, CDF, survival, quantile, support, sampling, mean/variance/std, raw moments, scalar numerical expectations |
| `bernoulli`, `binomial`, `poisson` | The same scalar operations, with mass/log mass in place of density/log density |
| `multivariate_normal`, `multivariate_student_t` | Joint density/log density, joint CDF, sampling, mean, variance, covariance, componentwise raw moments, marginals and coordinate conditioning |
| `empirical` | Scalar/vector masses, joint ECDF, sampling, exact moments/expectations, predicate probabilities/conditioning, arbitrary finite pushforwards; scalar inverse ECDF/support/survival; vector marginals/coordinate conditioning |
| `independent` | Vector product of scalar laws, joint density/CDF, sampling, moments/covariance, marginals and coordinate conditioning, according to component capabilities |
| `mixture` | Weighted density/log density, CDF, survival, support, sampling, moments/expectations, marginals and reweighted coordinate conditioning, according to component capabilities; numerical scalar continuous quantiles |
| `transformed`, `affine`, `exponential` | Scalar continuous change of variables, density/log density, sampling, CDF/survival/quantiles/support where the base supports them; numerical expectations when support is available; affine analytic means/variances/raw moments |
| `tensor.normal`, `student_t`, `lognormal`, `bernoulli`, `binomial`, `poisson`, `multivariate_normal` | Native tensor log density/density, sampling, mean/variance; Gaussian/lognormal CDF/quantile; multivariate Gaussian covariance; pathwise `rsample` when supported |
| `tensor.from_torch`, `independent`, `mixture`, `transformed` | Explicit native PyTorch adapters and composition, retaining the native law's implemented operations |

Each NumPy law exposes `capabilities`, an immutable set of supported operation
names. Unsupported operations raise `NotImplementedError`; there is no hidden
sampling approximation. Vector quantiles, arbitrary continuous conditioning,
general multivariate expectations and discrete mixture quantiles are not
claimed. Singular Gaussian/t laws, mixed continuous/atomic mixtures and product
laws with mixed reference measures are excluded. A finite discrete pushforward
uses `empirical(...).map(...)`, not a continuous Jacobian formula.

## Parameters, shapes and ownership

NumPy laws have one scalar event `()` or one vector event `(d,)`, with no implicit
parameter batches. `normal(loc, scale)` uses a standard deviation. `student_t`
uses degrees of freedom, location and **scale**, not standard deviation.
`lognormal(log_loc, log_scale)` means `exp(Normal(log_loc, log_scale))`.
`multivariate_student_t(df, loc, shape)` is an elliptical t law, with covariance
`shape * df/(df-2)` when `df > 2`. A diagonal t shape does not imply independence.
Vector covariance/shape matrices must be symmetric and positive definite.

`binomial(n, p)` requires a nonnegative integer `n`; probability parameters lie
in `[0,1]`. A Poisson rate may be zero. Degenerate discrete parameter choices
retain their actual singleton support. Continuous scale parameters must be
positive. Empirical values have shape `(n,)` or `(n,d)`; a single vector outcome
therefore has shape `(1,d)`. Weights are finite nonnegative relative masses with
positive total mass. Repeated atoms add mass; zero-weight rows are discarded.
Empirical variance is a population property of the supplied law, not an unbiased
sample estimator. No smoothing or tail extrapolation is performed.

| Operation | Input/output convention |
|---|---|
| Scalar density/CDF | Query shape `S` returns `S` |
| Vector density/CDF | Query shape `S + (d,)` returns `S` |
| `sample(size, rng=...)` | Integer or tuple `size` returns `sample_shape + event_shape`; default `()` draws one event; zero sample dimensions are valid |
| Mean/variance/raw moment | Returns `event_shape`; vector raw moments are componentwise, not cross-moment tensors |
| Covariance | Scalar variance or vector `(d,d)` matrix |
| `marginal(indices)` | Preserves requested order and always returns a vector, including `(1,)` |
| `condition(indices, values)` | Values are a finite vector aligned to indices; remaining coordinates retain original order and vector shape |

Parameters are copied to real float64 arrays and retained read-only. Numeric
strings, complex data, NaNs, invalid sizes and nonfinite parameters are rejected.
Evaluation points may be infinite; quantile probabilities must be finite.
Outputs are read-only NumPy arrays, including 0-D arrays for scalar values; use
`float(value)` for a Python scalar. Undefined or divergent moments retain backend
NaN/Inf semantics. In a vector t law with `1 < df <= 2`, marginal variances are
infinite and off-diagonal covariances are undefined (NaN). With `df <= 1`, means
and covariances are undefined. Representational overflow is not repaired by
inventing finite values.

Callbacks must be deterministic and must not mutate their arguments. An
expectation callback consumes **one event at a time**. Scalar parametric and
transformed expectations require a finite real scalar callback value at each
evaluation. Empirical expectations may return equally shaped scalar, vector or
matrix values. Predicate callbacks must return a single boolean per event.

## Probability, tails and conditioning

`density` means a PDF for continuous laws and an atom's mass for discrete laws.
A vector CDF is `P(X_i <= x_i for all i)`. It is not a vector of marginal CDFs.
`survival(x)` means `P(X > x)` for scalar laws, excluding mass at equality.
SciPy scalar survival uses the backend's stable survival function rather than
subtracting a rounded CDF from one.

Scalar quantiles use the lower inverse CDF in the interior. `quantile(0)` and
`quantile(1)` return support endpoints; unlike SciPy's discrete `ppf(0)`, zero
does not return an impossible value below support. Empirical quantiles do not
interpolate. Continuous mixture quantiles use library bisection of the CDF
threshold and return the left boundary of a plateau within floating precision.

NumPy sampling requires an explicit `numpy.random.Generator`; passing an integer
seed or relying on global state is rejected. Reusing a generator advances its
state. Reconstructing the same generator reproduces draws within a backend
version. SciPy vector Gaussian/t CDFs also require an explicit generator, since
their integration may be randomized. CDF accuracy follows SciPy defaults; no
error certificate is fabricated. Infinite orthant limits reduce to marginals.

`condition(indices, values)` supplies regular conditional Gaussian/t laws using
block solves and Schur complements. It does not divide by the probability of an
exact continuous observation, which is zero. For Student-t laws, conditioning
changes both degrees of freedom and scale. Mixtures update component weights
using the observed marginal density/mass in log space. Zero posterior weights
are discarded. Empirical coordinate conditioning uses exact atom matching and
rejects zero-probability observations. `given(predicate)` restricts an empirical
law to a positive-probability event without dropping coordinates.

```python
atoms = dist.empirical([[0, 1], [0, 3], [1, 4]], weights=[1, 3, 4])
atoms.condition([0], [0]).mean()       # [2.5]
atoms.given(lambda x: x[1] > 1).mean()
atoms.map(lambda x: x[0] + x[1])       # exact scalar pushforward

mixture = dist.mixture([dist.normal(-2, 1), dist.normal(4, 2)], [1, 3])
mixture.mean()                        # 2.5
mixture.variance()                    # 10, includes between-component variation
```

`expect` never infers existence of an integral. SciPy supplies scalar quadrature
or discrete summation; transformed laws integrate in the image coordinate to
avoid overflowing a forward transform before weighting it. Integration options
such as `epsabs`/`epsrel` for continuous laws or `maxcount`/`tolerance` for
discrete laws pass to SciPy. Bounds and conditional flags are excluded because
they would change the law being integrated. Backend warnings remain visible.
The returned value is a numerical approximation, without an invented error
estimate or convergence claim. Empirical expectations are finite weighted sums.

## Transform contracts

`MonotoneTransform` supplies an increasing or decreasing differentiable bijection
from the real line onto an open `image` interval. `forward`, `inverse`, and
`log_abs_derivative` consume and return equally shaped NumPy arrays.
The derivative callback is evaluated at the base coordinate:

```text
x = inverse(y)
log f_Y(y) = log f_X(x) - log |forward'(x)|
```

The caller owns invertibility and the derivative's correctness. Nonpositive
queries to `exponential(base)` return zero density/CDF without evaluating a
logarithm outside its domain. Densities at image endpoints are assigned zero;
endpoint density values do not change a continuous law. Negative affine scales
reverse CDF/quantile orientation correctly. General transformed moments are
obtained through an explicit integrable `expect` callback; no analytic moment
formula is assumed from arbitrary callbacks.

## Differentiable tensor laws

Install `python -m pip install -e '.[autodiff]'`. PyTorch is loaded only when a
tensor constructor or adapter is used. NumPy/SciPy remains the default family.

```python
import torch

loc = torch.tensor(2., dtype=torch.float64, requires_grad=True)
law = dist.tensor.normal(loc, torch.tensor(3., dtype=torch.float64))
loss = -law.log_density(torch.tensor(1., dtype=torch.float64))
loss.backward()                       # gradient reaches loc

with torch.random.fork_rng():
    torch.manual_seed(42)
    draws = law.rsample(100)          # pathwise gradients retained
```

Tensor results stay on their native device with their native dtype. Native
parameter broadcasting is explicit through `batch_shape`; sample shapes are
`sample_shape + batch_shape + event_shape`. `tensor.independent` reinterprets
trailing batch axes as event axes. `tensor.mixture` uses PyTorch's same-family
mixture convention, where the final component batch axis indexes components.
`tensor.transformed` accepts native PyTorch transforms.

`sample` uses native non-pathwise sampling; `rsample` is available only when
PyTorch declares it. Neither ordinary discrete draws nor categorical mixture
selection become pathwise differentiable. Tensor RNG control is native
`torch.manual_seed`/`torch.random.fork_rng`, not a NumPy generator. Native support
validation propagates; out-of-support tensor log-density queries can raise.
No tensor-to-NumPy conversion or silent backend fallback occurs. Parameters stay
caller-owned; rebuild the law after updates rather than mutating parameters
while a graph is active. Arbitrary native laws may raise `NotImplementedError`
for moments or other optional operations. `from_torch` requires explicit flags
to advertise a native CDF/quantile implementation.

## A custom law

An adapter receives validated inputs and returns arrays of the documented
shape. `Distribution` checks result shapes, CDF bounds and sampling outputs.
Mathematical correctness and truthful capability declarations belong to the
adapter. This minimal law can already participate in a discrete mixture:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PointMass:
    event_shape = ()
    measure: dist.Measure = "discrete"
    capabilities = frozenset({"log_density", "sample"})

    def evaluate(self, operation, *args, **kwargs):
        if operation == "sample":
            return np.full(args[0], 3.)
        return np.where(args[0] == 3., 0., -np.inf)

custom = dist.Distribution(PointMass())
combined = dist.mixture([custom, dist.empirical([0., 1.])])
combined.density([0., 1., 3.])         # [.25, .25, .5]
```

## Verification and library references

```bash
python -m pytest -q tests/math/distributions
python -m pyright --pythonpath "$(command -v python)"
python examples/distributions.py
```

Tests cover known moments, library agreement, probability identities, conditional
density factorizations, posterior mixture weights, scalar/vector/empty sample
shapes, exact empirical ties and quantile boundaries, parameter ownership,
invalid inputs, custom adapters, lazy imports and PyTorch parameter/pathwise
gradients. Examples use only supplied or generated mathematical values.

References: [SciPy scalar distributions](https://docs.scipy.org/doc/scipy/reference/stats.html),
[scalar expectations and their limitations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.expect.html),
[multivariate Gaussian](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.multivariate_normal.html),
[multivariate Student-t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html),
[PyTorch distributions and reparameterization](https://docs.pytorch.org/docs/2.14/distributions.html).
