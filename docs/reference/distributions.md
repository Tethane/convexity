# `convexity.math.distributions`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Probability distributions and conditional laws, independent of applications.

    from convexity.math import distributions as dist
    law = dist.normal(loc=2., scale=3.)
    law.quantile([.05, .5, .95])

SciPy and NumPy laws return checked arrays. Optional tensor laws preserve native
PyTorch tensors and autograd graphs. No data access, estimation or model fitting.

## `Distribution`

```python
Distribution(_adapter: 'DistributionAdapter') -> None
```

One NumPy probability law. Unsupported operations raise NotImplementedError.

density means PDF for continuous laws and PMF for discrete laws. Vector CDF
means P(X_i <= x_i for every i), not a vector of marginal CDFs. Outputs are
read-only arrays, including 0-D arrays for scalar outputs. NaN/Inf moments
retain the backend's undefined/divergent moment semantics.

### `Distribution.cdf`

```python
cdf(self, x: 'ArrayLike', *, rng: 'np.random.Generator | None' = None) -> 'FloatArray'
```

### `Distribution.condition`

```python
condition(self, indices: 'ArrayLike', values: 'ArrayLike') -> 'Distribution'
```

Law of the remaining coordinates given X[indices]=values.

Remaining coordinates retain original order, including a one-coordinate
vector. Continuous Gaussian/t laws use regular conditional densities;
empirical laws require positive mass at exact observed values.

### `Distribution.covariance`

```python
covariance(self) -> 'FloatArray'
```

Scalar variance or vector (d,d) covariance, never an estimated matrix.

### `Distribution.density`

```python
density(self, x: 'ArrayLike') -> 'FloatArray'
```

### `Distribution.expect`

```python
expect(self, function: 'Callable[[float], ArrayLike] | Callable[[FloatArray], ArrayLike]', **options: 'object') -> 'FloatArray'
```

Expectation of a one-event callback; no implicit sampling fallback.

Scalar SciPy laws integrate a scalar callback (options go to SciPy
expect). Empirical laws sum scalar/vector/matrix callback outputs exactly.
The caller must establish integrability; quadrature cannot prove it.

### `Distribution.given`

```python
given(self, event: 'Callable[[FloatArray], bool | BoolArray]') -> 'Distribution'
```

Restrict a finite empirical law to a positive-probability predicate.

### `Distribution.log_density`

```python
log_density(self, x: 'ArrayLike') -> 'FloatArray'
```

### `Distribution.map`

```python
map(self, function: 'Callable[[FloatArray], ArrayLike]') -> 'Distribution'
```

Exact finite pushforward; arbitrary scalar/vector maps of empirical laws.

### `Distribution.marginal`

```python
marginal(self, indices: 'ArrayLike') -> 'Distribution'
```

### `Distribution.mean`

```python
mean(self) -> 'FloatArray'
```

### `Distribution.moment`

```python
moment(self, order: 'int') -> 'FloatArray'
```

Raw E[X**order], componentwise for vector laws where supported.

### `Distribution.probability`

```python
probability(self, event: 'Callable[[FloatArray], bool | BoolArray]') -> 'FloatArray'
```

Exact predicate probability for finite empirical laws.

### `Distribution.quantile`

```python
quantile(self, probability: 'ArrayLike') -> 'FloatArray'
```

### `Distribution.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator') -> 'FloatArray'
```

Return size + event_shape draws using only the supplied generator.

### `Distribution.std`

```python
std(self) -> 'FloatArray'
```

### `Distribution.support`

```python
support(self) -> 'tuple[float, float]'
```

Scalar support endpoints; quantile(0/1) uses these endpoints.

### `Distribution.survival`

```python
survival(self, x: 'ArrayLike') -> 'FloatArray'
```

P(X > x) for scalar laws; SciPy laws use the stable backend SF.

### `Distribution.variance`

```python
variance(self) -> 'FloatArray'
```

## `DistributionAdapter`

Implement only advertised operations; no inheritance or registry required.

Events are scalar () or vector (d,). There are no parameter batch axes.
evaluate receives already checked inputs; see Distribution for output shapes.
Adapters must retain immutable parameters and not mutate callback arguments.

### `DistributionAdapter.evaluate`

```python
evaluate(self, operation: 'str', *args: 'object', **kwargs: 'object') -> 'ArrayLike | Distribution'
```

## `Measure`

```python
Measure = Literal['continuous', 'discrete']
```

Permitted values for this public type alias.

## `MonotoneTransform`

```python
MonotoneTransform(forward: 'Callable[[FloatArray], ArrayLike]', inverse: 'Callable[[FloatArray], ArrayLike]', log_abs_derivative: 'Callable[[FloatArray], ArrayLike]', increasing: 'bool' = True, image: 'tuple[float, float]' = (-inf, inf)) -> None
```

Differentiable bijection from R onto the open interval image.

All callbacks accept/return same-shaped NumPy arrays. log_abs_derivative is
log(|d forward(x)/dx|), evaluated at the BASE point x, not at forward(x).
Callers own monotonicity, inverse correctness, and the nonzero Jacobian.

## `normal`

```python
normal(loc: 'ScalarLike' = 0.0, scale: 'ScalarLike' = 1.0) -> 'Distribution'
```

Gaussian with location loc and strictly positive standard deviation scale.

## `student_t`

```python
student_t(df: 'float', loc: 'float' = 0.0, scale: 'float' = 1.0) -> 'Distribution'
```

Student-t; scale is not its standard deviation. df must be finite and > 0.

## `lognormal`

```python
lognormal(log_loc: 'float' = 0.0, log_scale: 'float' = 1.0) -> 'Distribution'
```

exp(Normal(log_loc, log_scale)); both parameters describe log(X).

## `bernoulli`

```python
bernoulli(p: 'float') -> 'Distribution'
```

Return a scalar Bernoulli law with success probability p in [0,1].

Support is {0,1}; density denotes probability mass and variance is p*(1-p).

## `binomial`

```python
binomial(n: 'int', p: 'float') -> 'Distribution'
```

Return the count of successes in n independent Bernoulli(p) trials.

n is a nonnegative integer, p in [0,1]; support is the integers from 0 to n.

## `poisson`

```python
poisson(rate: 'float') -> 'Distribution'
```

Return a scalar Poisson count law with finite nonnegative rate.

Mean and variance equal rate; density is probability mass, not continuous density.

## `multivariate_normal`

```python
multivariate_normal(mean: 'ArrayLike', covariance: 'ArrayLike') -> 'Distribution'
```

Gaussian with a positive-definite covariance; singular laws are excluded.

## `multivariate_student_t`

```python
multivariate_student_t(df: 'float', loc: 'ArrayLike', shape: 'ArrayLike') -> 'Distribution'
```

Elliptical Student-t with positive-definite shape (not covariance).

For df > 2, covariance = shape * df / (df - 2). This is a dependent
multivariate t, including when shape is diagonal, not independent t marginals.

## `empirical`

```python
empirical(values: 'ArrayLike', weights: 'ArrayLike | None' = None) -> 'Distribution'
```

Finite law on (n,) scalar or (n,d) vector values; repeated atoms add mass.

Weights are nonnegative relative masses. Zero-weight rows are discarded.
Quantiles are inverse ECDFs with no interpolation or inferred tails.

## `independent`

```python
independent(components: 'Sequence[Distribution]') -> 'Distribution'
```

Vector law of independent scalar components, all discrete or all continuous.

## `mixture`

```python
mixture(components: 'Sequence[Distribution]', weights: 'ArrayLike | None' = None) -> 'Distribution'
```

Finite mixture; component shapes and reference measures must agree.

Relative weights are normalized and zero-weight components are discarded.
Mixing a continuous density with atom masses is intentionally unsupported.

## `transformed`

```python
transformed(base: 'Distribution', transform: 'MonotoneTransform') -> 'Distribution'
```

Push a scalar base law through a monotone invertible transform.

Density uses the inverse Jacobian; decreasing maps reverse probability order.
Only capabilities justified by the base law and transform are exposed.

## `affine`

```python
affine(base: 'Distribution', shift: 'float' = 0.0, scale: 'float' = 1.0) -> 'Distribution'
```

Law of shift + scale*X for a continuous scalar X; negative scales allowed.

## `exponential`

```python
exponential(base: 'Distribution') -> 'Distribution'
```

Law of exp(X). Nonpositive density queries return zero, not log-domain NaNs.
