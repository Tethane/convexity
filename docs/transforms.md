# Transforms, decomposition and factor geometry

Family 4 supplies coordinate maps, sequence operators, decompositions and
subspaces in `convexity.math.transforms`. Inputs are arrays and mathematical
specifications. The family imports no data, research, forecasting, trading or
other mathematical family. It needs no market data.

## Finite acceptance catalog

| Mathematical task | Public objects and implementation |
|---|---|
| Scaling | `Standardize`, `RobustScale`, `MinMaxScale`; scikit-learn fits with inspectable, independent `FittedScaling` state |
| Affine coordinates | `AffineTransform`; explicit location, encoder, decoder and exact-inverse capability |
| Differences | `Difference`, `DifferenceResult`; ordinary, higher-order and seasonal differences with boundary conditions and continuation |
| Lag construction | `LagFeatures`, `FeatureMatrix`; explicit positional alignment and ordered lag/feature columns |
| Periodic features | `FourierFeatures`, `SeasonalFeatures`; statsmodels deterministic terms with explicit phase and baseline conventions |
| Principal components | `PCA`, `PCAResult`; scikit-learn full SVD, retained axes, singular values, explained variance and residuals |
| Whitening | `Whitening`; PCA or ZCA, explicit covariance divisor and rejection of zero-variance directions |
| Factor models | `GaussianFactorModel`, `FactorAnalysis`, `FactorAnalysisResult`; supplied or fitted Gaussian loadings and diagonal noise, posterior scores and covariance |
| Projection geometry | `Subspace`, `subspace`; SciPy SVD basis extraction, coordinates, orthogonal projections and residuals |
| Subspace alignment | `principal_angles`, `align_subspaces`, `Alignment`; SciPy principal angles and orthogonal Procrustes |
| Composition and extensibility | `Transform`, `TransformFitter`, `fit`, `Pipeline`, `FittedPipeline`; user-written maps without a registry or subclass requirement |

This catalog covers static linear factor geometry. Dynamic factors and seasonal
forecasting models belong to later families. Missing-value imputation, irregular
calendar handling, nonlinear manifold learning, ICA, NMF and streaming fits are
outside this catalog. No general covariance estimator or prediction stack is
introduced here.

Run the [problem-book script](../examples/transforms.py), or open the
[worked notebook](../notebooks/transforms.ipynb). Both use generated inputs,
analytic identities and numerical assertions.

## Three layers

- `interfaces/` defines first-class mathematical objects, array contracts,
  fitted state and extension protocols.
- `_backends/` adapts scikit-learn fitting, statsmodels periodic terms and SciPy
  subspace/linear algebra. Small finite-difference and lag algebra lives here too.
- `builtins/` exposes reusable specifications and convenient functions. All
  public names are re-exported by `convexity.math.transforms`.

NumPy loads with the family; SciPy, statsmodels and scikit-learn load on use.
Fitted objects retain copied numeric state, not mutable estimator instances.
This family has no application metadata or time-cutoff policy.

## Training, shapes and ownership

```python
import numpy as np
from convexity.math import transforms as tx

rng = np.random.default_rng(42)
x = rng.normal(size=(240, 4)) @ np.diag([4., 2., 1., .3])
fitted = tx.Pipeline((tx.Standardize(), tx.PCA(2))).fit(x[:180])
scores = fitted.transform(x[180:])
approximation = fitted.reconstruct(scores)
residuals = fitted.residuals(x[180:])
assert np.allclose(approximation + residuals, x[180:])
```

| Operation | Shape convention |
|---|---|
| Fit a scaling/decomposition tool | `(n,d)`; rows are observations, columns are coordinates |
| Apply a fitted map | `(...,d)` → `(...,k)`; vectors, matrices and batches all work |
| Scaling state | `location`, `scale`: `(d,)` |
| Affine state | `location`: `(d,)`, `encoder`: `(d,k)`, `decoder`: `(k,d)` |
| PCA axes / factor loadings | Columns of a `(d,k)` matrix |
| Sequence inputs | `(n,)` or `(n,d)`; first axis is ordered position |
| Lag/periodic feature result | `values`: `(m,p)`, `rows`: `(m,)`, `columns`: `p` unique names |
| Subspace basis | `(d,k)` orthonormal columns; `k=0` is a valid zero subspace |

Inputs must be finite real numeric arrays. Empty observations, NaNs, infinities,
complex values, numeric strings and boolean arrays are rejected. A univariate
fit takes `(n,1)`, not `(n,)`. Constructors copy arrays into read-only float64
storage; row indices use read-only integer storage. Outputs are also read-only.
Feature identity is positional: callers must preserve coordinate order.

Fitters are immutable specifications; each `.fit(training)` returns new state.
Transformation never refits. Callers select their training rows explicitly.
Pipelines fit each step on the preceding step's transformed training rows and
use the frozen steps on later inputs. Row-dropping sequence operators are used
separately, with their returned alignment, rather than hidden inside a pipeline.

## Exact inverses and reconstruction

An affine object states both maps:

```text
z = (x - location) @ encoder
x_reconstructed = z @ decoder + location
```

`is_invertible` checks a square inverse pair numerically. `inverse_transform`
requires that capability. `reconstruct` decodes coordinates even when information
was discarded. `residuals(x)` is `x - reconstruct(transform(x))`.

Reduced PCA reconstructs a projection and does not have an exact inverse over
the original ambient space. Full PCA can invert even when some training
directions have zero variance; whitening those directions is undefined.
Gaussian factor posterior means also discard information. A pipeline has an
exact inverse only when every stage supports one.

## Scaling

`standardize(x, center=True, scale=True, ddof=0)` uses training means and standard
deviations. Set `ddof=1` for the sample divisor. Centering and scaling are
independent choices; disabling centering does not change variance to a raw
second moment. Constant/numerically constant columns retain scikit-learn's
unit scale, so later deviations in those columns remain observable and invertible.

`robust_scale` centers by the median and scales by the interquantile range
(default 25th–75th percentiles). With `unit_variance=True`, scikit-learn adjusts
the scale so a normal reference has unit variance asymptotically. This does not
force an arbitrary sample's variance to one. Quantiles are strictly inside
`(0,100)`. The backend handles zero interquantile ranges without division by zero.

`minmax_scale(x, lower=0, upper=1)` fits an affine range map. Constant training
columns map to the lower endpoint with the backend's nonzero scale convention.
Future values can leave the fitted range: clipping would destroy invertibility.
`FittedScaling.location` is the location subtracted by the final map, including
the offset induced by a nonzero lower endpoint.

## Sequence algebra and alignment

```python
t = np.arange(80.)
series = .02 * t**2 + np.sin(2*np.pi*t/12)
delta = tx.difference(series[:60], lag=12, order=2)
assert np.allclose(delta.inverse_transform(), series[:60])

whole = tx.difference(series, lag=12, order=2)
future_differences = whole.values[60 - whole.row_offset:]
future_levels = delta.inverse_transform(future_differences, continuation=True)
assert np.allclose(future_levels, series[60:])

lagged = tx.lag_features(series, lags=(0, 1, 12))
aligned_response = series[lagged.rows]
```

`Difference(lag=s, order=q)` applies `(1-L**s)**q`. It discards the first `s*q`
rows and records the first and last `s` values at every intermediate difference
level. `initial` and `trailing` are ordered from original levels to the highest
intermediate differences. Ordinary inversion uses initial blocks and includes
them in the restored sequence. Continuation integrates **explicit future
differences** with trailing blocks and returns only new rows. The result is
immutable; repeated continuation calls start at the same boundary. Integration
does not supply a forecast or infer a frequency. High-order inversion can
amplify floating-point error, as repeated integration naturally does.

Lags are unique nonnegative integers in caller order. Lag zero means the current
row; positive lags read previous rows. Columns are lag-major, then feature-major.
Only complete rows are returned, starting at `max(lags)`. No padding, imputation
or future observations are used. If operators are chained, row indices are
relative to each supplied input; map them back explicitly (for example,
`delta.rows[tx.lag_features(delta.values).rows]`).

## Seasonal and Fourier designs

```python
seasonal = tx.seasonal_features(30, period=12, start=60, drop_first=True)
harmonics = tx.fourier_features(30, period=12, order=3, start=60)
```

`start` is the nonnegative integer position of the first row. Supply it when
continuing a design so phase does not restart. Periods are measured in row steps,
not inferred from dates. Seasonal columns indicate `t % period`; with
`drop_first=True`, season zero is the baseline. No intercept is inserted.

Fourier columns are `sin(2*pi*j*t/period), cos(2*pi*j*t/period)` for harmonic
`j=1,...,order`, interleaved sine then cosine. Positive real periods are allowed.
`order < period/2` prevents aliased harmonics and the degenerate Nyquist sine on
an integer grid. Short windows or concatenated designs can still be rank
deficient; columns are never silently removed.

## PCA and whitening

`pca(x, n_components=None, whiten=False, ddof=1)` centers and applies full SVD.
No standardization is implicit. The default retains `min(n,d)` axes; explicit
counts must be positive and at most that value. At least two rows and nonzero
total training variation are required. The result exposes column `components`,
`location`, `singular_values`, `explained_variance` and `explained_variance_ratio`.
Variance is `singular_values**2 / (n-ddof)`; ratios use total centered variance,
including omitted directions.

For retained orthonormal axes `Q`, scores are `(x-mean) @ Q`. Their decoded
reconstruction is `scores @ Q.T + mean`. Residuals are orthogonal to `Q`, and
training reconstruction error equals the sum of omitted squared singular values.
Eigenvector signs and bases within tied eigenspaces are not uniquely identified;
compare subspaces or use explicit alignment. No canonical tie-breaking claim
is made.

`whitening(x, method='pca', n_components=k, ddof=1)` divides each retained score
by its training standard deviation. The resulting covariance, with the same
divisor, is identity. `method='zca'` (default) rotates full whitened scores back
into the original axes; its encoder is symmetric. ZCA requires full column rank
and therefore more observations than dimensions after centering. PCA whitening
can explicitly select a lower rank.

Variances at or below `eps * max(n,d) * total_variance` are rejected for whitening.
There is no hidden ridge or eigenvalue floor: those would change the whitening
identity. Unwhitened PCA retains zero-variance directions when requested.

## Gaussian factor geometry

```python
model = tx.GaussianFactorModel(
    location=[0., 0., 0.],
    loadings=[[1.], [2.], [-1.]],
    noise_variance=[.2, .4, .3])
scores = model.transform([[1., 2., -1.]])
signal = model.reconstruct(scores)
fitted_factors = tx.factor_analysis(x, n_components=2, rotation='varimax')
```

The model is `x = mu + B f + epsilon`, with independent `f ~ N(0,I)` and
`epsilon ~ N(0,diag(psi))`. Noise variances must be strictly positive. It exposes
the implied observation covariance `B @ B.T + diag(psi)`. This is a consequence
of the supplied factor model, not a general covariance-estimation API.

Let `V = (I + B.T @ diag(1/psi) @ B)^-1`. `posterior_covariance` is `V` and
`transform(x)` returns posterior means `(x-mu) @ diag(1/psi) @ B @ V` using SciPy
positive-definite solves. `reconstruct(scores)` produces the common component
`mu + scores @ B.T`. The remainder is not generally orthogonal to `B`, and its
sample covariance is not automatically the model's diagonal noise variance.

`FactorAnalysis` uses scikit-learn's deterministic LAPACK fit, with optional
orthogonal varimax/quartimax rotation. It stores fitted means, loadings, noise
variances, the summed training log-likelihood history and convergence diagnostics.
The default tolerance is `1e-6`, maximum iterations 1000. Constant columns and
counts greater than `min(n-1,d)` are rejected. Convergence warnings raise
`ConvergenceError`, retaining `.result`; `check_convergence=False` returns the
failed result for inspection. Numerical convergence does not prove a unique
factorization or a global likelihood maximum. Orthogonal factor rotations leave
the implied observation covariance unchanged.

## Projections and alignment

```python
space = tx.subspace([[1., 2.], [0., 0.], [1., 2.]])  # dependent columns are allowed
projection = space.project([[2., 3., 4.]])
residual = space.residuals([[2., 3., 4.]])

a = tx.Subspace(np.eye(3)[:, :2])
b = tx.Subspace(a.basis[:, ::-1])
alignment = tx.align_subspaces(a, b)
assert np.allclose(alignment.aligned.basis, b.basis)
```

`subspace(A, rcond=None)` uses SciPy's SVD orthonormal basis for the range of `A`.
The default relative singular-value threshold is SciPy's machine-precision
threshold; an explicit `rcond` must lie in `[0,1)`. `Subspace(Q)` directly accepts
already orthonormal columns. The zero subspace is supported. Geometry is
Euclidean in the supplied coordinates; scaling changes that geometry. To remove
a sample-space design from an `(n,d)` matrix, project its transpose onto the
column space of the `(n,p)` design and transpose back.

`principal_angles(a,b)` returns `min(rank(a),rank(b))` angles in descending order,
in radians. Empty spaces give an empty array. Unequal-rank angles alone do not
measure unmatched directions. `align_subspaces` requires equal ranks and solves
`min_R ||Q_a R - Q_b||_F`, with `R.T R=I`; reflections are allowed. It reports the
rotation, aligned basis, Frobenius mismatch and principal angles. Rotation
cannot remove a difference between spanned spaces.

Scores rotate by the same `R`: `(scores @ R) @ (Q_a @ R).T = scores @ Q_a.T`.
`Alignment.transform` and `inverse_transform` implement this change of basis.

## Extending and checking the family

```python
class Center:
    def fit(self, samples):
        dimension = samples.shape[1]
        return tx.AffineTransform(samples.mean(axis=0),
                                  np.eye(dimension), np.eye(dimension))

custom = tx.fit(x, Center())
composed = tx.Pipeline((Center(), tx.PCA(2))).fit(x)
```

A fitter implements `.fit(samples)` and returns a `Transform` with
`input_dimension`, `output_dimension` and `.transform(values)`. `fit` checks input
dimension and the returned training shape. `checked_transform` and pipelines
also validate dimensions, finiteness and preservation of leading axes. To support
decoding, supply `.reconstruct`; an exact inverse additionally needs
`is_invertible` and `.inverse_transform`. Custom objects own their mathematical
claims, state ownership and convergence behavior.

```bash
python -m pytest -q tests/math/transforms
python -m pyright --pythonpath "$(command -v python)"
python examples/transforms.py
```

Tests cover scaling formulas, generated-data PCA spectra, optimal reconstruction
error, whitening covariance, Gaussian conditioning identities, backend factor
scores, differencing round trips/continuation, positional alignment, periodic
phase continuity, projection identities, rank deficiency, alignment invariance,
ownership, malformed inputs, convergence failures and custom tools.
The family suite also passes with the declared dependency floors: NumPy 1.26.4,
SciPy 1.15.3, scikit-learn 1.6.1 and statsmodels 0.14.4.

Backend references: [scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html),
[FactorAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html),
[statsmodels Fourier](https://www.statsmodels.org/stable/generated/statsmodels.tsa.deterministic.Fourier.html),
[Seasonality](https://www.statsmodels.org/stable/generated/statsmodels.tsa.deterministic.Seasonality.html),
[SciPy orth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orth.html),
[principal angles](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html),
[orthogonal Procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html).
