# `convexity.math.transforms`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Transforms, decomposition and factor geometry on supplied arrays.

    from convexity.math import transforms as tx
    fitted = tx.Pipeline((tx.Standardize(), tx.PCA(2))).fit(training_matrix)
    scores = fitted.transform(new_matrix)
    approximation = fitted.reconstruct(scores)

No calendars, data acquisition, forecasting or trading dependencies.

## `Difference`

```python
Difference(lag: 'int' = 1, order: 'int' = 1) -> None
```

Apply repeated lag differences; retained boundary data supports reconstruction.

## `FactorAnalysis`

```python
FactorAnalysis(n_components: 'int', rotation: 'str | None' = None, tolerance: 'float' = 1e-06, maxiter: 'int' = 1000, check_convergence: 'bool' = True) -> None
```

Fit latent factors with diagonal residual variance; n_components sets latent dimension and rotation is explicit.

### `FactorAnalysis.fit`

```python
fit(self, samples: 'ArrayLike') -> 'FactorAnalysisResult'
```

## `FittedPipeline`

```python
FittedPipeline(steps: 'Sequence[Transform]') -> None
```

Ordered fitted transforms. Coordinate order follows each transform; inverse requires every stage to support it.

### `FittedPipeline.inverse_transform`

```python
inverse_transform(self, coordinates)
```

### `FittedPipeline.reconstruct`

```python
reconstruct(self, coordinates)
```

### `FittedPipeline.residuals`

```python
residuals(self, values: 'ArrayLike') -> 'FloatArray'
```

### `FittedPipeline.transform`

```python
transform(self, values: 'ArrayLike')
```

## `FourierFeatures`

```python
FourierFeatures(period: 'float', order: 'int' = 1) -> None
```

sin/cos pairs on integer t, with period measured in row steps.

Require order < period/2: this avoids an identically zero sine column at
the Nyquist frequency. No intercept, rank pruning or phase reset is implicit.

## `LagFeatures`

```python
LagFeatures(lags: 'tuple[int, ...]' = (1,)) -> None
```

Nonnegative lags in caller order, with lag-major, then feature-major columns.

## `MinMaxScale`

```python
MinMaxScale(lower: 'float' = 0.0, upper: 'float' = 1.0) -> None
```

Fit an affine per-feature map to a requested output range using training extrema.

### `MinMaxScale.fit`

```python
fit(self, samples: 'ArrayLike') -> 'FittedScaling'
```

## `PCA`

```python
PCA(n_components: 'int | None' = None, whiten: 'bool' = False, ddof: 'int' = 1) -> None
```

Fit principal component axes from observation rows, with optional whitening and explicit covariance ddof.

### `PCA.fit`

```python
fit(self, samples: 'ArrayLike') -> 'PCAResult'
```

## `Pipeline`

```python
Pipeline(steps: 'Sequence[TransformFitter]') -> None
```

Fit each step only on the preceding step's transformed training rows.

### `Pipeline.fit`

```python
fit(self, samples: 'ArrayLike') -> 'FittedPipeline'
```

## `RobustScale`

```python
RobustScale(center: 'bool' = True, scale: 'bool' = True, quantile_range: 'tuple[float, float]' = (25.0, 75.0), unit_variance: 'bool' = False) -> None
```

Fit per-feature centering/scaling using medians and an explicit percentile range.

### `RobustScale.fit`

```python
fit(self, samples: 'ArrayLike') -> 'FittedScaling'
```

## `SeasonalFeatures`

```python
SeasonalFeatures(period: 'int', drop_first: 'bool' = False) -> None
```

Encode a supplied time coordinate using periodic features with explicit periods and harmonics.

## `Standardize`

```python
Standardize(center: 'bool' = True, scale: 'bool' = True, ddof: 'int' = 0) -> None
```

Fit per-feature mean and standard-deviation scaling; ddof selects the variance denominator.

### `Standardize.fit`

```python
fit(self, samples: 'ArrayLike') -> 'FittedScaling'
```

## `Whitening`

```python
Whitening(method: 'str' = 'zca', n_components: 'int | None' = None, ddof: 'int' = 1) -> None
```

PCA whitening may reduce rank; ZCA requires the full nonsingular basis.

### `Whitening.fit`

```python
fit(self, samples: 'ArrayLike') -> 'AffineTransform | PCAResult'
```

## `align_subspaces`

```python
align_subspaces(source: 'Subspace', target: 'Subspace') -> 'Alignment'
```

Solve min_R ||Q_source R - Q_target||_F with R.T R = I.

Reflections are allowed. Aligning the basis rotates scores with the same R
so that scores @ basis.T remains unchanged. Spanned spaces never change.

## `difference`

```python
difference(values: 'ArrayLike', *, lag: 'int' = 1, order: 'int' = 1) -> 'DifferenceResult'
```

Difference a time-major array using positive row lag and order.

Return DifferenceResult with differenced values and reconstruction boundary data.
The first lag*order rows are removed; this does not rescale by elapsed time.

## `factor_analysis`

```python
factor_analysis(samples: 'ArrayLike', n_components: 'int', *, rotation: 'str | None' = None, tolerance: 'float' = 1e-06, maxiter: 'int' = 1000, check_convergence: 'bool' = True) -> 'FactorAnalysisResult'
```

Fit a latent Gaussian factor model to samples (observation,feature).

Return fitted loadings, noise and transform; n_components selects latent dimension.
Rotation changes coordinates, not the represented covariance.

## `fourier_features`

```python
fourier_features(n: 'int', period: 'float', *, order: 'int' = 1, start: 'int' = 0) -> 'FeatureMatrix'
```

Return periodic sine/cosine features for n row indices starting at start.

Period is measured in observations; order sets harmonic count.

## `lag_features`

```python
lag_features(values: 'ArrayLike', lags: 'Sequence[int]' = (1,)) -> 'FeatureMatrix'
```

Return lagged feature columns and their retained source-row indices.

Input is time-major; lags are nonnegative row offsets. Leading unavailable rows
are discarded, preventing future observations from entering a lag feature.

## `minmax_scale`

```python
minmax_scale(samples: 'ArrayLike', *, lower: 'float' = 0.0, upper: 'float' = 1.0) -> 'FittedScaling'
```

Fit coordinatewise affine scaling of samples (observation,feature).

Training minima/maxima map to lower/upper; later values may fall outside that range.

## `pca`

```python
pca(samples: 'ArrayLike', n_components: 'int | None' = None, *, whiten: 'bool' = False, ddof: 'int' = 1) -> 'PCAResult'
```

Fit principal components to samples (observation,feature).

Columns are original coordinates; n_components selects retained dimension.
Whitening uses the supplied ddof; discarded components cannot be reconstructed.

## `principal_angles`

```python
principal_angles(source: 'Subspace', target: 'Subspace') -> 'FloatArray'
```

min(rank(source),rank(target)) angles, descending, in radians.

## `robust_scale`

```python
robust_scale(samples: 'ArrayLike', *, center: 'bool' = True, scale: 'bool' = True, quantile_range: 'tuple[float, float]' = (25.0, 75.0), unit_variance: 'bool' = False) -> 'FittedScaling'
```

Fit coordinatewise median/quantile-range scaling to sample rows.

quantile_range uses percentages; unit_variance applies the normal-reference correction.

## `seasonal_features`

```python
seasonal_features(n: 'int', period: 'int', *, start: 'int' = 0, drop_first: 'bool' = False) -> 'FeatureMatrix'
```

Return periodic indicator features for n observation indices.

Period counts rows; drop_first removes the reference category for full-rank regression.

## `standardize`

```python
standardize(samples: 'ArrayLike', *, center: 'bool' = True, scale: 'bool' = True, ddof: 'int' = 0) -> 'FittedScaling'
```

Fit coordinatewise mean/standard-deviation scaling to sample rows.

ddof determines the variance divisor; center/scale independently select operations.

## `subspace`

```python
subspace(spanning_columns: 'ArrayLike', *, rcond: 'float | None' = None) -> 'Subspace'
```

SVD range basis. Retain singular values > rcond * largest singular value.

## `whitening`

```python
whitening(samples: 'ArrayLike', *, method: 'str' = 'zca', n_components: 'int | None' = None, ddof: 'int' = 1) -> 'AffineTransform'
```

Fit a PCA or ZCA whitening map to samples (observation,feature).

ddof fixes the covariance normalization; rank-deficient directions require
explicit dimensional reduction rather than infinite inverse variance.

## `AffineTransform`

```python
AffineTransform(location: 'ArrayLike', encoder: 'ArrayLike', decoder: 'ArrayLike') -> 'None'
```

z = (x - location) @ encoder; reconstruction = z @ decoder + location.

encoder @ decoder need not be the identity. Only a square inverse pair
supports inverse_transform; dimensional reduction uses reconstruct.

### `AffineTransform.inverse_transform`

```python
inverse_transform(self, coordinates)
```

### `AffineTransform.reconstruct`

```python
reconstruct(self, coordinates: 'ArrayLike') -> 'FloatArray'
```

### `AffineTransform.residuals`

```python
residuals(self, values: 'ArrayLike') -> 'FloatArray'
```

### `AffineTransform.transform`

```python
transform(self, values: 'ArrayLike') -> 'FloatArray'
```

## `Alignment`

```python
Alignment(rotation: 'ArrayLike', aligned: 'Subspace', error: 'float', angles: 'ArrayLike') -> 'None'
```

source.basis @ rotation approximates target.basis, without translation/scaling.

### `Alignment.inverse_transform`

```python
inverse_transform(self, coordinates: 'ArrayLike') -> 'FloatArray'
```

### `Alignment.transform`

```python
transform(self, coordinates: 'ArrayLike') -> 'FloatArray'
```

## `ConvergenceError`

```python
ConvergenceError(result: 'FactorAnalysisResult') -> 'None'
```

Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics.

## `Diagnostics`

```python
Diagnostics(backend: 'str' = 'user', converged: 'bool' = True, message: 'str' = '', iterations: 'int | None' = None) -> None
```

Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable.

## `DifferenceResult`

```python
DifferenceResult(values: 'ArrayLike', initial: 'Sequence[ArrayLike]', trailing: 'Sequence[ArrayLike]', lag: 'int') -> 'None'
```

D_lag**order x and leading/trailing seeds for every integration level.

### `DifferenceResult.inverse_transform`

```python
inverse_transform(self, values: 'ArrayLike | None' = None, *, continuation: 'bool' = False) -> 'FloatArray'
```

Restore a sequence, or integrate future differences using trailing seeds.

Ordinary inversion includes initial rows. Continuation returns only
new rows and never mutates the stored boundary conditions.

## `FactorAnalysisResult`

```python
FactorAnalysisResult(location: 'ArrayLike', loadings: 'ArrayLike', noise_variance: 'ArrayLike', *, diagnostics: 'Diagnostics', log_likelihood: 'ArrayLike', nobs: 'int') -> 'None'
```

Fitted factor model with loadings, residual variances and diagnostics in training feature order.

### `FactorAnalysisResult.reconstruct`

```python
reconstruct(self, factors: 'ArrayLike') -> 'FloatArray'
```

### `FactorAnalysisResult.residuals`

```python
residuals(self, values: 'ArrayLike') -> 'FloatArray'
```

### `FactorAnalysisResult.transform`

```python
transform(self, values: 'ArrayLike') -> 'FloatArray'
```

## `FeatureMatrix`

```python
FeatureMatrix(values: 'ArrayLike', rows: 'ArrayLike', columns: 'tuple[str, ...]') -> 'None'
```

Constructed feature rows with retained column names and row alignment.

## `FittedScaling`

```python
FittedScaling(location: 'ArrayLike', scale: 'ArrayLike', *, diagnostics: 'Diagnostics | None' = None) -> 'None'
```

Fitted affine coordinate transform with its learned location and forward/inverse matrices.

### `FittedScaling.inverse_transform`

```python
inverse_transform(self, coordinates)
```

### `FittedScaling.reconstruct`

```python
reconstruct(self, coordinates: 'ArrayLike') -> 'FloatArray'
```

### `FittedScaling.residuals`

```python
residuals(self, values: 'ArrayLike') -> 'FloatArray'
```

### `FittedScaling.transform`

```python
transform(self, values: 'ArrayLike') -> 'FloatArray'
```

## `GaussianFactorModel`

```python
GaussianFactorModel(location: 'ArrayLike', loadings: 'ArrayLike', noise_variance: 'ArrayLike') -> 'None'
```

x = location + loadings @ f + epsilon, f~N(0,I), epsilon~N(0,diag(noise)).

Factors and noise are independent. transform returns E[f|x], not an
orthogonal projection or an exact inverse of the noisy generative model.

### `GaussianFactorModel.reconstruct`

```python
reconstruct(self, factors: 'ArrayLike') -> 'FloatArray'
```

### `GaussianFactorModel.residuals`

```python
residuals(self, values: 'ArrayLike') -> 'FloatArray'
```

### `GaussianFactorModel.transform`

```python
transform(self, values: 'ArrayLike') -> 'FloatArray'
```

## `PCAResult`

```python
PCAResult(location: 'ArrayLike', components: 'ArrayLike', explained_variance: 'ArrayLike', total_variance: 'float', *, nobs: 'int', ddof: 'int' = 1, whiten: 'bool' = False, diagnostics: 'Diagnostics | None' = None) -> 'None'
```

Principal axes in columns; variances use the explicit n-ddof divisor.

### `PCAResult.inverse_transform`

```python
inverse_transform(self, coordinates)
```

### `PCAResult.reconstruct`

```python
reconstruct(self, coordinates: 'ArrayLike') -> 'FloatArray'
```

### `PCAResult.residuals`

```python
residuals(self, values: 'ArrayLike') -> 'FloatArray'
```

### `PCAResult.transform`

```python
transform(self, values: 'ArrayLike') -> 'FloatArray'
```

## `Subspace`

```python
Subspace(basis: 'ArrayLike') -> 'None'
```

An orthonormal column basis Q; ambient row vectors project as x @ Q @ Q.T.

### `Subspace.coordinates`

```python
coordinates(self, values: 'ArrayLike') -> 'FloatArray'
```

### `Subspace.project`

```python
project(self, values: 'ArrayLike')
```

### `Subspace.reconstruct`

```python
reconstruct(self, coordinates: 'ArrayLike') -> 'FloatArray'
```

### `Subspace.residuals`

```python
residuals(self, values: 'ArrayLike') -> 'FloatArray'
```

## `Transform`

Coordinate map with declared input/output dimensions; transform preserves leading observation axes.

### `Transform.transform`

```python
transform(self, values: 'ArrayLike') -> 'FloatArray'
```

## `TransformFitter`

Fit a coordinate transform from observation rows without implicitly changing target semantics.

### `TransformFitter.fit`

```python
fit(self, samples: 'ArrayLike') -> 'Transform'
```

## `checked_transform`

```python
checked_transform(transform: 'Transform', values: 'ArrayLike') -> 'FloatArray'
```

Check a user-written map at the composition boundary.

## `fit`

```python
fit(samples: 'ArrayLike', fitter: 'TransformFitter') -> 'Transform'
```

Fit a user or built-in tool to an explicit (n,d) training matrix.
