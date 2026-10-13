# `convexity.math.calibration`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: calibration.

## `NativeCalibrationResult`

```python
NativeCalibrationResult(parameters: 'ArrayLike', residuals: 'ArrayLike', end_criteria: 'int', success: 'bool') -> 'None'
```

Native model calibration output and diagnostics; native handles are advanced interoperability objects.

### `NativeCalibrationResult.items`

```python
items(self)
```

### `NativeCalibrationResult.keys`

```python
keys(self)
```

## `CalibrationProblem`

```python
CalibrationProblem(predict: 'Callable[[FloatArray], ArrayLike]', observations: 'ArrayLike', initial: 'ArrayLike', weights: 'ArrayLike | None' = None, lower: 'ArrayLike' = -inf, upper: 'ArrayLike' = inf, regularization: 'float' = 0.0, prior: 'ArrayLike | None' = None) -> 'None'
```

Market targets, model response and parameter specification for numerical calibration.

## `CalibrationResult`

```python
CalibrationResult(parameters: 'ArrayLike', residuals: 'ArrayLike', jacobian: 'ArrayLike', singular_values: 'ArrayLike', rank: 'int', covariance: 'ArrayLike | None', success: 'bool', message: 'str', evaluations: 'int', active_bounds: 'ArrayLike') -> 'None'
```

Calibrated parameters and residual/solver diagnostics; a good fit does not prove identification.

## `Calibrator`

Solve a calibration problem and return owned parameters, residuals and numerical diagnostics.

### `Calibrator.fit`

```python
fit(self, problem: 'CalibrationProblem') -> 'CalibrationResult'
```

## `LeastSquaresCalibration`

```python
LeastSquaresCalibration(tolerance: 'float' = 1e-10, max_evaluations: 'int' = 3000, rank_tolerance: 'float' = 1e-07) -> None
```

Calibrate parameters by weighted residual minimization with explicit solver controls.

### `LeastSquaresCalibration.fit`

```python
fit(self, problem)
```

## `calibrate`

```python
calibrate(problem: 'CalibrationProblem', *, solver: 'Calibrator | None' = None, require_success: 'bool' = True) -> 'CalibrationResult'
```

Fit CalibrationProblem residuals using a built-in or custom solver.

Return parameters, residual/Jacobian diagnostics, rank and optional covariance.
require_success=True rejects failed solves; local convergence is not global optimality.

## `quantlib_calibrate`

```python
quantlib_calibrate(model: 'NativeCalibratedModel', helpers: 'Sequence[NativeCalibrationHelper]', valuation_date: 'DateLike | NativeDate', *, weights: 'ArrayLike | None' = None, fixed_parameters: 'Sequence[bool] | None' = None, max_iterations: 'int' = 1000) -> 'NativeCalibrationResult'
```

Mutate a supplied QL model using native helper errors and LM calibration.

Returns parameters, helper residuals and native stopping status. Native
helpers define relative-price/price/implied-vol errors themselves.
