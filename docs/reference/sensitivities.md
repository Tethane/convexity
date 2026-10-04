# `convexity.math.sensitivities`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: sensitivities.

## `HedgeResult`

```python
HedgeResult(holdings: 'ArrayLike', residual_exposure: 'ArrayLike', nullspace: 'ArrayLike', success: 'bool', message: 'str') -> 'None'
```

Hedge positions and remaining exposure in the declared instrument and risk-factor order.

## `sensitivities`

```python
sensitivities(function: 'Callable[[FloatArray], ArrayLike] | Callable[[NativeTensor], NativeTensor]', point: 'ArrayLike', *, order: 'int' = 1, autodiff: 'bool' = False) -> 'ArrayResult'
```

Per-unit derivatives: first order is (output,input); second is Hessian.

PyTorch callbacks must remain tensor-native; QuantLib is opaque to autograd.

## `bump_sensitivities`

```python
bump_sensitivities(function: 'Callable[[FloatArray], ScalarLike]', point: 'ArrayLike', *, steps: 'ArrayLike') -> 'SensitivityReport'
```

Central gradient/Hessian including cross-Greeks; absolute factor bumps.

## `hedge`

```python
hedge(exposures: 'ArrayLike', target: 'ArrayLike', *, weights: 'ArrayLike | None' = None, lower: 'ArrayLike' = -inf, upper: 'ArrayLike' = inf, equality: 'ArrayLike | None' = None, equality_target: 'ArrayLike | None' = None, ridge: 'float' = 0.0) -> 'HedgeResult'
```

Choose h to minimize ||sqrt(W)(A h + target)||² + ridge ||h||².

A has (risk,instrument) axes; equality is E h = b. Nullspace is the
unconstrained instrument-space exposure nullspace, independent of bounds.

## `curve_sensitivities`

```python
curve_sensitivities(value: 'Callable[[tuple[YieldCurve, ...]], ScalarLike]', curves: 'Sequence[YieldCurve]', *, bump: 'float' = 0.0001) -> 'SensitivityReport'
```

Parallel curve derivatives per unit decimal rate; DV01=-derivative*1e-4.
