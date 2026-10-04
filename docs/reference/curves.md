# `convexity.math.curves`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: curves.

## `DiscountCurve`

Discount-factor term structure measured from its reference date under an explicit day-count convention.

### `DiscountCurve.discount`

```python
discount(self, maturity: 'float | DateLike | NativeDate') -> 'float'
```

### `DiscountCurve.forward_rate`

```python
forward_rate(self, start: 'float | DateLike | NativeDate', end: 'float | DateLike | NativeDate') -> 'float'
```

### `DiscountCurve.time`

```python
time(self, maturity: 'float | DateLike | NativeDate') -> 'float'
```

### `DiscountCurve.zero_rate`

```python
zero_rate(self, maturity: 'float | DateLike | NativeDate') -> 'float'
```

## `CurveFitResult`

```python
CurveFitResult(reference_date: 'date', diagnostics: 'CalibrationResult') -> None
```

Owned discount-model fit; cashflow times are years from reference_date.

Parameters belong to the caller's supplied discount model. Diagnostics
describe a local least-squares solution, not a global optimum certificate.

## `YieldCurve`

```python
YieldCurve(structure: 'NativeYieldCurve') -> None
```

Own a QuantLib term structure. Extrapolation is opt-in at construction.

### `YieldCurve.discount`

```python
discount(self, maturity: 'float | DateLike | NativeDate')
```

### `YieldCurve.forward_rate`

```python
forward_rate(self, start, end)
```

### `YieldCurve.rolled`

```python
rolled(self, reference_date)
```

Conditional discount curve D(new,T)/D(old,new), fixed original curve.

### `YieldCurve.shifted`

```python
shifted(self, spread)
```

### `YieldCurve.time`

```python
time(self, maturity: 'float | DateLike | NativeDate')
```

### `YieldCurve.zero_rate`

```python
zero_rate(self, maturity: 'float | DateLike | NativeDate')
```

## `flat_curve`

```python
flat_curve(reference_date: 'DateLike | NativeDate', rate: 'float', *, convention: 'str' = 'actual365') -> 'YieldCurve'
```

Create a discount curve from a calendar reference date and continuous annual rate.

Times are year fractions under convention; discount(0)=1. Negative rates are allowed.

## `discount_curve`

```python
discount_curve(dates: 'Sequence[DateLike | NativeDate]', discounts: 'ArrayLike', *, convention: 'str' = 'actual365', extrapolate: 'bool' = False) -> 'YieldCurve'
```

Interpolate discount factors at increasing dates.

Discount factors are positive dimensionless present-value multipliers.
Queries beyond the final node require extrapolate=True.

## `zero_curve`

```python
zero_curve(dates: 'Sequence[DateLike | NativeDate]', rates: 'ArrayLike', *, convention: 'str' = 'actual365', extrapolate: 'bool' = False) -> 'YieldCurve'
```

Interpolate continuous annual zero rates at increasing dates.

Rates are decimal units and times follow convention; extrapolation is explicit.

## `bootstrap`

```python
bootstrap(reference_date: 'DateLike | NativeDate', helpers: 'Sequence[NativeRateHelper]', *, convention: 'str' = 'actual365', interpolation: 'str' = 'log_discount', extrapolate: 'bool' = False) -> 'YieldCurve'
```

Native rate helpers retain deposit/swap/OIS/futures/bond conventions.

## `forward_price`

```python
forward_price(spot: 'float', maturity: 'float | DateLike | NativeDate', financing: 'YieldCurve', *, dividend: 'YieldCurve | None' = None, borrowing: 'YieldCurve | None' = None, cash_dividends: 'Iterable[CashFlow]' = (), storage_rate: 'float' = 0.0, convenience_yield: 'float' = 0.0) -> 'float'
```

Deterministic carry: (S-PV(dividends))*D_div/(D_fin*D_borrow)*exp((u-y)T).

Borrowing is an additional positive carrying cost. Continuous yield and cash
dividends are mutually exclusive to prevent double counting.

## `futures_basis`

```python
futures_basis(futures: 'float', forward: 'float') -> 'float'
```

Return futures minus forward in their shared price units; no funding adjustment is inferred.

## `fit_discount_curve`

```python
fit_discount_curve(reference_date: 'DateLike | NativeDate', cashflow_times: 'ArrayLike', cashflows: 'ArrayLike', prices: 'ArrayLike', initial: 'ArrayLike', *, model: 'Callable[[FloatArray, FloatArray], ArrayLike]') -> 'CurveFitResult'
```

Fit model(times, parameters); times are years from reference_date.

Return CurveFitResult with owned parameters and local fit diagnostics.
Failed optimization raises RuntimeError; discounts must be positive.

## `deposit_quote`

```python
deposit_quote(rate: 'float', tenor: 'str', *, settlement_days: 'int' = 0, calendar: 'str' = 'none', adjustment: 'str' = 'unadjusted', convention: 'str' = 'actual365') -> 'NativeRateHelper'
```

Deposit quote for bootstrap; rate is decimal, tenor e.g. '3M'.

## `swap_quote`

```python
swap_quote(rate: 'float', tenor: 'str', *, settlement_days: 'int' = 0, calendar: 'str' = 'none', adjustment: 'str' = 'unadjusted', fixed_frequency: 'str' = 'annual', fixed_convention: 'str' = '30_360', floating_tenor: 'str' = '6M', floating_convention: 'str' = 'actual360', currency: 'str' = 'USD', discount: 'YieldCurve | None' = None) -> 'NativeRateHelper'
```

Par fixed/IBOR swap quote; optional external discount curve for dual-curve fitting.

## `ois_quote`

```python
ois_quote(rate: 'float', tenor: 'str', *, settlement_days: 'int' = 0, calendar: 'str' = 'weekends', currency: 'str' = 'USD', convention: 'str' = 'actual360', payment_frequency: 'str' = 'annual', payment_adjustment: 'str' = 'following', discount: 'YieldCurve | None' = None) -> 'NativeRateHelper'
```

Par overnight-indexed swap quote, with compounded overnight coupons.
