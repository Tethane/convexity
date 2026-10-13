# `convexity.math.fixed_income`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: fixed income.

## `CarryResult`

```python
CarryResult(initial_value: 'float', horizon_value: 'float', reinvested_coupons: 'float', carry: 'float', roll_down: 'float', total_return: 'float') -> None
```

Carry/roll decomposition under the supplied horizon and curve assumptions.

### `CarryResult.items`

```python
items(self)
```

### `CarryResult.keys`

```python
keys(self)
```

## `BondPricer`

Value a supplied bond under explicit date, curve and pricing conventions.

## `BondResult`

```python
BondResult(clean: 'float', dirty: 'float', accrued: 'float', settlement_date: 'DateLike | NativeDate', yield_to_maturity: 'float | None' = None, duration: 'float | None' = None, convexity: 'float | None' = None, dv01: 'float | None' = None) -> 'None'
```

Bond valuation and dated cash flows with the pricing conventions retained.

## `zero_coupon`

```python
zero_coupon(maturity: 'DateLike | NativeDate', *, face: 'float' = 100.0, settlement_days: 'int' = 0, calendar: 'str | NativeCalendar | None' = None, issue_date: 'DateLike | NativeDate | None' = None) -> 'NativeBond'
```

Construct a native zero-coupon bond with calendar maturity.

Face is redemption cash; settlement_days counts calendar-adjusted settlement lag.

## `fixed_bond`

```python
fixed_bond(schedule: 'NativeSchedule', coupon: 'float', *, face: 'float' = 100.0, settlement_days: 'int' = 0, convention: 'str | NativeDayCounter' = 'actual365', redemption: 'float' = 100.0) -> 'NativeBond'
```

Construct a native fixed-coupon bond from a schedule and annual decimal coupon.

Face and redemption set cash amounts; convention controls coupon accrual.

## `floating_bond`

```python
floating_bond(schedule: 'NativeSchedule', index: 'NativeIborIndex', *, face: 'float' = 100.0, spread: 'float' = 0.0, settlement_days: 'int' = 0, convention: 'str | NativeDayCounter' = 'actual360', fixing_days: 'int | None' = None) -> 'NativeBond'
```

Index carries forecast curve and historical fixings; missing fixings fail.

## `amortizing_bond`

```python
amortizing_bond(schedule: 'NativeSchedule', notionals: 'ArrayLike', coupon: 'float', *, settlement_days: 'int' = 0, convention: 'str | NativeDayCounter' = 'actual365') -> 'NativeBond'
```

Construct a native amortizing bond with notionals aligned to schedule periods.

Coupon is an annual decimal rate; principal changes generate repayment cashflows.

## `inflation_bond`

```python
inflation_bond(schedule: 'NativeSchedule', index: 'NativeInflationIndex', base_cpi: 'float', coupon: 'float', *, observation_lag_months: 'int' = 3, face: 'float' = 100.0, settlement_days: 'int' = 0, convention: 'str | NativeDayCounter' = 'actual365', growth_only: 'bool' = False, interpolation: 'str | int | None' = None) -> 'NativeBond'
```

Construct a native inflation-linked bond with explicit CPI index and base CPI.

Coupon is a real annual rate. Observation lag, interpolation and growth-only
redemption specify indexation; future CPI requires an explicit inflation curve.

## `callable_bond`

```python
callable_bond(schedule: 'NativeSchedule', coupon: 'float', callabilities: 'Sequence[tuple[DateLike | NativeDate, float, str]]', *, face: 'float' = 100.0, settlement_days: 'int' = 0, convention: 'str | NativeDayCounter' = 'actual365') -> 'NativeBond'
```

callabilities: (date, clean price per 100, 'call' or 'put').

## `cashflows`

```python
cashflows(bond: 'NativeBond', valuation_date: 'DateLike | NativeDate', *, currency: 'str' = 'USD') -> 'tuple[CashFlow, ...]'
```

Extract dated owned CashFlow records from a native bond at a valuation date.

The evaluation date is scoped to the call; amounts use the supplied currency label.

## `yield_from_price`

```python
yield_from_price(bond: 'NativeBond', clean_price: 'float', settlement_date: 'DateLike | NativeDate', *, convention: 'str | NativeDayCounter' = 'actual365', frequency: 'str | int | None' = None, compounding: 'str | int | None' = None, valuation_date: 'DateLike | NativeDate | None' = None) -> 'float'
```

Invert a bond clean price per 100 face into annual decimal yield.

Settlement date, day count, compounding and frequency determine the yield convention.

## `price_from_yield`

```python
price_from_yield(bond: 'NativeBond', yield_rate: 'float', settlement_date: 'DateLike | NativeDate', *, convention: 'str | NativeDayCounter' = 'actual365', frequency: 'str | int | None' = None, compounding: 'str | int | None' = None, valuation_date: 'DateLike | NativeDate | None' = None) -> 'float'
```

Return clean bond price per 100 face for an annual decimal yield.

Settlement date, day count, compounding and frequency must match yield_from_price.

## `bond_analytics`

```python
bond_analytics(bond: 'NativeBond', curve: 'YieldCurve', *, settlement_date: 'DateLike | NativeDate | None' = None, convention: 'str | NativeDayCounter' = 'actual365', frequency: 'str | int | None' = None, engine: 'NativePricingEngine | None' = None, compute_yield: 'bool' = True) -> 'BondResult'
```

Yield analytics describe promised cash flows, not option-adjusted risk.

With a callable engine, DV01/duration from this function remain yield-based;
use curve_sensitivities for effective callable-bond sensitivities.

## `key_rate_duration`

```python
key_rate_duration(bond: 'NativeBond', curve: 'YieldCurve', key_dates: 'Sequence[DateLike | NativeDate]', *, bump: 'float' = 0.0001, engine_factory: 'Callable[[YieldCurve], NativePricingEngine] | None' = None) -> 'FloatArray'
```

Triangular linearly interpolated zero-spread bumps; flat endpoint wings.

## `carry_roll_down`

```python
carry_roll_down(bond: 'NativeBond', curve: 'YieldCurve', horizon: 'DateLike | NativeDate', *, evolution: 'str' = 'constant_tenor') -> 'CarryResult'
```

Cash-unit decomposition with coupons reinvested at implied forwards.

carry = accrued financing of initial NPV. roll_down = terminal wealth minus
initial NPV minus carry. Uses valuation-date cash units, not clean quotes.
constant_tenor holds today's zero curve fixed as a function of remaining
tenor; implied_forward rolls today's dated curve and gives zero excess
return by construction. Neither convention is a forecast of future rates.

## `callable_bond_engine`

```python
callable_bond_engine(curve: 'YieldCurve', *, mean_reversion: 'float' = 0.1, volatility: 'float' = 0.01, steps: 'int' = 300) -> 'NativePricingEngine'
```

Hull–White exercise tree, usable directly or as a curve-risk factory.
