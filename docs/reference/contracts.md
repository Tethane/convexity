# `convexity.math.contracts`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: contracts.

## `evaluation_date`

```python
evaluation_date(value: str | datetime.date | datetime.datetime | QuantLib.QuantLib.Date) -> collections.abc.Iterator[None]
```

Restore all QuantLib settings, including on exceptions and nested calls.

## `day_count`

```python
day_count(name: str | QuantLib.QuantLib.DayCounter = 'actual365') -> QuantLib.QuantLib.DayCounter
```

Return the native QuantLib day counter for a supported convention name.

This is an explicit interoperability helper; year_fraction is the scalar facade.

## `to_quantlib_date`

```python
to_quantlib_date(value: str | datetime.date | datetime.datetime | QuantLib.QuantLib.Date) -> QuantLib.QuantLib.Date
```

Convert an ISO date, Python date/datetime or native date to QuantLib Date.

Dates are calendar dates; intraday components are not retained.

## `Payoff`

Callable terminal payoff from supplied underlying values; amounts inherit the contract's unit convention.

## `Exercise`

```python
Exercise(style: 'str', dates: 'Sequence[DateLike | NativeDate]') -> 'None'
```

Exercise style and allowed dates; calendar dates are distinct from model-time year fractions.

### `Exercise.to_quantlib`

```python
to_quantlib(self)
```

## `VanillaContract`

```python
VanillaContract(strike: 'float', exercise: 'Exercise', kind: 'str' = 'call', multiplier: 'float' = 1.0, settlement: 'str' = 'cash', currency: 'str' = 'USD') -> None
```

Plain option payoff and exercise specification, independent of the pricing model and engine.

### `VanillaContract.delivery`

```python
delivery(self, spot: 'float')
```

(underlying units, cash) at exercise; zero when out of the money.

### `VanillaContract.to_quantlib`

```python
to_quantlib(self)
```

## `CashFlow`

```python
CashFlow(date: 'DateLike | NativeDate', amount: 'float', currency: 'str' = 'USD') -> 'None'
```

Signed amount payable on a calendar date; currency/unit consistency belongs to the containing contract.

### `CashFlow.to_quantlib`

```python
to_quantlib(self)
```

## `payoff`

```python
payoff(spot: 'ArrayLike', strike: 'float', *, kind: 'str' = 'call', cash: 'float | None' = None) -> 'float | FloatArray'
```

Evaluate vanilla terminal payoff elementwise in spot units.

kind selects call/put; supplying cash selects a cash-or-nothing digital payoff.
The output is undiscounted and does not specify a probability measure.

## `schedule`

```python
schedule(start: 'DateLike | NativeDate', end: 'DateLike | NativeDate', *, months: 'int' = 6, calendar: 'str | NativeCalendar | None' = None, convention: 'str | int | None' = None, backwards: 'bool' = True, end_of_month: 'bool' = False, first_date: 'DateLike | NativeDate | None' = None, next_to_last_date: 'DateLike | NativeDate | None' = None) -> 'NativeSchedule'
```

Construct a native QuantLib schedule between calendar dates.

months is the regular tenor; calendar/convention specify business-day adjustment.
Stub dates and end-of-month handling are explicit. Use schedule_dates for Python dates.

## `year_fraction`

```python
year_fraction(start: 'DateLike | NativeDate', end: 'DateLike | NativeDate', convention: 'str | NativeDayCounter' = 'actual365') -> 'float'
```

Compute elapsed year fraction between dates under the selected day-count convention.

## `interest_factor`

```python
interest_factor(rate: 'float', time: 'float', *, compounding: 'str' = 'continuous', frequency: 'int' = 2) -> 'float'
```

Return accumulation over time years for a decimal annual rate.

Compounding is continuous, simple or periodic; frequency is periods per year.

## `fixed_cashflows`

```python
fixed_cashflows(dates: 'NativeSchedule', coupon: 'float', *, notional: 'float' = 100.0, convention: 'str' = 'actual365', redemption: 'float' = 100.0, currency: 'str' = 'USD') -> 'tuple[CashFlow, ...]'
```

Construct coupon and redemption CashFlows from ordered schedule dates.

Coupon is an annual decimal rate; each accrual uses the selected day count.
Notional and redemption are in the supplied currency.

## `present_value`

```python
present_value(cashflows: 'Sequence[CashFlow]', curve: 'YieldCurve', *, currency: 'str' = 'USD', fx: 'Mapping[str, float] | None' = None, include_reference: 'bool' = False) -> 'float'
```

FX maps source currency to valuation-currency units per source unit.

## `split_holdings`

```python
split_holdings(quantity: 'float', price: 'float', ratio: 'float') -> 'tuple[float, float]'
```

Apply a positive split ratio, multiplying quantity and dividing price.

Return adjusted holdings and price; total marked value is preserved.

## `dividend_adjustment`

```python
dividend_adjustment(quantity: 'float', price: 'float', dividend: 'float') -> 'tuple[float, float, float]'
```

Frictionless ex-dividend transformation: (shares, ex-price, cash).

## `to_date`

```python
to_date(value: 'DateLike | NativeDate') -> 'date'
```

Return a standard datetime.date from an ISO, Python or native date.

## `schedule_dates`

```python
schedule_dates(value: 'NativeSchedule | Sequence[DateLike | NativeDate]') -> 'tuple[date, ...]'
```

Inspect a schedule as a tuple of standard Python dates.
