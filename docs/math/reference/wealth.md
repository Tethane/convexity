# `convexity.math.wealth`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Wealth mathematics on supplied arrays and models.

## `Account`

```python
Account(holdings: 'ArrayLike', cash: 'float' = 0.0, collateral: 'float' = 0.0, pending_holdings: 'ArrayLike | None' = None, pending_cash: 'float' = 0.0) -> 'None'
```

Wealth/account state under explicit holdings, cash and valuation conventions.

### `Account.wealth`

```python
wealth(self, marks: 'ArrayLike') -> 'float'
```

## `Attribution`

```python
Attribution(exact: 'float', first_order: 'float', second_order: 'float', residual: 'float') -> None
```

Decomposition of wealth change into the components declared by the attribution calculation.

## `WealthMap`

Transform a supplied account state under explicitly modeled actions and market outcomes.

## `trade`

```python
trade(account: 'Account', quantities: 'ArrayLike', prices: 'ArrayLike', *, cost: 'float' = 0.0) -> 'Account'
```

Apply signed trades at current prices, debiting cash by trade value plus cost.

Quantities/prices align with account holdings; settlement follows Account conventions.

## `accrue`

```python
accrue(account: 'Account', *, lending_factor: 'float' = 1.0, borrowing_factor: 'float' = 1.0, collateral_factor: 'float' = 1.0, fees: 'float' = 0.0, distributions: 'ArrayLike' = 0.0) -> 'Account'
```

Positive growth factors over the interval; distributions per settled unit.

Pending claims do not accrue here. Supply entitlement-adjusted cash flows
separately when settlement calendars differ from the default held units.

## `collateral_transfer`

```python
collateral_transfer(account: 'Account', amount: 'float') -> 'Account'
```

Positive amount moves settled cash to owned posted collateral.

## `cash_flow`

```python
cash_flow(account: 'Account', amount: 'float') -> 'Account'
```

Add supplied entitlement/coupon cash once; external flows affect gains.

## `exercise_delivery`

```python
exercise_delivery(account: 'Account', option_index: 'int', signed_contracts: 'float', *, underlying_index: 'int', shares_per_contract: 'float', cash_per_contract: 'float') -> 'Account'
```

Remove exercised long (+) or assigned short (-) contracts into pending claims.

Per-contract delivery is supplied explicitly, e.g. contract.delivery(spot).
Option marks are per contract, already including their multiplier.

## `settle`

```python
settle(account: 'Account') -> 'Account'
```

Transfer all supplied pending claims once; repeating is idempotent.

## `futures_variation`

```python
futures_variation(account: 'Account', quantities: 'ArrayLike', old_prices: 'ArrayLike', new_prices: 'ArrayLike', *, multipliers: 'ArrayLike' = 1.0) -> 'Account'
```

Futures are zero-value positions outside the account's marked holdings.

## `financed_gain`

```python
financed_gain(initial_wealth: 'float', final_wealth: 'ArrayLike', *, funding_factor: 'float' = 1.0, terminal_external_flows: 'ArrayLike' = 0.0) -> 'FloatArray'
```

W1 - funding_factor*W0 - externally contributed flows valued at t1.

## `liquidation_wealth`

```python
liquidation_wealth(account: 'Account', marks: 'ArrayLike', cost: 'CostModel') -> 'float'
```

Pending claims must settle before this simple all-holdings liquidation.

## `pnl_attribution`

```python
pnl_attribution(value: 'Callable[[FloatArray], float]', initial: 'ArrayLike', final: 'ArrayLike', gradient: 'ArrayLike', hessian: 'ArrayLike | None' = None) -> 'Attribution'
```

Exact repricing versus local delta and .5*dx'Hdx; remainder explicit.
