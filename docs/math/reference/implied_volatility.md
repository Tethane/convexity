# `convexity.math.implied_volatility`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Financial mathematics: implied volatility.

## `VarianceSurface`

Implied-variance surface extension with explicit maturity and strike coordinates.

### `VarianceSurface.total_variance`

```python
total_variance(self, time: 'float', log_moneyness)
```

## `ArbitrageReport`

```python
ArbitrageReport(butterfly: 'ArrayLike', calendar: 'ArrayLike', minimum_variance: 'float') -> 'None'
```

Sampled violations, not a proof of arbitrage freedom outside the grid.

## `implied_volatility`

```python
implied_volatility(price: 'float', forward: 'float', strike: 'float', maturity: 'float', *, discount: 'float' = 1.0, kind: 'str' = 'call', model: 'str' = 'black76', displacement: 'float' = 0.0, tolerance: 'float' = 1e-12) -> 'float'
```

Invert a discounted forward-option price for its annualized volatility.

Maturity is years. Black-76 volatility is 1/sqrt(year); normal volatility is
price/sqrt(year). Return a scalar; unattainable prices or invalid domains fail.

## `SVI`

```python
SVI(a: 'float', b: 'float', rho: 'float', m: 'float', sigma: 'float') -> None
```

Raw SVI w(k)=a+b[rho(k-m)+sqrt((k-m)^2+sigma^2)].

### `SVI.butterfly`

```python
butterfly(self, k)
```

### `SVI.derivatives`

```python
derivatives(self, k)
```

### `SVI.fit`

```python
fit(k, variance, *, initial: 'ArrayLike' = (0.02, 0.1, -0.2, 0.0, 0.2), constraint_grid=None)
```

Constrain positivity, Lee wing slopes, and g(k)>=0 on a declared grid.

## `SSVI`

```python
SSVI(rho: 'float', eta: 'float', gamma: 'float' = 0.5) -> None
```

Power-law SSVI phi(theta)=eta/theta**gamma, on a bounded theta domain.

validate_domain checks sufficient butterfly bounds and calendar monotonicity
for 0<=gamma<=1/2, |rho|<1 and nondecreasing ATM total variance.

### `SSVI.validate_domain`

```python
validate_domain(self, theta_max)
```

## `SmileSurface`

```python
SmileSurface(times: 'ArrayLike', smiles: 'tuple[SVI, ...]') -> 'None'
```

Linear interpolation of SVI total variance between supplied expiries.

Extrapolation is rejected. Calendar and butterfly reports inspect the supplied
grid, including intermediate maturities; slice constraints alone do not
certify arbitrary interpolated surfaces.

### `SmileSurface.diagnose`

```python
diagnose(self, log_moneyness, *, time_points: 'int' = 21) -> 'ArbitrageReport'
```

### `SmileSurface.total_variance`

```python
total_variance(self, time: 'float', log_moneyness)
```

### `SmileSurface.volatility`

```python
volatility(self, time: 'float', log_moneyness)
```

## `sabr_volatility`

```python
sabr_volatility(strike: 'float', forward: 'float', maturity: 'float', alpha: 'float', beta: 'float', nu: 'float', rho: 'float', *, model: 'str' = 'black76', shift: 'float' = 0.0) -> 'float'
```

Evaluate the SABR implied-volatility approximation at a strike.

Maturity is years, beta in [0,1], rho in (-1,1); shift is in forward-price units.
model chooses lognormal Black-76 or normal volatility units, not an exact law.

## `black_variance_surface`

```python
black_variance_surface(reference_date: 'DateLike | NativeDate', dates: 'Sequence[DateLike | NativeDate]', strikes: 'ArrayLike', volatilities: 'ArrayLike', *, convention: 'str' = 'actual365', extrapolate: 'bool' = False) -> 'NativeVolatility'
```

QuantLib surface; vols shape (strike,expiry), bilinear total variance.

## `dupire_local_volatility`

```python
dupire_local_volatility(surface: 'NativeVolatility', discount: 'YieldCurve', dividend: 'YieldCurve', spot: 'float', time: 'float', strike: 'float') -> 'float'
```

Compute local volatility from a compatible implied-volatility surface and curves.

Time is years, strike/spot are prices. Numerical surface derivatives require
smoothness and arbitrage consistency; a negative local variance is an error.

## `risk_neutral_density`

```python
risk_neutral_density(call_price: 'Callable[[float], float]', strikes: 'ArrayLike', discount: 'float', *, step: 'float' = 0.01) -> 'FloatArray'
```

Breeden–Litzenberger finite difference C_KK/D; negative values are exposed.
