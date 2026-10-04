"""Usable financial workflows: only Convexity, NumPy and ordinary Python data."""

from functools import partial
import numpy as np
from convexity.math import contracts, curves, pricing, rates, fixed_income, credit
from convexity.math import processes, simulation, measures, sensitivities

reference = "2026-01-01"

# Market quotes are numbers with ordinary financial tenors, not vendor helpers.
curve = curves.bootstrap(
    reference,
    [
        curves.deposit_quote(0.03, "6M"),
        curves.deposit_quote(0.032, "1Y"),
        curves.swap_quote(0.035, "2Y"),
        curves.swap_quote(0.04, "5Y"),
    ],
)
print("Bootstrap errors:", curve.calibration_errors)
print("Forward:", curves.forward_price(100.0, "2027-01-01", curve))

# Contracts compose with the same curve; schedules accept readable conventions.
dates = contracts.schedule("2027-01-01", "2030-01-01", calendar="weekends")
index = rates.rate_index(curve, currency="USD", calendar="weekends")
bond = fixed_income.floating_bond(dates, index)
print("Floating bond:", fixed_income.bond_analytics(bond, curve, compute_yield=False))
cap = rates.cap_floor(dates, index, 0.04)
print("Cap:", rates.rate_option_price(cap, curve, volatility=0.2))
swap = rates.interest_rate_swap(dates, dates, index, 0.04)
print("Swap:", rates.swap_price(swap, curve))
option = rates.swaption(swap, ["2026-12-31"])
print(
    "Swaption:",
    rates.rate_option_price(option, curve, method="normal", volatility=0.01),
)

# The same factory handles a base price and every curve bump in effective risk.
callable_bond = fixed_income.callable_bond(
    contracts.schedule(reference, "2031-01-01"), 0.06, [("2028-01-01", 100.0, "call")]
)
engine = partial(fixed_income.callable_bond_engine, mean_reversion=0.1, volatility=0.01)
print(
    "Callable bond:",
    fixed_income.bond_analytics(
        callable_bond, curve, engine=engine(curve), compute_yield=False
    ),
)
print(
    "Callable key-rate durations:",
    fixed_income.key_rate_duration(
        callable_bond,
        curve,
        [reference, "2028-01-01", "2031-01-01"],
        engine_factory=engine,
    ),
)

# Historical fixings are ordinary mappings, isolated to each valuation.
current_curve = curves.flat_curve("2026-03-01", 0.04)
index_with_history = rates.rate_index(current_curve, fixings={reference: 0.037})
seasoned = fixed_income.floating_bond(
    contracts.schedule(reference, "2028-01-01"), index_with_history
)
print(
    "Seasoned floating bond:",
    fixed_income.bond_analytics(seasoned, current_curve, compute_yield=False),
)

# Credit calibration and pricing use an explicit trade date and recovery.
default = credit.bootstrap_cds(
    reference,
    [credit.cds_quote(0.01, "1Y", curve), credit.cds_quote(0.014, "5Y", curve)],
)
print(
    "CDS:",
    credit.cds_price(
        contracts.schedule(reference, "2031-01-01", months=3),
        0.014,
        curve,
        default,
        recovery=0.4,
        trade_date=reference,
    ),
)

# Existing process, scenario, forecast and distribution APIs compose unchanged.
hw = rates.ShortRateModel(
    "hull_white", {"kappa": 0.1, "sigma": 0.01}, curves.flat_curve(reference, 0.04)
)
print(
    "Short-rate transition mean:", processes.transition(hw.process(), 0.04, 1.0).mean()
)
paths = simulation.simulate(
    hw.process(), 0.04, [0.0, 0.5, 1.0], draws=100, rng=np.random.default_rng(42)
)
print("Scenario forecast mean:", paths.to_forecast().mean())

# P and Q remain deliberate economic choices, not a conversion of an empirical forecast.
pair = measures.gbm_pair(mu=0.08, rate=0.04, dividend=0.01, sigma=0.2)
model = pricing.BlackScholesModel.from_measure_pair(pair, 100.0, reference)
claim = contracts.VanillaContract(
    100.0, contracts.Exercise("european", ("2027-01-01",))
)
price = pricing.price(
    pricing.PricingProblem(claim, model, reference), pricing.VanillaEngine()
)
print("Vanilla price and Greeks:", price)
print("Delta hedge:", sensitivities.hedge([[1.0]], [price.greeks["delta"]]).holdings)
