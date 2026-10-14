"""explicitly supply P and Q, then compose Q with QuantLib."""

from convexity.math.measures import gbm_pair
from convexity.math.pricing import BlackScholesModel

pair = gbm_pair(mu=0.10, rate=0.04, dividend=0.01, sigma=0.20)
print("Market price of risk:", pair.market_price_of_risk(0.0, 100.0))
model = BlackScholesModel.from_measure_pair(pair, 100.0, "2026-01-01")
print("Q discount factor:", model.discount.discount(1.0))
