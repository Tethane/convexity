"""First Convexity calculation: future prices -> profit/loss -> risk -> share count.

Run from the repository root in the configured Python 3.14 environment:
    conda run --no-capture-output -n quant python docs/math/examples/getting_started.py

The drift and volatility below are invented teaching inputs. This example holds
shares for 21 trading days and omits dividends, fees, interest and financing.
It measures loss at the end of the holding period, not losses along the way.
"""

import numpy as np

from convexity.math import optimization, processes, risk


def main():
    # 1. Describe how a price can change. Time is measured in years, so mu and
    # sigma are annual parameters. A trading year has 252 days in this example.
    spot = 100.0
    holding_days = 21
    model = processes.GeometricBrownianMotion(mu=0.08, sigma=0.20)

    # A transition fixes the starting price and elapsed time. Its distribution
    # describes possible ending prices and their probabilities.
    terminal = model.transition(spot, holding_days / 252).distribution()
    print(f"Model mean ending price: ${float(terminal.mean()):.2f}")
    print("5%, 50%, 95% ending prices:", np.round(terminal.quantile([0.05, 0.50, 0.95]), 2))
    print(f"Model probability of finishing below ${spot:.0f}: {float(terminal.cdf(spot)):.2%}")

    # 2. Generate 20,000 equally likely scenarios. A fresh RNG with the same seed
    # reproduces the draws in the same software environment.
    future_prices = terminal.sample(20_000, rng=np.random.default_rng(42))
    print("Ending-price array shape:", future_prices.shape)

    # One entry per scenario: shares * price change = dollar profit/loss.
    shares = 50
    pnl = shares * (future_prices - spot)
    losses = -pnl  # Risk functions interpret positive numbers as losses.
    print(f"\nFor {shares} shares costing ${shares * spot:,.0f}:")
    print(f"  Average simulated profit: ${float(pnl.mean()):.2f}")
    print(f"  Simulated probability of a loss: {float(np.mean(pnl < 0)):.2%}")
    print(f"  95% VaR: ${risk.value_at_risk(losses, 0.95):.2f}")
    print(f"  95% expected shortfall: ${risk.expected_shortfall(losses, 0.95):.2f}")

    # VaR is a loss percentile. Expected shortfall averages the worst 5% here.
    # Neither statistic is a maximum-loss guarantee.
    #
    # 3. Supply a decision rule: maximize mean simulated profit, spend at most
    # $10,000, and keep 95% expected shortfall at or below $500.
    budget = 10_000.0
    risk_budget = 500.0
    max_shares = int(budget // spot)

    def objective(action):
        quantity = action[0]  # One decision variable: how many shares to buy.
        # enumerate_minimum minimizes, so negate the profit we want to maximize.
        return -float(np.mean(quantity * (future_prices - spot)))

    def feasible(action):
        quantity = action[0]
        scenario_losses = -quantity * (future_prices - spot)
        return bool(
            quantity * spot <= budget
            and risk.expected_shortfall(scenario_losses, 0.95) <= risk_budget
        )

    # All candidates use the SAME scenarios. The search checks every integer
    # share count from zero to max_shares, including the choice to buy nothing.
    result = optimization.enumerate_minimum(
        objective, [range(max_shares + 1)], feasible=feasible
    )
    if not result.success or result.solution is None:
        raise RuntimeError(f"Share-count search failed: {result.status}")
    chosen = int(result.solution[0])
    print(f"\nChosen share count: {chosen}")
    print("Shares | Cost       | Mean profit | 95% ES     | Allowed")
    candidates = sorted({0, 25, 50, max_shares, chosen, min(chosen + 1, max_shares)})
    for quantity in candidates:
        scenario_pnl = quantity * (future_prices - spot)
        es = risk.expected_shortfall(-scenario_pnl, 0.95)
        print(
            f"{quantity:6d} | ${quantity * spot:9,.2f} | "
            f"${float(scenario_pnl.mean()):10.2f} | ${es:9.2f} | "
            f"{feasible([quantity])}"
        )


if __name__ == "__main__":
    main()
