"""Family 34: funded accounts, collateral and physical option delivery."""

from convexity.math import wealth

account = wealth.Account([2.0, 0.0], 10000.0)
marks = [1000.0, 110.0]
print("Before exercise:", account.wealth(marks))
account = wealth.exercise_delivery(
    account, 0, 2, underlying_index=1, shares_per_contract=100, cash_per_contract=-10000
)
print(
    "Pending exercise:",
    account.wealth(marks),
    account.pending_holdings,
    account.pending_cash,
)
account = wealth.settle(account)
print("Settled:", account.wealth(marks), account.holdings, account.cash)
account = wealth.accrue(account, borrowing_factor=1.01, distributions=[0.0, 1.0])
print("After borrowing and dividend:", account.wealth(marks))
print(
    "Futures variation cash:",
    wealth.futures_variation(
        wealth.Account([0.0], 100), [2], [100], [101], multipliers=50
    ).cash,
)
print(
    "Attribution:",
    wealth.pnl_attribution(lambda x: x[0] ** 3, [2.0], [2.1], [12.0], [[12.0]]),
)
