"""Pure account equations, no external ledger or broker state."""

from dataclasses import replace


def trade(account, quantities, prices, cost):
    return replace(
        account,
        holdings=account.holdings + quantities,
        cash=float(account.cash - prices @ quantities - cost),
    )


def accrue(account, lend, borrow, collateral, fees, distributions):
    cash = (
        max(account.cash, 0) * lend
        - max(-account.cash, 0) * borrow
        - fees
        + account.holdings @ distributions
    )
    return replace(
        account, cash=float(cash), collateral=account.collateral * collateral
    )
