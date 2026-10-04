"""Self-financing wealth transformations with explicit cash versus preferences."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..._typing import FloatArray
from ...execution import CostModel

if TYPE_CHECKING:
    from .. import Account


from dataclasses import replace

import numpy as np

from ... import _decision as v
from .._backends import accounting
from ..interfaces import Account, Attribution


def trade(
    account: Account, quantities: ArrayLike, prices: ArrayLike, *, cost: float = 0.0
) -> Account:
    """Apply signed trades at current prices, debiting cash by trade value plus cost.

    Quantities/prices align with account holdings; settlement follows Account conventions."""
    q, p = v.array(quantities, "quantities", ndim=1), v.array(prices, "prices", ndim=1)
    if q.shape != account.holdings.shape or p.shape != q.shape:
        raise ValueError("trades, prices and holdings must align")
    return accounting.trade(account, q, p, v.scalar(cost, "cost", minimum=0))


def accrue(
    account: Account,
    *,
    lending_factor: float = 1.0,
    borrowing_factor: float = 1.0,
    collateral_factor: float = 1.0,
    fees: float = 0.0,
    distributions: ArrayLike = 0.0,
) -> Account:
    """Positive growth factors over the interval; distributions per settled unit.

    Pending claims do not accrue here. Supply entitlement-adjusted cash flows
    separately when settlement calendars differ from the default held units.
    """
    factors = [
        v.scalar(x, name, minimum=0, strict=True)
        for x, name in [
            (lending_factor, "lending factor"),
            (borrowing_factor, "borrowing factor"),
            (collateral_factor, "collateral factor"),
        ]
    ]
    d = v.vector(distributions, len(account.holdings), "distributions")
    return accounting.accrue(
        account,
        factors[0],
        factors[1],
        factors[2],
        v.scalar(fees, "fees", minimum=0),
        d,
    )


def collateral_transfer(account: Account, amount: float) -> Account:
    """Positive amount moves settled cash to owned posted collateral."""
    a = v.scalar(amount, "amount")
    if account.collateral + a < 0:
        raise ValueError("cannot release more collateral than posted")
    return replace(account, cash=account.cash - a, collateral=account.collateral + a)


def cash_flow(account: Account, amount: float) -> Account:
    """Add supplied entitlement/coupon cash once; external flows affect gains."""
    return replace(account, cash=account.cash + v.scalar(amount, "amount"))


def exercise_delivery(
    account: Account,
    option_index: int,
    signed_contracts: float,
    *,
    underlying_index: int,
    shares_per_contract: float,
    cash_per_contract: float,
) -> Account:
    """Remove exercised long (+) or assigned short (-) contracts into pending claims.

    Per-contract delivery is supplied explicitly, e.g. contract.delivery(spot).
    Option marks are per contract, already including their multiplier.
    """
    i, j = (
        v.integer(option_index, "option index"),
        v.integer(underlying_index, "underlying index"),
    )
    e = v.scalar(signed_contracts, "signed contracts")
    shares, cash = (
        v.scalar(shares_per_contract, "shares"),
        v.scalar(cash_per_contract, "cash"),
    )
    if i >= len(account.holdings) or j >= len(account.holdings) or i == j:
        raise ValueError("distinct valid option and underlying indices required")
    held = account.holdings[i]
    if e * held < 0 or abs(e) > abs(held) or e != np.floor(e):
        raise ValueError(
            "exercise must be integral and within same-signed owned/short contracts"
        )
    h, pending = account.holdings.copy(), account.pending_holdings.copy()
    h[i] -= e
    pending[j] += e * shares
    return replace(
        account,
        holdings=h,
        pending_holdings=pending,
        pending_cash=account.pending_cash + e * cash,
    )


def settle(account: Account) -> Account:
    """Transfer all supplied pending claims once; repeating is idempotent."""
    return replace(
        account,
        holdings=account.holdings + account.pending_holdings,
        cash=account.cash + account.pending_cash,
        pending_holdings=np.zeros_like(account.holdings),
        pending_cash=0.0,
    )


def futures_variation(
    account: Account,
    quantities: ArrayLike,
    old_prices: ArrayLike,
    new_prices: ArrayLike,
    *,
    multipliers: ArrayLike = 1.0,
) -> Account:
    """Futures are zero-value positions outside the account's marked holdings."""
    q, old, new = [
        v.array(x, name, ndim=1)
        for x, name in [
            (quantities, "quantities"),
            (old_prices, "old prices"),
            (new_prices, "new prices"),
        ]
    ]
    m = v.vector(multipliers, len(q), "multipliers")
    if old.shape != q.shape or new.shape != q.shape or np.any(m <= 0):
        raise ValueError("prices/quantities must align; multipliers positive")
    return cash_flow(account, float((q * m) @ (new - old)))


def financed_gain(
    initial_wealth: float,
    final_wealth: ArrayLike,
    *,
    funding_factor: float = 1.0,
    terminal_external_flows: ArrayLike = 0.0,
) -> FloatArray:
    """W1 - funding_factor*W0 - externally contributed flows valued at t1."""
    w0, w1 = (
        v.scalar(initial_wealth, "initial wealth"),
        v.array(final_wealth, "final wealth"),
    )
    f = v.scalar(funding_factor, "funding factor", minimum=0, strict=True)
    return v.array(
        w1 - f * w0 - v.array(terminal_external_flows, "external flows"),
        "financed gains",
    )


def liquidation_wealth(account: Account, marks: ArrayLike, cost: CostModel) -> float:
    """Pending claims must settle before this simple all-holdings liquidation."""
    if np.any(account.pending_holdings) or account.pending_cash:
        raise ValueError("settle pending delivery before liquidation")
    return account.wealth(marks) - v.scalar(
        cost(-account.holdings), "liquidation cost", minimum=0
    )


def pnl_attribution(
    value: Callable[[FloatArray], float],
    initial: ArrayLike,
    final: ArrayLike,
    gradient: ArrayLike,
    hessian: ArrayLike | None = None,
) -> Attribution:
    """Exact repricing versus local delta and .5*dx'Hdx; remainder explicit."""
    x, y, g = [
        v.array(z, n, ndim=1)
        for z, n in [(initial, "initial"), (final, "final"), (gradient, "gradient")]
    ]
    if y.shape != x.shape or g.shape != x.shape:
        raise ValueError("state and gradient dimensions must align")
    h = (
        np.zeros((len(x), len(x)))
        if hessian is None
        else v.array(hessian, "hessian", ndim=2)
    )
    if h.shape != (len(x), len(x)) or not np.allclose(h, h.T):
        raise ValueError("Hessian must be symmetric and aligned")
    dx = y - x
    exact = v.scalar(value(y), "final value") - v.scalar(value(x), "initial value")
    first, second = float(g @ dx), float(0.5 * dx @ h @ dx)
    return Attribution(exact, first, second, exact - first - second)
