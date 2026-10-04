from itertools import product
import numpy as np
import pytest
from convexity.math import policies, execution, wealth


@pytest.mark.parametrize(
    "mode,p,time,adverse,mandatory,scheduled",
    list(
        product(
            ["waiting", "active", "closed"],
            [0.1, 0.6, 0.75, 0.9, 1.0],
            [5.0, 10.0],
            [False, True],
            [False, True],
            [False, True],
        )
    ),
)
def test_lifecycle_exhaustive(mode, p, time, adverse, mandatory, scheduled):
    h = policies.Hysteresis(0.9, 0.6)
    state = policies.PolicyState(10.0, mode)
    new = h.update(
        state, p, time, adverse=adverse, mandatory=mandatory, scheduled=scheduled
    )
    assert new.deadline == state.deadline
    if (
        mode == "closed"
        or adverse
        or mandatory
        or time >= 10
        or (mode == "active" and p <= 0.6)
    ):
        assert new.mode == "closed"
    elif mode == "active" or (p >= 0.9 and scheduled):
        assert new.mode == "active"
    else:
        assert new.mode == "waiting"


def test_fixed_deadline_convergence_and_action_overrides():
    h = policies.Hysteresis(convergence_gap=0.01)
    s = policies.PolicyState(10.0)
    s = h.update(s, 0.95, 2.0)
    assert s.mode == "active"
    s = h.update(s, 0.8, 3.0)
    assert s.mode == "active"
    s = h.update(s, 0.99, 4.0, gap=0.01)
    assert s.mode == "closed" and s.reason == "convergence"
    assert h.update(s, 1.0, 5.0).mode == "closed"
    with pytest.raises(ValueError):
        h.update(s, 0.5, 1.0)
    assert policies.no_action([5.0, 6.0], switching=1.0).action == 0
    assert policies.no_action([5.0, 6.1], switching=1.0).action == 1
    assert policies.no_action([5.0, 3.0], feasible=[False, True]).action == 1
    assert policies.no_action([5.0, 3.0], mandatory=1).reason == "mandatory"
    with pytest.raises(ValueError):
        policies.no_action([1.0, 2.0], feasible=[False, False])


def test_cost_and_uniform_liquidation_reference():
    cost = execution.QuadraticCost([0.2, 0.3], [[1.0, 0.1], [0.1, 2.0]])
    assert cost([2.0, -1.0]) == pytest.approx(0.7 + 4 - 0.4 + 2)
    r = execution.liquidate([10.0], [0.25] * 4, impact=[[0.5]], covariance=[[1.0]])
    np.testing.assert_allclose(r.trades[:, 0], [2.5] * 4, atol=1e-5)
    assert r.expected_cost == pytest.approx(50.0, abs=1e-5)
    assert r.inventory_penalty == 0
    assert r.inventory[-1, 0] == pytest.approx(0, abs=1e-8)
    front = execution.liquidate(
        [10.0], [0.25] * 4, impact=[[0.5]], covariance=[[1.0]], aversion=10.0
    )
    assert front.trades[0, 0] > r.trades[0, 0]
    assert front.inventory_penalty > 0
    assert front.objective == pytest.approx(
        front.expected_cost + front.inventory_penalty
    )


def test_liquidation_rate_constraints_and_signed_inventory():
    r = execution.liquidate(
        [2.0, -1.0, 0.0],
        [0.5, 0.5],
        impact=np.eye(3),
        covariance=np.eye(3),
        maximum_rate=[2.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(r.trades, [[1.0, -0.5, 0.0]] * 2, atol=1e-6)
    with pytest.raises(RuntimeError):
        execution.liquidate(
            [10.0], [1.0], impact=[[1.0]], covariance=[[1.0]], maximum_rate=1.0
        )


def test_assignment_generated_recovery_and_sampling():
    rng = np.random.default_rng(71)
    x = np.column_stack([np.ones(3000), rng.normal(size=3000)])
    beta = np.array([-0.4, 0.8])
    p = 1 / (1 + np.exp(-x @ beta))
    n = np.full(3000, 8)
    y = rng.binomial(n, p)
    fit = execution.fit_assignment(x, y, n)
    np.testing.assert_allclose(fit.coefficients, beta, atol=0.06)
    assert fit.covariance is not None and fit.log_likelihood < 0
    np.testing.assert_allclose(fit.probability(x[:5]), p[:5], atol=0.025)
    draws = execution.assignment([10, 20], [0.2, 0.7], draws=20000, rng=rng)
    np.testing.assert_allclose(draws.mean(0), [2, 14], atol=0.06)
    with pytest.raises(ValueError):
        execution.assignment([1.5], [0.5], draws=1, rng=rng)


def test_self_financing_trade_funding_and_collateral_conservation():
    account = wealth.Account([2.0, -1.0], 100.0)
    prices = [10.0, 20.0]
    initial = account.wealth(prices)
    traded = wealth.trade(account, [1.0, 2.0], prices, cost=3.0)
    assert traded.wealth(prices) == initial - 3
    posted = wealth.collateral_transfer(traded, 25.0)
    assert posted.wealth(prices) == traded.wealth(prices)
    released = wealth.collateral_transfer(posted, -25.0)
    assert released.cash == traded.cash
    funded = wealth.accrue(
        wealth.Account([2.0], -100, collateral=20),
        lending_factor=1.01,
        borrowing_factor=1.02,
        collateral_factor=1.005,
        distributions=3,
        fees=1,
    )
    assert funded.cash == pytest.approx(-97.0)
    assert funded.collateral == pytest.approx(20.1)
    with pytest.raises(ValueError):
        wealth.collateral_transfer(account, -1.0)


@pytest.mark.parametrize("quantity", [2.0, -2.0])
def test_exercise_and_settlement_wealth_identity(quantity):
    # ATM+10 call, multiplier 100, mark at intrinsic immediately before exercise.
    account = wealth.Account([quantity, 0.0], 10000.0)
    marks = [1000.0, 110.0]
    before = account.wealth(marks)
    exercised = wealth.exercise_delivery(
        account,
        0,
        quantity,
        underlying_index=1,
        shares_per_contract=100,
        cash_per_contract=-10000,
    )
    assert exercised.wealth(marks) == before
    assert exercised.holdings[0] == 0
    assert exercised.pending_holdings[1] == quantity * 100
    settled = wealth.settle(exercised)
    assert settled.wealth(marks) == before
    again = wealth.settle(settled)
    assert again.cash == settled.cash
    np.testing.assert_array_equal(again.holdings, settled.holdings)
    with pytest.raises(ValueError):
        wealth.exercise_delivery(
            account,
            0,
            -quantity,
            underlying_index=1,
            shares_per_contract=100,
            cash_per_contract=-10000,
        )


def test_financed_gains_futures_and_taylor_attribution():
    account = wealth.Account([0.0], 100.0)
    varied = wealth.futures_variation(
        account, [2.0], [100.0], [101.0], multipliers=50.0
    )
    assert varied.cash == 200.0
    assert (
        wealth.financed_gain(100, 200, funding_factor=1.02, terminal_external_flows=10)
        == 88
    )
    r = wealth.pnl_attribution(lambda x: x[0] ** 3, [2.0], [2.1], [12.0], [[12.0]])
    assert r.exact == pytest.approx(2.1**3 - 8)
    assert r.first_order == pytest.approx(1.2)
    assert r.second_order == pytest.approx(0.06)
    assert r.residual == pytest.approx(0.001)
