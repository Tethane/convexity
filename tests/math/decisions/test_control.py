import numpy as np
import pytest
from convexity.math import control


def model():
    return control.FiniteMDP(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
        [[1.0, 5.0], [0.0, 0.0]],
        0.9,
        terminal=[False, True],
    )


def test_bellman_horizon_and_infinite_reference():
    mdp = model()
    r = control.finite_horizon(mdp, 2)
    np.testing.assert_allclose(r.values[:, 0], [5.5, 5.0, 0.0])
    np.testing.assert_array_equal(r.policy[:, 0], [0, 1])
    r = control.value_iteration(mdp)
    assert r.values[0] == pytest.approx(10.0, abs=1e-8)
    assert r.policy[0] == 0 and r.residual <= 1e-10


def test_feasible_actions_terminal_and_nonconvergence():
    mdp = control.FiniteMDP(
        [[[1.0], [1.0]]], [[1.0, 100.0]], 0.5, feasible=[[True, False]]
    )
    r = control.value_iteration(mdp)
    assert r.values[0] == pytest.approx(2.0)
    assert r.policy[0] == 0
    with pytest.raises(RuntimeError):
        control.value_iteration(mdp, maxiter=1)
    with pytest.raises(ValueError):
        control.value_iteration(control.FiniteMDP([[[1.0]]], [[1.0]], 1.0))
    with pytest.raises(ValueError):
        control.FiniteMDP([[[0.5]]], [[1.0]])


def test_belief_recursion_values_information_without_foresight():
    p = np.repeat(np.eye(2)[:, None, :], 2, axis=1)
    mdp = control.FiniteMDP(p, [[1.0, 0.0], [0.0, 1.0]], 1.0)
    informative = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    uninformative = np.ones((2, 2, 1))
    informed = control.belief_plan(mdp, informative, [0.5, 0.5], 2)
    blind = control.belief_plan(mdp, uninformative, [0.5, 0.5], 2)
    assert informed.value == pytest.approx(1.5)
    assert blind.value == pytest.approx(1.0)
    assert informed.action == 0
    np.testing.assert_allclose(
        control.belief_update([0.5, 0.5], np.eye(2), [0.9, 0.1]), [0.9, 0.1]
    )
    with pytest.raises(ValueError):
        control.belief_update([0.5, 0.5], np.eye(2), [0, 0])
    with pytest.raises(ValueError):
        control.belief_plan(mdp, informative, [0.5, 0.5], 10, max_nodes=5)


def test_lqr_scalar_riccati_identity():
    r = control.lqr([[1.0]], [[1.0]], [[1.0]], [[1.0]])
    golden = (1 + np.sqrt(5)) / 2
    assert r.riccati[0, 0] == pytest.approx(golden)
    assert r.gain[0, 0] == pytest.approx(golden / (1 + golden))
    assert r.residual < 1e-10
    assert abs(r.closed_loop[0, 0]) < 1


def test_tree_nonanticipativity_known_stochastic_solution():
    tree = control.ScenarioTree([-1, 0, 0], [1.0, 0.5, 0.5], [[0.0], [-1.0], [1.0]])
    r = control.scenario_mpc(
        tree, [[1.0]], [[1.0]], [[0.0]], [[1.0]], [2.0], terminal_cost=[[1.0]]
    )
    assert r.actions[0, 0] == pytest.approx(-1.0, abs=1e-6)
    assert r.objective == pytest.approx(3.0, abs=1e-6)
    np.testing.assert_allclose(r.states[:, 0], [2.0, 0.0, 2.0], atol=1e-6)
    # A clairvoyant scenario-specific root would achieve 2.5, which is excluded.
    assert r.objective > 2.5
    np.testing.assert_allclose(tree.masses(), [1, 0.5, 0.5])


def test_tree_limits_and_infeasibility():
    tree = control.ScenarioTree([-1, 0], [1.0, 1.0], [[0.0], [0.0]])
    r = control.scenario_mpc(
        tree,
        [[1.0]],
        [[1.0]],
        [[0.0]],
        [[1.0]],
        [2.0],
        terminal_cost=[[1.0]],
        lower_actions=-0.2,
    )
    assert r.actions[0, 0] == pytest.approx(-0.2, abs=1e-5)
    with pytest.raises(RuntimeError):
        control.scenario_mpc(
            tree, [[1.0]], [[1.0]], [[1.0]], [[1.0]], [2.0], upper_states=1.0
        )
    with pytest.raises(ValueError):
        control.ScenarioTree([-1, 0, 0], [1, 0.4, 0.4], [[0.0], [0.0], [0.0]])


def test_generator_hjb_and_impulse_control():
    g = np.array([[[-2.0, 2.0]], [[1.0, -1.0]]])
    reward = np.array([[1.0], [3.0]])
    r = control.generator_hjb(g, reward, 0.5)
    exact = np.linalg.solve(0.5 * np.eye(2) - g[:, 0, :], reward[:, 0])
    np.testing.assert_allclose(r.values, exact, atol=1e-8)
    assert r.residual < 1e-9
    mdp = control.FiniteMDP([[[1.0, 0.0]], [[0.0, 1.0]]], [[0.0], [1.0]], 0.9)
    r = control.impulse_control(mdp, [[1], [1]], [[1.0], [100.0]])
    np.testing.assert_allclose(r.values, [8.0, 10.0], atol=1e-8)
    assert r.policy[0] == 1
