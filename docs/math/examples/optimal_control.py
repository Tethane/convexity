"""exact Bellman, Riccati and nonanticipative stochastic MPC."""

from convexity.math import control

mdp = control.FiniteMDP(
    [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
    [[1.0, 5.0], [0.0, 0.0]],
    0.9,
    terminal=[False, True],
)
finite = control.finite_horizon(mdp, 4)
print("Finite-horizon values:", finite.values[:, 0], "actions:", finite.policy[:, 0])
print("Infinite-horizon values:", control.value_iteration(mdp).values)
print("LQR gain:", control.lqr([[1.0]], [[1.0]], [[1.0]], [[1.0]]).gain)
tree = control.ScenarioTree([-1, 0, 0], [1.0, 0.5, 0.5], [[0.0], [-1.0], [1.0]])
plan = control.scenario_mpc(
    tree, [[1.0]], [[1.0]], [[0.0]], [[1.0]], [2.0], terminal_cost=[[1.0]]
)
print("MPC first action:", plan.actions[0], "objective:", plan.objective)
