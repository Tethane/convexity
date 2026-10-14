import numpy as np
import pytest
from convexity.math import optimization as opt, robust, risk


def test_lp_known_solution_duality_and_owned_results():
    p = opt.QuadraticProgram(
        [-3.0, -2.0], inequality=[[1, 1], [1, 0]], upper=[4, 2], lower_bounds=0
    )
    r = opt.solve(p)
    assert r.solution is not None
    np.testing.assert_allclose(r.solution, [2, 2], atol=1e-6)
    assert r.objective == pytest.approx(-10.0, abs=1e-6)
    np.testing.assert_allclose(r.duals["inequality"], [2, 1], atol=1e-5)
    assert r.max_violation is not None and r.max_violation < 1e-6
    assert r.solution is not None
    assert not r.solution.flags.writeable


def test_qp_and_soc_known_solutions():
    r = opt.solve(opt.QuadraticProgram([-2.0, -4.0], quadratic=2 * np.eye(2)))
    assert r.solution is not None
    np.testing.assert_allclose(r.solution, [1, 2], atol=1e-6)
    cone = opt.SecondOrderCone(np.eye(2), [0, 0], [0, 0], 1)
    r = opt.solve(opt.QuadraticProgram([-3.0, -4.0], cones=(cone,)))
    assert r.solution is not None
    np.testing.assert_allclose(r.solution, [0.6, 0.8], atol=1e-6)


def test_integer_lp_matches_enumeration():
    p = opt.QuadraticProgram(
        [-3.0, -2.0],
        inequality=[[2, 1]],
        upper=[3],
        domains=["binary", "integer"],
        lower_bounds=0,
        upper_bounds=[1, 3],
    )
    r = opt.solve(p, opt.ConvexSolver("SCIPY"))
    exact = opt.enumerate_minimum(
        lambda x: -3 * x[0] - 2 * x[1],
        [[0, 1], [0, 1, 2, 3]],
        feasible=lambda x: bool(2 * x[0] + x[1] <= 3),
    )
    assert r.objective == pytest.approx(exact.objective)
    assert exact.solution is not None and r.solution is not None
    np.testing.assert_allclose(r.solution, exact.solution)


def test_failure_not_reported_as_solution_and_extension():
    p = opt.QuadraticProgram([1.0], inequality=[[1.0], [-1.0]], upper=[0.0, -1.0])
    with pytest.raises(opt.OptimizationError):
        opt.solve(p)
    assert not opt.solve(p, check_success=False).success
    r = opt.enumerate_minimum(lambda x: x[0], [[1, 2]], feasible=lambda x: False)
    assert r.status == "infeasible" and r.solution is None
    with pytest.raises(ValueError):
        opt.enumerate_minimum(sum, [range(100)] * 3, max_candidates=100)

    class CustomSolver:
        def solve(self, problem):
            return opt.SolveResult([1.0], 2.0, "optimal", "custom", 0.0)

    assert opt.solve(object(), CustomSolver()).objective == 2.0


@pytest.mark.parametrize(
    "domains,expected",
    [
        (["integer", "integer"], [1.0, 1.0]),
        (["binary", "binary"], [1.0, 1.0]),
        (["integer", "continuous", "integer"], [1.0, 1.5, 1.0]),
        (["binary", "integer", "binary"], [1.0, 1.0, 1.0]),
    ],
)
def test_multiple_discrete_coordinates(domains, expected):
    problem = opt.QuadraticProgram(
        [-1.0] * len(domains), lower_bounds=0.0, upper_bounds=1.5, domains=domains
    )
    result = opt.solve(problem, opt.ConvexSolver("SCIPY"))
    assert result.solution is not None
    np.testing.assert_allclose(result.solution, expected)
    assert result.max_violation == 0.0


def test_nonlinear_constrained_optimum():
    p = opt.NonlinearProblem(
        lambda x: (x[0] - 2) ** 2 + (x[1] - 2) ** 2,
        [0.0, 0.0],
        inequalities=lambda x: [x.sum() - 1],
    )
    r = opt.solve(p)
    assert r.solution is not None
    np.testing.assert_allclose(r.solution, [0.5, 0.5], atol=1e-5)
    assert r.status == "local_optimum"


def test_invalid_problem_rejected():
    with pytest.raises(ValueError):
        opt.QuadraticProgram([1.0, 2.0], quadratic=[[1, 3], [3, 1]])
    with pytest.raises(ValueError):
        opt.QuadraticProgram([1.0], lower_bounds=2, upper_bounds=1)
    with pytest.raises(ValueError):
        opt.QuadraticProgram([1.0], inequality=[[1.0]])


def test_support_functions_against_extrema():
    box = robust.Box([1.0, 2.0], [0.5, 1.0])
    d = np.array([-2.0, 3.0])
    exact = max(d @ [x, y] for x in [0.5, 1.5] for y in [1.0, 3.0])
    assert box.support(d) == pytest.approx(exact)
    ellipse = robust.Ellipsoid([1.0, 2.0], [[2.0, 0.0], [1.0, 1.0]], 2.0)
    u = ellipse.shape.T @ d
    extremum = ellipse.center + 2 * ellipse.shape @ (u / np.linalg.norm(u))
    assert ellipse.support(d) == pytest.approx(d @ extremum)


def test_minimax_cvar_and_robust_allocation():
    r = robust.minimax([[1.0, -1.0], [-1.0, 1.0]])
    assert r.solution is not None
    np.testing.assert_allclose(r.solution, [0.5, 0.5], atol=1e-5)
    returns = np.array([[0.1, -0.1], [-0.2, 0.2], [0.05, 0.03]])
    w = np.array([0.2, 0.4, 0.4])
    r = robust.mean_cvar(returns, weights=w, aversion=2, level=0.5)
    gains = returns @ r.solution
    assert r.objective == pytest.approx(
        -w @ gains + 2 * risk.expected_shortfall(-gains, 0.5, weights=w), abs=1e-6
    )
    m, c = np.array([0.1, 0.1]), np.eye(2)
    r = robust.robust_allocation(m, c, robust.Ellipsoid(m, np.eye(2), 0.2))
    assert r.solution is not None
    np.testing.assert_allclose(r.solution, [0.5, 0.5], atol=1e-5)


def test_gaussian_chance_and_kl_ambiguity_limits():
    cone = robust.gaussian_chance_constraint(
        [0.0], [[1.0]], 1.6448536269514722, probability=0.95
    )
    r = opt.solve(opt.QuadraticProgram([-1.0], cones=(cone,)))
    assert r.solution is not None
    assert r.solution[0] == pytest.approx(1.0, abs=1e-6)
    losses, weights = [0.0, 2.0, 100.0], [0.7, 0.3, 0.0]
    r0 = robust.kl_worst_case(losses, 0, weights=weights)
    r1 = robust.kl_worst_case(losses, 0.1, weights=weights)
    r2 = robust.kl_worst_case(losses, 2.0, weights=weights)
    assert r0.objective == pytest.approx(0.6)
    assert r1.objective is not None and r2.objective is not None
    assert 0.6 < r1.objective < r2.objective <= 2.00001
    assert r1.solution is not None
    assert r1.solution[2] == 0
    assert r2.objective == pytest.approx(2.0, abs=1e-6)
    assert robust.chance_probability([-1, 2], weights=[0.8, 0.2]) == pytest.approx(0.8)


def test_mean_cvar_mean_uncertainty_and_zero_radius():
    returns = np.array([[0.03, 0.01], [-0.04, -0.01], [0.06, 0.02], [0.01, 0.0]])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    mean = probabilities @ returns
    shape = np.diag([0.04, 0.01])
    nominal = robust.mean_cvar(returns, weights=probabilities, level=0.5)
    zero = robust.mean_cvar(
        returns,
        weights=probabilities,
        level=0.5,
        ambiguity=robust.Ellipsoid(mean, shape, 0),
    )
    assert zero.objective == pytest.approx(nominal.objective, abs=1e-7)

    uncertain = robust.mean_cvar(
        returns,
        weights=probabilities,
        level=0.5,
        ambiguity=robust.Ellipsoid(mean, shape, 2),
    )
    assert uncertain.solution is not None and nominal.solution is not None
    allocation = uncertain.solution
    expected = (
        -mean @ allocation
        + 2 * np.linalg.norm(shape.T @ allocation)
        + risk.expected_shortfall(-returns @ allocation, 0.5, weights=probabilities)
    )
    assert uncertain.objective == pytest.approx(expected, abs=1e-7)
    assert uncertain.solution[0] < nominal.solution[0]

    with pytest.raises(ValueError, match="scenario-weighted mean"):
        robust.mean_cvar(returns, ambiguity=robust.Ellipsoid(mean + 0.1, shape))
