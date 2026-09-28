import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import numerical as n


@pytest.mark.parametrize("method", ["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"])
def test_ivp_methods_and_dense_solution(method):
    times = np.linspace(0, 1, 11)
    result = n.solve_ivp(lambda t,y:-y, [1., 2.], (0., 1.), t_eval=times, tool=n.IVPSolver(method))
    assert_allclose(result.y, np.array([[1.], [2.]])*np.exp(-times), atol=1e-6)
    assert_allclose(result.solution(.35), np.array([1., 2.])*np.exp(-.35), atol=1e-6)
    assert result.diagnostics.converged
    with pytest.raises(ValueError, match="domain"):
        result.solution(1.1)


@pytest.mark.parametrize("method", ["Radau", "BDF", "LSODA"])
def test_stiff_ivp_with_jacobian(method):
    result = n.solve_ivp(lambda t,y:-1000*(y-np.cos(t))-np.sin(t), [1.], (0, 1),
                         jacobian=lambda t,y:np.array([[-1000.]]), tool=n.IVPSolver(method))
    assert_allclose(result.solution(np.linspace(0, 1, 9))[0], np.cos(np.linspace(0, 1, 9)), atol=2e-6)


def test_backward_ode_and_namespace_convenience():
    times = np.linspace(1, 0, 11)
    result = n.ode.solve(lambda t,y:-y, [np.exp(-1)], times)
    assert_allclose(result.y[0], np.exp(-times), atol=1e-6)
    assert_allclose(result.solution(.5), [np.exp(-.5)], atol=1e-6)


def test_terminal_event_before_first_output_and_event_direction():
    event = n.ODEEvent(lambda t,y:y[0]-.5, terminal=True, direction=-1)
    result = n.solve_ivp(lambda t,y:-y, [1.], (0, 2), t_eval=[1., 2.], events=(event,))
    assert result.terminated_by_event and result.diagnostics.converged
    assert result.y.shape == (1, 0)
    assert_allclose(result.events[0].t, [np.log(2)], atol=1e-7)
    assert_allclose(result.events[0].y, [[.5]], atol=1e-9)
    with pytest.raises(ValueError):
        result.solution(1.)
    no_event = n.ODEEvent(lambda t,y:y[0]-.5, direction=1)
    complete = n.solve_ivp(lambda t,y:-y, [1.], (0, 2), events=(no_event,))
    assert complete.events[0].t.size == 0
    assert complete.events[0].y.shape == (0, 1)


def test_oscillator_system_and_nonterminal_events():
    event = n.ODEEvent(lambda t,y:y[0], direction=-1)
    result = n.solve_ivp(lambda t,y:[y[1],-y[0]], [1., 0.], (0, 7), events=(event,),
                         tool=n.IVPSolver(method="DOP853", max_step=.2))
    assert_allclose(result.events[0].t, [np.pi/2], atol=1e-7)
    assert_allclose(result.solution(np.array([0., 1., 2.]))[0], np.cos([0, 1, 2]), atol=1e-7)


def test_ivp_failure_retains_partial_solution():
    with pytest.raises(n.ConvergenceError) as caught:
        n.solve_ivp(lambda t,y:y*y, [1.], (0, 2))
    result = caught.value.result
    assert not result.diagnostics.converged
    assert result.t[-1] < 2
    assert_allclose(result.solution(.5), [2.], atol=1e-5)


@pytest.mark.parametrize("factory", [
    lambda:n.solve_ivp(lambda t,y:y, [1.], (0, 1), t_eval=[.5, .2]),
    lambda:n.solve_ivp(lambda t,y:y, [1.], (0, 1), t_eval=[2.]),
    lambda:n.solve_ivp(lambda t,y:[1, 2], [1.], (0, 1)),
    lambda:n.solve_ivp(lambda t,y:y, [1.], (0, 1), jacobian=lambda t,y:[[1.]]),
    lambda:n.solve_ivp(lambda t,y:y, [1.], (0, 0)),
    lambda:n.solve_ivp(lambda t,y:[np.nan], [1.], (0, 1)),
])
def test_invalid_ivp_contracts(factory):
    with pytest.raises(ValueError):
        factory()


def test_boundary_value_analytic_solution_and_dense_output():
    # y'' = -pi^2 sin(pi*x), y(0)=y(1)=0.
    x = np.linspace(0, 1, 7)
    result = n.solve_bvp(lambda x,y:np.vstack([y[1],-np.pi**2*np.sin(np.pi*x)]),
                         lambda a,b:[a[0], b[0]], x, np.zeros((2, x.size)))
    points = np.linspace(0, 1, 31)
    assert_allclose(result.solution(points)[0], np.sin(np.pi*points), atol=2e-6)
    assert_allclose(result.solution(points)[1], np.pi*np.cos(np.pi*points), atol=2e-6)
    assert np.max(result.rms_residuals) <= 1e-6
    assert result.diagnostics.converged


def test_bvp_nonconvergence_and_boundary_shape():
    x = np.linspace(0, 1, 5)
    with pytest.raises(n.ConvergenceError) as caught:
        n.solve_bvp(lambda x,y:np.vstack([y[1],-100*np.sin(10*x)]),
                     lambda a,b:[a[0],b[0]-np.sin(10)], x, np.zeros((2, 5)),
                     tool=n.BVPSolver(tolerance=1e-10, max_nodes=5))
    assert not caught.value.result.diagnostics.converged
    with pytest.raises(ValueError, match="boundary residual"):
        n.solve_bvp(lambda x,y:y, lambda a,b:[0.], x, np.zeros((2, 5)))
