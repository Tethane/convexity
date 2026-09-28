"""One executable example for every numerical family-1 section; no external data."""
import argparse
import numpy as np
from convexity.math import numerical as cx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--autodiff', action='store_true')
    args = parser.parse_args()

    a = np.array([[3., 1.], [1., 2.]])
    print('Dense solve:', cx.solve(a, [9., 8.]).value)
    print('Least squares:', cx.lstsq([[1, 0], [0, 1], [1, 1]], [1, 2, 3]).value)
    print('QR factors:', tuple(cx.qr(a).factors))
    print('Generalized eigenvalues:', cx.generalized_eig(a, np.diag([1., 2.])).values)
    print('Condition number:', cx.condition_number(a).value)
    print('Log determinant:', cx.slogdet(a))

    diagonal = np.arange(1., 7.)
    operator = cx.linear_operator((6, 6), lambda x:diagonal*x, rmatvec=lambda x:diagonal*x)
    print('Matrix-free solve:', cx.solve(operator, diagonal).value)
    print('Partial singular values:', cx.sparse_svd(operator, 2).s)
    print('Fixed point:', cx.fixed_point(np.cos, 1.).value)
    print('Nonlinear system:', cx.root(lambda x:x*x-2, [1.]).value)

    function = lambda x:np.array([x[0]*x[1], x[0]**2])
    print('Jacobian-vector product:', cx.jvp(function, [1., 2.], [3., 4.]).value)
    print('Gradient check:', cx.check_derivative(lambda x:x @ x, [1., 2.], [2., 4.], kind='gradient').passed)
    print('2-D integral:', cx.nquad(lambda x:x[0]+x[1], [[0, 1], [0, 1]]).value)
    print('Gaussian integral:', cx.gauss_quad(lambda x:x**4, 0, 1, n=3).value)
    print('Supplied-density integral:', cx.integrate_density(lambda x:x*x, lambda x:np.exp(-x), 0, np.inf).value)

    grid = cx.linspace(0, 1, 9)
    xx, yy = cx.meshgrid(grid, grid)
    surface = cx.regular_grid((grid, grid), xx+2*yy)
    print('Regular-grid interpolation:', surface([.2, .3]))
    approximation = cx.polynomial_approximation(grid, 1+2*grid+grid**2, 2)
    print('Polynomial coefficients:', approximation.coefficients)

    event = cx.ODEEvent(lambda t,y:y[0]-.5, terminal=True, direction=-1)
    trajectory = cx.solve_ivp(lambda t,y:-y, [1.], (0, 2), events=(event,))
    print('ODE half-life event:', trajectory.events[0].t)
    print('Dense ODE solution at t=.3:', trajectory.solution(.3))
    boundary = cx.solve_bvp(lambda x,y:np.vstack([y[1], np.zeros_like(x)]),
                                lambda left,right:[left[0],right[0]-1], grid, np.zeros((2, grid.size)))
    print('Boundary-value solution at .5:', boundary.solution(.5))
    difference = cx.differentiation_matrix(grid)
    print('Derivative matrix on x^2:', difference @ grid**2)
    print('Laplacian shape:', cx.laplacian_matrix((5, 5)).shape)
    print('Machine epsilon:', cx.machine_precision().epsilon)
    print('Residual:', cx.residual(a, [2., 3.], [9., 8.]).absolute)

    def custom(problem: cx.LinearSystem) -> cx.ArrayResult:
        return cx.ArrayResult(np.linalg.solve(np.asarray(problem.a), np.asarray(problem.b)))
    backend = cx.scipy_backend().with_tools(solve=custom)
    print('Custom backend solve:', backend.run('solve', cx.LinearSystem(a, [9., 8.])).value)
    if args.autodiff:
        tools = cx.torch_backend()
        print('Automatic gradient:', cx.gradient(lambda x:(x*x).sum(), [1., 2.], tool=tools.tool('gradient')).value)
        print('Automatic JVP:', cx.jvp(lambda x:x*x, [1., 2.], [3., 4.], tool=tools.tool('jvp')).value)


if __name__ == '__main__':
    main()
