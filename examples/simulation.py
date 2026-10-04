"""Generated multi-asset paths, weighted laws, observations and integration error."""
import numpy as np
from convexity.math import simulation as sim, distributions as dist, processes


def main():
    rng = np.random.default_rng(42)
    times = np.linspace(0, 1, 13)
    noise = sim.brownian_increments(times, draws=4096, dimension=2,
                                    rng=rng, antithetic=True)
    paths = sim.geometric_brownian_paths([100, 80], [.03, .02],
                                         [[.2, 0], [.12, .16]], noise)
    basket = sim.estimate_scenarios(paths, lambda path: path[-1].mean())
    assert basket.standard_error is not None
    print('Terminal basket mean / antithetic standard error:',
          float(basket.value), float(basket.standard_error))
    print('Probability both assets finish above initial levels:',
          float(paths.probability(lambda path: np.all(path[-1] > path[0]))))
    tilted = paths.reweight(np.where(paths.values[:, -1, 0] < 100, 2., 1.))
    print('Reweighted terminal means:', tilted.mean()[-1])
    print('Weight effective size:', tilted.effective_size)
    print('Forecast retains weighted mean:', tilted.to_forecast().mean()[-1])

    observed = sim.simulate_observations(
        sim.PathScenarios(times, paths.values[:100]),
        lambda time, state: dist.multivariate_normal(state, np.eye(2)*.25), rng=rng)
    print('Joint latent/observation shape:', observed.joint().values.shape)

    ou = processes.OrnsteinUhlenbeck(1.2, theta=.5, sigma=.3)
    def coefficients(time, dt):
        decay = np.exp(-ou.kappa*dt)
        return decay, ou.theta*(1-decay), ou.sigma**2*(-np.expm1(-2*ou.kappa*dt))/(2*ou.kappa)
    law = sim.gaussian_path_distribution(0., times, coefficients)
    print('Exact OU terminal mean / variance:', law.mean()[-1], law.variance()[-1])

    qmc = sim.qmc_expectation(lambda z: np.exp(.2*z[:, 0]), 1,
                              draws=1024, replicates=8, rng=rng)
    assert qmc.standard_error is not None
    print('RQMC E[exp(.2Z)] / standard error / analytic:',
          float(qmc.value), float(qmc.standard_error), np.exp(.02))


if __name__ == '__main__':
    main()
