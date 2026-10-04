"""State-space innovation likelihood composed with the existing estimation family."""
import numpy as np

from convexity.math import estimation as est, state_space as ss


def main():
    generated = ss.simulate(ss.local_level(.15, .4), np.arange(500.), rng=np.random.default_rng(17))

    def objective(log_variances):
        q, r = np.exp(log_variances)
        model = ss.local_level(q, r)
        return -ss.KalmanFilter().filter(model, generated.observations).log_likelihood

    estimate = est.minimize_estimate(objective, np.log([.3, .6]))
    print('True process/observation variances:', [.15, .4])
    print('Estimated variances:', np.exp(estimate.parameters).round(4))
    print('Converged:', estimate.diagnostics.converged)


if __name__ == '__main__':
    main()
