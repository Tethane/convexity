"""Irregular noisy position observations and latent velocity, generated locally."""
import numpy as np

from convexity.math import state_space as ss


def main():
    rng = np.random.default_rng(12)
    times = np.r_[0., np.cumsum(rng.uniform(.2, 1.5, 99))]
    model = ss.local_linear_trend(.03, .5,
        initial=ss.GaussianState([0., .4], [[.1, 0.], [0., .1]]))
    generated = ss.simulate(model, times, rng=rng)
    observations = generated.observations.values.copy()
    observations[::7] = np.nan
    observations[40:48] = np.nan
    data = ss.TimeSeries(observations, times, names=('position',))
    result = ss.KalmanFilter().smooth(model, data)
    assert result.smoothed is not None
    print('Backend:', result.backend, '| log likelihood:', round(result.log_likelihood, 3))
    for name, states in [('filter', result.filtered), ('smoother', result.smoothed)]:
        rmse = np.sqrt(np.mean((states.mean-generated.states)**2, axis=0))
        print(name, 'position/velocity RMSE:', rmse.round(4))
    print('Final latent state:', result.filtered.distribution(len(data)-1).mean())
    print('Smoothed position/velocity covariance at row 40:', result.smoothed.covariance[40])
    print('Missing rows have zero log likelihood:', np.all(result.log_likelihoods[np.isnan(observations[:, 0])] == 0))


if __name__ == '__main__':
    main()
