"""Nonlinear Gaussian filters and a non-Gaussian count-observation particle model.

Install convexity[state-space] to run all three established inference backends.
"""

import numpy as np

from convexity.math import distributions as dist, state_space as ss


def main():
    transition = ss.GaussianTransition(
        1,
        lambda x, step: x - 0.25 * np.sin(x) * step.dt,
        lambda step: [[0.04 * step.dt]],
        lambda x, step: [[1 - 0.25 * np.cos(x[0]) * step.dt]],
    )
    model = ss.StateSpaceModel(
        ss.GaussianState([0.4], [[0.2]]),
        transition,
        ss.GaussianObservation(
            1,
            1,
            lambda x, step: x + 0.1 * x**3,
            [[0.15]],
            lambda x, step: [[1 + 0.3 * x[0] ** 2]],
        ),
        time_mode="elapsed",
    )
    times = np.cumsum(np.random.default_rng(1).uniform(0.3, 1.2, 50))
    generated = ss.simulate(model, times, rng=np.random.default_rng(2))
    y = generated.observations.values.copy()
    y[::8] = np.nan
    data = ss.TimeSeries(y, times)
    for estimator in (ss.ExtendedKalmanFilter(), ss.UnscentedKalmanFilter()):
        result = estimator.smooth(model, data)
        assert result.smoothed is not None
        print(
            result.method,
            "smoothed RMSE:",
            np.sqrt(np.mean((result.smoothed.mean - generated.states) ** 2)),
        )

    # Reuse the same latent transition with a count distribution.
    count_model = ss.StateSpaceModel(
        model.initial,
        transition,
        ss.DistributionObservation(
            1, 1, lambda x, step: dist.independent([dist.poisson(3 * np.exp(x[0]))])
        ),
        time_mode="elapsed",
    )
    counts = ss.simulate(count_model, times[:25], rng=np.random.default_rng(3))
    particle_filter = ss.ParticleFilter(500)
    filtered = particle_filter.filter(
        count_model, counts.observations, rng=np.random.default_rng(4)
    )
    smoothed = particle_filter.smooth_result(
        filtered, paths=100, method="ancestral", rng=np.random.default_rng(5)
    )
    print("Count model log likelihood estimate:", round(filtered.log_likelihood, 3))
    assert filtered.effective_sample_size is not None
    print(
        "Minimum effective sample size:",
        round(float(np.min(filtered.effective_sample_size)), 1),
    )
    assert smoothed.paths is not None
    print("Joint latent paths (draw,time,state):", smoothed.paths.shape)
    print("Posterior path mean:", smoothed.path_distribution().mean()[:5])


if __name__ == "__main__":
    main()
