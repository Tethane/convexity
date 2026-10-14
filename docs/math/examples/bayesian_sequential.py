"""Joint static drift and latent level inference with particles."""

import numpy as np
from convexity.math import bayesian as b, distributions as d, state_space as ss


def main():
    rng = np.random.default_rng(31)

    def state_model(theta):
        return ss.linear_gaussian(
            ss.GaussianState([0], [[0.1]]),
            [[1]],
            [[1]],
            [[0.03]],
            [[0.15]],
            transition_offset=[theta[0]],
        )

    observed = ss.simulate(state_model([0.2]), np.arange(10), rng=rng).observations
    model = b.StateParameterModel(
        d.independent([d.normal(0, 0.4)]), state_model, [0.0], ("drift",)
    )
    result = b.JointParticleFilter(particles=1200).smooth(
        model, observed, paths=100, rng=rng
    )
    print("Sequential posterior drift means:")
    print(
        [float(result.parameter_posterior(i).mean()[0]) for i in range(len(observed))]
    )
    print(
        "Final joint (drift, level) covariance:",
        result.joint_posterior(len(observed) - 1).covariance(),
    )
    print("ESS:", result.filtering.effective_sample_size)
    assert result.filtering.paths is not None
    print("Joint smoothing paths:", result.filtering.paths.shape)


if __name__ == "__main__":
    main()
