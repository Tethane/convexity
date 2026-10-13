"""Generated regimes, causal versus smoothed probabilities, and switching fits."""
import numpy as np
from convexity.math import distributions as d, regimes as r


def main():
    rng = np.random.default_rng(120)
    chain = r.MarkovChain([[.96, .04], [.08, .92]])
    model = r.HiddenMarkovModel(chain, [d.normal(-2, .6), d.normal(2, .8)], [.5, .5])
    generated = model.sample(800, rng=rng)
    y = generated.values[0]
    inferred = model.smooth(y)
    print('Stationary probabilities:', chain.stationary())
    print('Expected regime durations:', chain.durations())
    print('Final filtered probabilities:', inferred.filtered[-1])
    print('Exact known-model log likelihood:', inferred.log_likelihood)
    assert inferred.smoothed is not None
    print('Generated-label smoothing accuracy:', np.mean(inferred.smoothed.argmax(axis=1) == generated.regimes[0]))

    fitted = r.SwitchingRegression(switching_variance=True).fit(y)
    print('Fit converged:', fitted.converged)
    print('Regime means and variances:', fitted.coefficients[:, 0], fitted.variances)
    print('Transition rows:', fitted.chain.transition)
    forecast = fitted.predict(5, draws=3000, rng=rng)
    print('Future means:', forecast.mean().ravel())
    print('Joint forecast covariance:', forecast.covariance())

    dynamics = r.SwitchingDynamics(chain,
        lambda regime, state, step: d.normal(.8*state + [-.2, .3][regime], .1))
    print('User-defined switching path:', dynamics.sample(0., 0, 8, rng=rng).values[0])


if __name__ == '__main__':
    main()
