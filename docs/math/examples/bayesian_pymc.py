"""MCMC, ADVI, and posterior prediction with the bayesian extra installed."""

import numpy as np
from convexity.math import bayesian as b


def main():
    import pymc as pm

    rng = np.random.default_rng(21)
    observations = rng.normal(1.2, 0.7, 80)

    def build():
        with pm.Model() as model:
            mean = pm.Normal("mean", 0, 3)
            scale = pm.HalfNormal("scale", 2)
            pm.Normal("observations", mean, scale, observed=observations)
        return model

    model = b.PyMCModel(build)
    posterior = b.MCMC(draws=600, tune=600, chains=2).fit(model, rng=rng)
    print("MCMC mean, scale:", posterior.mean("mean"), posterior.mean("scale"))
    print("R-hat:", dict(posterior.diagnostics.rhat or {}))
    print("Bulk ESS:", dict(posterior.diagnostics.ess_bulk or {}))
    predictive = posterior.posterior_predictive(rng=rng)
    print("Predictive replicated data shape:", predictive.samples["observations"].shape)
    approximate = b.VariationalInference(iterations=8000, draws=2000).fit(
        model, rng=rng
    )
    print("ADVI mean, scale:", approximate.mean("mean"), approximate.mean("scale"))
    assert approximate.diagnostics.loss is not None
    print("Last negative ELBO:", approximate.diagnostics.loss[-1])


if __name__ == "__main__":
    main()
