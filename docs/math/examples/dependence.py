"""covariance, dependence and extreme tails."""

import numpy as np
from numpy.testing import assert_allclose

from convexity.math import dependence as dep, distributions as dist


def main():
    rng = np.random.default_rng(42)
    loadings = np.array([[1.0, 0.2], [0.5, 1.0], [1.5, -0.4], [-0.2, 0.8]])
    factors = rng.normal(size=(3000, 2))
    x = factors @ loadings.T + 0.4 * rng.normal(size=(3000, 4))

    sample = dep.sample_covariance(x)
    assert_allclose(sample.covariance, np.cov(x, rowvar=False))
    ewma = dep.ewma_covariance(x, decay=0.97, ddof=1)
    shrunk = dep.LedoitWolf().fit(x[:30])
    oracle = dep.OAS().fit(x[:30])
    fixed = dep.ShrunkCovariance(0.2).fit(x[:30])
    assert_allclose(
        dep.ewma_covariance(x, decay=1, ddof=1).covariance, sample.covariance
    )
    print("EWMA effective observations:", ewma.effective_nobs)
    print(
        "Shrinkage (Ledoit-Wolf, OAS, supplied):",
        shrunk.shrinkage,
        oracle.shrinkage,
        fixed.shrinkage,
    )
    weights = np.ones(4) / 4
    assert_allclose(sample.variance(weights), np.var(x @ weights, ddof=1))

    model = dep.FactorCovariance(loadings, np.eye(2), np.full(4, 0.4**2))
    fitted = dep.fit_factor_covariance(x, factors, residual="full")
    assert_allclose(fitted.covariance, sample.covariance, atol=1e-12)
    assert_allclose(fitted.loadings, loadings, atol=0.03)
    assert_allclose(
        dep.correlation_to_covariance(model.correlation, model.standard_deviation),
        model.covariance,
    )
    print("Factor loading recovery and full residual covariance identity verified.")
    print(
        "Pearson / Spearman / Kendall, coordinates 0 and 1:",
        [
            dep.correlation(x, method=m)[0, 1]
            for m in ("pearson", "spearman", "kendall")
        ],
    )

    indefinite = np.array([[1.0, 1.2], [1.2, 1.0]])
    spectral = dep.PositiveSemidefinite(min_eigenvalue=1e-6).project(indefinite)
    correlation = dep.nearest_correlation(indefinite, min_eigenvalue=1e-6)
    assert dep.PositiveSemidefinite().contains(spectral.matrix)
    assert_allclose(np.diag(correlation.matrix), 1)
    print(
        "PSD / correlation repair distances:", spectral.distance, correlation.distance
    )

    shape = np.array([[1.0, 0.6], [0.6, 1.0]])
    heavy = dep.multivariate_student_t(6, [0.0, 1.0], shape)
    observations = heavy.sample(5000, rng=rng)
    t_fit = dep.StudentT(df=5.0, estimate_df=True).fit(observations)
    assert_allclose(t_fit.location, [0.0, 1.0], atol=0.08)
    assert_allclose(t_fit.covariance, t_fit.shape * t_fit.df / (t_fit.df - 2))
    print("Student-t fitted df and shape diagonal:", t_fit.df, t_fit.shape.diagonal())

    copula = dep.StudentTCopula(shape, df=6)
    joint = dep.CopulaDistribution(
        copula, [dist.student_t(6), dist.student_t(6, loc=1)]
    )
    assert_allclose(
        joint.log_density(observations[:20]),
        heavy.log_density(observations[:20]),
        atol=1e-10,
    )
    fitted_copula = dep.fit_copula(observations, family="student_t", df=6)
    assert isinstance(fitted_copula.copula, dep.StudentTCopula)
    assert_allclose(fitted_copula.copula.correlation, shape, atol=0.04)
    print("Student-t lower / upper tail dependence:", copula.tail_dependence())
    print(
        "Clayton lower / upper tail dependence:", dep.ClaytonCopula(2).tail_dependence()
    )
    print(
        "Gumbel lower / upper tail dependence:", dep.GumbelCopula(2).tail_dependence()
    )
    independent = dep.IndependenceCopula(2)
    assert_allclose(independent.cdf([0.2, 0.7]), 0.14)
    gaussian = dep.GaussianCopula(shape)
    assert gaussian.tail_dependence() == (0.0, 0.0)

    excesses = dep.generalized_pareto(0.2, scale=1.5).sample(4000, rng=rng)
    tail_sample = np.r_[np.zeros(16000), 3 + excesses]
    pot = dep.peaks_over_threshold(tail_sample, threshold=3)
    periods = np.array([10.0, 100.0, 1000.0])
    levels = pot.return_level(periods)
    assert_allclose(pot.survival(levels), 1 / periods)
    print(
        "POT xi, scale and original-sample exceedance rate:",
        pot.tail.xi,
        pot.tail.scale,
        pot.exceedance_rate,
    )
    print("POT return levels at 10 / 100 / 1000 observations:", levels)

    blocks = dep.block_maxima(rng.exponential(size=12000), block_size=12)
    gev = dep.fit_gev(blocks)
    assert_allclose(gev.distribution.survival(gev.return_level(periods)), 1 / periods)
    print("GEV return levels at 10 / 100 / 1000 blocks:", gev.return_level(periods))

    class DiagonalEstimator:
        def fit(self, samples):
            return dep.CovarianceEstimate(
                samples.mean(axis=0),
                np.diag(np.var(samples, axis=0, ddof=1)),
                nobs=len(samples),
            )

    custom = dep.fit_covariance(x, DiagonalEstimator())
    assert_allclose(np.diag(custom.covariance), np.diag(sample.covariance))
    print("Custom estimator contract verified. All family-5 examples passed.")


if __name__ == "__main__":
    main()
