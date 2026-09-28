"""A small probability problem book. Run from an installed Convexity environment."""

import numpy as np
from numpy.testing import assert_allclose

from convexity.math import distributions as dist


def main():
    rng = np.random.default_rng(42)

    gaussian = dist.normal(2, 3)
    assert_allclose(gaussian.expect(lambda x: x*x), 13)
    print("Gaussian 5/50/95% quantiles:", gaussian.quantile([.05, .5, .95]))
    counts = dist.poisson(3)
    print("Poisson P(X <= 2):", float(counts.cdf(2)))

    atoms = dist.empirical([[0, 1], [0, 3], [1, 4]], [1, 3, 4])
    assert_allclose(atoms.condition([0], [0]).mean(), [2.5])
    print("Empirical expected sum:", float(atoms.map(lambda x: x.sum()).mean()))

    mixed = dist.mixture([dist.normal(-2, 1), dist.normal(4, 2)], [1, 3])
    assert_allclose(mixed.mean(), 2.5)
    assert_allclose(mixed.variance(), 10)
    p = np.array([.1, .5, .9])
    assert_allclose(mixed.cdf(mixed.quantile(p)), p)
    print("Mixture 10/50/90% quantiles:", mixed.quantile(p))
    print("Mixture sample shape:", mixed.sample((2, 3), rng=rng).shape)

    joint = dist.multivariate_normal([0, 0], [[1, .5], [.5, 1]])
    conditional = joint.condition([1], [2])
    assert_allclose(conditional.mean(), [1])
    assert_allclose(conditional.covariance(), [[.75]])
    print("Conditional Gaussian mean/covariance:", conditional.mean(), conditional.covariance())

    student = dist.multivariate_student_t(5, [0, 0], [[1, .5], [.5, 1]])
    conditional_t = student.condition([1], [2])
    assert_allclose(conditional_t.covariance(), [[1.6875]])
    assert_allclose(conditional_t.log_density([1]) + student.marginal([1]).log_density([2]),
                    student.log_density([1, 2]))
    print("Conditional Student-t covariance:", conditional_t.covariance())

    positive = dist.exponential(dist.normal(.2, .7))
    assert_allclose(positive.density([.5, 1, 2]), dist.lognormal(.2, .7).density([.5, 1, 2]))
    print("Transformed expectation:", float(positive.expect(lambda x: x)))


if __name__ == "__main__":
    main()
