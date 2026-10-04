import numpy as np
from scipy.special import expit, gammaln
from scipy.optimize import minimize
from ..interfaces import LogisticFit


def predict(x, beta):
    return expit(x @ beta)


def fit(x, y, n, lo, hi):
    def fun(beta):
        z = x @ beta
        return float(np.sum(n * np.logaddexp(0.0, z) - y * z))

    def jac(beta):
        return x.T @ (n * expit(x @ beta) - y)

    initial = np.clip(np.zeros(x.shape[1]), lo, hi)
    result = minimize(
        fun,
        initial,
        jac=jac,
        bounds=list(zip(lo, hi)),
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 10000},
    )
    if not result.success:
        raise RuntimeError(f"binomial logistic fit failed: {result.message}")
    p = predict(x, result.x)
    information = x.T @ ((n * p * (1 - p))[:, None] * x)
    covariance = None
    active = np.any(
        np.isclose(result.x, lo, atol=1e-6, rtol=0)
        | np.isclose(result.x, hi, atol=1e-6, rtol=0)
    )
    if not active and np.linalg.matrix_rank(information) == x.shape[1]:
        covariance = np.linalg.inv(information)
    combinatorial = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    return LogisticFit(
        result.x, covariance, float(combinatorial.sum() - result.fun), int(result.nit)
    )
