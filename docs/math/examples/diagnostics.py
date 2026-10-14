"""proper scores, residual diagnostics and block uncertainty."""

import numpy as np
from convexity.math import diagnostics

rng = np.random.default_rng(3500)
innovations = rng.normal(size=500)
series = np.zeros(500)
for t in range(1, len(series)):
    series[t] = 0.7 * series[t - 1] + innovations[t]
print("Autocorrelation test:", diagnostics.residual_test(series))
print("Unit-root test:", diagnostics.stationarity_test(series, lags=2))
print(
    "Weighted CRPS:", diagnostics.crps([-2.0, 1.0, 3.0], 2.0, weights=[0.1, 0.3, 0.6])
)
bootstrap = diagnostics.block_bootstrap(
    np.mean, series, block_length=15, draws=499, rng=rng
)
print("Block mean interval:", bootstrap.lower, bootstrap.upper)
print("HAC mean variance:", diagnostics.hac_mean_covariance(series, lags=14))
print("Holm adjusted:", diagnostics.adjust_pvalues([0.01, 0.04, 0.03]))
