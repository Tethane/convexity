import numpy as np
from typing import Any, cast
from inspect import signature
from scipy.stats import jarque_bera, norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import S_hac_simple
from ..interfaces import TestResult


def residual_test(x, method, lags):
    if method == "jarque_bera":
        r = cast(Any, jarque_bera(x))
        return TestResult(r.statistic, r.pvalue, method, "residuals are Gaussian")
    r = acorr_ljungbox(x, lags=lags, return_df=True)
    return TestResult(
        r.lb_stat.to_numpy(),
        r.lb_pvalue.to_numpy(),
        method,
        "no autocorrelation through each supplied lag",
    )


def stationarity(x, method, regression, lags):
    if method == "adf":
        # Preserve the stable tuple contract on both statsmodels 0.14 and 0.15.
        options = (
            {"result_object": False}
            if "result_object" in signature(adfuller).parameters
            else {}
        )
        r = cast(Any, adfuller)(
            x,
            maxlag=lags,
            regression=regression,
            autolag="AIC" if lags is None else None,
            **options,
        )
        return TestResult(r[0], r[1], "ADF", "unit root", {"lags": r[2], "nobs": r[3]})
    r = kpss(x, regression=regression, nlags="auto" if lags is None else lags)
    return TestResult(r[0], r[1], "KPSS", "level/trend stationarity", {"lags": r[2]})


def cointegration(y, x, trend, lags):
    r = coint(y, x, trend=trend, maxlag=lags, autolag="aic" if lags is None else None)
    return TestResult(r[0], r[1], "Engle-Granger", "no cointegration")


def hac(x, lags):
    centered = x - x.mean(axis=0)
    return S_hac_simple(centered, nlags=lags) / len(x) ** 2


def compare(difference, lags):
    variance = float(hac(difference[:, None], lags)[0, 0])
    if variance <= 0:
        raise ValueError("model comparison requires positive HAC variance")
    statistic = float(difference.mean() / np.sqrt(variance))
    return TestResult(
        statistic,
        float(2 * norm.sf(abs(statistic))),
        "HAC paired score comparison",
        "equal expected scores",
        {
            "mean_difference": float(difference.mean()),
            "standard_error": float(np.sqrt(variance)),
        },
    )


def adjust(p, method):
    return multipletests(p, method={"holm": "holm", "bh": "fdr_bh"}[method])[1]
