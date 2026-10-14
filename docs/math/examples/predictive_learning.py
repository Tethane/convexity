"""Regression, classification, and neural prediction. Run with --neural to include optional PyTorch fits."""

import argparse

import numpy as np
from numpy.testing import assert_allclose

from convexity.math import learning as learn, transforms as tx


def main(*, neural=False):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1200, 3))
    beta = np.array([1.5, -2.0, 0.5])
    y = 3 + x @ beta + 0.25 * rng.normal(size=len(x))
    train, test = x[:900], x[900:]
    ols = learn.linear_regression(train, y[:900])
    assert ols.coefficients is not None
    assert_allclose(ols.coefficients, beta, atol=0.04)
    penalized = [
        learn.Ridge(alpha=0.05),
        learn.Lasso(alpha=0.02),
        learn.ElasticNet(alpha=0.02),
    ]
    for tool in penalized:
        fit = learn.Pipeline((tx.Standardize(),), tool).fit(train, y[:900])
        assert np.mean((fit.predict_mean(test) - y[900:]) ** 2) < 0.13
    print("Linear recovery and train-only penalized pipelines verified.")

    offset = rng.uniform(-0.5, 0.5, len(x))
    mean = np.exp(0.4 + x @ [0.2, -0.15, 0.1] + offset)
    counts = rng.poisson(mean)
    poisson = learn.GeneralizedLinearModel("poisson").fit(x, counts, offset=offset)
    assert poisson.coefficients is not None
    assert_allclose(poisson.coefficients, [0.2, -0.15, 0.1], atol=0.08)
    assert poisson.coefficients is not None and poisson.intercept is not None
    assert_allclose(
        poisson.predict_mean(x[:5], offset=offset[:5]),
        np.exp(x[:5] @ poisson.coefficients + poisson.intercept + offset[:5]),
    )
    print("Poisson coefficients:", poisson.coefficients)

    from scipy.special import expit

    p = expit(-0.3 + x @ [0.8, -0.5, 0.2])
    trials = rng.integers(10, 30, len(x))
    successes = rng.binomial(trials, p)
    grouped = learn.GeneralizedLinearModel("binomial").fit(x, successes, trials=trials)
    assert grouped.coefficients is not None
    assert_allclose(grouped.coefficients, [0.8, -0.5, 0.2], atol=0.06)
    assert_allclose(grouped.predict_proba(test).sum(axis=-1), 1.0)
    labels = np.where(rng.random(len(x)) < p, "success", "failure")
    logistic = learn.LogisticRegression().fit(train, labels[:900])
    assert_allclose(logistic.predict_proba(test).sum(axis=-1), 1.0)
    print("Logistic probability column order:", logistic.classes)

    quantile = learn.QuantileRegression((0.1, 0.5, 0.9)).fit(train, y[:900])
    assert quantile.predict_quantiles(test).shape == (300, 3)
    assert quantile.coefficients is not None
    assert_allclose(
        quantile.coefficients, np.repeat(beta[:, None], 3, axis=1), atol=0.08
    )
    print(
        "Held-out quantile coverage:",
        np.mean(y[900:, None] <= quantile.predict_quantiles(test), axis=0),
    )

    nonlinear_x = rng.uniform(-2, 2, (900, 2))
    nonlinear_y = nonlinear_x[:, 0] ** 2 + (nonlinear_x[:, 1] > 0)
    for tool in [
        learn.DecisionTree(max_depth=6),
        learn.RandomForest(n_estimators=40),
        learn.ExtraTrees(n_estimators=40),
        learn.GradientBoosting(n_estimators=100),
    ]:
        fitted = tool.fit(nonlinear_x[:700], nonlinear_y[:700])
        mse = np.mean((fitted.predict_mean(nonlinear_x[700:]) - nonlinear_y[700:]) ** 2)
        assert mse < 0.1
        print(type(tool).__name__, "held-out MSE:", round(float(mse), 5))

    if neural:
        import torch

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            features = np.linspace(-2, 2, 120)[:, None]
            response = np.sin(features[:, 0])
            net = learn.NeuralRegressor(
                hidden_layers=(12,), seed=7, maxiter=3000, tolerance=3e-4
            ).fit(features, response)
            assert np.mean((net.predict_mean(features) - response) ** 2) < 0.001
            classifier = learn.NeuralClassifier(
                hidden_layers=(), maxiter=2000, tolerance=1e-5
            ).fit(train, labels[:900])
            assert_allclose(classifier.predict_proba(test).sum(axis=-1), 1.0)
            assert net.loss_history is not None
            print(
                "Neural final loss / updates:",
                net.loss_history[-1],
                net.diagnostics.iterations,
            )
        finally:
            torch.set_num_threads(previous_threads)
    print("All requested family-6 examples passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neural", action="store_true", help="also run optional PyTorch examples"
    )
    main(neural=parser.parse_args().neural)
