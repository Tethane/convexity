"""Optional PyTorch training, function snapshots, RNG isolation and failure reports."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import learning as learn


@pytest.fixture
def torch():
    module = pytest.importorskip("torch", exc_type=ImportError)
    previous = module.get_num_threads()
    module.set_num_threads(1)
    yield module
    module.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(), (1,), (2,)])
def test_neural_linear_recovery_weights_and_rng_isolation(torch, shape):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(120, 2))
    beta = rng.normal(size=(2,) + shape)
    y = 1.5 + x @ beta
    spec = learn.NeuralRegressor(hidden_layers=(), learning_rate=.05, maxiter=1500, tolerance=1e-6, seed=8)
    before = torch.random.get_rng_state().clone()
    model = spec.fit(x, y, sample_weight=np.arange(1, 121))
    assert torch.equal(torch.random.get_rng_state(), before)
    assert model.diagnostics.converged
    assert model.loss_history is not None
    assert model.loss_history[-1] < 1e-10
    assert_allclose(model.predict_mean(x), y, atol=1e-5)
    assert model.predict_mean(x[:6].reshape(2, 3, 2)).shape == (2, 3) + shape
    assert model.predict_mean(np.empty((0, 2))).shape == (0,) + shape
    repeat = spec.fit(x, y, sample_weight=np.arange(1, 121))
    assert_allclose(model.predict_mean(x), repeat.predict_mean(x), atol=0, rtol=0)


def test_neural_nonlinear_mean_recovery(torch):
    x = np.linspace(-2, 2, 100)[:, None]
    y = np.sin(x[:, 0])
    model = learn.NeuralRegressor(hidden_layers=(12,), seed=7, maxiter=2500, tolerance=3e-4).fit(x, y)
    test = np.linspace(-1.95, 1.95, 53)[:, None]
    assert np.mean((model.predict_mean(test) - np.sin(test[:, 0])) ** 2) < .001


def test_neural_class_probabilities_match_logistic_function(torch):
    # Nonseparable labels give a finite optimum, so the gradient criterion is meaningful.
    rng = np.random.default_rng(24)
    x = rng.normal(size=(400, 2))
    from scipy.special import expit
    p = expit(.3 + x @ [1., -.5])
    y = np.where(rng.random(len(x)) < p, "yes", "no")
    model = learn.NeuralClassifier(hidden_layers=(), learning_rate=.05, maxiter=1800, tolerance=1e-6).fit(x, y)
    assert model.classes is not None
    assert np.array_equal(model.classes, ["no", "yes"])
    probabilities = model.predict_proba(x)
    assert_allclose(probabilities.sum(axis=-1), 1)
    assert np.mean((probabilities[:, 1] - p) ** 2) < .01
    assert model.classes is not None
    assert np.array_equal(model.predict(x), model.classes[probabilities.argmax(axis=-1)])
    assert not probabilities.flags.writeable
    assert model.loss_history is not None
    assert not model.loss_history.flags.writeable


def test_neural_iteration_limit_and_retained_model(torch):
    x = np.arange(20.).reshape(10, 2) / 10
    y = np.arange(10.)
    spec = learn.NeuralRegressor(hidden_layers=(), maxiter=1, tolerance=1e-15)
    with pytest.raises(learn.ConvergenceError) as error:
        spec.fit(x, y)
    result = error.value.result
    assert not result.diagnostics.converged
    assert result.diagnostics.iterations == 1
    assert result.loss_history is not None
    assert len(result.loss_history) == 2  # initial and final losses
    assert result.predict_mean(x).shape == (10,)
    inspected = learn.NeuralRegressor(hidden_layers=(), maxiter=1, tolerance=1e-15, check_convergence=False).fit(x, y)
    assert_allclose(result.predict_mean(x), inspected.predict_mean(x))


def test_user_neural_function_snapshot_and_eval_mode(torch):
    layer = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2., -1.]], dtype=torch.float64))
        layer.bias.fill_(3.)
    module = torch.nn.Sequential(layer, torch.nn.Dropout(.9))
    module.train()
    model = learn.neural_function(module, 2)
    x = np.array([[1., 2.], [3., 4.]])
    assert_allclose(model.predict_mean(x), [3., 5.])
    assert module.training
    assert layer.weight.requires_grad
    with torch.no_grad():
        layer.weight.zero_()
    assert_allclose(model.predict_mean(x), [3., 5.])


def test_neural_function_logits_class_order_and_wrong_output(torch):
    module = torch.nn.Identity()
    classifier = learn.neural_function(module, 3, classes=["z", "a", "b"])
    x = np.array([[1., 2., 3.]])
    from scipy.special import softmax
    assert_allclose(classifier.predict_proba(x), softmax(x, axis=-1))
    assert classifier.predict(x)[0] == "b"
    wrong = learn.neural_function(module, 3)
    with pytest.raises(ValueError, match="module must return"):
        wrong.predict(x)
