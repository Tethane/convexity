"""Optional tensor graph, batching, backend fidelity and capability checks."""

import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import distributions as d


def test_import_does_not_load_torch():
    subprocess.run([sys.executable, "-c", "import sys; from convexity.math import distributions; "
                    "assert 'torch' not in sys.modules"], check=True)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_normal_parameter_gradients_and_batch_shapes(torch):
    loc = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    law = d.tensor.normal(loc, scale)
    x = torch.tensor([3., -1.], dtype=torch.float64)
    loc_grad, scale_grad = torch.autograd.grad(law.log_density(x).sum(), (loc, scale))
    assert_allclose(loc_grad, [2/4, -3/9])
    assert_allclose(scale_grad, [0, 0], atol=1e-15)
    assert law.batch_shape == (2,) and law.event_shape == ()
    assert law.sample((4, 3)).shape == (4, 3, 2)
    assert law.mean().dtype == torch.float64
    assert law.mean().device == loc.device
    joint = d.tensor.independent(law)
    assert joint.event_shape == (2,) and joint.batch_shape == ()
    assert joint.log_density(x).shape == ()
    assert joint.rsample(4).shape == (4, 2)


def test_reparameterized_sampling_retains_graph_and_reproducibility(torch):
    loc = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    law = d.tensor.normal(loc, torch.tensor(3., dtype=torch.float64))
    with torch.random.fork_rng():
        torch.manual_seed(3)
        a = law.rsample(10)
        torch.manual_seed(3)
        b = law.rsample(10)
    assert torch.equal(a, b)
    assert torch.autograd.grad(a.mean(), loc)[0] == 1
    assert not law.sample(5).requires_grad
    with pytest.raises(NotImplementedError):
        d.tensor.bernoulli(.5).rsample(2)


def test_tensor_catalog_and_cdf_agreement(torch):
    for tensor_law, array_law, x in [
        (d.tensor.student_t(5.), d.student_t(5), 1.),
        (d.tensor.lognormal(.2, .7), d.lognormal(.2, .7), 2.),
        (d.tensor.bernoulli(.3), d.bernoulli(.3), 1.),
        (d.tensor.binomial(5, .3), d.binomial(5, .3), 2.),
        (d.tensor.poisson(3.), d.poisson(3.), 2.),
    ]:
        assert_allclose(tensor_law.log_density(torch.tensor(x)), array_law.log_density(x), rtol=1e-6)
        assert_allclose(tensor_law.mean(), array_law.mean(), rtol=1e-6)
    law = d.tensor.normal(torch.tensor(0., dtype=torch.float64), 1.)
    p = torch.tensor([.1, .5, .9], dtype=torch.float64)
    assert_allclose(law.cdf(law.quantile(p)), p)
    with pytest.raises(ValueError):
        law.quantile(torch.tensor(2.))
    with pytest.raises(NotImplementedError):
        d.tensor.student_t(5.).cdf(torch.tensor(0.))


def test_vector_and_transformed_tensor_laws(torch):
    loc = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    matrix = torch.eye(2, dtype=torch.float64)
    law = d.tensor.multivariate_normal(loc, matrix)
    assert_allclose(law.log_density(torch.ones(2, dtype=torch.float64)).detach(), d.multivariate_normal([0, 0], np.eye(2)).log_density([1, 1]))
    assert law.covariance().shape == (2, 2)
    assert law.rsample(3).shape == (3, 2)
    base = d.tensor.normal(torch.tensor(0., requires_grad=True), 1.)
    positive = d.tensor.transformed(base, [torch.distributions.transforms.ExpTransform()])
    assert positive.rsample(10).requires_grad
    weights = torch.tensor([.25, .75], requires_grad=True)
    mix = d.tensor.mixture(weights, d.tensor.normal(torch.tensor([-2., 2.]), 1.))
    mix.log_density(torch.tensor(1.)).backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    assert "rsample" not in mix.capabilities
    wrapped = d.tensor.from_torch(torch.distributions.Normal(0., 1.), cdf=True, quantile=True)
    assert_allclose(wrapped.cdf(torch.tensor(0.)), .5)
