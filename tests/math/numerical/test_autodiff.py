import builtins
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import numerical as nm


def test_import_does_not_load_optional_torch():
    subprocess.run([sys.executable, "-c",
                    "import sys; from convexity.math import numerical; assert 'torch' not in sys.modules"],
                   check=True, capture_output=True, text=True)


def test_missing_torch_has_actionable_error(monkeypatch):
    original = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(ImportError, match=r"convexity\[autodiff\]"):
        nm.gradient(lambda x: x.sum(), [1, 2], tool=nm.AutoDiff())


def test_autodiff_scalar_gradient_jacobian_hessian():
    torch = pytest.importorskip("torch")
    tool = nm.AutoDiff()
    assert_allclose(nm.derivative(torch.sin, .3, tool=tool).value, np.cos(.3), atol=1e-14)
    assert_allclose(nm.gradient(lambda x: x @ x, [1., 2.], tool=tool).value, [2., 4.])
    function = lambda x: x[0]**3 + x[0]*x[1] + torch.exp(x[1])
    result = nm.hessian(function, [2., 0.], tool=tool)
    assert_allclose(result.value, [[12., 1.], [1., 1.]])
    assert result.diagnostics.error_estimate is None
    assert_allclose(nm.hessian(lambda x: x[0]**3 + x[0]*x[1] + np.exp(x[1]), [2., 0.]).value,
                    result.value, atol=1e-6)
    jac = nm.jacobian(lambda x: torch.stack([x[0]*x[1], x[0]**2, x[1]**2]), [2., 3.], tool=tool)
    assert_allclose(jac.value, [[3., 2.], [4., 0.], [0., 6.]])


def test_autodiff_constant_linear_and_invalid_output():
    torch = pytest.importorskip("torch")
    tool = nm.AutoDiff()
    assert_allclose(nm.gradient(lambda x: x.new_tensor(3.), [1., 2.], tool=tool).value, [0, 0])
    assert_allclose(nm.hessian(lambda x: x.sum(), [1., 2.], tool=tool).value, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="torch.Tensor"):
        nm.gradient(lambda x: 3., [1., 2.], tool=tool)
    with pytest.raises(ValueError, match="dimensions"):
        nm.gradient(lambda x: x*x, [1., 2.], tool=tool)
    with pytest.raises(ValueError, match="finite"):
        nm.gradient(lambda x: torch.log(-x).sum(), [1., 2.], tool=tool)
