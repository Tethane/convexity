import ast
import importlib
from pathlib import Path
import subprocess
import sys
import pytest

FAMILIES = "risk utility optimization robust control policies execution wealth diagnostics reinforcement".split()


@pytest.mark.parametrize("family", FAMILIES)
def test_public_layers_and_backend_boundaries(family):
    module = importlib.import_module(f"convexity.math.{family}")
    assert module.__file__ is not None
    root = Path(module.__file__).parent
    for layer in ("interfaces", "builtins", "_backends"):
        assert (root / layer / "__init__.py").is_file()
    for layer in ("interfaces", "builtins"):
        for path in (root / layer).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom) and not node.level
                    else []
                )
                assert not any(
                    name.split(".")[0]
                    in {
                        "cvxpy",
                        "scipy",
                        "statsmodels",
                        "torch",
                        "control",
                        "stable_baselines3",
                        "gymnasium",
                    }
                    for name in names
                ), path
    assert all(hasattr(module, name) for name in module.__all__)


def test_optional_rl_is_lazy():
    code = 'import sys; from convexity.math import reinforcement; assert "torch" not in sys.modules; assert "stable_baselines3" not in sys.modules'
    subprocess.run([sys.executable, "-c", code], check=True)
