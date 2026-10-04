"""Keep financial contracts separate from replaceable vendor execution."""

import ast
import importlib
from pathlib import Path

import pytest

FAMILIES = (
    "measures",
    "contracts",
    "curves",
    "pricing",
    "implied_volatility",
    "calibration",
    "stopping",
    "sensitivities",
    "fixed_income",
    "rates",
    "credit",
)


@pytest.mark.parametrize("family", FAMILIES)
def test_specialized_providers_are_behind_backend_boundary(family):
    module_file = importlib.import_module(f"convexity.math.{family}").__file__
    assert module_file is not None
    root = Path(module_file).parent
    for layer in ("interfaces", "builtins", "_backends"):
        assert (root / layer / "__init__.py").is_file()
    for layer in ("interfaces", "builtins"):
        for path in (root / layer).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    imports = [node.module or ""]
                else:
                    continue
                assert not any(
                    name.split(".")[0] in {"QuantLib", "scipy", "sklearn", "torch"}
                    for name in imports
                ), path
    providers = [
        node
        for path in (root / "_backends").glob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert providers, (
        f"{family} needs execution providers, not an empty backend directory"
    )


@pytest.mark.parametrize(
    "family,layer,name",
    [
        ("pricing", "builtins", "VanillaEngine"),
        ("curves", "builtins", "YieldCurve"),
        ("rates", "builtins", "ShortRateModel"),
        ("stopping", "builtins", "ExercisePolicy"),
        ("credit", "interfaces", "CDSResult"),
        ("calibration", "interfaces", "CalibrationResult"),
        ("rates", "interfaces", "RatePaths"),
        ("fixed_income", "interfaces", "BondResult"),
    ],
)
def test_public_types_are_owned_by_public_layers(family, layer, name):
    cls = getattr(importlib.import_module(f"convexity.math.{family}"), name)
    assert cls.__module__.startswith(f"convexity.math.{family}.{layer}.")
