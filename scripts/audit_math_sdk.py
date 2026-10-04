"""Inventory the installed public SDK; does not mistake imports for correctness.

Run with the desired interpreter: python scripts/audit_math_sdk.py --output PATH.
The JSON records public signatures, missing documentation and source hashes.
Numerical evidence comes separately from tests/math and executable examples.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import convexity.math as sdk


def inventory():
    families = []
    root = Path(sdk.__file__).parent
    for family in sdk.__all__:
        module = importlib.import_module(f"convexity.math.{family}")
        exports = []
        assert len(module.__all__) == len(set(module.__all__))
        for name in module.__all__:
            value = getattr(module, name)
            try:
                signature = str(inspect.signature(value))
            except (ValueError, TypeError):
                signature = None
            exports.append(dict(name=name, signature=signature,
                                module=getattr(value, '__module__', None),
                                doc=inspect.getdoc(value)))
        families.append(dict(name=family, exports=exports))
    dependencies = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'statsmodels', 'QuantLib',
                 'cvxpy', 'control', 'torch', 'arch', 'pymc', 'particles',
                 'filterpy', 'sdeint', 'sktime', 'neuralforecast',
                 'chronos-forecasting', 'timesfm', 'stable-baselines3'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.py'))}
    return dict(python=platform.python_version(), dependencies=dependencies,
                families=families, source_sha256=hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Inventoried {len(result['families'])} families into {args.output}")
