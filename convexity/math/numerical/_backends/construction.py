import numpy as np
from typing import Any, cast
from scipy.sparse import csr_array
from scipy.sparse.linalg import LaplacianNd
from .. import _validation as v


def grid(problem):
    if problem.kind == "range":
        value = np.arange(problem.start, problem.stop, problem.step, dtype=float)
    else:
        operation = {"linear": np.linspace, "log": np.logspace, "geometric": np.geomspace}[problem.kind]
        options = {"base": problem.base} if problem.kind == "log" else {}
        value = operation(problem.start, problem.stop, num=problem.num, endpoint=problem.endpoint, **options)
    return v.array(value, "grid", ndim=1, real=True)


def mesh(problem):
    arrays = tuple(np.meshgrid(*problem.axes, indexing="ij", sparse=problem.sparse))
    if problem.cartesian:
        return v.array(np.stack([a.ravel(order="C") for a in arrays], axis=-1), "Cartesian product", real=True)
    return tuple(v.array(a, "mesh", real=True) for a in arrays)


def difference(problem):
    from findiff import Diff
    operator = None
    for axis, (coordinates, order) in enumerate(zip(problem.axes, problem.orders)):
        if not order:
            continue
        spacing = float(coordinates[1]-coordinates[0]) if problem.periodic else coordinates
        derivative = Diff(axis, spacing, acc=problem.accuracy, periodic=problem.periodic) ** order
        operator = derivative if operator is None else operator * derivative
    assert operator is not None
    return csr_array(operator.matrix(tuple(x.size for x in problem.axes)))


def laplacian(problem) -> csr_array:
    return cast(csr_array, csr_array(LaplacianNd(problem.shape, boundary_conditions=problem.boundary,
                                 dtype=cast(Any, np.float64)).tosparse()) / problem.spacing**2)
