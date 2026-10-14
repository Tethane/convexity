import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from convexity.math import transforms as tx


@pytest.mark.parametrize("lag,order", [(1, 1), (1, 3), (4, 1), (4, 2), (5, 3)])
@pytest.mark.parametrize("dimension", [1, 3])
def test_differencing_round_trip_and_continuation(lag, order, dimension):
    x = np.random.default_rng(7).normal(size=(60, dimension)).cumsum(axis=0)
    if dimension == 1:
        x = x[:, 0]
    training = tx.difference(x[:40], lag=lag, order=order)
    whole = tx.difference(x, lag=lag, order=order)
    assert_allclose(training.inverse_transform(), x[:40], atol=1e-11)
    assert_array_equal(training.rows, np.arange(lag * order, 40))
    future = whole.values[40 - lag * order :]
    assert_allclose(
        training.inverse_transform(future, continuation=True), x[40:], atol=1e-11
    )
    expected = x
    for _ in range(order):
        expected = expected[lag:] - expected[:-lag]
    assert_allclose(whole.values, expected)
    with pytest.raises(ValueError, match="explicit future"):
        training.inverse_transform(continuation=True)


def test_lag_order_alignment_and_no_future_rows():
    x = np.arange(24).reshape(12, 2)
    result = tx.lag_features(x, (3, 0, 1))
    assert_array_equal(result.rows, np.arange(3, 12))
    assert result.columns == (
        "x0_lag3",
        "x1_lag3",
        "x0_lag0",
        "x1_lag0",
        "x0_lag1",
        "x1_lag1",
    )
    for j, lag in enumerate((3, 0, 1)):
        assert_array_equal(result.values[:, 2 * j : 2 * j + 2], x[result.rows - lag])
    changed = x.copy()
    changed[9:] = -100
    assert_array_equal(
        tx.lag_features(changed, (3, 0, 1)).values[:6], result.values[:6]
    )
    assert tx.lag_features(x[:, 0], (0, 2)).values.shape == (10, 2)


def test_fourier_formula_phase_continuation_and_orthogonality():
    feature = tx.FourierFeatures(12, 3)
    a, b, whole = feature(9), feature(15, start=9), feature(24)
    assert_allclose(np.concatenate((a.values, b.values)), whole.values, atol=1e-14)
    t = np.arange(9, 24)
    expected = np.column_stack(
        [f(2 * np.pi * j * t / 12) for j in range(1, 4) for f in (np.sin, np.cos)]
    )
    assert_allclose(b.values, expected, atol=1e-14)
    assert_allclose(whole.values.T @ whole.values, np.eye(6) * 12, atol=1e-13)
    assert_array_equal(b.rows, t)
    fractional = tx.fourier_features(6, 7.5, order=2, start=100)
    assert_allclose(
        fractional.values[:, 0],
        np.sin(2 * np.pi * np.arange(100, 106) / 7.5),
        atol=1e-13,
    )


def test_seasonal_dummies_baseline_and_continuation():
    full = tx.seasonal_features(9, 4, start=3)
    assert_array_equal(full.values, np.eye(4)[np.arange(3, 12) % 4])
    reduced = tx.seasonal_features(9, 4, start=3, drop_first=True)
    assert_array_equal(reduced.values, full.values[:, 1:])
    assert reduced.columns == ("season_1", "season_2", "season_3")


def test_rank_deficient_spanning_set_projection_pythagoras_and_zero_subspace():
    a = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    space = tx.subspace(a)
    assert space.dimension == 2
    x = np.random.default_rng(8).normal(size=(3, 4))
    p, r = space.project(x), space.residuals(x)
    assert_allclose(p + r, x)
    assert_allclose(p @ r.T, 0, atol=1e-14)
    assert_allclose(np.sum(x * x), np.sum(p * p) + np.sum(r * r))
    assert_allclose(space.project(p), p)
    assert_allclose(space.projector, np.diag([1.0, 1.0, 0.0, 0.0]), atol=1e-14)
    zero = tx.subspace(np.zeros((4, 2)))
    assert zero.dimension == 0
    assert zero.coordinates(x).shape == (3, 0)
    assert_allclose(zero.project(x), 0)
    assert_allclose(zero.residuals(x), x)
    assert tx.principal_angles(space, zero).shape == (0,)
    assert tx.align_subspaces(zero, zero).rotation.shape == (0, 0)


def test_alignment_rotates_scores_without_changing_reconstruction():
    source = tx.Subspace(np.eye(4)[:, :2])
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target = tx.Subspace(source.basis @ rotation)
    alignment = tx.align_subspaces(source, target)
    assert_allclose(alignment.rotation, rotation, atol=1e-14)
    assert_allclose(alignment.aligned.basis, target.basis, atol=1e-14)
    assert_allclose(alignment.angles, 0, atol=1e-14)
    assert alignment.error < 1e-14
    scores = np.array([[1.0, 2.0], [3.0, 4.0]])
    rotated = alignment.transform(scores)
    assert_allclose(
        alignment.aligned.reconstruct(rotated), source.reconstruct(scores), atol=1e-14
    )
    assert_allclose(alignment.inverse_transform(rotated), scores)
    other = tx.Subspace(np.eye(4)[:, 2:])
    assert_allclose(tx.principal_angles(source, other), np.pi / 2)
    assert_allclose(tx.align_subspaces(source, other).error, 2.0)


def test_principal_angles_with_unequal_ranks_and_cutoff():
    source = tx.subspace(np.diag([1.0, 1e-10, 0.0]), rcond=1e-8)
    target = tx.Subspace(np.eye(3)[:, :2])
    assert source.dimension == 1
    assert_allclose(tx.principal_angles(source, target), [0.0])
    with pytest.raises(ValueError, match="equal subspace"):
        tx.align_subspaces(source, target)
