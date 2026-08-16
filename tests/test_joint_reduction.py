"""The one decision: which reduction of ``Cov(Xw)`` a joint draw uses.

Picking the wrong route is a performance bug, not a correctness one, so
the decision table below is deliberately compact. The correctness
property that does matter is that declining leaves the draw bit-for-bit
unchanged, which is what keeps seeded Thompson sampling stable.
"""

from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

from bayesianbandits import NormalRegressor
from bayesianbandits import _estimators as E
from bayesianbandits._estimators import _RowSpaceDraw, build_joint_reduction
from bayesianbandits._support_covariance import SupportDraw


def _fit_dense(d, rows=None, seed=0):
    rng = np.random.default_rng(seed)
    est = NormalRegressor(alpha=1.0, beta=1.0, random_state=0)
    n = rows or max(300, d // 2)
    est.fit(rng.standard_normal((n, d)), rng.standard_normal(n))
    return est, rng


def _fit_sparse(d=20_000, nnz_per_row=20, rows=500, seed=0):
    rng = np.random.default_rng(seed)
    est = NormalRegressor(alpha=1.0, beta=1.0, random_state=0, sparse=True)
    r = np.repeat(np.arange(rows), nnz_per_row)
    c = rng.integers(0, d, size=r.size)
    X = sp.csc_array((rng.standard_normal(r.size), (r, c)), shape=(rows, d))
    est.fit(X, rng.standard_normal(rows))
    return est, rng, d


def _sparse_X(rng, n_rows, d, nnz_per_row=20, n_distinct=None):
    cols = (
        rng.integers(0, d, size=n_rows * nnz_per_row)
        if n_distinct is None
        else rng.choice(n_distinct, size=n_rows * nnz_per_row)
    )
    r = np.repeat(np.arange(n_rows), nnz_per_row)
    return sp.csc_array((rng.standard_normal(r.size), (r, cols)), shape=(n_rows, d))


def _route(est, X, size, d):
    return build_joint_reduction(
        est._precision_factor, X, d, size, est.sparse, est._precision_nnz
    )


class TestRouteSelection:
    """``min(size, n_rows, |U|)`` and nothing else."""

    def test_dense_decision_table(self):
        est, rng = _fit_dense(1000)
        # size=1 costs one solve, which no reduction can beat
        assert _route(est, rng.standard_normal((10, 1000)), 1, 1000) is None
        # few rows, many draws: the row side wins
        assert isinstance(
            _route(est, rng.standard_normal((10, 1000)), 1000, 1000), _RowSpaceDraw
        )
        # at n ~ d the row side does comparable work in many small calls
        small, rng2 = _fit_dense(100)
        assert _route(small, rng2.standard_normal((96, 100)), 10_000, 100) is None

    def test_sparse_picks_the_smaller_of_the_two_reductions(self):
        est, rng, d = _fit_sparse()
        wide = _sparse_X(rng, n_rows=4, d=d)  # |U| ~ 80 > n_rows 4
        narrow = _sparse_X(rng, n_rows=200, d=d, n_distinct=40)  # |U| ~ 40 < 200
        assert isinstance(_route(est, wide, 5000, d), _RowSpaceDraw)
        assert isinstance(_route(est, narrow, 5000, d), SupportDraw)


class TestDeclineIsExactlyWeightSpace:
    @pytest.mark.parametrize(
        ("d", "n_rows", "size"), [(1000, 96, 1), (1000, 96, 100), (1000, 320, 1)]
    )
    def test_declined_draws_are_bit_for_bit_unchanged(self, d, n_rows, size):
        est, rng = _fit_dense(d)
        X = rng.standard_normal((n_rows, d))
        assert _route(est, X, size, d) is None

        est.random_state_ = np.random.default_rng(5)
        routed = est.sample(X, size=size)
        with mock.patch.object(E, "build_joint_reduction", return_value=None):
            est.random_state_ = np.random.default_rng(5)
            weight = est.sample(X, size=size)
        assert np.array_equal(routed, weight)
