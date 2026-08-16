"""The one decision: which reduction of ``Cov(Xw)`` a joint draw uses.

``build_joint_reduction`` ranks three exact routes by the only cost they
share, triangular solves against the cached factor: ``size`` for weight
space, ``n_rows`` for the row side, ``|U|`` for the column side. These
check that it picks the smallest, that every route it picks agrees
distributionally with weight space, and that declining leaves the draw
bit-for-bit unchanged.
"""

from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from bayesianbandits import NormalRegressor
from bayesianbandits import _estimators as E
from bayesianbandits._estimators import _RowSpaceDraw, build_joint_reduction
from bayesianbandits._support_covariance import SupportDraw, support_of


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
    """Sparse rows touching ``n_distinct`` columns in total (default: up
    to ``n_rows * nnz_per_row``)."""
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
    def test_declines_when_weight_space_is_cheapest(self):
        """``size=1`` costs one solve, which no reduction can beat --
        this is the Thompson sampling hot path."""
        est, rng = _fit_dense(1000)
        for n_rows in (1, 10, 96):
            X = rng.standard_normal((n_rows, 1000))
            assert _route(est, X, 1, 1000) is None

    def test_takes_the_row_side_when_rows_beat_draws(self):
        est, rng = _fit_dense(1000)
        X = rng.standard_normal((10, 1000))
        assert isinstance(_route(est, X, 1000, 1000), _RowSpaceDraw)

    def test_declines_dense_when_rows_approach_features(self):
        """At ``n ~ d`` the two do comparable work, but the row side does
        it in many small latency-bound calls; ``2nk <= d^2`` stops it."""
        est, rng = _fit_dense(100)
        X = rng.standard_normal((96, 100))
        assert _route(est, X, 10_000, 100) is None

    def test_takes_the_column_side_when_support_is_smallest(self):
        """Many rows over few distinct columns: ``|U|`` beats both
        ``n_rows`` and ``size``."""
        est, rng, d = _fit_sparse()
        X = _sparse_X(rng, n_rows=200, d=d, n_distinct=40)
        n_u = support_of(X).size
        assert n_u < 200
        assert isinstance(_route(est, X, 500, d), SupportDraw)

    def test_takes_the_row_side_when_rows_are_smallest(self):
        """Few rows, each touching many distinct columns: ``n_rows``
        beats ``|U|``, so the row side wins even though both apply."""
        est, rng, d = _fit_sparse()
        X = _sparse_X(rng, n_rows=4, d=d)
        assert support_of(X).size > 4
        assert isinstance(_route(est, X, 1000, d), _RowSpaceDraw)

    def test_picks_the_smaller_of_the_two_reductions(self):
        """Same estimator, same ``size``: which side is chosen must
        follow ``min(n_rows, |U|)`` and nothing else."""
        est, rng, d = _fit_sparse()
        wide = _sparse_X(rng, n_rows=4, d=d)  # |U| ~ 80 > n_rows 4
        narrow = _sparse_X(rng, n_rows=200, d=d, n_distinct=40)  # |U| ~ 40 < 200
        assert support_of(wide).size > 4
        assert support_of(narrow).size < 200
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


class TestRoutesAgreeDistributionally:
    @pytest.mark.parametrize("route", ["row", "column"])
    def test_moments_match_weight_space(self, route):
        """Every route the gate can pick must reproduce weight space's
        mean and full covariance, not just its marginals."""
        size = 40_000
        if route == "row":
            est, rng = _fit_dense(60)
            X = rng.standard_normal((6, 60))
            d = 60
            expected = _RowSpaceDraw
        else:
            est, rng, d = _fit_sparse(d=5_000, rows=300)
            X = _sparse_X(rng, n_rows=12, d=d, nnz_per_row=5, n_distinct=8)
            expected = SupportDraw
        assert isinstance(_route(est, X, size, d), expected)

        est.random_state_ = np.random.default_rng(1)
        routed = np.asarray(est.sample(X, size=size))
        with mock.patch.object(E, "build_joint_reduction", return_value=None):
            est.random_state_ = np.random.default_rng(2)
            weight = np.asarray(est.sample(X, size=size))

        sd = weight.std(axis=0)
        z = np.abs(routed.mean(axis=0) - weight.mean(axis=0)) / (sd / np.sqrt(size))
        assert z.max() < 5.0
        # covariance, so cross-row dependence is checked and not just spread
        c_routed, c_weight = np.cov(routed, rowvar=False), np.cov(weight, rowvar=False)
        assert_allclose(
            c_routed, c_weight, rtol=0.10, atol=0.02 * np.abs(c_weight).max()
        )


class TestRowSpaceDraw:
    def test_draws_are_zero_mean_with_the_right_covariance(self):
        """``_RowSpaceDraw`` returns the zero-mean part only; callers add
        the predictive mean, exactly as ``SupportDraw`` requires."""
        rng = np.random.default_rng(0)
        L = np.tril(rng.standard_normal((4, 4))) + 2.0 * np.eye(4)
        draws = _RowSpaceDraw(L).joint(200_000, np.random.default_rng(3))
        assert draws.shape == (200_000, 4)
        assert np.abs(draws.mean(axis=0)).max() < 0.05
        assert_allclose(np.cov(draws, rowvar=False), L @ L.T, rtol=0.05, atol=0.02)
