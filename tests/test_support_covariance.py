"""Tests for the support-covariance sampling route.

The route replaces a per-draw (or per-row) solve against the precision
factor with ``|U|`` solves, where ``U`` is the set of columns the design
matrix touches. It is an exact reformulation, so every test here is
either an identity check against the path it replaces or a
distributional check that the two agree.
"""

from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.stats import ks_2samp

from bayesianbandits import (
    BayesianGLM,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from bayesianbandits import _support_covariance as sc
from bayesianbandits._estimators import _marginal_predictive_sd
from bayesianbandits._sparse_bayesian_linear_regression import create_sparse_factor


@pytest.fixture(autouse=True)
def suitesparse_envvar(sparse_solver):
    """Run every test in this module against both sparse backends."""
    yield


@contextmanager
def gate_off():
    """Force the pre-existing path, for A/B comparison."""
    with mock.patch.object(sc, "build", lambda *a, **k: None):
        yield


def make_precision(p, rng):
    off = rng.uniform(-0.4, 0.4, p - 1)
    return sp.diags([off, np.full(p, 2.0), off], [-1, 0, 1]).tocsc()


def make_design(p, n_rows, n_u, rng, nnz_per_row=6):
    """Sparse X whose columns are confined to a random support of size n_u."""
    support = np.sort(rng.choice(p, n_u, replace=False))
    rows, cols, vals = [], [], []
    for i in range(n_rows):
        pick = rng.choice(n_u, nnz_per_row, replace=False)
        rows += [i] * nnz_per_row
        cols += list(support[pick])
        vals += list(rng.standard_normal(nnz_per_row))
    X = sp.csr_array((vals, (rows, cols)), shape=(n_rows, p))
    return sp.csc_array(X), support


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------


def test_support_of_finds_exactly_the_touched_columns():
    rng = np.random.default_rng(0)
    X, support = make_design(400, 30, 12, rng)
    assert np.array_equal(sc.support_of(X), support)


def test_support_of_empty_matrix():
    assert sc.support_of(sp.csc_array((5, 100))).size == 0


def test_compact_columns_matches_fancy_indexing_without_copying():
    rng = np.random.default_rng(1)
    X, support = make_design(400, 30, 12, rng)
    compact = sc._compact_columns(X, support)

    assert_allclose(compact.todense(), X[:, support].todense())
    # data/indices are re-wrapped views, not copies
    assert np.shares_memory(compact.data, X.data)
    assert np.shares_memory(compact.indices, X.indices)


def test_support_covariance_equals_the_principal_submatrix():
    rng = np.random.default_rng(2)
    p = 300
    precision = make_precision(p, rng)
    factor = create_sparse_factor(precision)
    _, support = make_design(p, 10, 15, rng)

    S = sc.support_covariance(factor, support, p)
    expected = np.linalg.inv(precision.toarray())[np.ix_(support, support)]

    assert_allclose(S, expected, atol=1e-10)


def test_support_covariance_is_symmetric_and_positive_definite():
    rng = np.random.default_rng(3)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    _, support = make_design(p, 10, 20, rng)

    S = sc.support_covariance(factor, support, p)

    assert_allclose(S, S.T, atol=0)
    assert np.linalg.eigvalsh(S).min() > 0


def test_support_covariance_handles_a_single_column():
    """|U| == 1 exercises the 1-D solve-output branch."""
    rng = np.random.default_rng(4)
    p = 200
    precision = make_precision(p, rng)
    factor = create_sparse_factor(precision)
    support = np.array([7])

    S = sc.support_covariance(factor, support, p)

    assert S.shape == (1, 1)
    assert_allclose(S, np.linalg.inv(precision.toarray())[7:8, 7:8], atol=1e-10)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def test_build_declines_dense_input():
    rng = np.random.default_rng(5)
    factor = create_sparse_factor(make_precision(100, rng))
    assert sc.build(factor, np.ones((4, 100)), 100, budget=1000) is None


def test_build_declines_empty_input():
    rng = np.random.default_rng(6)
    factor = create_sparse_factor(make_precision(100, rng))
    assert sc.build(factor, sp.csc_array((4, 100)), 100, budget=1000) is None


@pytest.mark.parametrize("budget,fires", [(11, False), (12, False), (13, True)])
def test_build_fires_exactly_when_support_is_smaller_than_budget(budget, fires):
    rng = np.random.default_rng(7)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    X, _ = make_design(p, 40, 12, rng)

    assert (sc.build(factor, X, p, budget=budget) is not None) is fires


# --------------------------------------------------------------------------
# draws
# --------------------------------------------------------------------------


def test_sd_matches_the_half_solve_path():
    rng = np.random.default_rng(8)
    p = 400
    factor = create_sparse_factor(make_precision(p, rng))
    X, _ = make_design(p, 60, 15, rng)

    draw = sc.build(factor, X, p, budget=X.shape[0])
    assert draw is not None
    with gate_off():
        expected = _marginal_predictive_sd(factor, X)

    assert_allclose(draw.sd(), expected, rtol=1e-10)


def test_joint_covariance_matches_the_exact_predictive_covariance():
    rng = np.random.default_rng(9)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    X, support = make_design(p, 5, 12, rng)

    draw = sc.build(factor, X, p, budget=1000)
    assert draw is not None
    samples = draw.joint(400_000, np.random.default_rng(0))

    S = sc.support_covariance(factor, support, p)
    XU = np.asarray(X[:, support].todense())
    assert_allclose(np.cov(samples, rowvar=False), XU @ S @ XU.T, atol=0.02)


def test_duplicate_rows_are_perfectly_correlated_within_a_draw():
    """Identical rows are the same linear functional of the same weights."""
    rng = np.random.default_rng(10)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    X, _ = make_design(p, 4, 12, rng)
    doubled = sp.csc_array(sp.vstack([X, X], format="csr"))

    draw = sc.build(factor, doubled, p, budget=1000)
    assert draw is not None
    samples = draw.joint(64, np.random.default_rng(0))

    assert_allclose(samples[:, :4], samples[:, 4:], atol=0)


def test_duplicate_rows_leave_the_support_covariance_full_rank():
    """The n x n predictive covariance is singular here; S is not."""
    rng = np.random.default_rng(11)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    X, support = make_design(p, 4, 12, rng)
    doubled = sp.csc_array(sp.vstack([X, X], format="csr"))

    S = sc.support_covariance(factor, sc.support_of(doubled), p)
    XU = np.asarray(doubled[:, support].todense())

    assert np.linalg.matrix_rank(XU @ S @ XU.T) < doubled.shape[0]
    assert np.linalg.eigvalsh(S).min() > 0


# --------------------------------------------------------------------------
# estimator integration
# --------------------------------------------------------------------------


def fit_sparse(estimator, p, rng, n_obs=60):
    X = sp.csc_array(sp.random(n_obs, p, density=0.05, random_state=0))
    y = rng.standard_normal(n_obs)
    if isinstance(estimator, BayesianGLM):
        y = rng.integers(0, 2, n_obs).astype(float)
    estimator.fit(X, y)
    return estimator


@pytest.mark.parametrize(
    "make",
    [
        lambda: NormalRegressor(alpha=1.0, beta=1.0, sparse=True, random_state=0),
        lambda: NormalInverseGammaRegressor(sparse=True, random_state=0),
        lambda: BayesianGLM(alpha=1.0, sparse=True, random_state=0, link="logit"),
    ],
    ids=["normal", "nig", "glm"],
)
def test_sample_agrees_in_distribution_with_the_weight_space_path(make):
    rng = np.random.default_rng(12)
    p = 200
    est = fit_sparse(make(), p, rng)
    X, _ = make_design(p, 6, 10, rng)

    with_route = est.sample(X, size=30_000)
    with gate_off():
        without = est.sample(X, size=30_000)

    assert with_route.shape == without.shape
    for column in range(X.shape[0]):
        assert ks_2samp(with_route[:, column], without[:, column]).pvalue > 0.001


@pytest.mark.parametrize(
    "make",
    [
        lambda: NormalRegressor(alpha=1.0, beta=1.0, sparse=True, random_state=0),
        lambda: NormalInverseGammaRegressor(sparse=True, random_state=0),
        lambda: BayesianGLM(alpha=1.0, sparse=True, random_state=0, link="logit"),
    ],
    ids=["normal", "nig", "glm"],
)
def test_sample_marginal_sd_is_unchanged_by_the_route(make):
    """The route changes how the sd is computed, not what it is."""
    rng = np.random.default_rng(13)
    p = 200
    est = fit_sparse(make(), p, rng)
    X, _ = make_design(p, 80, 12, rng)

    factor = est._precision_factor
    draw = sc.build(factor, X, p, budget=X.shape[0])
    assert draw is not None
    with gate_off():
        expected = _marginal_predictive_sd(factor, X)

    assert_allclose(draw.sd(), expected, rtol=1e-10)


def test_thompson_sized_draw_is_bit_identical():
    """size=1 can never beat |U| solves, so the hot path must be untouched."""
    rng = np.random.default_rng(14)
    p = 200
    X, _ = make_design(p, 40, 10, rng)

    def draw():
        est = fit_sparse(
            NormalRegressor(alpha=1.0, beta=1.0, sparse=True, random_state=0),
            p,
            np.random.default_rng(14),
        )
        return est.sample(X, size=1)

    with_route = draw()
    with gate_off():
        without = draw()

    assert_allclose(with_route, without, atol=0, rtol=0)


def test_gate_declines_below_budget_two_without_scanning():
    """``|U| >= 1`` always, so ``budget < 2`` is decidable in O(1)."""
    rng = np.random.default_rng(16)
    p = 200
    factor = create_sparse_factor(make_precision(p, rng))
    X, _ = make_design(p, 40, 10, rng)

    with mock.patch.object(sc, "support_of", side_effect=AssertionError) as never:
        assert sc.build(factor, X, p, budget=1) is None
    never.assert_not_called()


def test_gate_declines_via_the_density_bound_without_scanning():
    """``|U| >= nnz / n_rows`` decides many declines without a column scan."""
    rng = np.random.default_rng(17)
    p = 200
    factor = create_sparse_factor(make_precision(p, rng))
    # 40 rows x 8 nonzeros = 320 nnz, so |U| >= ceil(320/40) = 8
    X, _ = make_design(p, 40, 10, rng, nnz_per_row=8)

    with mock.patch.object(sc, "support_of", side_effect=AssertionError) as never:
        assert sc.build(factor, X, p, budget=8) is None
    never.assert_not_called()


def test_route_survives_a_single_row_and_a_single_column():
    rng = np.random.default_rng(15)
    p = 200
    est = fit_sparse(
        NormalRegressor(alpha=1.0, beta=1.0, sparse=True, random_state=0), p, rng
    )
    X = sp.csc_array(sp.csr_array(([2.0], ([0], [3])), shape=(1, p)))

    drawn = est.sample(X, size=5_000)
    with gate_off():
        reference = est.sample(X, size=5_000)

    assert drawn.shape == (5_000, 1)
    assert ks_2samp(drawn[:, 0], reference[:, 0]).pvalue > 0.001
