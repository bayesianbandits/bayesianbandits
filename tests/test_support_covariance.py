"""Tests for the support-covariance route.

``Cov(Xw) = X_U (Λ⁻¹)_{U,U} X_Uᵀ`` is an identity, so the tests are
identity checks against the paths the route replaces (``S`` against the
explicit inverse, per-row sd against the half-solve path), plus the
composition each estimator applies after the reduction.
"""

from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from bayesianbandits import (
    BayesianGLM,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from bayesianbandits import _support_covariance as sc
from bayesianbandits._estimators import (
    _marginal_predictive_sd,
    _marginal_support_is_cheaper,
)
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


# -- the identity -------------------------------------------------------------


def test_support_of_finds_exactly_the_touched_columns():
    rng = np.random.default_rng(0)
    X, support = make_design(400, 30, 12, rng)
    assert np.array_equal(sc.support_of(X), support)


def test_compact_columns_matches_fancy_indexing():
    rng = np.random.default_rng(1)
    X, support = make_design(400, 30, 12, rng)
    assert_allclose(sc._compact_columns(X, support).todense(), X[:, support].todense())


def test_support_covariance_equals_the_principal_submatrix():
    rng = np.random.default_rng(2)
    p = 300
    precision = make_precision(p, rng)
    _, support = make_design(p, 10, 15, rng)

    S = sc.support_covariance(create_sparse_factor(precision), support, p)
    expected = np.linalg.inv(precision.toarray())[np.ix_(support, support)]

    assert_allclose(S, expected, atol=1e-10)


def test_support_covariance_handles_a_single_column():
    """``|U| == 1`` exercises the 1-D solve-output branch."""
    rng = np.random.default_rng(4)
    p = 200
    precision = make_precision(p, rng)
    S = sc.support_covariance(create_sparse_factor(precision), np.array([7]), p)
    assert S.shape == (1, 1)
    assert_allclose(S, np.linalg.inv(precision.toarray())[7:8, 7:8], atol=1e-10)


def test_support_covariance_solves_against_a_fortran_ordered_rhs():
    """CHOLMOD's dense format is column-major, so a C-ordered right-hand
    side is copied in full before the solve begins."""
    rng = np.random.default_rng(11)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    _, support = make_design(p, 10, 15, rng)
    seen = []

    real_solve = factor.solve
    with mock.patch.object(
        factor, "solve", lambda b: seen.append(b.flags.f_contiguous) or real_solve(b)
    ):
        sc.support_covariance(factor, support, p)

    assert seen and all(seen)


def test_draw_factor_reproduces_the_predictive_covariance():
    """``(X_U C)(X_U C)ᵀ`` is the predictive covariance, exactly."""
    rng = np.random.default_rng(9)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    X, support = make_design(p, 5, 12, rng)

    draw = sc.build(factor, X, p, budget=1000)
    assert draw is not None
    XU_C = np.asarray((draw._XU @ draw._C))
    S = sc.support_covariance(factor, support, p)
    XU = np.asarray(X[:, support].todense())
    assert_allclose(XU_C @ XU_C.T, XU @ S @ XU.T, atol=1e-10)


def test_sd_matches_the_half_solve_path():
    rng = np.random.default_rng(8)
    p = 400
    precision = make_precision(p, rng)
    factor = create_sparse_factor(precision)
    X, _ = make_design(p, 60, 15, rng)

    draw = sc.build(factor, X, p, budget=X.shape[0])
    assert draw is not None
    with gate_off():
        expected = _marginal_predictive_sd(factor, X, precision.nnz)

    assert_allclose(draw.sd(), expected, rtol=1e-10)


def test_marginal_predictive_sd_routes_through_the_support():
    """The wiring in ``_marginal_predictive_sd``, not just the helper."""
    rng = np.random.default_rng(18)
    p = 200
    precision = make_precision(p, rng)
    factor = create_sparse_factor(precision)
    X, _ = make_design(p, 80, 12, rng)  # 12 columns, 80 rows: fires

    routed = _marginal_predictive_sd(factor, X, precision.nnz)
    with gate_off():
        direct = _marginal_predictive_sd(factor, X, precision.nnz)

    assert_allclose(routed, direct, rtol=1e-10)


def test_marginal_predictive_sd_declines_a_support_it_barely_beats():
    """|U| just under n_rows saves nothing on solves and adds an
    O(|U|³) Cholesky, so the marginal path keeps its bounded scratch."""
    rng = np.random.default_rng(19)
    p = 400
    precision = make_precision(p, rng)
    factor = create_sparse_factor(precision)
    X, _ = make_design(p, 120, 100, rng)  # 100 columns, 120 rows

    with mock.patch.object(sc, "support_covariance") as never:
        gated = _marginal_predictive_sd(factor, X, precision.nnz)
    never.assert_not_called()

    with gate_off():
        direct = _marginal_predictive_sd(factor, X, precision.nnz)
    assert_allclose(gated, direct, rtol=1e-10)


@pytest.mark.parametrize(
    "n_u,n_rows,nnz,expected",
    [
        (12, 80, 598, True),  # cheap on every axis
        (100, 120, 1198, False),  # 4|U| > n_rows: no solves saved
        (200, 4000, 100, False),  # Cholesky outruns the solves it saves
        (4000, 100_000, 10**9, False),  # |U|² over the scratch budget
    ],
)
def test_marginal_support_gate(n_u, n_rows, nnz, expected):
    assert _marginal_support_is_cheaper(n_u, n_rows, nnz) is expected


# -- rank deficiency ----------------------------------------------------------


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
    """The n x n predictive covariance is singular here; S is not. This
    is why the reduction is on the feature side."""
    rng = np.random.default_rng(11)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    X, support = make_design(p, 4, 12, rng)
    doubled = sp.csc_array(sp.vstack([X, X], format="csr"))

    S = sc.support_covariance(factor, sc.support_of(doubled), p)
    XU = np.asarray(doubled[:, support].todense())

    assert np.linalg.matrix_rank(XU @ S @ XU.T) < doubled.shape[0]
    assert np.linalg.eigvalsh(S).min() > 0


# -- the gate -----------------------------------------------------------------


@pytest.mark.parametrize(
    "X",
    [np.ones((4, 100)), sp.csc_array((4, 100))],
    ids=["dense", "empty"],
)
def test_build_declines_input_it_cannot_reduce(X):
    rng = np.random.default_rng(5)
    factor = create_sparse_factor(make_precision(100, rng))
    assert sc.build(factor, X, 100, budget=1000) is None


@pytest.mark.parametrize("budget,fires", [(12, False), (13, True)])
def test_build_fires_exactly_when_support_is_smaller_than_budget(budget, fires):
    rng = np.random.default_rng(7)
    p = 300
    factor = create_sparse_factor(make_precision(p, rng))
    X, _ = make_design(p, 40, 12, rng)

    assert (sc.build(factor, X, p, budget=budget) is not None) is fires


# -- estimator composition ----------------------------------------------------


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
def test_sample_moments_survive_the_route(make):
    """Each estimator composes something on top of the reduction (the
    t-mixing scale, the inverse link). A dropped factor there is
    invisible to the identity checks above, so compare moments against
    the path the route replaces."""
    rng = np.random.default_rng(12)
    p = 200
    est = fit_sparse(make(), p, rng)
    X, _ = make_design(p, 6, 10, rng)

    est.random_state_ = np.random.default_rng(1)
    routed = est.sample(X, size=30_000)
    with gate_off():
        est.random_state_ = np.random.default_rng(2)
        direct = est.sample(X, size=30_000)

    sd = direct.std(axis=0)
    z = np.abs(routed.mean(axis=0) - direct.mean(axis=0)) / (sd / np.sqrt(30_000))
    assert z.max() < 5.0
    assert_allclose(routed.std(axis=0), sd, rtol=0.1)


def test_sparse_right_hand_side_still_uses_the_direct_solver(sparse_solver):
    """``solve`` keeps its sparse-RHS path; only dense input goes through
    the cached triangular factor."""
    rng = np.random.default_rng(19)
    p = 200
    precision = make_precision(p, rng)
    factor = create_sparse_factor(precision)
    rhs = sp.csc_array(sp.random(p, 3, density=0.2, random_state=5))

    got = np.asarray(factor.solve(rhs))
    if got.ndim == 0:  # some backends return a sparse container
        got = np.asarray(got.item().todense())
    expected = np.linalg.solve(precision.toarray(), rhs.toarray())
    assert_allclose(np.asarray(got).reshape(expected.shape), expected, atol=1e-8)
