"""Tests for the observed / never-observed partition inside the sparse factors.

``Λ = αI + Σ x_t x_tᵀ`` is exactly block diagonal between the features
some ``x_t`` has touched and the ones none has: a never-observed feature
stores only its diagonal, so it is an independent scalar and needs no
factorization. The factors partition on that structure and factor only
the observed block. Everything here checks the partitioned factor against
dense linear algebra on the full matrix, so the partition has to be
invisible to every caller of the factor protocol.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal

from bayesianbandits import NormalRegressor
from bayesianbandits._sparse_bayesian_linear_regression import (
    SparseHalfSolve,
    create_sparse_factor,
    scale_factor,
    takahashi_diagonal,
    trivial_columns,
)


@pytest.fixture(autouse=True)
def suitesparse_envvar(sparse_solver):
    """Run every test in this module against both sparse backends."""
    yield


def make_precision(p, m, rng, block=4):
    """SPD precision over ``p`` features of which ``m`` (a random subset)
    are coupled in ``block``-sized groups; the rest carry only a
    diagonal, with distinct values so per-entry scaling is exercised."""
    m -= m % block
    observed = np.sort(rng.choice(p, m, replace=False))
    A = rng.standard_normal((m // block, block, block))
    B = np.einsum("nij,nkj->nik", A, A) + block * np.eye(block)
    grid = observed.reshape(-1, block)
    r = np.repeat(grid[:, :, None], block, axis=2)
    core = sp.coo_array(
        (B.ravel(), (r.ravel(), r.transpose(0, 2, 1).ravel())), shape=(p, p)
    )
    diag = sp.diags(rng.uniform(1.0, 3.0, p))
    return sp.csc_array(core + diag), observed


def dense_inverse(precision):
    return np.linalg.inv(precision.toarray())


# -- detection ---------------------------------------------------------------


def test_trivial_columns_are_exactly_the_uncoupled_ones():
    rng = np.random.default_rng(0)
    p = 60
    precision, observed = make_precision(p, 24, rng)
    expected = np.setdiff1d(np.arange(p), observed)
    assert_array_equal(trivial_columns(precision), expected)


def test_detection_errs_toward_the_block():
    """Three ways a column can look trivial without being one, or be
    trivial without being safe to treat so, each left in the block: a
    stored explicit zero off the diagonal (structure as far as the
    pattern knows), triangular storage (the coupling sits in the other
    column), and a non-positive diagonal (left to the backends' non-SPD
    behaviour, unchanged)."""
    rng = np.random.default_rng(1)
    p = 30
    precision, observed = make_precision(p, 12, rng)
    j = np.setdiff1d(np.arange(p), observed)[0]
    k = observed[0]
    coo = sp.coo_array(precision)
    coo = sp.coo_array(
        (
            np.append(coo.data, [0.0, 0.0]),
            (np.append(coo.row, [j, k]), np.append(coo.col, [k, j])),
        ),
        shape=(p, p),
    )
    with_zero = sp.csc_array(coo)
    assert with_zero.nnz == precision.nnz + 2
    assert j not in trivial_columns(with_zero)

    off = np.full(19, -0.3)
    upper = sp.csc_array(sp.triu(sp.diags([off, np.full(20, 2.0), off], [-1, 0, 1])))
    assert (upper.indptr[1:] - upper.indptr[:-1] == 1).any()  # looks trivial
    assert trivial_columns(upper).size == 0

    d = np.full(10, 2.0)
    d[3] = -1.0
    assert 3 not in trivial_columns(sp.csc_array(sp.diags(d)))


# -- the protocol against dense truth ----------------------------------------


@pytest.mark.parametrize("m", [24, 0, 60])
def test_solve_matches_the_dense_inverse(m):
    """m=24: partitioned; m=0: all trivial; m=60: nothing trivial."""
    rng = np.random.default_rng(2)
    p = 60
    precision, _ = make_precision(p, m, rng)
    factor = create_sparse_factor(precision)
    inv = dense_inverse(precision)

    b = rng.standard_normal(p)
    assert_allclose(factor.solve(b), inv @ b, rtol=1e-10)
    B = rng.standard_normal((p, 5))
    assert_allclose(factor.solve(B), inv @ B, rtol=1e-10)


@pytest.mark.parametrize("m", [24, 60])
@pytest.mark.parametrize(
    "fmt,index_dtype", [("csc", np.int32), ("csc", np.int64), ("csr", np.int32)]
)
def test_solve_with_a_sparse_rhs_matches_dense(m, fmt, index_dtype):
    """A sparse right-hand side comes back sparse and correct, whatever
    its format or index dtype: the backends never see it, so an
    int64-indexed operand against an int32 factor cannot be misread
    (which CHOLMOD's own sparse solve does silently), and the RVGA Gram
    matrix's CSR ``Xᵀ`` is fine. Columns hit observed and never-observed
    rows both."""
    rng = np.random.default_rng(21)
    p = 60
    precision, observed = make_precision(p, m, rng)
    factor = create_sparse_factor(precision)
    unobserved = np.setdiff1d(np.arange(p), observed)
    rows = (
        np.concatenate([observed[:3], unobserved[:2]])
        if unobserved.size
        else observed[:5]
    )
    B = sp.csc_array(
        (np.ones(rows.size), (rows, np.arange(rows.size))), shape=(p, rows.size)
    )
    B = sp.csc_array(
        (B.data, B.indices.astype(index_dtype), B.indptr.astype(index_dtype)),
        shape=B.shape,
    )
    B = B.tocsr() if fmt == "csr" else B
    assert B.indices.dtype == index_dtype

    out = factor.solve(B)
    assert sp.issparse(out)
    assert_allclose(out.toarray(), dense_inverse(precision) @ B.toarray(), rtol=1e-10)


def test_half_solve_dense_rhs_gram_is_the_predictive_covariance():
    """A dense operand populates every row, never-observed features
    included, and comes back dense in factor-row order."""
    rng = np.random.default_rng(6)
    p = 48
    precision, observed = make_precision(p, 20, rng)
    factor = create_sparse_factor(precision)
    X = rng.standard_normal((7, p))
    B = np.asarray(factor.half_solve(X.T))
    assert B.shape == (p, 7)
    assert_allclose(B.T @ B, X @ dense_inverse(precision) @ X.T, atol=1e-10)


# -- sample_at ---------------------------------------------------------------


def test_sample_at_covariance_is_the_inverse_submatrix():
    """Mixed observed / never-observed features, unsorted; and every
    feature for ``None``."""
    rng = np.random.default_rng(7)
    p = 48
    precision, observed = make_precision(p, 20, rng)
    factor = create_sparse_factor(precision)
    unobserved = np.setdiff1d(np.arange(p), observed)
    inv = dense_inverse(precision)
    for features in (
        np.array([observed[3], unobserved[5], unobserved[0], observed[0]]),
        None,
    ):
        G = factor.sample_at(features, 40_000, np.random.default_rng(0))
        want = inv if features is None else inv[np.ix_(features, features)]
        assert G.shape == (want.shape[0], 40_000)
        assert_allclose(np.cov(G), want, atol=0.05 * np.abs(want).max())


def test_sample_at_repeats_a_feature_exactly():
    """One draw per feature: asking twice returns the same rows, for an
    observed and a never-observed feature alike."""
    rng = np.random.default_rng(27)
    p = 48
    precision, observed = make_precision(p, 20, rng)
    factor = create_sparse_factor(precision)
    j = np.setdiff1d(np.arange(p), observed)[2]
    k = observed[4]
    G = factor.sample_at(np.array([j, k, j, k]), 5, np.random.default_rng(0))
    assert_array_equal(G[0], G[2])
    assert_array_equal(G[1], G[3])


@pytest.mark.parametrize("m", [24, 60])
def test_sample_at_draws_only_the_normals_it_needs(m):
    """``n_factored`` for the block plus one per distinct never-observed
    feature asked for -- and not one per never-observed feature there
    is: the generator advances by exactly that many. With nothing
    trivial (``m == p``) that is every feature."""
    rng = np.random.default_rng(37)
    p = 60
    precision, observed = make_precision(p, m, rng)
    factor = create_sparse_factor(precision)
    unobserved = np.setdiff1d(np.arange(p), observed)
    if unobserved.size:
        features = np.array([observed[1], unobserved[3], unobserved[3], unobserved[9]])
        distinct_trivial = 2
    else:
        features = np.array([3, 1])
        distinct_trivial = 0
    size = 7

    used, twin = np.random.default_rng(5), np.random.default_rng(5)
    factor.sample_at(features, size, used)
    twin.standard_normal(size * (factor.n_factored + distinct_trivial))
    assert used.random() == twin.random()


def test_logdet_trace_inv_and_get_L_csc_match_dense():
    """``logdet`` and ``trace_inv`` against dense truth, and ``get_L_csc``
    is a Cholesky factor of some permutation of the matrix: same
    eigenvalues, and its diagonal and Takahashi trace reproduce the two."""
    rng = np.random.default_rng(8)
    p = 48
    precision, _ = make_precision(p, 20, rng)
    factor = create_sparse_factor(precision)
    _, ld = np.linalg.slogdet(precision.toarray())
    assert_allclose(factor.logdet(), ld, rtol=1e-10)
    assert_allclose(factor.trace_inv(), np.trace(dense_inverse(precision)), rtol=1e-10)
    L = factor.get_L_csc()
    assert_allclose(
        np.sort(np.linalg.eigvalsh((L @ L.T).toarray())),
        np.sort(np.linalg.eigvalsh(precision.toarray())),
        rtol=1e-8,
    )
    assert_allclose(2.0 * np.sum(np.log(L.diagonal())), ld, rtol=1e-10)
    assert_allclose(np.sum(takahashi_diagonal(L)), factor.trace_inv(), rtol=1e-10)


def test_scaled_factor_composes_over_the_partition():
    rng = np.random.default_rng(11)
    p = 48
    precision, _ = make_precision(p, 20, rng)
    s = 2.5
    factor = scale_factor(create_sparse_factor(precision), s)
    inv = dense_inverse(precision) / s

    b = rng.standard_normal(p)
    assert_allclose(factor.solve(b), inv @ b, rtol=1e-10)
    Mt = np.asarray(factor.half_solve(np.eye(p)))
    assert_allclose(Mt.T @ Mt, inv, atol=1e-10)
    X = sp.csc_array(sp.random(6, p, density=0.2, random_state=44))
    B = factor.half_solve(X.T)  # sparse rhs, split result
    assert isinstance(B, SparseHalfSolve)
    gram = B.block.T @ B.block + (B.trivial.T @ B.trivial).toarray()
    assert_allclose(gram, X.toarray() @ inv @ X.toarray().T, atol=1e-10)
    G = factor.sample_at(None, 40_000, np.random.default_rng(0))
    assert_allclose(np.cov(G), inv, atol=0.05 * np.abs(inv).max())
    _, ld = np.linalg.slogdet(s * precision.toarray())
    assert_allclose(factor.logdet(), ld, rtol=1e-10)
    assert_allclose(factor.trace_inv(), np.trace(inv), rtol=1e-10)


# -- refactorization ---------------------------------------------------------


def test_refactorize_follows_the_new_matrix():
    """Same pattern with new values (the trivial diagonal must be
    re-read), then a pattern where a later update couples a feature that
    was trivial before (the partition must be redone)."""
    rng = np.random.default_rng(12)
    p = 40
    A1, observed = make_precision(p, 16, rng)
    factor = create_sparse_factor(A1)
    b = rng.standard_normal(p)

    A2 = sp.csc_array(A1 * 1.7)  # scales every entry, block and trivial alike
    same = factor.refactorize(A2)
    assert_allclose(same.solve(b), dense_inverse(A2) @ b, rtol=1e-10)

    j, k = np.setdiff1d(np.arange(p), observed)[0], observed[0]
    A3 = sp.lil_array(A1)
    A3[j, k] = A3[k, j] = 0.4
    A3[j, j] += 1.0
    A3 = sp.csc_array(A3)
    grown = factor.refactorize(A3)
    assert grown.n_factored == observed.size + 1
    assert_allclose(grown.solve(b), dense_inverse(A3) @ b, rtol=1e-10)
    assert_allclose(grown.trace_inv(), np.trace(dense_inverse(A3)), rtol=1e-10)


# -- the property that made this worth doing ---------------------------------


# -- stacked contexts ---------------------------------------------------------


def test_stacked_contexts_with_distinct_supports_have_the_right_joint_law():
    """Several contexts stacked, each over its own features, some of them
    never observed: a row must get nothing from a feature it does not
    touch, rows touching one feature must share its draw, and the
    covariance between contexts must be exactly ``X_a Λ⁻¹ X_bᵀ`` --
    coupled through observed features, zero through never-observed ones.
    Checked as the full ``n x n`` covariance of the weight-space draw
    against dense truth."""
    from unittest import mock

    from bayesianbandits import _estimators as E

    rng = np.random.default_rng(31)
    p = 120
    n_ctx, n_arms = 3, 4
    # observed: 0..29; everything else never observed
    X_fit = sp.csc_array(sp.random(50, 30, density=0.3, random_state=31))
    X_fit = sp.csc_array(sp.hstack([X_fit, sp.csc_array((50, p - 30))]))
    est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True, random_state=0)
    est.fit(X_fit, rng.standard_normal(50))
    trivial = trivial_columns(est.cov_inv_)
    assert 30 <= trivial.min()

    rows, cols, vals = [], [], []
    for c in range(n_ctx):
        # two observed features shared by every context, two of its own,
        # one never-observed feature of its own, and one never-observed
        # feature shared by contexts 0 and 1 only
        own = [0, 1, 10 + 2 * c, 11 + 2 * c, 50 + c] + ([90] if c < 2 else [])
        for a in range(n_arms):
            for j in own:
                rows.append(c * n_arms + a)
                cols.append(j)
                vals.append(rng.standard_normal())
    X = sp.csc_array(sp.csr_array((vals, (rows, cols)), shape=(n_ctx * n_arms, p)))

    # force the weight-space path (the reductions are exact too, but this
    # is the path under test)
    with mock.patch.object(E, "build_joint_reduction", return_value=None):
        est.random_state_ = np.random.default_rng(1)
        draws = est.sample(X, size=40_000)

    Xd = X.toarray()
    want = Xd @ np.linalg.inv(est.cov_inv_.toarray()) @ Xd.T
    assert_allclose(np.cov(draws.T), want, atol=0.05 * np.abs(want).max())
    assert_allclose(draws.mean(axis=0), Xd @ est.coef_, atol=0.05 * np.abs(want).max())

    # the structure the numbers above must carry: contexts 0 and 1 share
    # never-observed feature 90, context 2 does not, and a never-observed
    # feature couples nothing else
    cov = np.cov(draws.T)
    inv = np.linalg.inv(est.cov_inv_.toarray())
    for j in (50, 51, 52, 90):
        assert inv[j, j] == 1.0 / est.cov_inv_[j, j]
        assert np.count_nonzero(inv[j]) == 1
    # so cross-context covariance is the observed-feature part plus, for
    # contexts 0 and 1 only, feature 90's shared draw
    x0, x1, x2 = (Xd[c * n_arms : (c + 1) * n_arms] for c in range(n_ctx))
    obs = np.arange(30)
    for a, b, share_90 in ((x0, x1, True), (x0, x2, False), (x1, x2, False)):
        expect = a[:, obs] @ inv[np.ix_(obs, obs)] @ b[:, obs].T
        if share_90:
            expect = expect + np.outer(a[:, 90], b[:, 90]) / est.cov_inv_[90, 90]
        got = cov[
            np.ix_(
                range(n_arms * (0 if a is x0 else 1), n_arms * (1 if a is x0 else 2)),
                range(n_arms * (1 if b is x1 else 2), n_arms * (2 if b is x1 else 3)),
            )
        ]
        assert_allclose(got, expect, atol=0.05 * np.abs(want).max())


# -- half_solve with a sparse right-hand side, and n_factored -----------------


@pytest.mark.parametrize("m", [24, 60])
def test_n_factored_and_solve_cost_count_only_the_block(m):
    """The observed block for a sparse factor, unchanged by scaling;
    every feature for a dense one; the per-solve cost is the block's
    stored entries."""
    from scipy.linalg import cholesky

    from bayesianbandits._sparse_bayesian_linear_regression import DenseFactor

    rng = np.random.default_rng(41)
    p = 60
    precision, observed = make_precision(p, m, rng)
    factor = create_sparse_factor(precision)
    assert factor.n_factored == observed.size
    assert factor.solve_cost == precision.nnz - (p - observed.size)
    assert scale_factor(factor, 2.0).n_factored == observed.size
    U = cholesky(precision.toarray(), lower=False)
    assert DenseFactor(_U=U, _n_features=p).n_factored == p


@pytest.mark.parametrize("m", [24, 60])
def test_half_solve_sparse_rhs_splits_with_the_same_gram(m):
    """For a sparse ``b`` the result is the dense block half-solve plus a
    sparse trivial part with one row per distinct never-observed feature
    ``b`` touches -- nothing else, and no denser than ``b``'s trivial
    entries -- and the two Grams add to ``bᵀ Λ⁻¹ b``. With nothing
    trivial the block is the dense half-solve of the same operand."""
    rng = np.random.default_rng(43)
    p = 60
    precision, observed = make_precision(p, m, rng)
    factor = create_sparse_factor(precision)
    unobserved = np.setdiff1d(np.arange(p), observed)
    # 5 columns; two of them share never-observed feature j, one touches
    # another never-observed feature, the rest observed only
    X = sp.random(5, p, density=0.15, random_state=43).tolil()
    X[:, unobserved] = 0.0
    if unobserved.size:
        j, k = unobserved[1], unobserved[4]
        X[0, j] = 0.7
        X[3, j] = -1.1
        X[2, k] = 0.4
    X = sp.csc_array(X)
    b = X.T  # (p, 5), sparse

    B = factor.half_solve(b)
    assert isinstance(B, SparseHalfSolve)
    touched = 2 if unobserved.size else 0
    assert B.block.shape == (factor.n_factored, 5)
    assert B.trivial.shape == (touched, 5)
    assert B.trivial.nnz == (3 if unobserved.size else 0)
    Xd = X.toarray()
    gram = B.block.T @ B.block + (B.trivial.T @ B.trivial).toarray()
    assert_allclose(gram, Xd @ dense_inverse(precision) @ Xd.T, atol=1e-10)
    if unobserved.size:
        # trivial rows are value / sqrt(Λ_jj), ascending by feature
        diag = precision.diagonal()
        want = np.zeros((2, 5))
        want[0, 0], want[0, 3] = 0.7 / np.sqrt(diag[j]), -1.1 / np.sqrt(diag[j])
        want[1, 2] = 0.4 / np.sqrt(diag[k])
        assert_allclose(B.trivial.toarray(), want, rtol=1e-12)
    else:
        assert_allclose(B.block, factor.half_solve(b.toarray()), rtol=1e-12)


def test_marginal_sd_chunks_the_half_solve_over_the_block():
    """The marginal path over a partitioned factor with the block budget
    forced small: many chunks, each handing the factor a sparse block of
    rows, and the result is dense truth."""
    from unittest import mock

    from bayesianbandits import _estimators as E
    from bayesianbandits import _support_covariance as sc
    from bayesianbandits._estimators import _marginal_predictive_sd

    rng = np.random.default_rng(45)
    p = 80
    precision, _ = make_precision(p, 32, rng)
    factor = create_sparse_factor(precision)
    inv = dense_inverse(precision)

    X = sp.csc_array(sp.random(37, p, density=0.1, random_state=45))
    Xd = X.toarray()
    with (
        mock.patch.object(sc, "build", lambda *a, **k: None),
        mock.patch.object(E, "_MARGINAL_SD_BLOCK_ELEMS", 3 * factor.n_factored),
    ):
        seen = []
        real = factor.half_solve
        with mock.patch.object(
            factor, "half_solve", lambda b: seen.append(b.shape[1]) or real(b)
        ):
            got = _marginal_predictive_sd(factor, X)
    assert_allclose(got, np.sqrt(np.einsum("ij,jk,ik->i", Xd, inv, Xd)), rtol=1e-10)
    assert len(seen) == 13 and max(seen) == 3  # 37 rows in blocks of 3


def test_a_feature_seen_alone_stays_trivial_and_seen_together_joins_the_block():
    """``Λ`` couples two features only when one observation touches
    both. A feature observed on its own gains a diagonal entry and
    nothing else, so it stays an independent scalar; observed together
    with another, it joins the block. The partition follows exactly."""
    rng = np.random.default_rng(51)
    p = 40
    X = sp.csc_array(sp.random(50, 12, density=0.4, random_state=51))
    X = sp.csc_array(sp.hstack([X, sp.csc_array((50, p - 12))]))
    est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True, random_state=0)
    est.fit(X, rng.standard_normal(50))
    assert trivial_columns(est.cov_inv_).size == p - 12

    alone = sp.csc_array(sp.csr_array(([1.0], ([0], [35])), shape=(1, p)))
    est.partial_fit(alone, np.array([0.5]))
    assert 35 in trivial_columns(est.cov_inv_)
    assert est._precision_factor.n_factored == 12

    together = sp.csc_array(
        sp.csr_array(([1.0, 1.0], ([0, 0], [30, 31])), shape=(1, p))
    )
    est.partial_fit(together, np.array([0.5]))
    still = trivial_columns(est.cov_inv_)
    assert 30 not in still and 31 not in still and 35 in still
    assert est._precision_factor.n_factored == 14
    b = rng.standard_normal(p)
    assert_allclose(
        est._precision_factor.solve(b),
        np.linalg.solve(est.cov_inv_.toarray(), b),
        rtol=1e-8,
    )
