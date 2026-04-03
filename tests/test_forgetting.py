"""Tests for forgetting rules: exponential, stabilized (KZ), and SIFt."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, eigvalsh

from bayesianbandits._forgetting import (
    ExponentialForgetting,
    SiftForgetting,
    StabilizedForgetting,
    _filter_batch_sparse,
    _sift_downdate_dense,
    _sift_downdate_sparse,
    filter_batch,
)


def _random_pd(n, rng, cond=10.0):
    """Generate a random positive definite matrix with bounded condition number."""
    A = rng.standard_normal((n, n))
    R = A.T @ A + np.eye(n)
    # Bound condition number
    eigvals = eigvalsh(R)
    shift = max(0, eigvals[-1] / cond - eigvals[0])
    return R + shift * np.eye(n)


def _naive_sift_downdate(R, X_bar, lam):
    """Naive implementation: R - (1-lam) * R Xb^T inv(Xb R Xb^T) Xb R."""
    w = R @ X_bar.T
    H = X_bar @ w
    return R - (1 - lam) * w @ np.linalg.solve(H, w.T)


def _sym(A):
    """Mirror upper triangle to lower, producing a full symmetric matrix.

    ``_sift_downdate_dense`` returns upper-triangle-only (dsyrk convention).
    Tests that need full-matrix operations (eigvalsh, matmul, element access
    in the lower triangle) should call ``_sym`` first.
    """
    return np.triu(A) + np.triu(A, 1).T


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["exponential", "stabilized", "sift"])
def rule(request):
    if request.param == "exponential":
        return ExponentialForgetting()
    elif request.param == "stabilized":
        return StabilizedForgetting(alpha=1.0)
    else:
        return SiftForgetting(eps=1e-10)


# ---------------------------------------------------------------------------
# Shared interface tests
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_data():
    rng = np.random.default_rng(42)
    R = _random_pd(5, rng)
    X = rng.standard_normal((3, 5))
    y = rng.standard_normal(3)
    return R, X, y


class TestSharedInterface:
    """Tests that apply to all forgetting rules."""

    def test_return_type(self, rule, shared_data):
        R, X, y = shared_data
        result = rule(R, X, y, 0.95)
        assert result is not None
        R_bar, X_eff, y_eff = result
        assert isinstance(R_bar, np.ndarray) or sparse.issparse(R_bar)
        assert isinstance(X_eff, np.ndarray)
        assert isinstance(y_eff, np.ndarray)

    def test_passthrough_at_lam_1(self, rule, shared_data):
        R, X, y = shared_data
        result = rule(R, X, y, 1.0)
        assert result is not None
        R_bar, _, _ = result
        assert_allclose(np.asarray(R_bar), R, atol=1e-12)

    def test_format_preservation_dense(self, rule, shared_data):
        R, X, y = shared_data
        result = rule(R, X, y, 0.95)
        if result is not None:
            R_bar, _, _ = result
            assert isinstance(R_bar, np.ndarray)

    def test_format_preservation_sparse(self, rule, shared_data):
        R, X, y = shared_data
        R_sparse = sparse.csc_array(R)
        result = rule(R_sparse, X, y, 0.95)
        if result is not None:
            R_bar, _, _ = result
            if not isinstance(rule, SiftForgetting):
                assert sparse.issparse(R_bar)


# ---------------------------------------------------------------------------
# filter_batch tests
# ---------------------------------------------------------------------------


class TestFilterBatch:
    def test_orthonormality(self):
        """X_bar @ X_bar.T should be diag(surviving eigenvalues)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 10))
        y = rng.standard_normal(3)

        result = filter_batch(X, y, eps=1e-12)
        assert result is not None
        X_bar, y_bar = result

        gram_bar = X_bar @ X_bar.T
        # Should be diagonal (eigenvectors are orthonormal)
        off_diag = gram_bar - np.diag(np.diag(gram_bar))
        assert_allclose(off_diag, 0, atol=1e-10)

        # Diagonal entries should be the surviving eigenvalues of X @ X.T
        gram_orig = X @ X.T
        eigvals_orig = np.sort(eigvalsh(gram_orig))
        eigvals_bar = np.sort(np.diag(gram_bar))
        surviving = eigvals_orig[eigvals_orig >= 1e-12]
        assert_allclose(eigvals_bar, surviving, rtol=1e-10)

    @pytest.mark.parametrize("use_sparse", [False, True], ids=["dense", "sparse"])
    def test_q_zero_returns_none(self, use_sparse):
        """When no eigenvalues survive thresholding, return None."""
        rng = np.random.default_rng(42)
        if use_sparse:
            X = sparse.csc_array((3, 10))
        else:
            X = rng.standard_normal((3, 10)) * 1e-15
        y = rng.standard_normal(3)

        result = filter_batch(X, y, eps=1e-10)
        assert result is None

    def test_sparse_rank_zero_returns_none(self):
        """Sparse X with explicit stored zeros produces a zero Gram;
        dpstrf returns rank 0."""
        # Explicit stored zeros pass the active-column check but yield
        # a zero Gram, hitting the rank == 0 early exit in _filter_batch_sparse.
        X = sparse.csc_array(([0.0, 0.0], ([0, 1], [0, 2])), shape=(3, 10))
        y = np.zeros(3)

        result = filter_batch(X, y, eps=1e-10)
        assert result is None

    def test_dense_and_sparse_agree(self):
        """filter_batch should give same results for dense and sparse X."""
        rng = np.random.default_rng(42)
        X_dense = rng.standard_normal((3, 10))
        # Make it somewhat sparse
        X_dense[np.abs(X_dense) < 0.5] = 0
        X_sparse = sparse.csr_matrix(X_dense)
        y = rng.standard_normal(3)

        result_dense = filter_batch(X_dense, y, eps=1e-12)
        result_sparse = filter_batch(X_sparse, y, eps=1e-12)

        assert result_dense is not None
        assert result_sparse is not None

        X_bar_d, y_bar_d = result_dense
        X_bar_s, y_bar_s = result_sparse

        # Dense and sparse paths use different factorizations (eigh vs pivoted
        # Cholesky), so X_bar itself differs. Check the sufficient statistics
        # that the caller actually uses: X_bar.T @ X_bar and X_bar.T @ y_bar.
        xtx_d = X_bar_d.T @ X_bar_d
        xtx_s = X_bar_s.T @ X_bar_s
        if sparse.issparse(xtx_s):
            xtx_s = xtx_s.toarray()
        assert_allclose(xtx_d, xtx_s, atol=1e-10)

        xty_d = X_bar_d.T @ y_bar_d
        xty_s = X_bar_s.T @ y_bar_s
        if sparse.issparse(xty_s):
            xty_s = np.asarray(xty_s).ravel()
        assert_allclose(xty_d, xty_s, atol=1e-10)

    def test_projection_preserves_sufficient_statistics(self):
        """When q = p (no thresholding), X_bar preserves both X.T@X and X.T@y."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 10))
        y = rng.standard_normal(3)

        result = filter_batch(X, y, eps=0.0)
        assert result is not None
        X_bar, y_bar = result
        assert_allclose(X_bar.T @ X_bar, X.T @ X, atol=1e-10)
        assert_allclose(X_bar.T @ y_bar, X.T @ y, atol=1e-10)


# ---------------------------------------------------------------------------
# ExponentialForgetting tests
# ---------------------------------------------------------------------------


class TestExponentialForgetting:
    def test_scalar_decay(self):
        rng = np.random.default_rng(42)
        R = _random_pd(5, rng)
        X = rng.standard_normal((3, 5))
        y = rng.standard_normal(3)
        lam = 0.95

        rule = ExponentialForgetting()
        R_bar, X_eff, y_eff = rule(R, X, y, lam)

        assert_allclose(R_bar, lam * R)
        assert_allclose(X_eff, X)
        assert_allclose(y_eff, y)


# ---------------------------------------------------------------------------
# StabilizedForgetting tests
# ---------------------------------------------------------------------------


class TestStabilizedForgetting:
    def test_floor_reinjection(self):
        rng = np.random.default_rng(42)
        R = _random_pd(5, rng)
        X = rng.standard_normal((3, 5))
        y = rng.standard_normal(3)
        alpha = 2.0
        lam = 0.9

        rule = StabilizedForgetting(alpha=alpha)
        R_bar, _, _ = rule(R, X, y, lam)

        expected = lam * R + (1 - lam) * alpha * np.eye(5)
        assert_allclose(R_bar, expected, atol=1e-12)

    def test_eigenvalue_floor(self):
        """lambda_min(R_bar) >= (1-lam) * alpha regardless of input."""
        rng = np.random.default_rng(42)
        alpha = 1.5
        lam = 0.8

        rule = StabilizedForgetting(alpha=alpha)

        for _ in range(20):
            R = _random_pd(8, rng)
            X = rng.standard_normal((3, 8))
            y = rng.standard_normal(3)

            R_bar, _, _ = rule(R, X, y, lam)
            min_eig = eigvalsh(np.asarray(R_bar))[0]
            assert min_eig >= (1 - lam) * alpha - 1e-10

    def test_sparse_format(self):
        rng = np.random.default_rng(42)
        R = sparse.csc_array(_random_pd(5, rng))
        X = rng.standard_normal((3, 5))
        y = rng.standard_normal(3)

        rule = StabilizedForgetting(alpha=1.0)
        R_bar, _, _ = rule(R, X, y, 0.9)
        assert sparse.issparse(R_bar)


# ---------------------------------------------------------------------------
# SiftForgetting / _sift_downdate tests
# ---------------------------------------------------------------------------


def _downdate(R, X, y, lam, use_sparse):
    """Run the appropriate downdate path and return (R_bar_dense, R_dense)."""
    if use_sparse:
        R_sparse = sparse.csc_array(R)
        # Build sparse X with nonzeros only in a few columns
        X_sparse = sparse.csc_array(X)
        result = _filter_batch_sparse(X_sparse, y, eps=1e-12)
        if result is None:
            return None
        R_bar = _sift_downdate_sparse(R_sparse, result.active_cols, result.X_bar_a, lam)
        return R_bar.toarray(), np.asarray(R)
    else:
        result = filter_batch(X, y, eps=1e-12)
        if result is None:
            return None
        X_bar, _ = result
        R_bar = _sym(_sift_downdate_dense(R, X_bar, lam))
        return R_bar, np.asarray(R)


@pytest.mark.parametrize("use_sparse", [False, True], ids=["dense", "sparse"])
class TestSiftDowndate:
    def test_matches_naive(self, use_sparse):
        """Downdate matches naive R - (1-lam) * R Xb^T inv(Xb R Xb^T) Xb R."""
        rng = np.random.default_rng(42)
        n = 20
        R = _random_pd(n, rng)

        # Sparse path needs X with a few active columns
        X = np.zeros((3, n))
        X[:, :4] = rng.standard_normal((3, 4))
        y = rng.standard_normal(3)
        lam = 0.9

        # Reference: naive dense computation
        result_dense = filter_batch(X, y, eps=1e-12)
        assert result_dense is not None
        X_bar_dense, _ = result_dense
        R_bar_naive = _naive_sift_downdate(R, X_bar_dense, lam)

        pair = _downdate(R, X, y, lam, use_sparse)
        assert pair is not None
        R_bar, _ = pair

        assert_allclose(R_bar, R_bar_naive, atol=1e-10)

    def test_positive_definite(self, use_sparse):
        """R_bar must be PD (Corollary 1: R_bar >= lam * R)."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            n = rng.integers(10, 30) if use_sparse else rng.integers(5, 15)
            p = rng.integers(1, min(5, n)) if use_sparse else rng.integers(1, n)
            cond = rng.choice([10.0, 1e4, 1e8])
            R = _random_pd(n, rng, cond=cond)

            if use_sparse:
                active = rng.choice(n, size=p, replace=False)
                X = np.zeros((3, n))
                X[:, active] = rng.standard_normal((3, p))
            else:
                X = rng.standard_normal((p, n))
            y = rng.standard_normal(X.shape[0])
            lam = rng.uniform(0.8, 0.99)

            pair = _downdate(R, X, y, lam, use_sparse)
            if pair is None:
                continue
            R_bar, R_dense = pair

            # Symmetric
            assert_allclose(R_bar, R_bar.T, atol=1e-10)

            min_eig = eigvalsh(R_bar)[0]
            assert min_eig > 0, f"R_bar not PD: min eigenvalue = {min_eig}"

            # Corollary 1: R_bar - lam * R should be PSD
            diff = R_bar - lam * R_dense
            min_eig_diff = eigvalsh(diff)[0]
            assert min_eig_diff >= -1e-10, (
                f"R_bar - lam*R not PSD: min eigenvalue = {min_eig_diff}"
            )

    def test_directional_forgetting(self, use_sparse):
        """Features outside X's support are untouched when R is diagonal."""
        rng = np.random.default_rng(42)
        n = 10
        R = np.diag(rng.uniform(1.0, 5.0, n))

        X = np.zeros((3, n))
        X[:, :2] = rng.standard_normal((3, 2))
        y = rng.standard_normal(3)
        lam = 0.9

        pair = _downdate(R, X, y, lam, use_sparse)
        assert pair is not None
        R_bar, R_dense = pair

        # Features 2..n-1 should be completely unchanged
        assert_allclose(R_bar[2:, 2:], R_dense[2:, 2:], atol=1e-10)
        # Cross terms between excited and unexcited should also be unchanged
        assert_allclose(R_bar[2:, :2], R_dense[2:, :2], atol=1e-10)
        assert_allclose(R_bar[:2, 2:], R_dense[:2, 2:], atol=1e-10)

    def test_no_mutation(self, use_sparse):
        """Input precision must not be mutated."""
        rng = np.random.default_rng(42)
        n = 10
        R = _random_pd(n, rng)
        R_orig = R.copy()

        X = np.zeros((3, n))
        X[:, :3] = rng.standard_normal((3, 3))
        y = rng.standard_normal(3)

        _downdate(R, X, y, 0.9, use_sparse)
        assert_allclose(R, R_orig)


class TestSiftForgetting:
    @pytest.mark.parametrize("use_sparse", [False, True], ids=["dense", "sparse"])
    def test_returns_none_when_no_excitation(self, use_sparse):
        rng = np.random.default_rng(42)
        R = _random_pd(5, rng)
        if use_sparse:
            X = sparse.csc_array((3, 5))
        else:
            X = rng.standard_normal((3, 5)) * 1e-15
        y = rng.standard_normal(3)

        rule = SiftForgetting(eps=1e-10)
        result = rule(R, X, y, 0.9)
        assert result is None

    @pytest.mark.parametrize(
        "p_regime",
        ["p_ge_n", "p_lt_n"],
        ids=["full-excitation", "partial-excitation"],
    )
    def test_sift_vs_scalar_precision(self, p_regime):
        """Proposition 5: SIFt + update vs scalar forgetting + update.

        When p >= n (full excitation), they are equal.
        When p < n (partial excitation), SIFt retains strictly more precision.
        """
        rng = np.random.default_rng(42)

        for _ in range(10):
            if p_regime == "p_ge_n":
                n = rng.integers(3, 8)
                p = rng.integers(n, n + 5)
            else:
                n = rng.integers(6, 12)
                p = rng.integers(1, n - 1)

            R = _random_pd(n, rng)
            X = rng.standard_normal((p, n))
            y = rng.standard_normal(p)
            lam = rng.uniform(0.8, 0.99)

            rule = SiftForgetting(eps=1e-12)
            result = rule(R, X, y, lam)
            assert result is not None
            R_bar, X_bar, y_bar = result

            advantage = (_sym(R_bar) + X_bar.T @ X_bar) - (lam * R + X.T @ X)
            eigs = eigvalsh(advantage)
            assert eigs[0] >= -1e-10, f"advantage not PSD: min eig = {eigs[0]}"

            if p_regime == "p_ge_n":
                # Equal: advantage should be zero
                assert_allclose(advantage, 0, atol=1e-8)
            else:
                # Strict: SIFt kept extra precision
                assert eigs[-1] > 1e-10, "advantage is zero, SIFt gave no benefit"

    def test_directional_forgetting_general(self):
        """For general R, verify correction is confined to col(R @ X_bar.T)."""
        rng = np.random.default_rng(42)
        n = 10

        R = _random_pd(n, rng)

        X = np.zeros((3, n))
        X[:, :2] = rng.standard_normal((3, 2))
        y = rng.standard_normal(3)
        lam = 0.9

        result = filter_batch(X, y, eps=1e-12)
        assert result is not None
        X_bar, _ = result

        R_bar = _sym(_sift_downdate_dense(R, X_bar, lam))

        # The correction R - R_bar should live in col(R @ X_bar.T)
        w = R @ X_bar.T  # (n, q)
        correction = R - R_bar

        # Project correction columns onto the orthogonal complement of col(w)
        Q, _ = np.linalg.qr(w)  # (n, q) orthonormal basis for col(w)
        P_perp = np.eye(n) - Q @ Q.T
        residual = P_perp @ correction

        assert_allclose(residual, 0, atol=1e-10)

    def test_eigenvalue_bounds_theorem_3(self):
        """Over 100+ steps, lambda_min(R_k) >= min(eps/(1-lam), lambda_min(R_0))."""
        rng = np.random.default_rng(42)
        n = 6
        lam = 0.95
        eps = 0.1

        R = _random_pd(n, rng, cond=5.0)
        lam_min_R0 = eigvalsh(R)[0]
        lower_bound = min(eps / (1 - lam), lam_min_R0)

        rule = SiftForgetting(eps=eps)

        updates_applied = 0
        for step in range(150):
            p = rng.integers(1, 4)
            X = rng.standard_normal((p, n))
            y = rng.standard_normal(p)

            result = rule(R, X, y, lam)
            if result is None:
                continue
            R_bar, X_bar, y_bar = result
            R_bar_full = _sym(R_bar) if isinstance(R_bar, np.ndarray) else R_bar
            # Forgetting actually modified R (non-vacuous check)
            assert not np.allclose(R_bar_full, R), (
                f"Step {step}: R_bar == R, no forgetting"
            )
            R = np.asarray(R_bar_full + X_bar.T @ X_bar)
            updates_applied += 1

            min_eig = eigvalsh(R)[0]
            assert min_eig >= lower_bound - 1e-8, (
                f"Step {step}: lambda_min={min_eig:.6f} < bound={lower_bound:.6f}"
            )

        assert updates_applied > 100, f"Only {updates_applied}/150 updates applied"

    def test_end_to_end_small_problem(self):
        """Full downdate + update + solve on a small problem matches naive."""
        rng = np.random.default_rng(42)
        n = 10
        p = 3
        lam = 0.9

        R = _random_pd(n, rng)
        theta = rng.standard_normal(n)
        X = rng.standard_normal((p, n))
        y = rng.standard_normal(p)

        # SIFt path
        rule = SiftForgetting(eps=1e-12)
        result = rule(R, X, y, lam)
        assert result is not None
        R_bar_raw, X_bar, y_bar = result
        R_bar = _sym(R_bar_raw)
        R_new = R_bar + X_bar.T @ X_bar
        eta = R_bar @ theta + X_bar.T @ y_bar
        cho = cho_factor(R_new)
        theta_new = cho_solve(cho, eta)

        # Naive path: use _naive_sift_downdate
        filtered = filter_batch(X, y, eps=1e-12)
        assert filtered is not None
        X_bar_n, y_bar_n = filtered
        R_bar_n = _naive_sift_downdate(R, X_bar_n, lam)
        R_new_n = R_bar_n + X_bar_n.T @ X_bar_n
        eta_n = R_bar_n @ theta + X_bar_n.T @ y_bar_n
        theta_new_n = np.linalg.solve(R_new_n, eta_n)

        assert_allclose(R_new, R_new_n, atol=1e-10)
        assert_allclose(theta_new, theta_new_n, atol=1e-10)

    @pytest.mark.parametrize("use_sparse", [False, True], ids=["dense", "sparse"])
    def test_duplicate_rows_in_batch(self, use_sparse):
        """Duplicate X rows produce rank-deficient batch; SIFt handles gracefully.

        Duplicate rows excite the same direction, so q = p - 1. Both copies
        still contribute to X.T @ X and X.T @ y (sufficient statistics are
        preserved), and the downdate operates in the correct lower-dimensional
        information subspace.
        """
        rng = np.random.default_rng(42)
        n = 8
        R = _random_pd(n, rng)

        # Build batch with one duplicate row
        x = rng.standard_normal((1, n))
        X = np.vstack([x, x, rng.standard_normal((2, n))])  # row 0 == row 1
        y = rng.standard_normal(4)
        lam = 0.9

        if use_sparse:
            R = sparse.csc_array(R)
            X = sparse.csc_array(X)

        rule = SiftForgetting(eps=1e-10)
        result = rule(R, X, y, lam)
        assert result is not None
        R_bar, X_bar, y_bar = result

        R_bar_dense = R_bar.toarray() if sparse.issparse(R_bar) else np.asarray(R_bar)
        X_bar_dense = X_bar.toarray() if sparse.issparse(X_bar) else np.asarray(X_bar)
        X_dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
        R_dense = R.toarray() if sparse.issparse(R) else np.asarray(R)

        # q should be 3 (not 4) due to rank deficiency from duplicate row
        assert X_bar_dense.shape[0] == 3

        # Sufficient statistics preserved despite rank deficiency
        assert_allclose(X_bar_dense.T @ X_bar_dense, X_dense.T @ X_dense, atol=1e-10)
        assert_allclose(X_bar_dense.T @ y_bar, X_dense.T @ y, atol=1e-10)

        # R_bar still PD
        R_bar_full = _sym(R_bar_dense) if not use_sparse else R_bar_dense
        assert eigvalsh(R_bar_full)[0] > 0

        # R_bar >= lam * R (Corollary 1)
        diff = R_bar_full - lam * R_dense
        assert eigvalsh(diff)[0] >= -1e-10
