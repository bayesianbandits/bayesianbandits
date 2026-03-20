"""Tests for empirical Bayes helper functions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import cho_factor, solve
from scipy.sparse import csc_array
from scipy.special import expit, gammaln

from bayesianbandits._empirical_bayes import (
    _diagonal_trace_approx,
    _factorization_stats,
    _takahashi_diagonal,
    _takahashi_trace,
    accumulate_sufficient_stats,
    mackay_update_glm,
    mackay_update_nig,
    mackay_update_normal_online,
)
from bayesianbandits._sparse_bayesian_linear_regression import (
    DenseFactor,
    create_sparse_factor,
    scale_factor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dense_factor(precision: NDArray[np.float64]) -> DenseFactor:
    """Create a DenseFactor from a dense precision matrix."""
    cho = cho_factor(precision, lower=False, check_finite=False)
    return DenseFactor(_U=cho[0], _n_features=cho[0].shape[0])


def _compute_posterior_normal(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float,
    beta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute posterior mean and precision for NormalRegressor."""
    p = X.shape[1]
    precision = alpha * np.eye(p) + beta * X.T @ X
    mu_n = solve(precision, beta * X.T @ y, assume_a="pos")
    return mu_n, precision


def _compute_posterior_nig(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float,
    a0: float,
    b0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Compute NIG posterior parameters."""
    p = X.shape[1]
    n = X.shape[0]
    precision = alpha * np.eye(p) + X.T @ X
    mu_n = solve(precision, X.T @ y, assume_a="pos")
    an = a0 + 0.5 * n
    bn = b0 + 0.5 * (float(y @ y) - float(mu_n @ precision @ mu_n))
    return mu_n, precision, an, bn


def _solve_glm_map(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float,
    link: str,
    n_iter: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve for MAP estimate via IRLS and return (theta, Hessian)."""
    p = X.shape[1]
    theta = np.zeros(p)
    for _ in range(n_iter):
        eta = X @ theta
        if link == "logit":
            mu = expit(eta)
            W = mu * (1 - mu) + 1e-8
        elif link == "log":
            mu = np.exp(np.clip(eta, -500, 500))
            W = mu + 1e-8
        else:
            raise ValueError(f"Unknown link: {link}")
        z = eta + (y - mu) / W
        H = alpha * np.eye(p) + X.T @ (W[:, None] * X)
        theta = solve(H, X.T @ (W * z), assume_a="pos")
    # Recompute Hessian at final theta
    eta = X @ theta
    if link == "logit":
        mu = expit(eta)
        W = mu * (1 - mu) + 1e-8
    else:
        mu = np.exp(np.clip(eta, -500, 500))
        W = mu + 1e-8
    H = alpha * np.eye(p) + X.T @ (W[:, None] * X)
    return theta, H


def _generate_glm_data(
    rng: np.random.Generator,
    n: int,
    p: int,
    link: str,
    coef_scale: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate GLM data for a given link function."""
    w = rng.standard_normal(p) * coef_scale
    X = rng.standard_normal((n, p))
    eta = X @ w
    if link == "logit":
        y = (rng.random(n) < expit(eta)).astype(float)
    elif link == "log":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        raise ValueError(f"Unknown link: {link}")
    return X, y, w


def _log_evidence_normal_hand(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    mu_n: NDArray[np.float64],
    precision: NDArray[np.float64],
    alpha: float,
    beta: float,
) -> float:
    """Hand-compute log evidence for Normal model (for test reference)."""
    n = y.shape[0]
    p = precision.shape[0]
    residual = y - X @ mu_n
    ld = float(np.linalg.slogdet(precision)[1])
    return float(
        0.5 * p * np.log(alpha)
        + 0.5 * n * np.log(beta)
        - 0.5 * ld
        - 0.5 * (beta * float(residual @ residual) + alpha * float(mu_n @ mu_n))
        - 0.5 * n * np.log(2 * np.pi)
    )


def _log_evidence_nig_hand(
    n_obs: int,
    p: int,
    precision: NDArray[np.float64],
    alpha: float,
    a0: float,
    b0: float,
    an: float,
    bn: float,
) -> float:
    """Hand-compute log evidence for NIG model (for test reference)."""
    ld = float(np.linalg.slogdet(precision)[1])
    return float(
        gammaln(an)
        - gammaln(a0)
        + a0 * np.log(b0)
        - an * np.log(bn)
        + 0.5 * p * np.log(alpha)
        - 0.5 * ld
        - 0.5 * n_obs * np.log(2 * np.pi)
    )


def _log_evidence_glm_hand(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    H: NDArray[np.float64],
    alpha: float,
    link: str,
) -> float:
    """Hand-compute log evidence for GLM Laplace (for test reference)."""
    p = H.shape[0]
    eta = X @ theta
    if link == "logit":
        mu = expit(eta)
        log_lik = float(
            np.sum(y * np.log(mu + 1e-15) + (1 - y) * np.log(1 - mu + 1e-15))
        )
    else:
        log_lik = float(np.sum(y * eta - np.exp(eta) - gammaln(y + 1)))
    log_prior = 0.5 * p * np.log(alpha / (2 * np.pi)) - 0.5 * alpha * float(
        theta @ theta
    )
    ld = float(np.linalg.slogdet(H)[1])
    laplace = 0.5 * p * np.log(2 * np.pi) - 0.5 * ld
    return float(log_lik + log_prior + laplace)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_spd_matrix() -> NDArray[np.float64]:
    """A small 5x5 SPD matrix."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((5, 5))
    return A.T @ A + 5.0 * np.eye(5)


# ---------------------------------------------------------------------------
# 1. ScaledSparseFactor logdet
# ---------------------------------------------------------------------------


class TestScaledSparseFactorLogdet:
    def test_scaled_logdet(self, small_spd_matrix: NDArray[np.float64]) -> None:
        """Verify log|sP| = p·log(s) + log|P|."""
        scale = 3.7
        p = small_spd_matrix.shape[0]

        sparse_prec = csc_array(small_spd_matrix)
        factor = create_sparse_factor(sparse_prec)
        scaled_factor = scale_factor(factor, scale)

        expected = p * np.log(scale) + factor.logdet()
        np.testing.assert_allclose(scaled_factor.logdet(), expected, atol=1e-10)

        # Also verify against dense
        dense_expected = float(np.linalg.slogdet(scale * small_spd_matrix)[1])
        np.testing.assert_allclose(scaled_factor.logdet(), dense_expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Takahashi diagonal recursion
# ---------------------------------------------------------------------------


class TestTakahashiDiagonal:
    @pytest.mark.cython
    def test_dense_spd_matches_inv(self) -> None:
        """Takahashi diagonal matches np.linalg.inv diagonal for small dense SPD."""
        rng = np.random.default_rng(42)
        p = 6
        A = rng.standard_normal((p, p))
        M = A.T @ A + 3.0 * np.eye(p)

        expected_diag = np.diag(np.linalg.inv(M))

        L_dense = np.linalg.cholesky(M)
        L_csc = csc_array(L_dense)
        result_diag = _takahashi_diagonal(L_csc)

        np.testing.assert_allclose(result_diag, expected_diag, atol=1e-10)

    @pytest.mark.cython
    def test_sparse_tridiagonal_no_factor(self) -> None:
        """Takahashi on a tridiagonal L constructed directly (no suitesparse)."""
        p = 50
        diag = np.full(p, 4.0)
        off_diag = np.full(p - 1, -1.0)
        M_dense = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

        expected_diag = np.diag(np.linalg.inv(M_dense))

        L_dense = np.linalg.cholesky(M_dense)
        L_csc = csc_array(L_dense)
        result_diag = _takahashi_diagonal(L_csc)

        np.testing.assert_allclose(result_diag, expected_diag, atol=1e-10)

    @pytest.mark.cython
    def test_identity_no_factor(self) -> None:
        """Takahashi on identity without SparseFactor: diag(I⁻¹) = ones."""
        p = 10
        L_csc = csc_array(np.eye(p))
        result_diag = _takahashi_diagonal(L_csc)
        np.testing.assert_allclose(result_diag, np.ones(p), atol=1e-14)

    def test_sparse_spd_cholmod_or_superlu(self) -> None:
        """Takahashi via SparseFactor matches dense inverse diagonal."""
        rng = np.random.default_rng(77)
        p = 8
        A = rng.standard_normal((p, p))
        M = A.T @ A + 5.0 * np.eye(p)

        expected_diag = np.diag(np.linalg.inv(M))
        expected_trace = float(np.sum(expected_diag))

        sparse_M = csc_array(M)
        factor = create_sparse_factor(sparse_M)
        result_trace = _takahashi_trace(factor)

        np.testing.assert_allclose(result_trace, expected_trace, atol=1e-10)

    def test_truly_sparse_matrix(self) -> None:
        """Takahashi on a sparse tridiagonal matrix."""
        p = 20
        diag = np.full(p, 4.0)
        off_diag = np.full(p - 1, -1.0)
        M_dense = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

        expected_diag = np.diag(np.linalg.inv(M_dense))

        sparse_M = csc_array(M_dense)
        factor = create_sparse_factor(sparse_M)
        L = factor.get_L_csc()
        result_diag = _takahashi_diagonal(L)

        # Trace is permutation-invariant, so compare sorted diagonals
        np.testing.assert_allclose(
            sorted(result_diag), sorted(expected_diag), atol=1e-10
        )

    def test_scaled_sparse_factor(self) -> None:
        """Takahashi trace through ScaledSparseFactor."""
        rng = np.random.default_rng(99)
        p = 5
        A = rng.standard_normal((p, p))
        M = A.T @ A + 4.0 * np.eye(p)
        s = 2.5

        expected_trace = float(np.trace(np.linalg.inv(s * M)))

        sparse_M = csc_array(M)
        factor = create_sparse_factor(sparse_M)
        scaled = scale_factor(factor, s)
        result_trace = _takahashi_trace(scaled)

        np.testing.assert_allclose(result_trace, expected_trace, atol=1e-10)

    def test_identity(self) -> None:
        """Takahashi on identity: diag(I⁻¹) = [1, 1, ..., 1]."""
        p = 10
        M = csc_array(np.eye(p))
        factor = create_sparse_factor(M)
        L = factor.get_L_csc()
        result_diag = _takahashi_diagonal(L)
        np.testing.assert_allclose(result_diag, np.ones(p), atol=1e-14)


# ---------------------------------------------------------------------------
# 4. Diagonal trace approximation
# ---------------------------------------------------------------------------


class TestDiagonalTraceApprox:
    def test_diagonal_matrix_exact(self) -> None:
        """For a diagonal matrix, the approximation is exact."""
        diag_vals = np.array([2.0, 5.0, 10.0, 0.5])
        M = np.diag(diag_vals)
        expected = float(np.sum(1.0 / diag_vals))
        result = _diagonal_trace_approx(M)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_sparse_diagonal_matrix_exact(self) -> None:
        """Sparse diagonal: approximation is exact."""
        diag_vals = np.array([2.0, 5.0, 10.0, 0.5])
        M = csc_array(np.diag(diag_vals))
        expected = float(np.sum(1.0 / diag_vals))
        result = _diagonal_trace_approx(M)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_diagonally_dominant_close(self) -> None:
        """For strongly diag-dominant matrices, approx is close to exact."""
        rng = np.random.default_rng(42)
        p = 5
        A = rng.standard_normal((p, p)) * 0.1
        M = A.T @ A + 10.0 * np.eye(p)  # strongly dominant

        exact = float(np.trace(np.linalg.inv(M)))
        approx = _diagonal_trace_approx(M)

        # Should be within ~10% for strongly dominant
        np.testing.assert_allclose(approx, exact, rtol=0.15)


# ---------------------------------------------------------------------------
# 5. _factorization_stats with trace_method
# ---------------------------------------------------------------------------


class TestFactorizationStatsTraceMethod:
    def test_auto_dense_matches_cholesky(self) -> None:
        rng = np.random.default_rng(42)
        p = 5
        A = rng.standard_normal((p, p))
        M = A.T @ A + 3.0 * np.eye(p)

        ld, tr_inv = _factorization_stats(M, _make_dense_factor(M), trace_method="auto")
        expected_tr = float(np.trace(np.linalg.inv(M)))
        np.testing.assert_allclose(tr_inv, expected_tr, atol=1e-10)

    def test_diagonal_method(self) -> None:
        rng = np.random.default_rng(42)
        p = 5
        A = rng.standard_normal((p, p))
        M = A.T @ A + 10.0 * np.eye(p)

        ld, tr_inv = _factorization_stats(
            M, _make_dense_factor(M), trace_method="diagonal"
        )
        expected = _diagonal_trace_approx(M)
        np.testing.assert_allclose(tr_inv, expected, atol=1e-14)

    def test_auto_sparse_exact(self) -> None:
        rng = np.random.default_rng(42)
        p = 5
        A = rng.standard_normal((p, p))
        M = A.T @ A + 3.0 * np.eye(p)

        sparse_M = csc_array(M)
        ld, tr_inv = _factorization_stats(
            sparse_M, create_sparse_factor(sparse_M), trace_method="auto"
        )
        expected_tr = float(np.trace(np.linalg.inv(M)))
        np.testing.assert_allclose(tr_inv, expected_tr, atol=1e-10)


# ---------------------------------------------------------------------------
# 6-9. Removed: Tests that used the deleted mackay_update_normal function.
# The underlying logic is now tested via mackay_update_normal_online and
# the estimator-level tests in test_eb_estimators.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10. Log evidence vs hand computation (folded into update tests)
# ---------------------------------------------------------------------------


class TestLogEvidenceFromUpdates:
    def test_log_evidence_normal(self) -> None:
        """Verify log evidence returned by mackay_update_normal_online."""
        rng = np.random.default_rng(7)
        n, p = 20, 3
        alpha, beta = 1.5, 3.0

        X = rng.standard_normal((n, p))
        w = rng.standard_normal(p)
        y = X @ w + rng.standard_normal(n) * 0.5
        mu_n, precision = _compute_posterior_normal(X, y, alpha, beta)

        _, _, result = mackay_update_normal_online(
            mu_n,
            precision,
            alpha,
            beta,
            prior_scalar=alpha,
            effective_n=float(n),
            eff_yTy=float(y @ y),
            eff_XTy=np.asarray(X.T @ y, dtype=np.float64),
            factor=_make_dense_factor(precision),
        )
        expected = _log_evidence_normal_hand(X, y, mu_n, precision, alpha, beta)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_log_evidence_nig(self) -> None:
        """Verify NIG evidence against hand computation."""
        rng = np.random.default_rng(8)
        n, p = 20, 3
        alpha_lam = 2.0
        a0, b0 = 1.0, 1.0

        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)

        mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha_lam, a0, b0)

        _, result = mackay_update_nig(
            mu_n,
            precision,
            alpha_lam,
            an,
            bn,
            a0,
            b0,
            factor=_make_dense_factor(precision),
            n_obs=n,
        )
        expected = _log_evidence_nig_hand(n, p, precision, alpha_lam, a0, b0, an, bn)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_log_evidence_glm(self, link: str) -> None:
        """Verify GLM Laplace evidence against hand computation."""
        rng = np.random.default_rng(9)
        p = 3
        alpha = 1.0

        X, y, _ = _generate_glm_data(rng, n=30, p=p, link=link, coef_scale=0.3)
        theta, H = _solve_glm_map(X, y, alpha, link)

        _, result = mackay_update_glm(
            theta, H, alpha, X, y, link, factor=_make_dense_factor(H)
        )
        expected = _log_evidence_glm_hand(X, y, theta, H, alpha, link)
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 11. Sufficient statistics
# ---------------------------------------------------------------------------


class TestSufficientStats:
    def test_accumulated_stats_sparse_X(self) -> None:
        """accumulate_sufficient_stats with sparse X matches dense."""
        rng = np.random.default_rng(55)
        n, p = 15, 4

        X_dense = rng.standard_normal((n, p))
        X_sparse = csc_array(X_dense)
        y = rng.standard_normal(n)

        eff_n_d = 0.0
        eff_yTy_d = 0.0
        eff_XTy_d = np.zeros(p)

        eff_n_s = 0.0
        eff_yTy_s = 0.0
        eff_XTy_s = np.zeros(p)

        for i in range(n):
            eff_n_d, eff_yTy_d, eff_XTy_d = accumulate_sufficient_stats(
                eff_n_d,
                eff_yTy_d,
                eff_XTy_d,
                X_dense[[i]],
                y[[i]],
                prior_decay=0.99,
            )
            eff_n_s, eff_yTy_s, eff_XTy_s = accumulate_sufficient_stats(
                eff_n_s,
                eff_yTy_s,
                eff_XTy_s,
                csc_array(X_sparse[[i]]),
                y[[i]],
                prior_decay=0.99,
            )

        np.testing.assert_allclose(eff_n_s, eff_n_d, atol=1e-10)
        np.testing.assert_allclose(eff_yTy_s, eff_yTy_d, atol=1e-10)
        np.testing.assert_allclose(eff_XTy_s, eff_XTy_d, atol=1e-10)

    def test_online_beta_fallback(self) -> None:
        """When RSS <= 0, mackay_update_normal_online returns beta unchanged."""
        rng = np.random.default_rng(33)
        p = 3
        alpha, beta = 1.0, 2.0
        mu_n = rng.standard_normal(p)
        precision = np.eye(p) * alpha + np.eye(p) * beta

        # Craft sufficient stats so RSS = yTy - 2*m'XTy + m'XTXm <= 0.
        # Set eff_yTy = 0 and eff_XTy large so RSS goes negative.
        alpha_new, beta_new, _ = mackay_update_normal_online(
            mu_n,
            precision,
            alpha,
            beta,
            prior_scalar=alpha,
            effective_n=1.0,
            eff_yTy=0.0,
            eff_XTy=mu_n * 1000.0,
            factor=_make_dense_factor(precision),
        )
        assert beta_new == beta

    def test_online_rejects_pathological_beta_alpha_ratio(self) -> None:
        """When beta_new/alpha_new would exceed the guardrail, keep old hyperparams."""
        p = 3
        alpha, beta = 1.0, 1.0
        # Large mu_n => alpha_new = gamma / ||mu||^2 is tiny
        mu_n = np.ones(p) * 1e8
        precision = alpha * np.eye(p) + beta * np.eye(p)
        # Sufficient stats crafted so RSS is tiny positive => beta_new huge
        # RSS = yTy - 2*m'XTy + m'XTXm; with XTX = (Λ - α·I)/β = I,
        # m'XTXm = ||m||^2 = 3e16. Set yTy and XTy so RSS is tiny.
        mTXTXm = float(mu_n @ mu_n)  # 3e16
        eff_XTy = mu_n.copy()  # mTXTy = ||m||^2 = 3e16
        # RSS = yTy - 2*3e16 + 3e16 = yTy - 3e16
        # Set yTy = 3e16 + 1e-20 so RSS = 1e-20
        eff_yTy = mTXTXm + 1e-20

        alpha_new, beta_new, _ = mackay_update_normal_online(
            mu_n,
            precision,
            alpha,
            beta,
            prior_scalar=alpha,
            effective_n=100.0,
            eff_yTy=eff_yTy,
            eff_XTy=eff_XTy,
            factor=_make_dense_factor(precision),
        )
        # beta_new/alpha_new would be astronomically large; guardrail rejects
        assert alpha_new == alpha
        assert beta_new == beta

    def test_online_sparse_precision_matvec(self) -> None:
        """mackay_update_normal_online handles sparse precision @ dense mu_n."""
        rng = np.random.default_rng(88)
        n, p = 20, 4
        alpha, beta = 1.5, 2.5

        X = rng.standard_normal((n, p))
        w = rng.standard_normal(p)
        y = X @ w + rng.standard_normal(n) * 0.3

        mu_n, precision_dense = _compute_posterior_normal(X, y, alpha, beta)
        precision_sparse = csc_array(precision_dense)

        # Dense reference
        alpha_dense, beta_dense, _ = mackay_update_normal_online(
            mu_n,
            precision_dense,
            alpha,
            beta,
            prior_scalar=alpha,
            effective_n=float(n),
            eff_yTy=float(y @ y),
            eff_XTy=X.T @ y,
            factor=_make_dense_factor(precision_dense),
        )

        # Sparse path (now exact via Takahashi)
        alpha_sparse, beta_sparse, _ = mackay_update_normal_online(
            mu_n,
            precision_sparse,
            alpha,
            beta,
            prior_scalar=alpha,
            effective_n=float(n),
            eff_yTy=float(y @ y),
            eff_XTy=X.T @ y,
            factor=create_sparse_factor(precision_sparse),
        )

        np.testing.assert_allclose(alpha_sparse, alpha_dense, atol=1e-10)
        np.testing.assert_allclose(beta_sparse, beta_dense, atol=1e-10)


# ---------------------------------------------------------------------------
# 12. Sparse/dense parity
# ---------------------------------------------------------------------------


class TestSparseDenseParity:
    @pytest.mark.parametrize("sparse", [True, False])
    def test_mackay_update_nig_parity(self, sparse: bool) -> None:
        rng = np.random.default_rng(77)
        n, p = 50, 4
        alpha = 1.0
        a0, b0 = 1.0, 1.0
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)

        if sparse:
            sp = csc_array(precision)
            result, ev = mackay_update_nig(
                mu_n,
                sp,
                alpha,
                an,
                bn,
                a0,
                b0,
                factor=create_sparse_factor(sp),
                n_obs=n,
            )
        else:
            result, ev = mackay_update_nig(
                mu_n,
                precision,
                alpha,
                an,
                bn,
                a0,
                b0,
                factor=_make_dense_factor(precision),
                n_obs=n,
            )

        exact, ev_exact = mackay_update_nig(
            mu_n,
            precision,
            alpha,
            an,
            bn,
            a0,
            b0,
            factor=_make_dense_factor(precision),
            n_obs=n,
        )
        np.testing.assert_allclose(result, exact, atol=1e-10)
        np.testing.assert_allclose(ev, ev_exact, rtol=0.01)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_mackay_update_glm_parity(self, sparse: bool) -> None:
        rng = np.random.default_rng(99)
        n, p = 50, 4
        alpha = 1.0
        X, y, _ = _generate_glm_data(rng, n=n, p=p, link="logit")
        theta, H = _solve_glm_map(X, y, alpha, "logit")

        if sparse:
            sp = csc_array(H)
            result, ev = mackay_update_glm(
                theta,
                sp,
                alpha,
                X,
                y,
                "logit",
                factor=create_sparse_factor(sp),
            )
        else:
            result, ev = mackay_update_glm(
                theta,
                H,
                alpha,
                X,
                y,
                "logit",
                factor=_make_dense_factor(H),
            )

        exact, ev_exact = mackay_update_glm(
            theta,
            H,
            alpha,
            X,
            y,
            "logit",
            factor=_make_dense_factor(H),
        )
        np.testing.assert_allclose(result, exact, atol=1e-10)
        np.testing.assert_allclose(ev, ev_exact, rtol=0.01)


# ---------------------------------------------------------------------------
# 13. MacKay NIG
# ---------------------------------------------------------------------------


class TestMackayNIG:
    def test_basic_update(self) -> None:
        rng = np.random.default_rng(77)
        p = 4
        alpha = 1.0
        an, bn = 5.0, 2.0
        a0, b0 = 1.0, 1.0
        mu_n = rng.standard_normal(p)
        A = rng.standard_normal((p, p))
        precision = A.T @ A + 3.0 * np.eye(p)

        alpha_new, _ = mackay_update_nig(
            mu_n,
            precision,
            alpha,
            an,
            bn,
            a0,
            b0,
            factor=_make_dense_factor(precision),
        )
        assert alpha_new > 0

    def test_identity_design_analytical(self) -> None:
        """X=I: verify against closed-form."""
        p = 5
        alpha = 2.0
        a0, b0 = 3.0, 2.0
        rng = np.random.default_rng(99)

        X = np.eye(p)
        y = rng.standard_normal(p)

        mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)
        sigma_sq = bn / an

        # With X=I: precision = (alpha+1)I, tr(inv) = p/(alpha+1)
        _, tr_inv_actual = _factorization_stats(
            precision, _make_dense_factor(precision)
        )
        tr_inv = p / (alpha + 1.0)
        np.testing.assert_allclose(tr_inv_actual, tr_inv, atol=1e-10)

        # Verify alpha_new = p*sigma^2 / (||mu_n||^2 + sigma^2 * tr_inv)
        expected_alpha = p * sigma_sq / (float(mu_n @ mu_n) + sigma_sq * tr_inv)
        alpha_new, _ = mackay_update_nig(
            mu_n,
            precision,
            alpha,
            an,
            bn,
            a0,
            b0,
            factor=_make_dense_factor(precision),
        )
        np.testing.assert_allclose(alpha_new, expected_alpha, atol=1e-10)

    def test_evidence_nondecreasing(self) -> None:
        """NIG MacKay iterations must not decrease log evidence."""
        rng = np.random.default_rng(123)
        n, p = 100, 3
        X = rng.standard_normal((n, p))
        w = rng.standard_normal(p)
        y = X @ w + rng.standard_normal(n) * 0.5

        alpha = 0.1
        a0, b0 = 1.0, 1.0
        evidences: list[float] = []

        for _ in range(30):
            mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)
            alpha, ev = mackay_update_nig(
                mu_n,
                precision,
                alpha,
                an,
                bn,
                a0,
                b0,
                factor=_make_dense_factor(precision),
                n_obs=n,
            )
            evidences.append(ev)

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-6, (
                f"NIG evidence decreased at step {i}: "
                f"{evidences[i - 1]:.6f} -> {evidences[i]:.6f}"
            )

    def test_gradient_at_convergence(self) -> None:
        """At NIG MacKay fixed point, d/dalpha log_evidence_nig ~ 0."""
        rng = np.random.default_rng(456)
        n, p = 100, 3
        X = rng.standard_normal((n, p))
        w = rng.standard_normal(p)
        y = X @ w + rng.standard_normal(n) * 0.5

        alpha = 0.1
        a0, b0 = 1.0, 1.0

        for _ in range(200):
            mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)
            alpha_new, _ = mackay_update_nig(
                mu_n,
                precision,
                alpha,
                an,
                bn,
                a0,
                b0,
                factor=_make_dense_factor(precision),
            )
            if abs(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        eps = 1e-5

        def ev(a: float) -> float:
            m, P, an_, bn_ = _compute_posterior_nig(X, y, a, a0, b0)
            _, e = mackay_update_nig(
                m,
                P,
                a,
                an_,
                bn_,
                a0,
                b0,
                factor=_make_dense_factor(P),
                n_obs=n,
            )
            return e

        d_alpha = (ev(alpha + eps) - ev(alpha - eps)) / (2 * eps)
        np.testing.assert_allclose(d_alpha, 0.0, atol=1e-3)

    def test_fixed_point_idempotent(self) -> None:
        """At convergence, the update is a fixed point: alpha_new ~ alpha."""
        rng = np.random.default_rng(456)
        n, p = 100, 3
        X = rng.standard_normal((n, p))
        w = rng.standard_normal(p)
        y = X @ w + rng.standard_normal(n) * 0.5

        alpha = 0.1
        a0, b0 = 1.0, 1.0

        for _ in range(200):
            mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)
            alpha_new, _ = mackay_update_nig(
                mu_n,
                precision,
                alpha,
                an,
                bn,
                a0,
                b0,
                factor=_make_dense_factor(precision),
            )
            if abs(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        # One more update should return the same value
        mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)
        alpha_check, _ = mackay_update_nig(
            mu_n,
            precision,
            alpha,
            an,
            bn,
            a0,
            b0,
            factor=_make_dense_factor(precision),
        )
        np.testing.assert_allclose(alpha_check, alpha, rtol=1e-8)

    def test_recovery(self) -> None:
        """Iterate NIG MacKay and recover approximately correct alpha."""
        rng = np.random.default_rng(789)
        n, p = 500, 3
        alpha_true = 2.0
        sigma_true = 0.5

        w = rng.normal(0, 1.0 / np.sqrt(alpha_true), size=p)
        X = rng.standard_normal((n, p))
        y = X @ w + rng.normal(0, sigma_true, size=n)

        alpha = 0.1
        a0, b0 = 0.01, 0.01  # weak prior on sigma

        for _ in range(100):
            mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)
            alpha, _ = mackay_update_nig(
                mu_n,
                precision,
                alpha,
                an,
                bn,
                a0,
                b0,
                factor=_make_dense_factor(precision),
            )

        # NIG recovery is less precise than Normal due to joint sigma^2 estimation
        np.testing.assert_allclose(alpha, alpha_true, rtol=1.0)


# ---------------------------------------------------------------------------
# 14. MacKay GLM (parameterized over link)
# ---------------------------------------------------------------------------


class TestMackayGLM:
    def test_basic_update(self) -> None:
        rng = np.random.default_rng(88)
        n, p = 20, 4
        alpha = 1.0
        theta = rng.standard_normal(p) * 0.5
        A = rng.standard_normal((p, p))
        precision = A.T @ A + alpha * np.eye(p)
        X = rng.standard_normal((n, p))
        y = (rng.random(n) < 0.5).astype(float)

        alpha_new, _ = mackay_update_glm(
            theta,
            precision,
            alpha,
            X,
            y,
            "logit",
            factor=_make_dense_factor(precision),
        )
        assert alpha_new > 0

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_evidence_nondecreasing(self, link: str) -> None:
        """GLM MacKay iterations must not decrease Laplace evidence."""
        rng = np.random.default_rng(111)
        X, y, _ = _generate_glm_data(rng, n=200, p=3, link=link)

        alpha = 0.1
        evidences: list[float] = []

        for _ in range(30):
            theta, H = _solve_glm_map(X, y, alpha, link)
            alpha, ev = mackay_update_glm(
                theta,
                H,
                alpha,
                X,
                y,
                link,
                factor=_make_dense_factor(H),
            )
            evidences.append(ev)

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-4, (
                f"GLM ({link}) evidence decreased at step {i}: "
                f"{evidences[i - 1]:.6f} -> {evidences[i]:.6f}"
            )

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_gradient_at_convergence(self, link: str) -> None:
        """At GLM MacKay fixed point, d/dalpha log_evidence_glm ~ 0."""
        rng = np.random.default_rng(222)
        X, y, _ = _generate_glm_data(rng, n=200, p=3, link=link)

        alpha = 1.0

        for _ in range(50):
            theta, H = _solve_glm_map(X, y, alpha, link)
            alpha_new, _ = mackay_update_glm(
                theta,
                H,
                alpha,
                X,
                y,
                link,
                factor=_make_dense_factor(H),
            )
            if abs(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        eps = 1e-5

        def ev(a: float) -> float:
            t, h = _solve_glm_map(X, y, a, link)
            _, e = mackay_update_glm(
                t,
                h,
                a,
                X,
                y,
                link,
                factor=_make_dense_factor(h),
            )
            return e

        d_alpha = (ev(alpha + eps) - ev(alpha - eps)) / (2 * eps)
        np.testing.assert_allclose(d_alpha, 0.0, atol=1e-2)

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_fixed_point_idempotent(self, link: str) -> None:
        """At convergence, the update is a fixed point: alpha_new ~ alpha."""
        rng = np.random.default_rng(222)
        X, y, _ = _generate_glm_data(rng, n=200, p=3, link=link)

        alpha = 1.0

        for _ in range(50):
            theta, H = _solve_glm_map(X, y, alpha, link)
            alpha_new, _ = mackay_update_glm(
                theta,
                H,
                alpha,
                X,
                y,
                link,
                factor=_make_dense_factor(H),
            )
            if abs(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        # One more update should return the same value
        theta, H = _solve_glm_map(X, y, alpha, link)
        alpha_check, _ = mackay_update_glm(
            theta,
            H,
            alpha,
            X,
            y,
            link,
            factor=_make_dense_factor(H),
        )
        np.testing.assert_allclose(alpha_check, alpha, rtol=1e-6)


# ---------------------------------------------------------------------------
# 15. Guard clauses
# ---------------------------------------------------------------------------


class TestGuardClauses:
    def test_mackay_nig_zero_denom_returns_old_alpha(self) -> None:
        """When denom <= 0, mackay_update_nig should return the old alpha."""
        p = 3
        alpha = 5.0
        mu_n = np.zeros(p)
        # sigma_sq = bn/an = 0, and mu_n = 0 => denom = 0
        precision = np.eye(p)
        an = 1.0
        bn = 0.0  # sigma_sq = 0 exactly

        alpha_new, _ = mackay_update_nig(
            mu_n,
            precision,
            alpha,
            an,
            bn,
            a0=1.0,
            b0=1.0,
            factor=_make_dense_factor(precision),
        )
        assert alpha_new == alpha

    def test_mackay_glm_zero_theta_returns_old_alpha(self) -> None:
        """When theta_MAP is zero, alpha_new should fall back to the old alpha."""
        p = 4
        alpha = 3.0
        theta_MAP = np.zeros(p)
        rng = np.random.default_rng(42)
        A = rng.standard_normal((p, p))
        precision = A.T @ A + alpha * np.eye(p)
        X = rng.standard_normal((10, p))
        y = (rng.random(10) < 0.5).astype(float)

        alpha_new, _ = mackay_update_glm(
            theta_MAP,
            precision,
            alpha,
            X,
            y,
            "logit",
            factor=_make_dense_factor(precision),
        )
        assert alpha_new == alpha


# ---------------------------------------------------------------------------
# 16. Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_mackay_glm_unknown_link_raises(self) -> None:
        rng = np.random.default_rng(42)
        p = 3
        X = rng.standard_normal((10, p))
        y = rng.standard_normal(10)
        theta = rng.standard_normal(p)
        H = np.eye(p)

        with pytest.raises(ValueError, match="Unknown link function"):
            mackay_update_glm(
                theta,
                H,
                1.0,
                X,
                y,
                "unknown",
                factor=_make_dense_factor(H),
            )
