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
    _dirichlet_multinomial_log_evidence,
    _factorization_stats,
    accumulate_sufficient_stats,
    mackay_update_normal_online,
    minka_update_dirichlet_multinomial,
    negbin_update_gamma_poisson,
)
from bayesianbandits._sparse_bayesian_linear_regression import (
    DenseFactor,
    create_sparse_factor,
    scale_factor,
    takahashi_diagonal,
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
# 1. Scaled factor logdet
# ---------------------------------------------------------------------------


class TestScaledFactorLogdet:
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
        result_diag = takahashi_diagonal(L_csc)

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
        result_diag = takahashi_diagonal(L_csc)

        np.testing.assert_allclose(result_diag, expected_diag, atol=1e-10)

    @pytest.mark.cython
    def test_identity_no_factor(self) -> None:
        """Takahashi on identity without SparseFactor: diag(I⁻¹) = ones."""
        p = 10
        L_csc = csc_array(np.eye(p))
        result_diag = takahashi_diagonal(L_csc)
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
        result_trace = factor.trace_inv()

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
        result_diag = takahashi_diagonal(L)

        # Trace is permutation-invariant, so compare sorted diagonals
        np.testing.assert_allclose(
            sorted(result_diag), sorted(expected_diag), atol=1e-10
        )

    def test_scaled_sparse_factor(self) -> None:
        """Takahashi trace through a scaled factor."""
        rng = np.random.default_rng(99)
        p = 5
        A = rng.standard_normal((p, p))
        M = A.T @ A + 4.0 * np.eye(p)
        s = 2.5

        expected_trace = float(np.trace(np.linalg.inv(s * M)))

        sparse_M = csc_array(M)
        factor = create_sparse_factor(sparse_M)
        scaled = scale_factor(factor, s)
        result_trace = scaled.trace_inv()

        np.testing.assert_allclose(result_trace, expected_trace, atol=1e-10)

    def test_identity(self) -> None:
        """Takahashi on identity: diag(I⁻¹) = [1, 1, ..., 1]."""
        p = 10
        M = csc_array(np.eye(p))
        factor = create_sparse_factor(M)
        L = factor.get_L_csc()
        result_diag = takahashi_diagonal(L)
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


# ---------------------------------------------------------------------------
# 13. MacKay NIG
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 14. MacKay GLM (parameterized over link)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 15. Guard clauses
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 16. Error paths
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Minka Dirichlet-Multinomial
# ---------------------------------------------------------------------------


def _make_dirichlet_groups(
    rng: np.random.Generator,
    n_groups: int,
    n_per_group: int,
    K: int,
    true_alpha: NDArray[np.float64],
) -> dict[int, NDArray[np.float64]]:
    """Generate synthetic Dirichlet-Multinomial data as posterior alphas.

    For each group, draw a probability vector from Dir(true_alpha),
    then draw n_per_group multinomial counts.  Return posterior alphas
    = prior + counts for an arbitrary starting prior.
    """
    prior = np.ones(K, dtype=np.float64)  # uniform initial prior
    posterior_alphas: dict[int, NDArray[np.float64]] = {}
    for g in range(n_groups):
        p = rng.dirichlet(true_alpha)
        counts = rng.multinomial(n_per_group, p).astype(np.float64)
        posterior_alphas[g] = prior + counts
    return posterior_alphas


class TestMinkaDirichletMultinomial:
    def test_basic_update(self) -> None:
        """Returns positive alphas and finite log evidence."""
        rng = np.random.default_rng(42)
        prior = np.array([1.0, 1.0, 1.0])
        posteriors = _make_dirichlet_groups(rng, 10, 20, 3, np.array([2.0, 5.0, 1.0]))

        alpha_new, log_ev, _ = minka_update_dirichlet_multinomial(posteriors, prior)

        assert np.all(alpha_new > 0)
        assert np.isfinite(log_ev)

    def test_evidence_nondecreasing(self) -> None:
        """Iterated Minka steps never decrease log evidence."""
        rng = np.random.default_rng(123)
        true_alpha = np.array([5.0, 1.0])
        posteriors = _make_dirichlet_groups(rng, 20, 50, 2, true_alpha)
        prior = np.ones(2, dtype=np.float64)

        evidences: list[float] = []
        alpha = prior.copy()
        for _ in range(30):
            # Recompute posteriors with current prior
            current_posteriors = {}
            for g, post in posteriors.items():
                counts = post - np.ones(2)  # extract counts from original prior=1
                current_posteriors[g] = alpha + counts
            alpha, ev, _ = minka_update_dirichlet_multinomial(
                current_posteriors, alpha, n_iter=1
            )
            evidences.append(ev)

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-6, (
                f"Evidence decreased at step {i}: "
                f"{evidences[i - 1]:.6f} -> {evidences[i]:.6f}"
            )

    def test_fixed_point_idempotent(self) -> None:
        """At convergence, one more step returns same alpha."""
        rng = np.random.default_rng(456)
        true_alpha = np.array([3.0, 1.0, 2.0])
        posteriors = _make_dirichlet_groups(rng, 15, 30, 3, true_alpha)
        prior = np.ones(3, dtype=np.float64)

        # Iterate to convergence with many outer steps
        alpha = prior.copy()
        for _ in range(500):
            current = {g: alpha + (p - prior) for g, p in posteriors.items()}
            alpha_new, _, _ = minka_update_dirichlet_multinomial(
                current, alpha, n_iter=1
            )
            alpha = alpha_new

        # One more step should be approximately idempotent.
        # The (s, m) alternation means one step doesn't fully converge,
        # but the change should be very small.
        current = {g: alpha + (p - prior) for g, p in posteriors.items()}
        alpha_check, _, _ = minka_update_dirichlet_multinomial(current, alpha, n_iter=1)
        np.testing.assert_allclose(alpha_check, alpha, atol=1e-3, rtol=1e-3)

    def test_gradient_at_convergence(self) -> None:
        """At fixed point, ∂log_ev/∂α_k ≈ 0 (via finite differences)."""
        rng = np.random.default_rng(789)
        true_alpha = np.array([4.0, 2.0])
        posteriors = _make_dirichlet_groups(rng, 20, 40, 2, true_alpha)
        prior = np.ones(2, dtype=np.float64)

        # Iterate to convergence
        alpha = prior.copy()
        for _ in range(100):
            current = {g: alpha + (p - prior) for g, p in posteriors.items()}
            alpha_new, _, _ = minka_update_dirichlet_multinomial(
                current, alpha, n_iter=1
            )
            alpha = alpha_new

        # Check gradient via finite differences
        current = {g: alpha + (p - prior) for g, p in posteriors.items()}
        counts = np.array(list(current.values())) - alpha[np.newaxis, :]
        count_totals = counts.sum(axis=1)

        eps = 1e-5
        for k in range(len(alpha)):
            alpha_plus = alpha.copy()
            alpha_plus[k] += eps
            alpha_minus = alpha.copy()
            alpha_minus[k] -= eps

            ev_plus = _dirichlet_multinomial_log_evidence(
                counts, count_totals, alpha_plus
            )
            ev_minus = _dirichlet_multinomial_log_evidence(
                counts, count_totals, alpha_minus
            )
            grad_k = (ev_plus - ev_minus) / (2 * eps)
            assert abs(grad_k) < 0.1, (
                f"Gradient w.r.t. alpha[{k}] = {grad_k:.6f} at convergence"
            )

    def test_uniform_counts_preserve_symmetry(self) -> None:
        """Symmetric counts preserve symmetric prior."""
        K = 3
        prior = np.ones(K, dtype=np.float64)
        # All groups have identical symmetric counts
        posteriors = {g: prior + np.array([10.0, 10.0, 10.0]) for g in range(5)}

        alpha_new, _, _ = minka_update_dirichlet_multinomial(
            posteriors, prior, n_iter=10
        )

        # All components should be equal
        np.testing.assert_allclose(
            alpha_new / alpha_new.sum(),
            np.ones(K) / K,
            atol=1e-6,
        )

    def test_skewed_counts_shift_prior(self) -> None:
        """Skewed counts shift the base measure toward majority class."""
        prior = np.array([1.0, 1.0])
        # All groups have mostly class 0
        posteriors = {g: prior + np.array([50.0, 5.0]) for g in range(10)}

        alpha_new, _, _ = minka_update_dirichlet_multinomial(
            posteriors, prior, n_iter=10
        )

        # Prior should shift toward class 0
        assert alpha_new[0] > alpha_new[1], (
            f"Expected alpha[0] > alpha[1], got {alpha_new}"
        )

    def test_empty_posteriors(self) -> None:
        """Empty dict returns prior unchanged."""
        prior = np.array([1.0, 2.0, 3.0])
        alpha_new, log_ev, converged = minka_update_dirichlet_multinomial({}, prior)
        np.testing.assert_array_equal(alpha_new, prior)
        assert log_ev == -np.inf
        assert converged

    def test_zero_count_groups_return_prior_with_evidence(self) -> None:
        """Groups with posteriors == prior (zero counts) return prior unchanged.

        When all groups have zero effective counts (posterior = prior),
        there is no data to update from. The function should return the
        prior unchanged, with log evidence equal to the Dirichlet-Multinomial
        evidence evaluated at zero counts (which is 0.0 since all
        gammaln terms cancel).
        """
        prior = np.array([2.0, 3.0])
        # Posteriors equal to prior => zero counts for every group
        posteriors = {0: prior.copy(), 1: prior.copy(), 2: prior.copy()}
        alpha_new, log_ev, converged = minka_update_dirichlet_multinomial(
            posteriors, prior, n_iter=10
        )
        np.testing.assert_array_equal(alpha_new, prior)
        assert converged
        # Zero counts => all gammaln terms cancel => log_ev = 0
        assert log_ev == pytest.approx(0.0, abs=1e-12)

    def test_denominator_near_zero_converges_early(self) -> None:
        """Huge prior concentration with tiny counts triggers early exit.

        When s >> c_g for all groups, ψ(c_g + s) - ψ(s) ≈ 0, making
        the shared denominator fall below EPS. The function should
        declare convergence and return a valid result.
        """
        # Very large prior so s = sum(prior) >> counts
        prior = np.array([1e15, 1e15])
        # Each group has 1 observation: tiny counts relative to s
        posteriors = {
            0: prior + np.array([1.0, 0.0]),
            1: prior + np.array([0.0, 1.0]),
        }
        alpha_new, log_ev, converged = minka_update_dirichlet_multinomial(
            posteriors, prior, n_iter=100
        )
        assert converged, "Should converge early when denominator ≈ 0"
        assert np.isfinite(log_ev)
        # With negligible data relative to the prior, alpha should barely move
        np.testing.assert_allclose(alpha_new, prior, rtol=1e-4)


# ---------------------------------------------------------------------------
# NegBin Gamma-Poisson
# ---------------------------------------------------------------------------


class TestNegbinGammaPoisson:
    def test_empty_posteriors(self) -> None:
        """Empty dict returns prior unchanged."""
        prior = np.array([2.0, 3.0])
        prior_new, log_ev, converged = negbin_update_gamma_poisson({}, prior)
        np.testing.assert_array_equal(prior_new, prior)
        assert log_ev == -np.inf
        assert converged

    def test_zero_exposure_groups_return_prior(self) -> None:
        """Groups with posteriors == prior (zero exposure) return prior unchanged."""
        prior = np.array([2.0, 3.0])
        posteriors = {0: prior.copy(), 1: prior.copy()}
        prior_new, log_ev, converged = negbin_update_gamma_poisson(
            posteriors, prior, n_iter=10
        )
        np.testing.assert_array_equal(prior_new, prior)
        assert converged

    def test_basic_update(self) -> None:
        """Returns positive params and finite log evidence."""
        rng = np.random.default_rng(42)
        prior = np.array([1.0, 1.0])
        posteriors = {}
        for g in range(10):
            rate = rng.gamma(3.0, 1.0)
            counts = rng.poisson(rate, size=20)
            posteriors[g] = prior + np.array([float(counts.sum()), float(len(counts))])

        prior_new, log_ev, _ = negbin_update_gamma_poisson(
            posteriors, prior, n_iter=10, tol=1e-6
        )
        assert np.all(prior_new > 0)
        assert np.isfinite(log_ev)

    def test_evidence_nondecreasing(self) -> None:
        """Iterated EM steps never decrease log evidence."""
        rng = np.random.default_rng(123)
        prior = np.array([1.0, 1.0])
        posteriors = {}
        for g in range(15):
            rate = rng.gamma(5.0, 1.0 / 2.0)
            counts = rng.poisson(rate, size=30)
            posteriors[g] = prior + np.array([float(counts.sum()), float(len(counts))])

        evidences: list[float] = []
        alpha = prior.copy()
        for _ in range(20):
            current = {g: alpha + (p - prior) for g, p in posteriors.items()}
            alpha, ev, _ = negbin_update_gamma_poisson(current, alpha, n_iter=1)
            evidences.append(ev)

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-6, (
                f"Evidence decreased at step {i}: "
                f"{evidences[i - 1]:.6f} -> {evidences[i]:.6f}"
            )
