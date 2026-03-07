"""Tests for empirical Bayes helper functions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.sparse import csc_array, eye as speye
from scipy.special import expit, gammaln

from bayesianbandits._empirical_bayes import (
    effective_num_parameters,
    log_evidence_glm_laplace,
    log_evidence_nig,
    log_evidence_normal,
    logdet,
    mackay_update_glm,
    mackay_update_nig,
    mackay_update_normal,
    trace_of_inverse,
)
from bayesianbandits._sparse_bayesian_linear_regression import (
    create_sparse_factor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_spd_matrix() -> NDArray[np.float64]:
    """A small 5x5 SPD matrix."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((5, 5))
    return A.T @ A + 5.0 * np.eye(5)


@pytest.fixture
def regression_data() -> (
    tuple[NDArray[np.float64], NDArray[np.float64], float, float]
):
    """Generate regression data with known hyperparameters."""
    rng = np.random.default_rng(123)
    n, p = 100, 3
    alpha_true = 2.0
    beta_true = 5.0

    w_true = rng.normal(0, 1.0 / np.sqrt(alpha_true), size=p)
    X = rng.standard_normal((n, p))
    y = X @ w_true + rng.normal(0, 1.0 / np.sqrt(beta_true), size=n)
    return X, y, alpha_true, beta_true


# ---------------------------------------------------------------------------
# 1. logdet correctness
# ---------------------------------------------------------------------------


class TestLogdet:
    def test_dense_matches_slogdet(self, small_spd_matrix: NDArray[np.float64]) -> None:
        expected = float(np.linalg.slogdet(small_spd_matrix)[1])
        result = logdet(small_spd_matrix, sparse=False)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sparse_matches_dense(self, small_spd_matrix: NDArray[np.float64]) -> None:
        dense_ld = logdet(small_spd_matrix, sparse=False)
        sparse_prec = csc_array(small_spd_matrix)
        sparse_ld = logdet(sparse_prec, sparse=True)
        np.testing.assert_allclose(sparse_ld, dense_ld, atol=1e-10)

    def test_sparse_with_precomputed_factor(
        self, small_spd_matrix: NDArray[np.float64]
    ) -> None:
        sparse_prec = csc_array(small_spd_matrix)
        factor = create_sparse_factor(sparse_prec)
        result = logdet(sparse_prec, factor=factor, sparse=True)
        expected = float(np.linalg.slogdet(small_spd_matrix)[1])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_not_pd_raises(self) -> None:
        bad = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="not positive definite"):
            logdet(bad, sparse=False)


# ---------------------------------------------------------------------------
# 2. ScaledSparseFactor logdet
# ---------------------------------------------------------------------------


class TestScaledSparseFactorLogdet:
    def test_scaled_logdet(self, small_spd_matrix: NDArray[np.float64]) -> None:
        """Verify log|sP| = p·log(s) + log|P|."""
        from bayesianbandits._sparse_bayesian_linear_regression import scale_factor

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
# 3. Hutchinson trace estimator
# ---------------------------------------------------------------------------


class TestTraceOfInverse:
    def test_dense_exact(self, small_spd_matrix: NDArray[np.float64]) -> None:
        expected = float(np.trace(np.linalg.inv(small_spd_matrix)))
        result = trace_of_inverse(small_spd_matrix, sparse=False)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_hutchinson_converges(
        self, small_spd_matrix: NDArray[np.float64]
    ) -> None:
        """Stochastic trace converges to exact trace (rtol ~5% at 50 probes)."""
        exact = float(np.trace(np.linalg.inv(small_spd_matrix)))
        sparse_prec = csc_array(small_spd_matrix)
        result = trace_of_inverse(
            sparse_prec, sparse=True, n_probes=200, rng=np.random.default_rng(0)
        )
        np.testing.assert_allclose(result, exact, rtol=0.1)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_identity_trace(self, sparse: bool) -> None:
        """tr(I^{-1}) = p."""
        p = 10
        if sparse:
            prec = csc_array(speye(p, format="csc"))
            result = trace_of_inverse(
                prec, sparse=True, n_probes=100, rng=np.random.default_rng(42)
            )
        else:
            prec = np.eye(p)
            result = trace_of_inverse(prec, sparse=False)
        np.testing.assert_allclose(result, float(p), rtol=0.05)


# ---------------------------------------------------------------------------
# 4. effective_num_parameters
# ---------------------------------------------------------------------------


class TestEffectiveNumParameters:
    def test_bounds(self, small_spd_matrix: NDArray[np.float64]) -> None:
        """gamma must satisfy 0 <= gamma <= p."""
        p = small_spd_matrix.shape[0]
        gamma = effective_num_parameters(1.0, small_spd_matrix, sparse=False)
        assert 0 <= gamma <= p

    def test_analytical_identity_case(self) -> None:
        """X=I: precision=(alpha+beta)I, so tr(inv)=p/(alpha+beta), gamma=p*beta/(alpha+beta)."""
        p = 5
        alpha, beta = 2.0, 8.0
        precision = (alpha + beta) * np.eye(p)
        gamma = effective_num_parameters(alpha, precision, sparse=False)
        expected = p * beta / (alpha + beta)
        np.testing.assert_allclose(gamma, expected, atol=1e-10)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_sparse_dense_parity(
        self, small_spd_matrix: NDArray[np.float64], sparse: bool
    ) -> None:
        if sparse:
            prec = csc_array(small_spd_matrix)
            gamma = effective_num_parameters(
                1.0, prec, sparse=True, n_probes=500, rng=np.random.default_rng(0)
            )
        else:
            gamma = effective_num_parameters(1.0, small_spd_matrix, sparse=False)
        exact = effective_num_parameters(1.0, small_spd_matrix, sparse=False)
        np.testing.assert_allclose(gamma, exact, rtol=0.1)


# ---------------------------------------------------------------------------
# 5. MacKay analytical case (X=I)
# ---------------------------------------------------------------------------


class TestMackayAnalytical:
    def test_identity_design(self) -> None:
        """X=I: verify each intermediate against closed-form."""
        p = 5
        alpha, beta = 1.0, 10.0
        rng = np.random.default_rng(99)

        w_true = rng.normal(0, 1.0 / np.sqrt(alpha), size=p)
        X = np.eye(p)
        y = X @ w_true + rng.normal(0, 1.0 / np.sqrt(beta), size=p)

        # Posterior
        precision = (alpha + beta) * np.eye(p)
        mu_n = solve(precision, beta * y, assume_a="pos")

        # Verify trace
        tr_inv = trace_of_inverse(precision, sparse=False)
        np.testing.assert_allclose(tr_inv, p / (alpha + beta), atol=1e-10)

        # Verify gamma
        gamma = effective_num_parameters(alpha, precision, sparse=False)
        np.testing.assert_allclose(gamma, p * beta / (alpha + beta), atol=1e-10)

        # Run MacKay update
        alpha_new, beta_new = mackay_update_normal(
            mu_n, precision, X, y, alpha, beta, sparse=False
        )

        # Verify alpha_new = gamma / ||mu_n||^2
        expected_alpha = gamma / float(mu_n @ mu_n)
        np.testing.assert_allclose(alpha_new, expected_alpha, atol=1e-10)

        # Verify beta_new = (n - gamma) / ||y - X mu_n||^2
        residual = y - X @ mu_n
        expected_beta = (p - gamma) / float(residual @ residual)
        np.testing.assert_allclose(beta_new, expected_beta, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. MacKay recovery
# ---------------------------------------------------------------------------


class TestMackayRecovery:
    def test_converges_to_true_hyperparams(
        self,
        regression_data: tuple[
            NDArray[np.float64], NDArray[np.float64], float, float
        ],
    ) -> None:
        """Iterate MacKay updates and verify convergence to true hyperparams."""
        X, y, alpha_true, beta_true = regression_data

        # Start from wrong hyperparams
        alpha, beta = 0.1, 0.1

        for _ in range(100):
            mu_n, precision = _compute_posterior_normal(X, y, alpha, beta)
            alpha, beta = mackay_update_normal(
                mu_n, precision, X, y, alpha, beta, sparse=False
            )

        # Should be in the right ballpark (not exact due to finite data)
        np.testing.assert_allclose(alpha, alpha_true, rtol=0.5)
        np.testing.assert_allclose(beta, beta_true, rtol=0.5)


# ---------------------------------------------------------------------------
# 7. Evidence monotonicity
# ---------------------------------------------------------------------------


class TestEvidenceMonotonicity:
    def test_evidence_nondecreasing(
        self,
        regression_data: tuple[
            NDArray[np.float64], NDArray[np.float64], float, float
        ],
    ) -> None:
        """Each EM iteration must not decrease log evidence (EM invariant)."""
        X, y, _, _ = regression_data
        alpha, beta = 0.1, 0.1
        evidences: list[float] = []

        for _ in range(30):
            mu_n, precision = _compute_posterior_normal(X, y, alpha, beta)
            ev = log_evidence_normal(X, y, mu_n, precision, alpha, beta)
            evidences.append(ev)
            alpha, beta = mackay_update_normal(
                mu_n, precision, X, y, alpha, beta, sparse=False
            )

        # Evidence should be non-decreasing (allow tiny floating-point dips)
        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-6, (
                f"Evidence decreased at step {i}: {evidences[i-1]:.6f} -> {evidences[i]:.6f}"
            )


# ---------------------------------------------------------------------------
# 8. Gradient check at convergence
# ---------------------------------------------------------------------------


class TestGradientAtConvergence:
    def test_gradients_near_zero(
        self,
        regression_data: tuple[
            NDArray[np.float64], NDArray[np.float64], float, float
        ],
    ) -> None:
        """At MacKay fixed point, d/dalpha and d/dbeta of log evidence ~ 0."""
        X, y, _, _ = regression_data
        alpha, beta = 0.1, 0.1

        # Run to convergence
        for _ in range(200):
            mu_n, precision = _compute_posterior_normal(X, y, alpha, beta)
            alpha_new, beta_new = mackay_update_normal(
                mu_n, precision, X, y, alpha, beta, sparse=False
            )
            if abs(alpha_new - alpha) < 1e-10 and abs(beta_new - beta) < 1e-10:
                break
            alpha, beta = alpha_new, beta_new

        # Compute evidence at convergence
        mu_n, precision = _compute_posterior_normal(X, y, alpha, beta)

        # Finite-difference gradients
        eps = 1e-5

        def ev(a: float, b: float) -> float:
            m, P = _compute_posterior_normal(X, y, a, b)
            return log_evidence_normal(X, y, m, P, a, b)

        d_alpha = (ev(alpha + eps, beta) - ev(alpha - eps, beta)) / (2 * eps)
        d_beta = (ev(alpha, beta + eps) - ev(alpha, beta - eps)) / (2 * eps)

        np.testing.assert_allclose(d_alpha, 0.0, atol=1e-3)
        np.testing.assert_allclose(d_beta, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 9. Log evidence functions vs hand computation
# ---------------------------------------------------------------------------


class TestLogEvidenceFunctions:
    def test_log_evidence_normal(self) -> None:
        """Verify against direct numpy computation."""
        rng = np.random.default_rng(7)
        n, p = 20, 3
        alpha, beta = 1.5, 3.0

        X = rng.standard_normal((n, p))
        w = rng.standard_normal(p)
        y = X @ w + rng.standard_normal(n) * 0.5
        mu_n, precision = _compute_posterior_normal(X, y, alpha, beta)

        result = log_evidence_normal(X, y, mu_n, precision, alpha, beta)

        # Hand computation
        residual = y - X @ mu_n
        ld = float(np.linalg.slogdet(precision)[1])
        expected = (
            0.5 * p * np.log(alpha)
            + 0.5 * n * np.log(beta)
            - 0.5 * ld
            - 0.5 * (beta * float(residual @ residual) + alpha * float(mu_n @ mu_n))
            - 0.5 * n * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_log_evidence_nig(self) -> None:
        """Verify NIG evidence against hand computation."""
        rng = np.random.default_rng(8)
        n, p = 20, 3
        alpha_lam = 2.0
        a0, b0 = 1.0, 1.0

        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)

        # Compute NIG posterior
        precision = alpha_lam * np.eye(p) + X.T @ X
        mu_n = solve(precision, X.T @ y, assume_a="pos")
        an = a0 + 0.5 * n
        bn = b0 + 0.5 * (float(y @ y) - float(mu_n @ precision @ mu_n))

        result = log_evidence_nig(
            X, y, mu_n, precision, alpha_lam, a0, b0, an, bn
        )

        # Hand computation
        ld = float(np.linalg.slogdet(precision)[1])
        expected = (
            gammaln(an) - gammaln(a0)
            + a0 * np.log(b0) - an * np.log(bn)
            + 0.5 * p * np.log(alpha_lam)
            - 0.5 * ld
            - 0.5 * n * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_log_evidence_glm(self, link: str) -> None:
        """Verify GLM Laplace evidence against hand computation."""
        rng = np.random.default_rng(9)
        p = 3
        alpha = 1.0

        X, y, _ = _generate_glm_data(rng, n=30, p=p, link=link, coef_scale=0.3)
        theta, H = _solve_glm_map(X, y, alpha, link)

        result = log_evidence_glm_laplace(X, y, theta, H, alpha, link)

        # Hand computation
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
        expected = log_lik + log_prior + laplace

        np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 10. Sparse/dense parity
# ---------------------------------------------------------------------------


class TestSparseDenseParity:
    @pytest.mark.parametrize("sparse", [True, False])
    def test_logdet_parity(
        self, small_spd_matrix: NDArray[np.float64], sparse: bool
    ) -> None:
        if sparse:
            prec = csc_array(small_spd_matrix)
            result = logdet(prec, sparse=True)
        else:
            result = logdet(small_spd_matrix, sparse=False)
        expected = float(np.linalg.slogdet(small_spd_matrix)[1])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_mackay_update_normal_parity(self, sparse: bool) -> None:
        rng = np.random.default_rng(55)
        n, p = 50, 4
        alpha, beta = 1.0, 2.0
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        mu_n, precision = _compute_posterior_normal(X, y, alpha, beta)

        if sparse:
            sp = csc_array(precision)
            sX = csc_array(X)
            a_new, b_new = mackay_update_normal(
                mu_n, sp, sX, y, alpha, beta,
                sparse=True, n_probes=500, rng=np.random.default_rng(0),
            )
        else:
            a_new, b_new = mackay_update_normal(
                mu_n, precision, X, y, alpha, beta, sparse=False
            )

        # Compare to dense exact
        a_exact, b_exact = mackay_update_normal(
            mu_n, precision, X, y, alpha, beta, sparse=False
        )
        np.testing.assert_allclose(a_new, a_exact, rtol=0.1)
        np.testing.assert_allclose(b_new, b_exact, rtol=0.1)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_log_evidence_normal_parity(self, sparse: bool) -> None:
        rng = np.random.default_rng(66)
        n, p = 30, 3
        alpha, beta = 1.0, 2.0
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        mu_n, precision = _compute_posterior_normal(X, y, alpha, beta)

        if sparse:
            sp = csc_array(precision)
            result = log_evidence_normal(X, y, mu_n, sp, alpha, beta, sparse=True)
        else:
            result = log_evidence_normal(
                X, y, mu_n, precision, alpha, beta, sparse=False
            )

        expected = log_evidence_normal(
            X, y, mu_n, precision, alpha, beta, sparse=False
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)

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
            result = mackay_update_nig(
                mu_n, sp, alpha, an, bn,
                sparse=True, n_probes=500, rng=np.random.default_rng(0),
            )
        else:
            result = mackay_update_nig(mu_n, precision, alpha, an, bn, sparse=False)

        exact = mackay_update_nig(mu_n, precision, alpha, an, bn, sparse=False)
        np.testing.assert_allclose(result, exact, rtol=0.1)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_log_evidence_nig_parity(self, sparse: bool) -> None:
        rng = np.random.default_rng(88)
        n, p = 30, 3
        alpha = 2.0
        a0, b0 = 1.0, 1.0
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)

        if sparse:
            sp = csc_array(precision)
            result = log_evidence_nig(
                X, y, mu_n, sp, alpha, a0, b0, an, bn, sparse=True
            )
        else:
            result = log_evidence_nig(
                X, y, mu_n, precision, alpha, a0, b0, an, bn, sparse=False
            )

        expected = log_evidence_nig(
            X, y, mu_n, precision, alpha, a0, b0, an, bn, sparse=False
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_mackay_update_glm_parity(self, sparse: bool) -> None:
        rng = np.random.default_rng(99)
        n, p = 50, 4
        alpha = 1.0
        X, y, _ = _generate_glm_data(rng, n=n, p=p, link="logit")
        theta, H = _solve_glm_map(X, y, alpha, "logit")

        if sparse:
            sp = csc_array(H)
            result = mackay_update_glm(
                theta, sp, alpha,
                sparse=True, n_probes=500, rng=np.random.default_rng(0),
            )
        else:
            result = mackay_update_glm(theta, H, alpha, sparse=False)

        exact = mackay_update_glm(theta, H, alpha, sparse=False)
        np.testing.assert_allclose(result, exact, rtol=0.1)

    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_log_evidence_glm_parity(self, sparse: bool, link: str) -> None:
        rng = np.random.default_rng(100)
        n, p = 30, 3
        alpha = 1.0
        X, y, _ = _generate_glm_data(rng, n=n, p=p, link=link, coef_scale=0.3)
        theta, H = _solve_glm_map(X, y, alpha, link)

        if sparse:
            sp = csc_array(H)
            result = log_evidence_glm_laplace(
                X, y, theta, sp, alpha, link, sparse=True
            )
        else:
            result = log_evidence_glm_laplace(
                X, y, theta, H, alpha, link, sparse=False
            )

        expected = log_evidence_glm_laplace(
            X, y, theta, H, alpha, link, sparse=False
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 11. MacKay NIG
# ---------------------------------------------------------------------------


class TestMackayNIG:
    def test_basic_update(self) -> None:
        rng = np.random.default_rng(77)
        p = 4
        alpha = 1.0
        an, bn = 5.0, 2.0
        mu_n = rng.standard_normal(p)
        A = rng.standard_normal((p, p))
        precision = A.T @ A + 3.0 * np.eye(p)

        alpha_new = mackay_update_nig(mu_n, precision, alpha, an, bn, sparse=False)
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
        tr_inv = p / (alpha + 1.0)
        np.testing.assert_allclose(
            trace_of_inverse(precision, sparse=False), tr_inv, atol=1e-10
        )

        # Verify alpha_new = p*sigma^2 / (||mu_n||^2 + sigma^2 * tr_inv)
        expected_alpha = p * sigma_sq / (float(mu_n @ mu_n) + sigma_sq * tr_inv)
        alpha_new = mackay_update_nig(mu_n, precision, alpha, an, bn, sparse=False)
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
            ev = log_evidence_nig(X, y, mu_n, precision, alpha, a0, b0, an, bn)
            evidences.append(ev)
            alpha = mackay_update_nig(mu_n, precision, alpha, an, bn, sparse=False)

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-6, (
                f"NIG evidence decreased at step {i}: "
                f"{evidences[i-1]:.6f} -> {evidences[i]:.6f}"
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
            alpha_new = mackay_update_nig(
                mu_n, precision, alpha, an, bn, sparse=False
            )
            if abs(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        eps = 1e-5

        def ev(a: float) -> float:
            m, P, an_, bn_ = _compute_posterior_nig(X, y, a, a0, b0)
            return log_evidence_nig(X, y, m, P, a, a0, b0, an_, bn_)

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
            alpha_new = mackay_update_nig(
                mu_n, precision, alpha, an, bn, sparse=False
            )
            if abs(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        # One more update should return the same value
        mu_n, precision, an, bn = _compute_posterior_nig(X, y, alpha, a0, b0)
        alpha_check = mackay_update_nig(mu_n, precision, alpha, an, bn, sparse=False)
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
            alpha = mackay_update_nig(mu_n, precision, alpha, an, bn, sparse=False)

        # NIG recovery is less precise than Normal due to joint sigma^2 estimation
        np.testing.assert_allclose(alpha, alpha_true, rtol=1.0)


# ---------------------------------------------------------------------------
# 12. MacKay GLM (parameterized over link)
# ---------------------------------------------------------------------------


class TestMackayGLM:
    def test_basic_update(self) -> None:
        rng = np.random.default_rng(88)
        p = 4
        alpha = 1.0
        theta = rng.standard_normal(p) * 0.5
        A = rng.standard_normal((p, p))
        precision = A.T @ A + alpha * np.eye(p)

        alpha_new = mackay_update_glm(theta, precision, alpha, sparse=False)
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
            ev = log_evidence_glm_laplace(X, y, theta, H, alpha, link)
            evidences.append(ev)
            alpha = mackay_update_glm(theta, H, alpha, sparse=False)

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-4, (
                f"GLM ({link}) evidence decreased at step {i}: "
                f"{evidences[i-1]:.6f} -> {evidences[i]:.6f}"
            )

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_gradient_at_convergence(self, link: str) -> None:
        """At GLM MacKay fixed point, d/dalpha log_evidence_glm ~ 0."""
        rng = np.random.default_rng(222)
        X, y, _ = _generate_glm_data(rng, n=200, p=3, link=link)

        alpha = 1.0

        for _ in range(50):
            theta, H = _solve_glm_map(X, y, alpha, link)
            alpha_new = mackay_update_glm(theta, H, alpha, sparse=False)
            if abs(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        eps = 1e-5

        def ev(a: float) -> float:
            t, h = _solve_glm_map(X, y, a, link)
            return log_evidence_glm_laplace(X, y, t, h, a, link)

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
            alpha_new = mackay_update_glm(theta, H, alpha, sparse=False)
            if abs(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        # One more update should return the same value
        theta, H = _solve_glm_map(X, y, alpha, link)
        alpha_check = mackay_update_glm(theta, H, alpha, sparse=False)
        np.testing.assert_allclose(alpha_check, alpha, rtol=1e-6)
