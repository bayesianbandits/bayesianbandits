# tests/test_rvga_approximator.py
"""Tests for RVGAApproximator — mathematical correctness and software engineering."""

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.special import expit

from bayesianbandits import BayesianGLM
from bayesianbandits._gaussian import (
    GaussianPosterior,
    LaplaceApproximator,
    RVGAApproximator,
    gh_expected_weights,
    log_expected_weights,
    probit_expected_weights,
    update_gaussian_posterior_laplace,
    update_gaussian_posterior_rvga,
)

# ===================================================================
# T-M1: GH quadrature matches analytical E[exp(eta)] = exp(m + v/2)
# ===================================================================


class TestGHQuadratureAccuracy:
    """Gauss-Hermite quadrature accuracy tests."""

    @pytest.mark.parametrize(
        "m, v",
        [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.5),
            (-2.0, 3.0),
            (0.0, 10.0),
        ],
    )
    def test_gh_log_link_matches_analytical(self, m, v):
        """For log link, E[exp(eta)] = exp(m + v/2) exactly."""
        m_arr = np.array([m])
        v_arr = np.array([v])
        E_mu, E_W = gh_expected_weights(m_arr, v_arr, "log", n_gh_nodes=20)

        # For log link, h(eta) = exp(eta) and h'(eta) = exp(eta),
        # so E_mu == E_W == exp(m + v/2).
        expected = np.exp(m + v / 2.0)
        assert_allclose(E_mu[0], expected, rtol=1e-10)
        assert_allclose(E_W[0], expected, rtol=1e-10)

    @pytest.mark.parametrize(
        "m, v",
        [
            (0.0, 1.0),
            (1.0, 0.5),
            (-1.5, 2.0),
            (3.0, 0.1),
        ],
    )
    def test_gh_logit_matches_scipy_quad(self, m, v):
        """GH with 20 nodes matches scipy.integrate.quad for logit link."""
        from scipy.integrate import quad
        from scipy.stats import norm

        m_arr = np.array([m])
        v_arr = np.array([v])
        E_mu_gh, E_W_gh = gh_expected_weights(m_arr, v_arr, "logit", n_gh_nodes=20)

        # Numerical integration: E[sigma(eta)] where eta ~ N(m, v)
        std = np.sqrt(v)
        E_mu_quad, _ = quad(
            lambda x: expit(x) * norm.pdf(x, m, std), m - 8 * std, m + 8 * std
        )
        # E[sigma'(eta)] = E[sigma(eta)(1 - sigma(eta))]
        E_W_quad, _ = quad(
            lambda x: expit(x) * (1 - expit(x)) * norm.pdf(x, m, std),
            m - 8 * std,
            m + 8 * std,
        )

        assert_allclose(E_mu_gh[0], E_mu_quad, rtol=1e-6)
        assert_allclose(E_W_gh[0], E_W_quad, rtol=1e-6)

    def test_gh_logit_symmetry(self):
        """E[sigma(eta)] = 0.5 when eta ~ N(0, v) by symmetry."""
        v_arr = np.array([0.5, 1.0, 5.0])
        E_mu, _ = gh_expected_weights(np.zeros(3), v_arr, "logit", n_gh_nodes=20)
        assert_allclose(E_mu, 0.5, atol=1e-12)

    def test_gh_degenerate_variance_zero(self):
        """When v=0, E[f(eta)] = f(m) (no averaging)."""
        m_arr = np.array([1.5])
        v_arr = np.array([0.0])

        E_mu, E_W = gh_expected_weights(m_arr, v_arr, "logit", n_gh_nodes=20)
        assert_allclose(E_mu[0], expit(1.5), rtol=1e-10)
        assert_allclose(E_W[0], expit(1.5) * (1 - expit(1.5)), rtol=1e-10)

        E_mu, E_W = gh_expected_weights(m_arr, v_arr, "log", n_gh_nodes=20)
        assert_allclose(E_mu[0], np.exp(1.5), rtol=1e-10)
        assert_allclose(E_W[0], np.exp(1.5), rtol=1e-10)


# ===================================================================
# T-M4: Probit approximation accuracy
# ===================================================================


class TestProbitApproximation:
    @pytest.mark.parametrize(
        "m, v",
        [(0.0, 1.0), (1.0, 0.5), (-1.5, 2.0), (3.0, 0.1), (0.0, 5.0)],
    )
    def test_probit_matches_gh_approximately(self, m, v):
        """Probit approximation within 5% of GH for logit link."""
        m_arr = np.array([m])
        v_arr = np.array([v])
        E_mu_p, E_W_p = probit_expected_weights(m_arr, v_arr)
        E_mu_gh, E_W_gh = gh_expected_weights(m_arr, v_arr, "logit", n_gh_nodes=50)
        assert_allclose(E_mu_p[0], E_mu_gh[0], rtol=0.05)
        assert_allclose(E_W_p[0], E_W_gh[0], rtol=0.05)

    def test_probit_degenerate_v_zero(self):
        """When v=0, probit k=1 and reduces to point estimate."""
        m_arr = np.array([1.5])
        v_arr = np.array([0.0])
        E_mu, E_W = probit_expected_weights(m_arr, v_arr)
        assert_allclose(E_mu[0], expit(1.5), rtol=1e-14)
        assert_allclose(E_W[0], expit(1.5) * (1 - expit(1.5)), rtol=1e-14)


class TestLogExpectedWeights:
    """Closed-form lognormal mean for the log link."""

    @pytest.mark.parametrize(
        "m, v",
        [(0.0, 1.0), (1.0, 0.5), (-2.0, 3.0), (0.5, 10.0), (-1.0, 0.0)],
    )
    def test_matches_lognormal_mean(self, m, v):
        """E[exp(eta)] = exp(m + v/2), and E_mu == E_W for log link."""
        E_mu, E_W = log_expected_weights(np.array([m]), np.array([v]))
        assert_allclose(E_mu[0], np.exp(m + v / 2), rtol=1e-14)
        assert_allclose(E_W[0], np.exp(m + v / 2), rtol=1e-14)

    def test_outputs_do_not_alias(self):
        """Callers clamp E_W in place; E_mu must not change with it."""
        E_mu, E_W = log_expected_weights(np.array([-800.0]), np.array([1.0]))
        np.maximum(E_W, 1e-10, out=E_W)
        assert E_mu[0] < 1e-10

    def test_extreme_inputs_finite(self):
        """Exponent is clipped, so no overflow to inf."""
        E_mu, E_W = log_expected_weights(
            np.array([500.0, -2000.0]), np.array([1000.0, 0.0])
        )
        assert np.all(np.isfinite(E_mu))
        assert np.all(np.isfinite(E_W))
        assert np.all(E_mu > 0)


# ===================================================================
# T-M6: Expected weight bounds
# ===================================================================


class TestExpectedWeightProperties:
    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 5.0])
    def test_logit_expected_weight_bounded_by_quarter(self, v):
        """E_q[sigma'(eta)] <= 0.25 for any m (global max of sigma')."""
        m_arr = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        v_arr = np.full_like(m_arr, v)
        _, E_W = gh_expected_weights(m_arr, v_arr, "logit", n_gh_nodes=30)
        assert np.all(E_W <= 0.25 + 1e-12)

    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 5.0])
    def test_log_expected_weight_ge_point_weight(self, v):
        """E_q[exp(eta)] >= exp(m) for log link (convexity of exp)."""
        m_arr = np.array([-1.0, 0.0, 1.0])
        v_arr = np.full_like(m_arr, v)
        _, E_W = gh_expected_weights(m_arr, v_arr, "log", n_gh_nodes=30)
        assert np.all(E_W >= np.exp(m_arr) - 1e-12)

    def test_logit_expected_weight_monotone_decreasing_in_v_at_zero(self):
        """At m=0, E[sigma'(eta)] strictly decreases as v increases.

        sigma'(eta) is unimodal symmetric about eta=0 with max 0.25.
        Increasing v spreads mass into tails where sigma' is smaller.

        NB: This monotonicity does NOT hold for general m. At m=3,
        sigma' is locally convex (f''(3)>0), so increasing v from 0
        initially increases E[sigma'(eta)] via the heat equation.
        """
        variances = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        _, E_Ws = gh_expected_weights(
            np.zeros_like(variances), variances, "logit", n_gh_nodes=30
        )
        assert np.all(np.diff(E_Ws) < 0), f"E_W not monotonically decreasing: {E_Ws}"

    def test_expected_weights_equal_point_when_v_zero(self):
        """When v=0, expected weights = point-estimate weights."""
        m_arr = np.array([-2.0, 0.0, 1.5])
        v_arr = np.zeros(3)
        _, E_W_logit = gh_expected_weights(m_arr, v_arr, "logit", n_gh_nodes=20)
        _, E_W_log = gh_expected_weights(m_arr, v_arr, "log", n_gh_nodes=20)
        assert_allclose(E_W_logit, expit(m_arr) * (1 - expit(m_arr)), rtol=1e-10)
        assert_allclose(E_W_log, np.exp(m_arr), rtol=1e-10)


# ===================================================================
# T-M7: Posterior precision properties
# ===================================================================


class TestPosteriorPrecisionProperties:
    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_precision_is_spd(self, link):
        """Posterior precision must be symmetric positive definite.

        dsyrk only fills the upper triangle, so reconstruct the full
        symmetric matrix before checking.
        """
        np.random.seed(42)
        X = np.random.randn(30, 4)
        if link == "logit":
            y = np.random.binomial(1, 0.5, 30).astype(float)
        else:
            y = np.random.poisson(3.0, 30).astype(float)
        posterior = update_gaussian_posterior_rvga(
            X,
            y,
            np.zeros(4),
            np.eye(4) * 0.5,
            link=link,
            n_iter=10,
            use_probit=False,
        )
        P = np.asarray(posterior.precision)
        # Reconstruct full symmetric matrix from upper triangle
        P_full = np.triu(P) + np.triu(P, 1).T
        assert_allclose(P_full, P_full.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(P_full)
        assert np.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals.min()}"


# ===================================================================
# T-M9 & T-M10: Posterior quality -- R-VGA vs Laplace
# ===================================================================


class TestRVGAvsLaplace:
    def test_rvga_wider_posterior_logit(self):
        """R-VGA precision <= Laplace precision (wider posterior) for logit.

        Posterior variance spreads mass into tails where sigma'(eta) < 0.25,
        so expected weights E[sigma'(eta)] < sigma'(MAP), giving less precision.
        """
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        prior_mean = np.zeros(1)
        prior_precision = np.eye(1)

        lap = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="logit", n_iter=20
        )
        rvga = update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean,
            prior_precision,
            link="logit",
            n_iter=20,
            use_probit=False,
            n_gh_nodes=30,
        )

        # R-VGA precision should be <= Laplace precision
        assert rvga.precision[0, 0] <= lap.precision[0, 0]
        # Both should have positive coefficients
        assert rvga.mean[0] > 0
        assert lap.mean[0] > 0

    def test_rvga_kl_le_laplace_1d_logit(self):
        """In 1D logit, R-VGA closer to true posterior than Laplace."""
        from scipy.integrate import quad
        from scipy.stats import norm

        X = np.array([[1.0], [1.0], [1.0], [1.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        prior_mean = np.array([0.0])
        prior_precision = np.array([[0.1]])

        lap = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="logit", n_iter=50
        )
        rvga = update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean,
            prior_precision,
            link="logit",
            n_iter=50,
            use_probit=False,
            n_gh_nodes=30,
        )

        # True unnormalized log posterior
        def log_unnorm(beta):
            eta = X[:, 0] * beta
            ll = np.sum(y * eta - np.log(1 + np.exp(eta)))
            lp = norm.logpdf(beta, 0, np.sqrt(1 / 0.1))
            return ll + lp

        def unnorm(beta):
            return np.exp(log_unnorm(beta))

        Z, _ = quad(unnorm, -15, 15)

        def kl_gaussian_to_true(mu, sigma_sq):
            """KL(q || p_true) where q = N(mu, sigma_sq)."""
            sigma = np.sqrt(sigma_sq)

            def integrand(beta):
                q = norm.pdf(beta, mu, sigma)
                if q < 1e-300:
                    return 0.0
                log_q = norm.logpdf(beta, mu, sigma)
                log_p = log_unnorm(beta) - np.log(Z)
                return q * (log_q - log_p)

            kl, _ = quad(integrand, mu - 8 * sigma, mu + 8 * sigma)
            return kl

        sigma_sq_lap = 1.0 / lap.precision[0, 0]
        sigma_sq_rvga = 1.0 / rvga.precision[0, 0]

        kl_lap = kl_gaussian_to_true(lap.mean[0], sigma_sq_lap)
        kl_rvga = kl_gaussian_to_true(rvga.mean[0], sigma_sq_rvga)

        assert kl_rvga <= kl_lap + 1e-6  # R-VGA at least as good


# ===================================================================
# T-M12 & T-M13: Single-step analytical verification
# ===================================================================


class TestSingleStepAnalytical:
    def test_single_rvga_step_logit_probit(self):
        """One R-VGA step from w=0, logit, probit mode. Hand-computed.

        X = [[-2],[-1],[1],[2]], y = [0,0,1,1], alpha=1
        At mu=0, Lambda=I:
          m_i = 0 for all i
          v_i = x_i^2 / 1 = 4, 1, 1, 4
          beta = sqrt(8/pi)
          k_i = beta / sqrt(v_i + beta^2)
          E_W[i] = k_i * sigma'(k_i * 0) = k_i * 0.25
        """
        beta = np.sqrt(8.0 / np.pi)
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        posterior = update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean=np.zeros(1),
            prior_precision=np.eye(1),
            link="logit",
            n_iter=1,
            use_probit=True,
        )

        # Hand-compute k values
        v = np.array([4.0, 1.0, 1.0, 4.0])
        k = beta / np.sqrt(v + beta**2)
        E_W = k * 0.25  # sigma'(k*0) = sigma'(0) = 0.25

        # Expected precision: alpha*I + sum_i E_W[i] * x_i^2
        expected_prec = 1.0 + np.sum(E_W * X[:, 0] ** 2)
        assert_allclose(posterior.precision[0, 0], expected_prec, rtol=1e-10)

        # E_mu[i] = sigma(k_i * 0) = 0.5 for all i
        # z_i = m_i + (y_i - 0.5) / E_W[i]
        # gradient = sum_i E_W[i] * z_i * x_i + prior_eta(=0)
        E_mu = np.full(4, 0.5)
        z = 0.0 + (y - E_mu) / E_W
        expected_eta = np.sum(E_W * z * X[:, 0])
        expected_coef = expected_eta / expected_prec
        assert_allclose(posterior.mean[0], expected_coef, rtol=1e-10)

    def test_single_rvga_step_2d_log_link(self):
        """One R-VGA step in 2D, log link. Verifies matrix assembly.

        X = [[1,0],[0,1],[1,1]], y = [3,7,5], alpha=I
        At mu=[0,0], Lambda=I:
          m = [0,0,0], v = [||x_i||^2] = [1, 1, 2]
          E_W = exp(v/2) = [exp(0.5), exp(0.5), exp(1)]
          X^T diag(W) X = [[e05+e1, e1], [e1, e05+e1]]  (off-diagonal!)
        """
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([3.0, 7.0, 5.0])

        posterior = update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean=np.zeros(2),
            prior_precision=np.eye(2),
            link="log",
            n_iter=1,
            use_probit=False,
            n_gh_nodes=30,
        )

        v = np.array([1.0, 1.0, 2.0])
        E_W = np.exp(v / 2)  # [exp(0.5), exp(0.5), exp(1)]
        E_mu = E_W.copy()

        # X^T diag(W) X:
        # row 0: x=[1,0], W=exp(0.5) -> contributes [[exp(0.5),0],[0,0]]
        # row 1: x=[0,1], W=exp(0.5) -> contributes [[0,0],[0,exp(0.5)]]
        # row 2: x=[1,1], W=exp(1)   -> contributes [[exp(1),exp(1)],[exp(1),exp(1)]]
        e05 = np.exp(0.5)
        e1 = np.exp(1.0)
        expected_prec = np.eye(2) + np.array([[e05 + e1, e1], [e1, e05 + e1]])
        # dsyrk only fills upper triangle; reconstruct full matrix
        P = np.asarray(posterior.precision)
        P_full = np.triu(P) + np.triu(P, 1).T
        assert_allclose(P_full, expected_prec, rtol=1e-6)

        # Working response and natural parameter
        z = (y - E_mu) / E_W  # m_i = 0, so z = 0 + (y - E_mu)/E_W
        Wz = E_W * z  # = y - E_mu
        expected_eta = X.T @ Wz  # + prior_eta = 0
        expected_coef = np.linalg.solve(expected_prec, expected_eta)  # 2D solve!
        assert_allclose(posterior.mean, expected_coef, rtol=1e-6)

    def test_single_rvga_step_log_link(self):
        """One R-VGA step from w=0, log link. Hand-computed.

        X = [[1],[2]], y = [3,7], alpha=1
        At mu=0, Lambda=I:
          m_i = 0, v_i = x_i^2 = 1, 4
          E[exp(eta_i)] = exp(0 + v_i/2) = exp(0.5), exp(2.0)
          Both E_mu and E_W equal exp(v_i/2) for log link
        """
        X = np.array([[1.0], [2.0]])
        y = np.array([3.0, 7.0])

        posterior = update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean=np.zeros(1),
            prior_precision=np.eye(1),
            link="log",
            n_iter=1,
            use_probit=False,
            n_gh_nodes=30,
        )

        v = np.array([1.0, 4.0])
        E_W = np.exp(v / 2)  # exp(0.5), exp(2.0)
        E_mu = E_W.copy()  # same for log link

        expected_prec = 1.0 + np.sum(E_W * X[:, 0] ** 2)
        assert_allclose(posterior.precision[0, 0], expected_prec, rtol=1e-6)

        z = 0.0 + (y - E_mu) / E_W
        expected_eta = np.sum(E_W * z * X[:, 0])
        expected_coef = expected_eta / expected_prec
        assert_allclose(posterior.mean[0], expected_coef, rtol=1e-6)


# ===================================================================
# T-M16 & T-M17: Convergence
# ===================================================================


class TestRVGAConvergence:
    def test_tol_causes_early_stopping(self):
        """Loose tol stops early, giving close but not identical results."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X @ [0.5, -1.0, 0.3] + np.random.randn(50) * 0.5 > 0).astype(float)
        prior_precision = np.eye(3) * 0.1

        r_tight = update_gaussian_posterior_rvga(
            X,
            y,
            np.zeros(3),
            prior_precision,
            link="logit",
            n_iter=100,
            tol=1e-12,
            use_probit=False,
        )
        r_loose = update_gaussian_posterior_rvga(
            X,
            y,
            np.zeros(3),
            prior_precision,
            link="logit",
            n_iter=100,
            tol=1e-2,
            use_probit=False,
        )
        # Close to same fixed point
        assert_allclose(r_loose.mean, r_tight.mean, rtol=0.05)
        # But not bitwise identical (early stopping fired for loose tol)
        assert not np.array_equal(r_loose.mean, r_tight.mean)

    def test_more_iterations_improve_log_likelihood(self):
        """More R-VGA iterations give higher log-likelihood."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_coef = np.array([0.5, -1.0, 0.3])
        y = (X @ true_coef + np.random.randn(100) * 0.5 > 0).astype(float)

        prior_precision = np.eye(3) * 0.1

        r1 = update_gaussian_posterior_rvga(
            X,
            y,
            np.zeros(3),
            prior_precision,
            link="logit",
            n_iter=1,
            use_probit=False,
        )
        r10 = update_gaussian_posterior_rvga(
            X,
            y,
            np.zeros(3),
            prior_precision,
            link="logit",
            n_iter=10,
            use_probit=False,
        )

        def log_lik(coef):
            eta = X @ coef
            mu = expit(eta)
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            return np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

        assert log_lik(r10.mean) > log_lik(r1.mean)

    def test_rvga_mean_near_map(self):
        """R-VGA mean is near the MAP (not identical -- different objective).

        With more data, posteriors concentrate and R-VGA mean approaches
        the MAP because expected curvature -> point curvature as v -> 0.
        """
        from scipy.optimize import minimize

        np.random.seed(42)
        X = np.random.randn(200, 3)
        true_coef = np.array([0.5, -1.0, 0.3])
        y = (X @ true_coef + np.random.randn(200) * 0.5 > 0).astype(float)
        prior_precision = np.eye(3) * 0.1

        def neg_log_posterior(beta):
            eta = X @ beta
            mu = expit(eta)
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            ll = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
            lp = -0.5 * beta @ prior_precision @ beta
            return -(ll + lp)

        result = minimize(neg_log_posterior, np.zeros(3), method="L-BFGS-B")

        rvga = update_gaussian_posterior_rvga(
            X,
            y,
            np.zeros(3),
            prior_precision,
            link="logit",
            n_iter=100,
            use_probit=False,
        )

        # R-VGA mean is close but not identical to MAP (different objective)
        assert_allclose(rvga.mean, result.x, rtol=0.05)


# ===================================================================
# Software engineering tests
# ===================================================================


class TestRVGAIntegration:
    """BayesianGLM integration tests with RVGAApproximator."""

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_fit_predict_sample(self, link):
        """End-to-end fit/predict/sample works for both links."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        if link == "logit":
            y = np.random.binomial(1, 0.5, 50).astype(float)
        else:
            y = np.random.poisson(3.0, 50).astype(float)

        glm = BayesianGLM(
            link=link,
            approximator=RVGAApproximator(),
            random_state=0,
        )
        glm.fit(X, y)
        pred = glm.predict(X[:3])
        samples = glm.sample(X[:3], size=10)

        assert pred.shape == (3,)
        assert samples.shape == (10, 3)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(samples))

    def test_partial_fit(self):
        """partial_fit works incrementally."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = np.random.binomial(1, 0.6, 20).astype(float)

        glm = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(),
            random_state=0,
        )
        glm.fit(X[:10], y[:10])
        coef_after_fit = glm.coef_.copy()
        glm.partial_fit(X[10:], y[10:])
        # Coefficients should change after partial_fit
        assert not np.allclose(glm.coef_, coef_after_fit)

    def test_default_approximator_unchanged(self):
        """BayesianGLM() still defaults to LaplaceApproximator."""
        glm = BayesianGLM()
        glm.fit(np.array([[1.0]]), np.array([1.0]))
        assert isinstance(glm.approximator_, LaplaceApproximator)

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_sparse_dense_equivalence(self, link):
        """Sparse and dense R-VGA produce same results."""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        if link == "logit":
            y = np.random.binomial(1, 0.5, 30).astype(float)
        else:
            y = np.random.poisson(2.0, 30).astype(float)

        glm_dense = BayesianGLM(
            link=link,
            approximator=RVGAApproximator(use_probit=False),
            sparse=False,
            random_state=0,
        )
        glm_sparse = BayesianGLM(
            link=link,
            approximator=RVGAApproximator(use_probit=False),
            sparse=True,
            random_state=0,
        )
        glm_dense.fit(X, y)
        glm_sparse.fit(sp.csc_array(X), y)

        assert_allclose(glm_dense.coef_, glm_sparse.coef_, rtol=1e-5)

    def test_sparse_partial_fit_threads_prior_factor(self):
        """Sparse partial_fit with non-diagonal prior reuses the cached factor.

        After the first fit the precision picks up off-diagonal entries
        from X^T W X, so the second call exercises the non-diagonal
        Gram-matrix path.  Result must match a fresh-from-scratch run.
        """
        np.random.seed(0)
        X1 = sp.csc_array(np.random.randn(20, 4))
        X2 = sp.csc_array(np.random.randn(20, 4))
        y1 = np.random.binomial(1, 0.5, 20).astype(float)
        y2 = np.random.binomial(1, 0.5, 20).astype(float)

        glm_seq = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(use_probit=False, n_gh_nodes=30),
            sparse=True,
            random_state=0,
        )
        glm_seq.fit(X1, y1)
        # At this point the cached _precision_factor is non-diagonal;
        # partial_fit should thread it through.
        assert "_precision_factor" in glm_seq.__dict__
        glm_seq.partial_fit(X2, y2)

        # Reference: same inputs, no cached factor (force fresh factor).
        glm_ref = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(use_probit=False, n_gh_nodes=30),
            sparse=True,
            random_state=0,
        )
        glm_ref.fit(X1, y1)
        glm_ref.__dict__.pop("_precision_factor", None)  # force recreation
        glm_ref.partial_fit(X2, y2)

        assert_allclose(glm_seq.coef_, glm_ref.coef_, rtol=1e-10)

    def test_empty_data_returns_prior(self):
        """Empty data returns prior unchanged."""
        prior_mean = np.array([1.0, -2.0])
        prior_precision = np.array([[2.0, 0.5], [0.5, 3.0]])
        X = np.zeros((0, 2))
        y = np.zeros(0)

        posterior = update_gaussian_posterior_rvga(
            X, y, prior_mean, prior_precision, link="logit"
        )
        assert_allclose(posterior.mean, prior_mean)
        assert_allclose(posterior.precision, prior_precision)

    def test_single_observation_moves_mean(self):
        """Single positive observation with positive features -> positive coef."""
        posterior = update_gaussian_posterior_rvga(
            np.array([[1.0, 0.5]]),
            np.array([1.0]),
            np.zeros(2),
            np.eye(2) * 0.1,
            link="logit",
        )
        assert np.all(posterior.mean > 0)

    def test_logit_predictions_bounded(self):
        """Logit predictions in [0, 1]."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.random.binomial(1, 0.5, 30).astype(float)
        glm = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(),
            random_state=0,
        )
        glm.fit(X, y)
        samples = glm.sample(X, size=10_000)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_log_predictions_positive(self):
        """Log-link predictions > 0."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.random.poisson(3.0, 30).astype(float)
        glm = BayesianGLM(
            link="log",
            approximator=RVGAApproximator(use_probit=False),
            random_state=0,
        )
        glm.fit(X, y)
        samples = glm.sample(X[:5], size=10_000)
        assert np.all(samples > 0.0)

    def test_sample_weights(self):
        """Upweighting positive class increases posterior mean."""
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([0.0, 0.0, 1.0])
        prior_mean = np.zeros(1)
        prior_precision = np.array([[1.0]])

        r_equal = update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean,
            prior_precision,
            link="logit",
            sample_weight=np.ones(3),
        )
        r_weighted = update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean,
            prior_precision,
            link="logit",
            sample_weight=np.array([1.0, 1.0, 5.0]),
        )
        assert r_weighted.mean[0] > r_equal.mean[0]

    def test_learning_rate_decay(self):
        """Decay increases posterior variance (decreases precision)."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = np.random.binomial(1, 0.6, 20).astype(float)

        glm = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(),
            learning_rate=0.9,
            random_state=0,
        )
        glm.fit(X, y)
        prec_before = glm.cov_inv_.copy()
        glm.decay(X[:5])
        prec_after = glm.cov_inv_.copy()
        # Precision should decrease (uncertainty increases)
        assert np.all(np.diag(prec_after) < np.diag(prec_before))

    def test_probit_vs_gh_similar_logit(self):
        """use_probit=True and False give similar results for logit."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.binomial(1, 0.6, 50).astype(float)

        glm_probit = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(use_probit=True),
            random_state=0,
        )
        glm_gh = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(use_probit=False, n_gh_nodes=30),
            random_state=0,
        )
        glm_probit.fit(X, y)
        glm_gh.fit(X, y)
        assert_allclose(glm_probit.coef_, glm_gh.coef_, rtol=0.05)

    def test_probit_ignored_for_log_link(self):
        """use_probit=True with log link doesn't error (uses closed form)."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = np.random.poisson(3.0, 20).astype(float)
        glm = BayesianGLM(
            link="log",
            approximator=RVGAApproximator(use_probit=True),
            random_state=0,
        )
        glm.fit(X, y)  # Should not raise
        assert np.all(np.isfinite(glm.coef_))

    def test_log_link_independent_of_gh_nodes(self):
        """Log link uses the closed form, so n_gh_nodes is irrelevant.

        With quadrature, 2 nodes would integrate exp badly; identical
        results prove the analytic path is taken.
        """
        np.random.seed(7)
        X = np.random.randn(30, 3)
        y = np.random.poisson(2.0, 30).astype(float)

        coefs = []
        for n_nodes in (2, 50):
            glm = BayesianGLM(
                link="log",
                approximator=RVGAApproximator(use_probit=False, n_gh_nodes=n_nodes),
                random_state=0,
            )
            glm.fit(X, y)
            coefs.append(glm.coef_.copy())
        assert_allclose(coefs[0], coefs[1], rtol=1e-14)

    def test_perfect_separation_bounded(self):
        """Perfect separation doesn't produce infinite coefficients."""
        X = np.array([[-5.0], [-4.0], [4.0], [5.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        posterior = update_gaussian_posterior_rvga(
            X,
            y,
            np.zeros(1),
            np.array([[1.0]]),
            link="logit",
            n_iter=50,
        )
        assert np.abs(posterior.mean[0]) < 10
        assert posterior.mean[0] > 0

    def test_deterministic(self):
        """R-VGA produces identical results on repeated calls."""
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        args: dict[str, Any] = dict(
            prior_mean=np.zeros(1),
            prior_precision=np.eye(1),
            link="logit",
            n_iter=5,
            use_probit=False,
        )
        r1 = update_gaussian_posterior_rvga(X, y, **args)
        r2 = update_gaussian_posterior_rvga(X, y, **args)
        assert_allclose(r1.mean, r2.mean)
        assert_allclose(r1.precision, r2.precision)

    def test_batch_vs_sequential_close(self):
        """Batch fit and sequential partial_fit give similar posteriors.

        Unlike conjugate updates, R-VGA's variational fixed point depends
        on which data is in each IRLS batch (expected curvature is computed
        over the current batch). So exact equivalence does NOT hold; the
        mean converges well (~2%) but precision differs more (~10%) because
        curvature is more sensitive to the approximation quality of the
        intermediate posterior.
        """
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = np.random.poisson(3.0, 20).astype(float)

        approx = RVGAApproximator(n_iter=100, tol=1e-8, use_probit=False, n_gh_nodes=30)

        glm_batch = BayesianGLM(link="log", approximator=approx, random_state=0)
        glm_batch.fit(X, y)

        glm_seq = BayesianGLM(link="log", approximator=approx, random_state=0)
        glm_seq.fit(X[:10], y[:10])
        glm_seq.partial_fit(X[10:], y[10:])

        assert_allclose(glm_seq.coef_, glm_batch.coef_, rtol=0.05)

    def test_pickle_roundtrip(self):
        """Pickle roundtrip preserves predictions."""
        import pickle

        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.random.binomial(1, 0.5, 30).astype(float)
        glm = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(),
            random_state=0,
        )
        glm.fit(X, y)
        pred_before = glm.predict(X[:5])

        glm2 = pickle.loads(pickle.dumps(glm))
        pred_after = glm2.predict(X[:5])
        assert_allclose(pred_before, pred_after)


class TestSequentialMinibatching:
    """batch_size chunking in the sparse path."""

    @staticmethod
    def _make_sparse_case(n, p, seed=0, nnz_per_row=5):
        rng = np.random.default_rng(seed)
        X = sp.random(n, p, density=nnz_per_row / p, random_state=seed, format="csc")
        y = rng.binomial(1, 0.5, n).astype(np.float64)
        return sp.csc_array(X), y

    def test_chunk_loop_matches_manual_chaining(self):
        """batch_size chunking is exactly equivalent to manually threading
        chunk posteriors, including decay and sample-weight slicing."""
        X, y = self._make_sparse_case(100, 30)
        sw = np.random.default_rng(1).uniform(0.5, 2.0, 100)
        prior_mean = np.zeros(30)
        prior_prec = sp.csc_array(sp.eye(30, format="csc"))
        kwargs: dict[str, Any] = dict(
            link="logit", learning_rate=0.99, sparse=True, n_iter=3, tol=0.0
        )

        auto = update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean,
            prior_prec,
            sample_weight=sw,
            batch_size=30,
            **kwargs,
        )

        manual = GaussianPosterior(prior_mean, prior_prec, None)
        for s in range(0, 100, 30):
            manual = update_gaussian_posterior_rvga(
                sp.csc_array(X.tocsr()[s : s + 30]),
                y[s : s + 30],
                manual.mean,
                manual.precision,
                sample_weight=sw[s : s + 30],
                prior_factor=manual.factor,
                **kwargs,
            )

        assert_allclose(np.asarray(auto.mean), np.asarray(manual.mean), rtol=1e-12)
        assert_allclose(
            np.asarray(auto.precision.todense()),
            np.asarray(manual.precision.todense()),
            rtol=1e-12,
        )

    def test_batch_size_geq_n_is_joint(self):
        """No chunking when the batch fits: identical to batch_size=None."""
        X, y = self._make_sparse_case(50, 20)
        prior_mean = np.zeros(20)
        prior_prec = sp.csc_array(sp.eye(20, format="csc"))
        kwargs: dict[str, Any] = dict(link="logit", sparse=True, n_iter=3, tol=0.0)
        joint = update_gaussian_posterior_rvga(
            X, y, prior_mean, prior_prec, batch_size=None, **kwargs
        )
        capped = update_gaussian_posterior_rvga(
            X, y, prior_mean, prior_prec, batch_size=50, **kwargs
        )
        assert_allclose(np.asarray(joint.mean), np.asarray(capped.mean), rtol=1e-14)

    def test_sequential_close_to_joint(self):
        """Sequential posterior stays close to the joint one in the
        sparse regime (benchmarked drift ~1e-3 at scale)."""
        X, y = self._make_sparse_case(400, 200)
        prior_mean = np.zeros(200)
        prior_prec = sp.csc_array(sp.eye(200, format="csc"))
        kwargs: dict[str, Any] = dict(link="logit", sparse=True, n_iter=5, tol=1e-8)
        joint = update_gaussian_posterior_rvga(
            X, y, prior_mean, prior_prec, batch_size=None, **kwargs
        )
        seq = update_gaussian_posterior_rvga(
            X, y, prior_mean, prior_prec, batch_size=100, **kwargs
        )
        assert np.max(np.abs(np.asarray(seq.mean) - np.asarray(joint.mean))) < 0.05

    def test_dense_ignores_batch_size(self):
        """Dense path never chunks."""
        np.random.seed(3)
        X = np.random.randn(60, 4)
        y = np.random.binomial(1, 0.5, 60).astype(np.float64)
        kwargs: dict[str, Any] = dict(link="logit", sparse=False, n_iter=3, tol=0.0)
        ref = update_gaussian_posterior_rvga(
            X, y, np.zeros(4), np.eye(4), batch_size=None, **kwargs
        )
        chunked = update_gaussian_posterior_rvga(
            X, y, np.zeros(4), np.eye(4), batch_size=10, **kwargs
        )
        assert_allclose(np.asarray(ref.mean), np.asarray(chunked.mean), rtol=1e-14)

    def test_glm_fit_with_batching(self):
        """BayesianGLM sparse fit through a small batch_size works and
        matches manually chained partial_fits with the same chunks."""
        X, y = self._make_sparse_case(90, 25, seed=11)
        batched = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(batch_size=30, tol=0.0),
            sparse=True,
            random_state=0,
        )
        batched.fit(X, y)

        chained = BayesianGLM(
            link="logit",
            approximator=RVGAApproximator(batch_size=30, tol=0.0),
            sparse=True,
            random_state=0,
        )
        Xr = X.tocsr()
        chained.fit(sp.csc_array(Xr[:30]), y[:30])
        chained.partial_fit(sp.csc_array(Xr[30:60]), y[30:60])
        chained.partial_fit(sp.csc_array(Xr[60:]), y[60:])

        assert_allclose(batched.coef_, chained.coef_, rtol=1e-10)
        assert np.all(np.isfinite(batched.predict(X)))
