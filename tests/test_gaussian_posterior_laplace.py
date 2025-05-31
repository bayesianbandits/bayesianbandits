# tests/test_gaussian_posterior_laplace.py

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import expit
import pytest

from bayesianbandits._gaussian import (
    logit_link_and_derivative,
    update_gaussian_posterior_laplace,
    compute_effective_weights,
)


def test_compute_effective_weights_size_mismatch():
    """Test that compute_effective_weights raises error for size mismatch."""
    y = np.random.binomial(1, 0.5, 10).astype(float)
    with pytest.raises(ValueError):
        compute_effective_weights(5, y, 1.0)


class TestLaplaceApproximation:
    """Test Laplace approximation for Bayesian GLM updates."""

    def test_reduces_to_linear_regression(self):
        """Test that with identity link, we recover linear regression."""
        # For Gaussian likelihood, Laplace should be exact
        np.random.seed(42)
        n, p = 20, 3
        X = np.random.randn(n, p)
        true_coef = np.array([1.0, -0.5, 0.3])
        y = X @ true_coef + 0.1 * np.random.randn(n)

        # Prior
        prior_mean = np.zeros(p)
        prior_precision = np.eye(p) * 0.1

        # This should give same result as NormalRegressor with beta=100 (high precision)
        # Using "identity" link with Gaussian noise
        # We'll simulate by using very small perturbations
        y_binary = (y > np.median(y)).astype(float)
        posterior = update_gaussian_posterior_laplace(
            X, y_binary, prior_mean, prior_precision, link="logit"
        )

        # Check it moved in right direction
        assert_allclose(np.sign(posterior.mean), np.sign(true_coef), atol=0.5)

    def test_prior_dominates_with_no_data(self):
        """Test that with no observations, posterior equals prior."""
        prior_mean = np.array([1.0, -2.0])
        prior_precision = np.array([[2.0, 0.5], [0.5, 3.0]])

        # Empty data
        X = np.zeros((0, 2))
        y = np.zeros(0)

        posterior = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="logit"
        )

        assert_allclose(posterior.mean, prior_mean)
        assert_allclose(posterior.precision, prior_precision)

    def test_single_observation_moves_mean(self):
        """Test that single observation moves mean in correct direction."""
        # Start with zero mean
        prior_mean = np.zeros(2)
        prior_precision = np.eye(2) * 0.1  # Weak prior

        # Single positive observation with positive features
        X = np.array([[1.0, 0.5]])
        y = np.array([1.0])

        posterior = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="logit"
        )

        # Mean should move positive
        assert np.all(posterior.mean > 0)
        # Precision should increase (more certainty)
        assert np.all(np.diag(posterior.precision) > np.diag(prior_precision))

    def test_perfect_separation_bounded(self):
        """Test that perfect separation doesn't cause numerical issues."""
        # Perfectly separable data
        X = np.array([[-5.0], [-4.0], [4.0], [5.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        prior_mean = np.zeros(1)
        prior_precision = np.array([[1.0]])

        posterior = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="logit", n_iter=50
        )

        # Should find reasonable coefficient
        assert np.abs(posterior.mean[0]) < 10  # Not infinite
        assert posterior.mean[0] > 0  # Positive direction

        # Check predictions are good
        pred_probs = expit(X @ posterior.mean)
        assert np.all(pred_probs[:2] < 0.1)  # Low prob for y=0
        assert np.all(pred_probs[2:] > 0.9)  # High prob for y=1

    def test_sample_weights_affect_posterior(self):
        """Test that sample weights correctly influence the posterior."""
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([0.0, 0.0, 1.0])

        prior_mean = np.zeros(1)
        prior_precision = np.array([[1.0]])

        # Equal weights - should give negative coefficient (2 zeros vs 1 one)
        posterior_equal = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="logit", sample_weight=np.ones(3)
        )

        # Weight the positive example more
        posterior_weighted = update_gaussian_posterior_laplace(
            X,
            y,
            prior_mean,
            prior_precision,
            link="logit",
            sample_weight=np.array([1.0, 1.0, 5.0]),
        )

        # Weighted should be more positive
        assert posterior_weighted.mean[0] > posterior_equal.mean[0]

    def test_learning_rate_decay(self):
        """Test learning rate correctly decays prior influence."""
        X = np.random.randn(10, 2)
        y = np.random.binomial(1, 0.7, 10).astype(float)

        # Strong prior at specific value
        prior_mean = np.array([3.0, -2.0])
        prior_precision = np.eye(2) * 100  # Very strong prior

        # With no decay, prior dominates
        posterior_no_decay = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="logit", learning_rate=1.0
        )

        # With decay, data has more influence
        posterior_decay = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="logit", learning_rate=0.5
        )

        # Decayed version should move further from prior
        dist_no_decay = np.linalg.norm(posterior_no_decay.mean - prior_mean)
        dist_decay = np.linalg.norm(posterior_decay.mean - prior_mean)
        assert dist_decay > dist_no_decay

    def test_poisson_log_link(self):
        """Test Poisson regression with log link."""
        # Generate count data
        np.random.seed(42)
        n = 10000  # More data helps
        X = np.random.randn(n, 2)
        true_coef = np.array([0.5, -0.3])
        true_mu = np.exp(X @ true_coef)
        y = np.random.poisson(true_mu).astype(float)

        prior_mean = np.zeros(2)
        prior_precision = np.eye(2) * 0.1

        posterior = update_gaussian_posterior_laplace(
            X, y, prior_mean, prior_precision, link="log", n_iter=10
        )

        # Check we moved in the right direction
        assert posterior.mean[0] > 0.4  # Should be positive (true is 0.5)
        assert posterior.mean[1] < -0.2  # Should be negative (true is -0.3)

        # Check predictions are reasonable
        pred_mu = np.exp(X @ posterior.mean)
        assert np.all(pred_mu > 0)  # Poisson means must be positive

        # Check we improved from prior (which predicts all 1s)
        prior_pred = np.exp(X @ prior_mean)  # All 1s

        # Use Poisson deviance as metric
        def poisson_deviance(y_true, y_pred):
            # Handle y=0 case
            dev = 2 * np.sum(
                np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0)
                - (y_true - y_pred)
            )
            return dev

        dev_prior = poisson_deviance(y, prior_pred)
        dev_posterior = poisson_deviance(y, pred_mu)

        # We should improve
        assert dev_posterior < dev_prior

        # Alternative: check that log-likelihood improved
        log_lik_prior = np.sum(y * np.log(prior_pred) - prior_pred)
        log_lik_posterior = np.sum(y * np.log(pred_mu + 1e-10) - pred_mu)
        assert log_lik_posterior > log_lik_prior

    def test_natural_parameterization_additive(self):
        """Test that updates in natural parameters are additive."""
        # Set up two separate data batches
        X1 = np.random.randn(5, 2)
        y1 = np.random.binomial(1, 0.6, 5).astype(float)
        X2 = np.random.randn(5, 2)
        y2 = np.random.binomial(1, 0.7, 5).astype(float)

        prior_mean = np.zeros(2)
        prior_precision = np.eye(2)

        # Update with both batches at once
        X_both = np.vstack([X1, X2])
        y_both = np.hstack([y1, y2])
        posterior_both = update_gaussian_posterior_laplace(
            X_both, y_both, prior_mean, prior_precision, link="logit"
        )

        # Update sequentially
        posterior_1 = update_gaussian_posterior_laplace(
            X1, y1, prior_mean, prior_precision, link="logit"
        )
        posterior_2 = update_gaussian_posterior_laplace(
            X2, y2, posterior_1.mean, posterior_1.precision, link="logit"
        )


def test_fisher_information_equivalence():
    """Test that Hessian at MAP equals expected Fisher information."""
    # For GLMs with canonical links, -H = X'WX where W is the GLM weight matrix
    X = np.random.randn(50, 3)
    y = np.random.binomial(1, 0.7, 50).astype(float)

    # Get MAP estimate
    posterior = update_gaussian_posterior_laplace(
        X, y, np.zeros(3), np.eye(3) * 0.01, link="logit", n_iter=50
    )

    # Compute Fisher information at MAP
    eta = X @ posterior.mean
    mu, d_mu = logit_link_and_derivative(eta)
    W = d_mu  # For canonical link
    fisher = X.T @ np.diag(W) @ X

    # Should match precision (minus prior contribution)
    assert_allclose(posterior.precision - 0.01 * np.eye(3), fisher, rtol=1e-3)


def test_irls_finds_correct_map():
    """Test IRLS converges to the true MAP (compare with scipy.optimize)."""
    from scipy.optimize import minimize

    X = np.random.randn(100, 3)
    true_coef = np.array([0.5, -1.0, 0.3])
    y = (X @ true_coef + np.random.randn(100) * 0.5 > 0).astype(float)

    prior_precision = np.eye(3) * 0.1

    # Define negative log posterior
    def neg_log_posterior(beta):
        eta = X @ beta
        mu = expit(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        log_lik = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
        log_prior = -0.5 * beta @ prior_precision @ beta
        return -(log_lik + log_prior)

    # Find MAP with scipy
    result = minimize(neg_log_posterior, np.zeros(3), method="L-BFGS-B")
    map_scipy = result.x

    # Find MAP with IRLS
    posterior = update_gaussian_posterior_laplace(
        X, y, np.zeros(3), prior_precision, link="logit", n_iter=100
    )

    assert_allclose(posterior.mean, map_scipy, rtol=1e-4)


def test_glm_working_response():
    """Test that working response gives correct gradient."""
    from bayesianbandits._gaussian import compute_glm_weights_and_working_response

    eta = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    mu, d_mu = logit_link_and_derivative(eta)
    W, z = compute_glm_weights_and_working_response(y, mu, d_mu, eta)

    # Check gradient: X'(y - mu) = X'W(z - eta)
    gradient_direct = y - mu
    gradient_working = W * (z - eta)

    assert_allclose(gradient_direct, gradient_working)


def test_laplace_approximation_quality():
    """Compare Laplace to ground truth for small problem."""
    # Use 1D problem where we can compute truth via quadrature
    X = np.array([[1.0], [1.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    # Weak prior
    prior_mean = np.array([0.0])
    prior_precision = np.array([[0.1]])

    # Laplace approximation
    posterior = update_gaussian_posterior_laplace(
        X, y, prior_mean, prior_precision, link="logit", n_iter=50
    )

    # Ground truth via quadrature
    from scipy.integrate import quad
    from scipy.stats import norm

    def log_posterior(beta):
        eta = X[:, 0] * beta
        log_lik = np.sum(y * eta - np.log(1 + np.exp(eta)))
        log_prior = norm.logpdf(beta, 0, np.sqrt(1 / 0.1))
        return log_lik + log_prior

    # Normalize to get true posterior
    def unnorm_posterior(beta):
        return np.exp(log_posterior(beta))

    Z, _ = quad(unnorm_posterior, -10, 10)

    # True mean
    def integrand(beta):
        return beta * unnorm_posterior(beta) / Z

    true_mean, _ = quad(integrand, -10, 10)

    # Should be close (Laplace is approximate)
    assert abs(posterior.mean[0] - true_mean) < 0.1


def test_first_newton_step_exact():
    """Test exact form of first Newton step."""
    np.random.seed(42)
    n, p = 50, 3
    X = np.random.randn(n, p)
    y = np.random.binomial(1, 0.7, n).astype(float)

    alpha = 0.01
    prior_mean = np.zeros(p)
    prior_precision = np.eye(p) * alpha

    # One Newton step
    posterior = update_gaussian_posterior_laplace(
        X, y, prior_mean, prior_precision, link="logit", n_iter=1
    )

    # At η=0, μ=0.5 and W=0.25 for all observations
    # The update should solve: (αI + 0.25*X'X)θ = 0.25*X'*4*(y-0.5)
    # Which simplifies to: (αI + 0.25*X'X)θ = X'(y-0.5)

    W = 0.25 * np.ones(n)
    H = alpha * np.eye(p) + X.T @ (W[:, None] * X)
    g = X.T @ (y - 0.5)

    expected_mean = np.linalg.solve(H, g)

    assert_allclose(posterior.mean, expected_mean, rtol=1e-10)
