import numpy as np
from numpy.testing import assert_allclose

from bayesianbandits._gaussian import (
    log_link_and_derivative,
    logit_link_and_derivative,
)


class TestLogitLink:
    """Test logit link function."""

    def test_known_values(self):
        """Test at specific points."""
        # sigmoid(0) = 0.5
        mu, d_mu = logit_link_and_derivative(np.array([0.0]))
        assert_allclose(mu, 0.5)
        assert_allclose(d_mu, 0.25)

        # sigmoid(∞) → 1, sigmoid(-∞) → 0
        mu_pos, d_mu_pos = logit_link_and_derivative(np.array([100.0]))
        mu_neg, d_mu_neg = logit_link_and_derivative(np.array([-100.0]))
        assert_allclose(mu_pos, 1.0, atol=1e-10)
        assert_allclose(mu_neg, 0.0, atol=1e-10)
        assert_allclose(d_mu_pos, 0.0, atol=1e-10)
        assert_allclose(d_mu_neg, 0.0, atol=1e-10)

    def test_symmetry(self):
        """Test sigmoid(-x) = 1 - sigmoid(x)."""
        x = np.array([0.5, 1.0, 2.0, 3.0])
        mu_pos, _ = logit_link_and_derivative(x)
        mu_neg, _ = logit_link_and_derivative(-x)
        assert_allclose(mu_neg, 1 - mu_pos)

    def test_derivative_finite_difference(self):
        """Verify derivative numerically."""
        eta = np.linspace(-5, 5, 50)
        mu, d_mu = logit_link_and_derivative(eta)

        # Central difference (more accurate than forward difference)
        eps = 1e-7
        mu_plus, _ = logit_link_and_derivative(eta + eps)
        mu_minus, _ = logit_link_and_derivative(eta - eps)
        d_mu_numerical = (mu_plus - mu_minus) / (2 * eps)

        assert_allclose(d_mu, d_mu_numerical, rtol=1e-5)

    def test_bounds(self):
        """Test output is in (0, 1)."""
        eta = np.random.randn(100) * 10
        mu, d_mu = logit_link_and_derivative(eta)

        assert np.all(
            mu >= 0
        )  # Because of machine precision, they can be essentially 0 or 1
        assert np.all(mu <= 1)
        assert np.all(d_mu >= 0)
        assert np.all(d_mu <= 0.25)  # Max at mu=0.5


class TestLogLink:
    """Test log link function."""

    def test_known_values(self):
        """Test at specific points."""
        # exp(0) = 1
        mu, d_mu = log_link_and_derivative(np.array([0.0]))
        assert_allclose(mu, 1.0)
        assert_allclose(d_mu, 1.0)

        # exp(ln(2)) = 2
        mu, d_mu = log_link_and_derivative(np.array([np.log(2)]))
        assert_allclose(mu, 2.0)
        assert_allclose(d_mu, 2.0)

        # exp(ln(x)) = x
        x = np.array([0.5, 1.0, 2.0, 10.0])
        mu, d_mu = log_link_and_derivative(np.log(x))
        assert_allclose(mu, x)
        assert_allclose(d_mu, x)

    def test_derivative_equals_function(self):
        """Test that derivative equals the function value."""
        eta = np.random.randn(100)
        mu, d_mu = log_link_and_derivative(eta)
        assert_allclose(mu, d_mu)

    def test_overflow_protection(self):
        """Test extreme values don't overflow."""
        # These should be clipped
        mu_large, _ = log_link_and_derivative(np.array([800.0]))
        mu_clip, _ = log_link_and_derivative(np.array([700.0]))
        assert_allclose(mu_large, mu_clip)

        # Negative values should be safe
        mu_neg, _ = log_link_and_derivative(np.array([-800.0]))
        assert mu_neg > 0
        assert np.isfinite(mu_neg)

    def test_positivity(self):
        """Test output is always positive."""
        eta = np.random.randn(100) * 100
        mu, d_mu = log_link_and_derivative(eta)

        assert np.all(mu > 0)
        assert np.all(d_mu > 0)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(d_mu))
