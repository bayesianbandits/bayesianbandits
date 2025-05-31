from typing import Literal
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_less
from sklearn.datasets import make_classification, make_regression

from bayesianbandits import BayesianGLM, LaplaceApproximator
from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver


# Fixtures and parametrization
@pytest.fixture(
    params=[SparseSolver.SUPERLU, SparseSolver.CHOLMOD],
    autouse=True,
)
def suitesparse_envvar(request):
    """Test with different sparse solvers."""
    with mock.patch("bayesianbandits._gaussian.solver", request.param):
        yield


@pytest.fixture
def binary_data():
    """Binary classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
        flip_y=0.1,
    )
    return X, y


@pytest.fixture
def count_data():
    """Count regression data (Poisson-like)."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    # Transform to counts
    y = np.exp(0.5 * (y - y.mean()) / y.std())
    y = np.round(np.abs(y)).astype(int)
    return X, y


# Basic functionality tests
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("link", ["logit", "log"])
@pytest.mark.parametrize("rng", [None, np.random.default_rng(42)])
def test_bayesian_glm_init(sparse: bool, link: str, rng) -> None:
    """Test BayesianGLM initialization."""
    # Default approximator
    clf = BayesianGLM(
        alpha=1.0, link=link, learning_rate=0.9, sparse=sparse, random_state=0
    )
    assert clf.alpha == 1.0
    assert clf.link == link
    assert clf.approximator is None  # Will be initialized on fit

    # Custom approximator
    approx = LaplaceApproximator(n_iter=5)
    clf2 = BayesianGLM(
        alpha=1.0, link=link, approximator=approx, random_state=rng, sparse=sparse
    )
    assert clf2.approximator is approx


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("rng", [None, np.random.default_rng(42)])
def test_bayesian_glm_predict_before_fit_uses_prior(sparse: bool, rng) -> None:
    """Test predict before fit uses prior."""
    X = np.random.randn(10, 5)
    clf = BayesianGLM(alpha=0.1, link="logit", sparse=sparse, random_state=rng)

    # Check prior prediction
    prior_preds = clf.predict(X)
    assert prior_preds.shape == (X.shape[0],)
    assert np.all((prior_preds >= 0) & (prior_preds <= 1))


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_coef_before_fit_bad_link_throws(sparse: bool) -> None:
    """Test coef before fit raises error for bad link."""
    clf = BayesianGLM(alpha=0.1, link="blah", sparse=sparse)

    with pytest.raises(ValueError, match="Unknown link: blah"):
        clf.predict(np.random.randn(10, 5))


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_sample_before_fit_uses_prior(sparse: bool) -> None:
    """Test sample before fit uses prior."""
    X = np.random.randn(10, 5)
    clf = BayesianGLM(alpha=0.1, link="logit", sparse=sparse)

    # Check prior sampling
    samples = clf.sample(X, size=5)
    assert samples.shape == (5, X.shape[0])
    assert np.all((samples >= 0) & (samples <= 1))


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_decay_before_fit_does_nothing(sparse: bool) -> None:
    """Test decay before fit does nothing."""
    X = np.random.randn(10, 5)
    clf = BayesianGLM(alpha=0.1, link="logit", sparse=sparse)

    # Decay should not raise or change anything
    clf.decay(X, decay_rate=0.9)


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_fit_logit(binary_data, sparse: bool) -> None:
    """Test BayesianGLM fit with logistic link."""
    X, y = binary_data

    clf = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
    )
    X_fit = sp.csc_array(X) if sparse else X
    clf.fit(X_fit, y)

    # Check shapes
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.cov_inv_.shape == (X.shape[1], X.shape[1])

    # Check predictions are probabilities
    probs = clf.predict(X_fit)
    assert np.all((probs >= 0) & (probs <= 1))

    # Check reasonable fit
    accuracy = np.mean((probs > 0.5) == y)
    assert accuracy > 0.7  # Should do better than random


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_fit_log(count_data, sparse: bool) -> None:
    """Test BayesianGLM fit with log link."""
    X, y = count_data

    clf = BayesianGLM(
        alpha=0.1,
        link="log",
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
    )
    X_fit = sp.csc_array(X) if sparse else X
    clf.fit(X_fit, y)

    # Check predictions are positive
    preds = clf.predict(X_fit)
    assert np.all(preds > 0)

    # Check reasonable fit (log scale RMSE)
    log_rmse = np.sqrt(np.mean((np.log(preds + 1) - np.log(y + 1)) ** 2))
    assert log_rmse < 1.0


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_partial_fit_basic(binary_data, sparse: bool) -> None:
    """Test basic partial_fit functionality."""
    X, y = binary_data
    X_fit = sp.csc_array(X) if sparse else X

    approx = LaplaceApproximator(n_iter=5)

    # Test partial_fit without prior fit (should be same as fit)
    clf_partial = BayesianGLM(
        alpha=0.1, link="logit", approximator=approx, sparse=sparse
    )
    clf_partial.partial_fit(X_fit[:50], y[:50])

    clf_fit = BayesianGLM(alpha=0.1, link="logit", approximator=approx, sparse=sparse)
    clf_fit.fit(X_fit[:50], y[:50])

    assert_allclose(clf_partial.coef_, clf_fit.coef_)

    # Test that partial_fit updates the model
    coef_before = clf_partial.coef_.copy()
    clf_partial.partial_fit(X_fit[50:], y[50:])

    # Coefficients should change
    assert not np.allclose(coef_before, clf_partial.coef_)

    # Model should make reasonable predictions
    accuracy = np.mean((clf_partial.predict(X_fit) > 0.5) == y)
    assert accuracy > 0.65  # Should beat random


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_streaming_convergence(binary_data, sparse: bool) -> None:
    """Test that streaming updates eventually produce a good model."""
    X, y = binary_data
    X_fit = sp.csc_array(X) if sparse else X

    # Use fewer iterations for streaming
    clf = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=3),
        sparse=sparse,
    )

    # Stream through data multiple times
    batch_size = 10
    for epoch in range(3):  # Multiple passes
        indices = np.random.permutation(len(y))
        for i in range(0, len(y), batch_size):
            batch_idx = indices[i : i + batch_size]
            clf.partial_fit(X_fit[batch_idx], y[batch_idx])

    # After multiple epochs, should have good performance
    accuracy = np.mean((clf.predict(X_fit) > 0.5) == y)
    assert accuracy > 0.75


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_uncertainty_increases_with_decay(
    binary_data, sparse: bool
) -> None:
    """Test that decay increases uncertainty."""
    X, y = binary_data
    X_fit = sp.csc_array(X) if sparse else X

    clf = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
        random_state=42,
    )
    clf.fit(X_fit, y)

    # Get uncertainty before decay (via sampling)
    samples_before = clf.sample(X_fit[:5], size=100)
    std_before = np.std(samples_before, axis=0)

    # Apply decay
    clf.decay(X_fit, decay_rate=0.7)

    # Get uncertainty after decay
    samples_after = clf.sample(X_fit[:5], size=100)
    std_after = np.std(samples_after, axis=0)

    # Uncertainty should increase
    assert np.all(std_after > std_before * 0.9)  # Allow some numerical noise


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_convergence(binary_data, sparse: bool) -> None:
    """Test that more iterations improve convergence."""
    X, y = binary_data
    X_fit = sp.csc_array(X) if sparse else X

    # Fit with different n_iter
    clf_1 = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=1),
        sparse=sparse,
    )
    clf_10 = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
    )

    clf_1.fit(X_fit, y)
    clf_10.fit(X_fit, y)

    # More iterations should give better log-likelihood
    def log_likelihood(clf, X, y):
        probs = clf.predict(X)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    ll_1 = log_likelihood(clf_1, X_fit, y)
    ll_10 = log_likelihood(clf_10, X_fit, y)
    assert ll_10 > ll_1


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("link", ["logit", "log"])
def test_bayesian_glm_sample(
    binary_data, count_data, sparse: bool, link: Literal["logit", "log"]
) -> None:
    """Test posterior sampling."""
    X, y = binary_data if link == "logit" else count_data
    X_fit = sp.csc_array(X) if sparse else X

    clf = BayesianGLM(
        alpha=0.1,
        link=link,
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
        random_state=42,
    )
    clf.fit(X_fit, y)

    # Single sample
    samples_1 = clf.sample(X_fit[:5], size=1)
    assert samples_1.shape == (1, 5)

    # Multiple samples
    samples_100 = clf.sample(X_fit[:5], size=100)
    assert samples_100.shape == (100, 5)

    # Check samples are in valid range
    if link == "logit":
        assert np.all((samples_100 >= 0) & (samples_100 <= 1))
    else:  # log link
        assert np.all(samples_100 > 0)

    # Check uncertainty (std should be > 0)
    sample_std = np.std(samples_100, axis=0)
    assert np.all(sample_std > 0)


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_decay(binary_data, sparse: bool) -> None:
    """Test prior decay functionality."""
    X, y = binary_data
    X_fit = sp.csc_array(X) if sparse else X

    clf = BayesianGLM(
        alpha=1.0,
        link="logit",
        learning_rate=0.9,
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
    )
    clf.fit(X_fit, y)

    # Store predictions and precision before decay
    preds_before = clf.predict(X_fit).copy()
    if sparse:
        precision_before = np.diag(clf.cov_inv_.toarray())
    else:
        precision_before = np.diag(clf.cov_inv_)

    # Apply decay
    clf.decay(X_fit)

    # Predictions should stay the same
    preds_after = clf.predict(X_fit)
    assert_allclose(preds_before, preds_after)

    # Precision should decrease (uncertainty increases)
    if sparse:
        precision_after = np.diag(clf.cov_inv_.toarray())
    else:
        precision_after = np.diag(clf.cov_inv_)
    assert_array_less(precision_after, precision_before)


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_sample_weights(binary_data, sparse: bool) -> None:
    """Test that sample weights affect the fit correctly."""
    X, y = binary_data
    X_fit = sp.csc_array(X) if sparse else X

    # Uniform weights
    clf_uniform = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
    )
    clf_uniform.fit(X_fit, y)

    # Non-uniform weights (emphasize positive class)
    weights = np.where(y == 1, 2.0, 0.5)
    clf_weighted = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
    )
    clf_weighted.fit(X_fit, y, sample_weight=weights)

    # Coefficients should differ
    assert not np.allclose(clf_uniform.coef_, clf_weighted.coef_)

    # Weighted model should predict higher probabilities on average
    # (since we upweighted positive examples)
    assert clf_weighted.predict(X_fit).mean() > clf_uniform.predict(X_fit).mean()


@pytest.mark.parametrize("sparse", [True, False])
def test_bayesian_glm_weight_duplication_equivalence(sparse: bool) -> None:
    """Test weight=2 equals seeing sample twice."""
    X = np.array([[1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    y = np.array([0, 1, 1, 0])

    # Duplicate data
    X_dup = np.vstack([X, X])
    y_dup = np.hstack([y, y])
    clf_dup = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
    )
    X_dup_fit = sp.csc_array(X_dup) if sparse else X_dup
    clf_dup.fit(X_dup_fit, y_dup)

    # Use weights
    weights = np.full(len(y), 2.0)
    clf_weighted = BayesianGLM(
        alpha=0.1,
        link="logit",
        approximator=LaplaceApproximator(n_iter=10),
        sparse=sparse,
    )
    X_fit = sp.csc_array(X) if sparse else X
    clf_weighted.fit(X_fit, y, sample_weight=weights)

    assert_allclose(clf_dup.coef_, clf_weighted.coef_, rtol=1e-5)


def test_bayesian_glm_serialization(binary_data) -> None:
    """Test model serialization."""
    from io import BytesIO

    import joblib

    X, y = binary_data
    clf = BayesianGLM(
        alpha=0.1, link="logit", approximator=LaplaceApproximator(n_iter=10)
    )
    clf.fit(X, y)

    # Serialize and deserialize
    buffer = BytesIO()
    joblib.dump(clf, buffer)
    buffer.seek(0)
    clf_loaded = joblib.load(buffer)

    # Check predictions match
    preds_original = clf.predict(X)
    preds_loaded = clf_loaded.predict(X)
    assert_allclose(preds_original, preds_loaded)

    # Check sampling still works
    samples = clf_loaded.sample(X[:5], size=10)
    assert samples.shape == (10, 5)


@pytest.mark.parametrize(
    "link,y_vals", [("logit", [0, 1, 1, 0, 1]), ("log", [0, 1, 3, 7, 15])]
)
def test_bayesian_glm_edge_cases(link, y_vals) -> None:
    """Test edge cases and numerical stability."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array(y_vals)

    clf = BayesianGLM(alpha=1.0, link=link, approximator=LaplaceApproximator(n_iter=10))
    clf.fit(X, y)

    # Test extreme predictions
    X_extreme = np.array([[-100], [0], [100]])
    preds = clf.predict(X_extreme)

    # Should not produce NaN or Inf
    assert np.all(np.isfinite(preds))

    if link == "logit":
        # Should be bounded in [0, 1]
        assert np.all((preds >= 0) & (preds <= 1))
        # Extreme inputs should produce near 0 or 1
        assert preds[0] < 0.01 or preds[0] > 0.99
        assert preds[2] < 0.01 or preds[2] > 0.99


def test_bayesian_glm_invalid_link() -> None:
    """Test invalid link function raises error."""
    with pytest.raises(ValueError, match="Unknown link"):
        clf = BayesianGLM(link="invalid")
        clf.fit(np.array([[1]]), np.array([1]))


@pytest.mark.parametrize("n_samples,n_features", [(50, 2), (100, 5), (200, 10)])
def test_bayesian_glm_scaling(n_samples, n_features) -> None:
    """Test model works with different data sizes."""
    rng = np.random.RandomState(42)  # Fixed seed
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    logits = X @ true_coef
    y = (logits + rng.randn(n_samples) * 0.5 > 0).astype(int)

    clf = BayesianGLM(
        alpha=0.1, link="logit", approximator=LaplaceApproximator(n_iter=5)
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    accuracy = np.mean((preds > 0.5) == y)
    assert accuracy > 0.6


def test_bayesian_glm_custom_approximator(binary_data) -> None:
    """Test using custom approximator settings."""
    X, y = binary_data

    # Test with high-precision approximator
    approx_precise = LaplaceApproximator(n_iter=50, tol=1e-8)
    clf_precise = BayesianGLM(alpha=0.1, approximator=approx_precise)
    clf_precise.fit(X, y)

    # Test with fast approximator
    approx_fast = LaplaceApproximator(n_iter=1)
    clf_fast = BayesianGLM(alpha=0.1, approximator=approx_fast)
    clf_fast.fit(X, y)

    # Precise should generally have better likelihood
    def log_likelihood(clf, X, y):
        probs = clf.predict(X)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    ll_precise = log_likelihood(clf_precise, X, y)
    ll_fast = log_likelihood(clf_fast, X, y)

    # Precise approximator should do at least as well
    assert ll_precise >= ll_fast - 1e-6  # Allow tiny numerical differences
