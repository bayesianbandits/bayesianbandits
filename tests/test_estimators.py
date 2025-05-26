import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray

from bayesianbandits import (
    DirichletClassifier,
    GammaRegressor,
)


@pytest.fixture
def X() -> NDArray[np.int_]:
    """Return X."""
    return np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(-1, 1)


@pytest.fixture
def y() -> NDArray[np.int_]:
    """Return y."""
    return np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])


def test_dirichletclassifier_init() -> None:
    """Test DirichletClassifier init."""
    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    assert clf is not None


def test_dirichletclassifier_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test DirichletClassifier fit."""

    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf.fit(X, y)
    assert_almost_equal(clf.classes_, np.array([1, 2, 3]))
    assert clf.n_features_ == 1
    assert_almost_equal(clf.prior_, np.array([1.0, 1.0, 1.0]))

    assert_almost_equal(clf.known_alphas_[1], np.array([3, 2, 1]))
    assert_almost_equal(clf.known_alphas_[2], np.array([1, 3, 2]))
    assert_almost_equal(clf.known_alphas_[3], np.array([1, 1, 4]))


def test_dirichletclassifier_partial_fit_never_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test DirichletClassifier partial_fit."""

    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf.partial_fit(X, y)
    assert_almost_equal(clf.classes_, np.array([1, 2, 3]))
    assert clf.n_features_ == 1
    assert_almost_equal(clf.prior_, np.array([1.0, 1.0, 1.0]))

    assert_almost_equal(clf.known_alphas_[1], np.array([3, 2, 1]))
    assert_almost_equal(clf.known_alphas_[2], np.array([1, 3, 2]))
    assert_almost_equal(clf.known_alphas_[3], np.array([1, 1, 4]))


def test_dirichleclassifier_partial_fit_already_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test DirichletClassifier partial_fit."""

    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf.partial_fit(X, y)
    clf.partial_fit(X, y)
    assert_almost_equal(clf.classes_, np.array([1, 2, 3]))
    assert clf.n_features_ == 1
    assert_almost_equal(clf.prior_, np.array([1.0, 1.0, 1.0]))

    assert_almost_equal(clf.known_alphas_[1], np.array([5, 3, 1]))
    assert_almost_equal(clf.known_alphas_[2], np.array([1, 5, 3]))
    assert_almost_equal(clf.known_alphas_[3], np.array([1, 1, 7]))


def test_dirichletclassifier_predict_proba(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test DirichletClassifier predict_proba."""

    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf.fit(X, y)

    assert_almost_equal(
        clf.predict_proba(X),
        np.array(
            [
                [0.5, 0.33333333, 0.16666667],
                [0.5, 0.33333333, 0.16666667],
                [0.5, 0.33333333, 0.16666667],
                [0.16666667, 0.5, 0.33333333],
                [0.16666667, 0.5, 0.33333333],
                [0.16666667, 0.5, 0.33333333],
                [0.16666667, 0.16666667, 0.66666667],
                [0.16666667, 0.16666667, 0.66666667],
                [0.16666667, 0.16666667, 0.66666667],
            ]
        ),
    )


def test_dirichletclassifier_predict_proba_no_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test DirichletClassifier predict_proba."""

    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    assert_almost_equal(
        clf.predict_proba(X),
        np.array(
            [
                [0.33333333, 0.33333333, 0.33333333],
                [0.33333333, 0.33333333, 0.33333333],
                [0.33333333, 0.33333333, 0.33333333],
                [0.33333333, 0.33333333, 0.33333333],
                [0.33333333, 0.33333333, 0.33333333],
                [0.33333333, 0.33333333, 0.33333333],
                [0.33333333, 0.33333333, 0.33333333],
                [0.33333333, 0.33333333, 0.33333333],
                [0.33333333, 0.33333333, 0.33333333],
            ]
        ),
    )


def test_dirichletclassifier_predict(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test DirichletClassifier predict."""

    clf = DirichletClassifier(
        alphas={1: 1, 2: 1, 3: 1}, random_state=np.random.default_rng(0)
    )
    clf.fit(X, y)
    assert_almost_equal(clf.predict(X), np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))


def test_dirichletclassifier_sample(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test DirichletClassifier sample."""

    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf.fit(X, y)

    assert_almost_equal(
        clf.sample(X).squeeze(),
        np.array(
            [
                [0.47438369, 0.43488058, 0.09073572],
                [0.26262622, 0.2555389, 0.48183488],
                [0.60124073, 0.03608761, 0.36267166],
                [0.17052082, 0.37621552, 0.45326367],
                [0.00475738, 0.71717348, 0.27806914],
                [0.12992228, 0.51114609, 0.35893164],
                [0.31837443, 0.05969183, 0.62193374],
                [0.13646493, 0.21666226, 0.64687281],
                [0.60657175, 0.0318367, 0.36159154],
            ]
        ),
    )


def test_dirichletclassifier_sample_no_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test DirichletClassifier sample."""

    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)

    assert_almost_equal(
        clf.sample(X).squeeze(),
        np.array(
            [
                [3.95461990e-01, 5.93018059e-01, 1.15199510e-02],
                [1.03975805e-03, 2.52155602e-01, 7.46804640e-01],
                [1.58651734e-01, 1.77899202e-01, 6.63449064e-01],
                [6.48202142e-01, 3.51660064e-01, 1.37794082e-04],
                [6.65230007e-01, 2.12541311e-02, 3.13515862e-01],
                [1.95029163e-01, 7.23642534e-01, 8.13283026e-02],
                [1.67241370e-01, 8.12607512e-01, 2.01511183e-02],
                [6.97553771e-02, 5.31008838e-01, 3.99235785e-01],
                [6.68451176e-01, 1.59130848e-01, 1.72417976e-01],
            ]
        ),
    )


def test_dirichletclassifier_decay(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor decay only increases variance."""

    clf = DirichletClassifier(
        alphas={1: 1, 2: 1, 3: 1}, learning_rate=0.9, random_state=0
    )
    clf.fit(X, y)

    pre_decay = clf.predict(X)

    clf.decay(X)

    assert_almost_equal(clf.predict(X), pre_decay)


def test_dirichletclassifier_manual_decay(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor decay only increases variance."""

    clf = DirichletClassifier(
        alphas={1: 1, 2: 1, 3: 1}, learning_rate=1.0, random_state=0
    )
    clf.fit(X, y)

    pre_decay = clf.predict(X)

    clf.decay(X, decay_rate=0.9)

    assert_almost_equal(clf.predict(X), pre_decay)


def test_dirichletclassifier_fit_with_weights(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test that sample weights affect the posterior correctly."""
    clf1 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf2 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)

    # Uniform weights (implicit)
    clf1.fit(X, y)

    # Non-uniform weights
    weights = np.array([2.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 2.0])
    clf2.fit(X, y, sample_weight=weights)

    # Posteriors should be different
    assert not np.allclose(clf1.known_alphas_[1], clf2.known_alphas_[1])
    assert not np.allclose(clf1.known_alphas_[2], clf2.known_alphas_[2])
    assert not np.allclose(clf1.known_alphas_[3], clf2.known_alphas_[3])

    # Check specific values for weighted case
    # For group 1: weights [2.0, 1.0, 0.5], y encodings [[1,0,0], [1,0,0], [0,1,0]]
    # Expected: prior [1,1,1] + [2*1 + 1*1, 2*0 + 1*0, 0] + [0.5*0, 0.5*1, 0.5*0] = [4, 1.5, 1]
    assert_almost_equal(clf2.known_alphas_[1], np.array([4.0, 1.5, 1.0]))


def test_dirichletclassifier_weight_equals_duplication(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test that weight=2 is equivalent to seeing a sample twice."""
    # Test with a single sample
    clf1 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    X_double = np.array([1, 1]).reshape(-1, 1)
    y_double = np.array([1, 1])
    clf1.fit(X_double, y_double)

    clf2 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    X_single = np.array([1]).reshape(-1, 1)
    y_single = np.array([1])
    clf2.fit(X_single, y_single, sample_weight=np.array([2.0]))

    assert_almost_equal(clf1.known_alphas_[1], clf2.known_alphas_[1])

    # Test with multiple samples
    clf3 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    X_many = np.array([1, 1, 2, 2, 3, 3]).reshape(-1, 1)
    y_many = np.array([1, 1, 2, 2, 3, 3])
    clf3.fit(X_many, y_many)

    clf4 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    X_few = np.array([1, 2, 3]).reshape(-1, 1)
    y_few = np.array([1, 2, 3])
    clf4.fit(X_few, y_few, sample_weight=np.array([2.0, 2.0, 2.0]))

    assert_almost_equal(clf3.known_alphas_[1], clf4.known_alphas_[1])
    assert_almost_equal(clf3.known_alphas_[2], clf4.known_alphas_[2])
    assert_almost_equal(clf3.known_alphas_[3], clf4.known_alphas_[3])


def test_dirichletclassifier_zero_weights(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test that samples with weight=0 are effectively ignored."""
    clf1 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    # Only fit on subset
    X_subset = np.array([1, 1, 2, 2, 3, 3]).reshape(-1, 1)
    y_subset = np.array([1, 2, 2, 3, 3, 3])
    clf1.fit(X_subset, y_subset)

    clf2 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    # Fit on full data but with zeros for excluded samples
    weights = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    clf2.fit(X, y, sample_weight=weights)

    assert_almost_equal(clf1.known_alphas_[1], clf2.known_alphas_[1])
    assert_almost_equal(clf1.known_alphas_[2], clf2.known_alphas_[2])
    assert_almost_equal(clf1.known_alphas_[3], clf2.known_alphas_[3])


def test_dirichletclassifier_all_zero_weights(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test edge case where all weights are zero."""
    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    weights = np.zeros(len(y))
    clf.fit(X, y, sample_weight=weights)

    # Should only have the prior
    assert_almost_equal(clf.known_alphas_[1], np.array([1.0, 1.0, 1.0]))
    assert_almost_equal(clf.known_alphas_[2], np.array([1.0, 1.0, 1.0]))
    assert_almost_equal(clf.known_alphas_[3], np.array([1.0, 1.0, 1.0]))


def test_dirichletclassifier_partial_fit_with_weights(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test partial_fit with sample weights."""
    # Test initial partial_fit
    clf1 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    weights = np.array([2.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 2.0])
    clf1.partial_fit(X, y, sample_weight=weights)

    # Should be same as fit with weights
    clf2 = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf2.fit(X, y, sample_weight=weights)

    assert_almost_equal(clf1.known_alphas_[1], clf2.known_alphas_[1])
    assert_almost_equal(clf1.known_alphas_[2], clf2.known_alphas_[2])
    assert_almost_equal(clf1.known_alphas_[3], clf2.known_alphas_[3])

    # Test sequential partial_fit with learning_rate=1 (no decay)
    clf3 = DirichletClassifier(
        alphas={1: 1, 2: 1, 3: 1},
        learning_rate=1.0,  # No decay
        random_state=0,
    )
    # First batch
    X1 = X[:5]
    y1 = y[:5]
    w1 = weights[:5]
    clf3.partial_fit(X1, y1, sample_weight=w1)

    # Second batch
    X2 = X[5:]
    y2 = y[5:]
    w2 = weights[5:]
    clf3.partial_fit(X2, y2, sample_weight=w2)

    # With learning_rate=1, sequential partial_fit should equal single fit
    clf4 = DirichletClassifier(
        alphas={1: 1, 2: 1, 3: 1}, learning_rate=1.0, random_state=0
    )
    clf4.fit(X, y, sample_weight=weights)

    assert_almost_equal(clf3.known_alphas_[1], clf4.known_alphas_[1])
    assert_almost_equal(clf3.known_alphas_[2], clf4.known_alphas_[2])
    assert_almost_equal(clf3.known_alphas_[3], clf4.known_alphas_[3])


def test_dirichletclassifier_weights_learning_rate_interaction() -> None:
    """Test that sample weights and learning_rate interact correctly."""
    # Create simple data
    X = np.array([1, 1, 1]).reshape(-1, 1)
    y = np.array([1, 2, 3])

    # Test with learning_rate < 1
    clf1 = DirichletClassifier(
        alphas={1: 1, 2: 1, 3: 1}, learning_rate=0.5, random_state=0
    )
    weights = np.array([2.0, 1.0, 0.5])
    clf1.fit(X, y, sample_weight=weights)

    # The decay applies to all values including prior
    # Expected calculation:
    # Stack: prior [1, 1, 1] + weighted data [[2, 0, 0], [0, 1, 0], [0, 0, 0.5]]
    # Decay indices for 4 elements: [3, 2, 1, 0]
    # Decay factors: [0.5^3, 0.5^2, 0.5^1, 0.5^0] = [0.125, 0.25, 0.5, 1.0]
    # Result: [1, 1, 1]*0.125 + [2, 0, 0]*0.25 + [0, 1, 0]*0.5 + [0, 0, 0.5]*1.0
    #       = [0.125, 0.125, 0.125] + [0.5, 0, 0] + [0, 0.5, 0] + [0, 0, 0.5]
    #       = [0.625, 0.625, 0.625]
    assert_almost_equal(clf1.known_alphas_[1], np.array([0.625, 0.625, 0.625]))

    # Also verify that weights and decay compose properly
    # If we double all weights, the posterior should scale accordingly
    clf2 = DirichletClassifier(
        alphas={1: 1, 2: 1, 3: 1}, learning_rate=0.5, random_state=0
    )
    weights_double = weights * 2
    clf2.fit(X, y, sample_weight=weights_double)

    # Prior contribution stays same: [0.125, 0.125, 0.125]
    # Data contributions double: [1.0, 0, 0] + [0, 1.0, 0] + [0, 0, 1.0]
    # Total: [1.125, 1.125, 1.125]
    assert_almost_equal(clf2.known_alphas_[1], np.array([1.125, 1.125, 1.125]))


def test_dirichletclassifier_importance_sampling_scenario() -> None:
    """Test a simple importance sampling scenario."""
    # Simulate biased sampling where class 1 is undersampled
    # True distribution: [0.4, 0.3, 0.3]
    # Sampling distribution: [0.1, 0.45, 0.45]
    np.random.seed(42)

    # Generate biased samples
    n_samples = 100
    sampling_probs = [0.1, 0.45, 0.45]
    classes = [1, 2, 3]
    y_biased = np.random.choice(classes, size=n_samples, p=sampling_probs)
    X_biased = np.ones((n_samples, 1))  # All same context

    # Calculate importance weights
    true_probs = np.array([0.4, 0.3, 0.3])
    sampling_probs_array = np.array(sampling_probs)
    importance_weights = np.zeros(n_samples)
    for i, y_val in enumerate(y_biased):
        class_idx = y_val - 1
        importance_weights[i] = true_probs[class_idx] / sampling_probs_array[class_idx]

    # Fit with importance weights
    clf_weighted = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf_weighted.fit(X_biased, y_biased, sample_weight=importance_weights)

    # Fit without weights (biased)
    clf_biased = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    clf_biased.fit(X_biased, y_biased)

    # The weighted classifier should recover closer to true distribution
    weighted_probs = clf_weighted.predict_proba(np.array([[1]]))[0]
    biased_probs = clf_biased.predict_proba(np.array([[1]]))[0]

    # Check that weighted is closer to true distribution than biased
    weighted_error = np.abs(weighted_probs - true_probs).sum()
    biased_error = np.abs(biased_probs - true_probs).sum()
    assert weighted_error < biased_error

    # The biased estimate should be close to sampling distribution
    assert np.abs(biased_probs[0] - 0.1) < 0.1  # Class 1 should be underestimated

    # The weighted estimate should move toward the true distribution
    # Check that class 1 probability increased with weighting
    assert weighted_probs[0] > biased_probs[0]


def test_dirichletclassifier_weight_validation() -> None:
    """Test that weight validation works correctly."""
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])

    clf = DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)

    # Wrong shape should raise ValueError
    with pytest.raises(ValueError, match="sample_weight.shape"):
        clf.fit(X, y, sample_weight=np.array([1.0, 2.0]))  # Too few weights

    with pytest.raises(ValueError, match="sample_weight.shape"):
        clf.fit(X, y, sample_weight=np.array([1.0, 2.0, 3.0, 4.0]))  # Too many weights


def test_dirichletclassifier_fractional_weights() -> None:
    """Test that fractional weights work correctly."""
    X = np.array([1, 1, 2, 2]).reshape(-1, 1)
    y = np.array([1, 2, 1, 2])

    # Use fractional weights
    weights = np.array([0.7, 1.3, 0.4, 1.6])

    clf = DirichletClassifier(alphas={1: 1, 2: 1}, random_state=0)
    clf.fit(X, y, sample_weight=weights)

    # Check the math
    # For X=1: prior [1,1] + [0.7, 0] + [0, 1.3] = [1.7, 2.3]
    # For X=2: prior [1,1] + [0.4, 0] + [0, 1.6] = [1.4, 2.6]
    assert_almost_equal(clf.known_alphas_[1], np.array([1.7, 2.3]))
    assert_almost_equal(clf.known_alphas_[2], np.array([1.4, 2.6]))


def test_gamma_regressor_init() -> None:
    """Test GammaRegressor init."""

    clf = GammaRegressor(alpha=1, beta=1, random_state=0)
    assert clf is not None


def test_gamma_regressor_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor fit."""

    clf = GammaRegressor(alpha=1, beta=1, random_state=0)
    clf.fit(X, y)
    assert_almost_equal(clf.coef_[1], np.array([5, 4]))
    assert_almost_equal(clf.coef_[2], np.array([8, 4]))
    assert_almost_equal(clf.coef_[3], np.array([10, 4]))


def test_gamma_regressor_partial_fit_never_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor partial_fit."""

    clf = GammaRegressor(alpha=1, beta=1, random_state=0)
    clf.partial_fit(X, y)
    assert_almost_equal(clf.coef_[1], np.array([5, 4]))
    assert_almost_equal(clf.coef_[2], np.array([8, 4]))
    assert_almost_equal(clf.coef_[3], np.array([10, 4]))


def test_gamma_regressor_partial_fit_already_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor partial_fit."""

    clf = GammaRegressor(alpha=1, beta=1, random_state=0)
    clf.fit(X, y)
    clf.partial_fit(X, y)
    assert_almost_equal(clf.coef_[1], np.array([9, 7]))
    assert_almost_equal(clf.coef_[2], np.array([15, 7]))
    assert_almost_equal(clf.coef_[3], np.array([19, 7]))


def test_gamma_regressor_predict(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor predict."""

    clf = GammaRegressor(alpha=1, beta=1, random_state=0)
    clf.fit(X, y)
    assert_almost_equal(
        clf.predict(X), np.array([1.25, 1.25, 1.25, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
    )


def test_gamma_regressor_predict_no_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor predict."""

    clf = GammaRegressor(alpha=1, beta=1, random_state=0)
    assert_almost_equal(clf.predict(X), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_gamma_regressor_sample(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor sample."""

    clf = GammaRegressor(alpha=1, beta=1, random_state=0)
    clf.fit(X, y)

    assert_almost_equal(
        clf.sample(X),
        np.array(
            [
                1.2358946,
                1.5478387,
                0.9006251,
                2.9684361,
                1.4696328,
                1.5167874,
                1.0225134,
                1.5718412,
                2.0178289,
            ]
        )[np.newaxis, :],
    )


def test_gamma_regressor_sample_no_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor sample."""

    clf = GammaRegressor(alpha=1, beta=1, random_state=0)

    assert_almost_equal(
        clf.sample(X),
        np.array(
            [
                6.7993190e-01,
                1.0195971e00,
                1.9806663e-02,
                2.2693267e-03,
                5.5034287e-01,
                1.6299404e00,
                6.7358295e-01,
                7.5530136e-01,
                2.8167860e00,
            ]
        )[np.newaxis, :],
    )


def test_gamma_regressor_decay(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor decay only increases variance."""

    clf = GammaRegressor(alpha=1, beta=1, learning_rate=0.9, random_state=0)
    clf.fit(X, y)

    pre_decay = clf.predict(X)

    clf.decay(X)

    assert_almost_equal(clf.predict(X), pre_decay)


def test_gamma_regressor_manual_decay(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
) -> None:
    """Test GammaRegressor decay only increases variance."""

    clf = GammaRegressor(alpha=1, beta=1, learning_rate=1.0, random_state=0)
    clf.fit(X, y)

    pre_decay = clf.predict(X)

    clf.decay(X, decay_rate=0.9)

    assert_almost_equal(clf.predict(X), pre_decay)
