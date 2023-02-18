import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray

from bayesianbandits import DirichletClassifier


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
        clf.sample(X),
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
        clf.sample(X),
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
