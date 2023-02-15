from collections import defaultdict
from typing import Dict, Union

import numpy as np
from scipy.stats import dirichlet
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
    NotFittedError,
)


class DirichletClassifier(BaseEstimator, ClassifierMixin):
    """Dirichlet Classifier

    Parameters
    ----------
    alphas : Dict[Union[int, str], float]
        Prior alphas for each class. Keys must be the same as the classes in the
        training data.
    learning_rate : float, default=1.0
        Learning rate for the Dirichlet distribution. Higher values will give
        more weight to recent data.
    random_state : Union[int, np.random.Generator, None], default=None
        Random state for sampling from the Dirichlet distribution.

    Attributes
    ----------
    classes_ : np.ndarray
        The classes seen during fit.
    n_classes_ : int
        The number of classes seen during fit.
    n_features_ : int
        The number of features seen during fit.
    prior_ : np.ndarray
        The prior alphas for each class.
    known_alphas_ : Dict[Union[int, str], np.ndarray]
        The posterior alphas for each class seen during fit.

    Examples
    --------

    This classifier is used in the same way as any other scikit-learn classifier.

    >>> from bayesianbandits import DirichletClassifier
    >>> X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(-1, 1)
    >>> y = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
    >>> clf = DirichletClassifier({1: 1, 2: 1, 3: 1}, random_state=0)
    >>> clf.fit(X, y)
    DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
    >>> clf.predict_proba(X)
    array([[0.5       , 0.33333333, 0.16666667],
           [0.5       , 0.33333333, 0.16666667],
           [0.5       , 0.33333333, 0.16666667],
           [0.16666667, 0.5       , 0.33333333],
           [0.16666667, 0.5       , 0.33333333],
           [0.16666667, 0.5       , 0.33333333],
           [0.16666667, 0.16666667, 0.66666667],
           [0.16666667, 0.16666667, 0.66666667],
           [0.16666667, 0.16666667, 0.66666667]])

    This classifier also implements `partial_fit` to update the posterior,
    which can be useful for online learning.

    >>> clf.partial_fit(X, y)
    DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)

    This classifier also implements `sample` to sample from the posterior,
    which can be useful for uncertainty estimation and Thompson sampling.

    >>> clf.sample(X)
    array([[0.52877785, 0.41235606, 0.05886609],
           [0.34152373, 0.28178422, 0.37669205],
           [0.70292861, 0.07890427, 0.21816712],
           [0.10838237, 0.45793671, 0.43368092],
           [0.00318876, 0.71391831, 0.28289293],
           [0.07336816, 0.57424303, 0.35238881],
           [0.20754162, 0.03891185, 0.75354653],
           [0.08269207, 0.13128832, 0.78601961],
           [0.41846435, 0.02196364, 0.55957201]])
    """

    def __init__(
        self,
        alphas: Dict[Union[int, str], float],
        *,
        learning_rate: float = 1.0,
        random_state: Union[int, np.random.Generator, None] = None
    ):
        self.alphas = alphas
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        """
        X, y = check_X_y(X, y, copy=True, ensure_2d=True)

        if isinstance(self.random_state, int):
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        self.classes_ = np.array(list(self.alphas.keys()))

        y = (y[:, np.newaxis] == self.classes_).astype(int)

        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.prior_ = np.array(list(self.alphas.values()))
        if self.n_features_ > 1:
            raise NotImplementedError("Only one feature supported")

        self.known_alphas_ = defaultdict(lambda: self.prior_)

        self._fit_helper(X, y)

        return self

    def partial_fit(self, X, y):
        """
        Update the model using X as training data and y as target values.
        """
        try:
            check_is_fitted(self)
        except NotFittedError:
            return self.fit(X, y)

        X, y = check_X_y(X, y, copy=True, ensure_2d=True)
        y = (y[:, np.newaxis] == self.classes_).astype(int)

        self._fit_helper(X, y)
        return self

    def _fit_helper(self, X, y):
        sort_keys = X[:, 0].argsort()
        X, y = X[sort_keys], y[sort_keys]

        groups, start_indexes = np.unique(X[:, 0], return_index=True)

        for group, arr in zip(groups, np.split(y, start_indexes)[1:]):
            key = group.item()
            vals = np.row_stack((self.known_alphas_[key], arr))
            posterior = (
                vals
                * (self.learning_rate ** np.flip(np.arange(len(vals))))[:, np.newaxis]
            )
            self.known_alphas_[key] = posterior.sum(axis=0)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        check_is_fitted(self)
        X = check_array(X, copy=True, ensure_2d=True)

        alphas = np.row_stack(list(self.known_alphas_[x.item()] for x in X))
        return alphas / alphas.sum(axis=1)[:, np.newaxis]

    def predict(self, X):
        """
        Predict class for X.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def sample(self, X, size=1):
        """
        Sample from the posterior for X.
        """
        alphas = list(self.known_alphas_[x.item()] for x in X)
        return np.squeeze(
            np.stack(
                list(
                    dirichlet.rvs(alpha, size=size, random_state=self.random_state_)
                    for alpha in alphas
                ),
            )
        )
