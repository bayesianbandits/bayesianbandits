from collections import defaultdict
from functools import partial
from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import dirichlet, gamma  # type: ignore
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore
from sklearn.utils.validation import check_X_y  # type: ignore
from sklearn.utils.validation import NotFittedError, check_is_fitted
from typing_extensions import Self


class DirichletClassifier(BaseEstimator, ClassifierMixin):  # type: ignore
    """
    Intercept-only Dirichlet Classifier

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
    ) -> None:
        self.alphas = alphas
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> Self:
        """
        Fit the model using X as training data and y as target values.
        """
        X, y = check_X_y(X, y, copy=True, ensure_2d=True)  # type: ignore

        self._initialize_prior()

        y_encoded: NDArray[np.int_] = (y[:, np.newaxis] == self.classes_).astype(int)

        self.n_features_ = X.shape[1]

        if self.n_features_ > 1:
            raise NotImplementedError("Only one feature supported")

        self._fit_helper(X, y_encoded)

        return self

    def _initialize_prior(self) -> None:
        if not hasattr(self, "prior_"):
            if isinstance(self.random_state, int):
                self.random_state_ = np.random.default_rng(self.random_state)
            else:
                self.random_state_ = self.random_state

            self.classes_ = np.array(list(self.alphas.keys()))
            self.n_classes_ = len(self.classes_)
            self.prior_ = np.array(list(self.alphas.values()))

            self.known_alphas_: Dict[Any, NDArray[np.float_]] = defaultdict(
                self._return_prior
            )

    def _return_prior(self) -> NDArray[np.float_]:
        return self.prior_

    def partial_fit(self, X: NDArray[Any], y: NDArray[Any]):
        """
        Update the model using X as training data and y as target values.
        """
        try:
            check_is_fitted(self, "n_features_")
        except NotFittedError:
            return self.fit(X, y)

        X_fit, y = check_X_y(X, y, copy=True, ensure_2d=True)
        y = (y[:, np.newaxis] == self.classes_).astype(int)

        self._fit_helper(X_fit, y)
        return self

    def _fit_helper(self, X: NDArray[Any], y: NDArray[Any]):
        sort_keys = X[:, 0].argsort()  # type: ignore
        X, y = X[sort_keys], y[sort_keys]  # type: ignore

        groups, start_indexes = np.unique(X[:, 0], return_index=True)

        for group, arr in zip(groups, np.split(y, start_indexes)[1:]):
            key = group.item()
            vals = np.row_stack((self.known_alphas_[key], arr))

            decay_idx = np.flip(np.arange(len(vals)))  # type: ignore

            posterior = vals * (self.learning_rate**decay_idx)[:, np.newaxis]
            self.known_alphas_[key] = posterior.sum(axis=0)  # type: ignore

    def predict_proba(self, X: NDArray[Any]) -> Any:
        """
        Predict class probabilities for X.
        """
        try:
            check_is_fitted(self, "n_features_")
        except NotFittedError:
            self._initialize_prior()

        X_pred = check_array(X, copy=True, ensure_2d=True)

        alphas = np.row_stack(list(self.known_alphas_[x.item()] for x in X_pred))
        return alphas / alphas.sum(axis=1)[:, np.newaxis]  # type: ignore

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict class for X.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def sample(self, X: NDArray[Any], size: int = 1) -> NDArray[np.float64]:
        """
        Sample from the posterior for X.
        """
        try:
            check_is_fitted(self, "n_features_")
        except NotFittedError:
            self._initialize_prior()

        alphas = list(self.known_alphas_[x.item()] for x in X)
        return np.squeeze(
            np.stack(
                list(
                    dirichlet.rvs(alpha, size, self.random_state_)  # type: ignore
                    for alpha in alphas
                ),
            )
        )

    def decay(self, X: NDArray[Any]) -> None:
        """
        Decay the prior by a factor of `learning_rate`.
        """
        if not hasattr(self, "known_alphas_"):
            self._initialize_prior()
        for x in X:
            self.known_alphas_[x.item()] *= self.learning_rate


class GammaRegressor(BaseEstimator, RegressorMixin):
    """
    Intercept-only Gamma regression model.

    Parameters
    ----------
    alpha : float
        Shape parameter of the gamma distribution.
    beta : float
        Inverse scale parameter of the gamma distribution.
    learning_rate : float, default=1.0
        Learning rate for the model.
    random_state : int, np.random.Generator, default=None
        Random state for the model.

    Attributes
    ----------
    coef_ : float
        Dictionary of coefficients for the model.

    Examples
    --------

    This regressor is intended to be used with a single feature. It is
    useful for modeling count data, where the target is the number of
    occurrences of an event.

    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> model = GammaRegressor(alpha=1, beta=1, random_state=0)
    >>> model.fit(X, y)
    GammaRegressor(alpha=1, beta=1, random_state=0)

    >>> model.predict(X)
    array([1. , 1.5, 2. , 2.5, 3. ])

    This model implements a partial fit method, which can be used to
    update the model with new data.

    >>> model.partial_fit(X, y)
    GammaRegressor(alpha=1, beta=1, random_state=0)

    >>> model.predict(X)
    array([1.        , 1.66666667, 2.33333333, 3.        , 3.66666667])

    The `sample` method can be used to sample from the posterior.

    >>> model.sample(X)
    array([0.95909923, 2.06378492, 1.7923389 , 4.36674677, 2.84313527])

    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        *,
        learning_rate: float = 1.0,
        random_state: Union[int, np.random.Generator, None] = None
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> Self:
        """
        Fit the model using X as training data and y as target values. y must be
        count data.
        """
        X, y = check_X_y(X, y, copy=True, ensure_2d=True)  # type: ignore

        self._initialize_prior()

        y_encoded: NDArray[np.int_] = y.astype(int)

        self.n_features_ = X.shape[1]

        if self.n_features_ > 1:
            raise NotImplementedError("Only one feature supported")

        self._fit_helper(X, y_encoded)

        return self

    def _initialize_prior(self) -> None:
        if not hasattr(self, "prior_"):
            if isinstance(self.random_state, int):
                self.random_state_ = np.random.default_rng(self.random_state)
            else:
                self.random_state_ = self.random_state

            self.prior_ = np.array([self.alpha, self.beta], dtype=np.float_)

            self.coef_: Dict[Any, NDArray[np.float_]] = defaultdict(self._return_prior)

    def _return_prior(self) -> NDArray[np.float_]:
        return self.prior_

    def _fit_helper(self, X: NDArray[Any], y: NDArray[Any]):
        # Performs the gamma-poisson update

        # Sort the data by the first column
        sort_keys = X[:, 0].argsort()  # type: ignore
        X, y = X[sort_keys], y[sort_keys]  # type: ignore

        # Get the unique groups and their start indexes
        groups, start_indexes = np.unique(X[:, 0], return_index=True)

        for group, arr in zip(groups, np.split(y, start_indexes)[1:]):
            key = group.item()

            # the update is computed by stacking the prior with the data,
            # where alpha and the counts are stacked together and beta is
            # stacked with ones

            data = np.column_stack((arr, np.ones_like(arr)))

            vals = np.row_stack((self.coef_[key], data))

            # Calculate the decay index for nonstationary models
            decay_idx = np.flip(np.arange(len(vals)))  # type: ignore
            # Calculate the posterior
            posterior = vals * (self.learning_rate**decay_idx)[:, np.newaxis]
            # Calculate the coefficient
            self.coef_[key] = posterior.sum(axis=0)  # type: ignore

    def partial_fit(self, X: NDArray[Any], y: NDArray[Any]):
        """
        Update the model using X as training data and y as target values.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            return self.fit(X, y)

        X_fit, y = check_X_y(X, y, copy=True, ensure_2d=True)
        y_encoded: NDArray[np.int_] = y.astype(int)

        self._fit_helper(X_fit, y_encoded)
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict class for X.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior()

        X_pred = check_array(X, copy=True, ensure_2d=True)

        shape_params = np.row_stack(list(self.coef_[x.item()] for x in X_pred))
        return shape_params[:, 0] / shape_params[:, 1]

    def sample(self, X: NDArray[Any], size: int = 1) -> NDArray[np.float64]:
        """
        Sample from the posterior for X.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior()

        shape_params = list(self.coef_[x.item()] for x in X)

        rv_gen = partial(gamma.rvs, size=size, random_state=self.random_state_)

        return np.squeeze(
            np.stack(
                list(rv_gen(alpha, scale=1 / beta) for alpha, beta in shape_params),
            )
        )

    def decay(self, X: NDArray[Any]) -> None:
        """
        Decay the prior by a factor of `learning_rate`.
        """
        if not hasattr(self, "coef_"):
            self._initialize_prior()
        for x in X:
            self.coef_[x.item()] *= self.learning_rate
