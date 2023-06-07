from __future__ import annotations

from collections import defaultdict
from functools import cached_property, partial
from typing import Any, Dict, Optional, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cholesky, solve
from scipy.stats import (
    Covariance,
    dirichlet,
    gamma,
    multivariate_normal,
    multivariate_t,
)
from scipy.stats._multivariate import _squeeze_output
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin  # type: ignore
from sklearn.utils.validation import (
    NotFittedError,
    check_array,  # type: ignore
    check_is_fitted,
    check_X_y,  # type: ignore
)
from typing_extensions import Self

from ._np_utils import groupby_array


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
        more weight to recent data. This transforms the model into a recursive
        Bayesian estimator.
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

    Notes
    -----
    This model implements the Dirichlet-Multinomial model described in Chapter 3
    of ref [1].

    References
    ----------
    [1] Murphy, Kevin P. "Machine Learning: A Probabilistic Perspective."

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
        random_state: Union[int, np.random.Generator, None] = None,
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
        if isinstance(self.random_state, int):
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        self.classes_ = np.array(list(self.alphas.keys()))
        self.n_classes_ = len(self.classes_)
        self.prior_ = np.array(list(self.alphas.values()), dtype=np.float_)

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
        for group, arr in groupby_array(X[:, 0], y, by=X[:, 0]):
            key = group[0].item()
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

    def decay(self, X: NDArray[Any], *, decay_rate: Optional[float] = None) -> None:
        """
        Decay the prior by a factor of `learning_rate`.
        """
        if not hasattr(self, "known_alphas_"):
            self._initialize_prior()

        if decay_rate is None:
            decay_rate = self.learning_rate

        for x in X:
            self.known_alphas_[x.item()] *= decay_rate


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
        Learning rate for the model. This transforms the model into a recursive
        Bayesian estimator.
    random_state : int, np.random.Generator, default=None
        Random state for the model.

    Attributes
    ----------
    coef_ : float
        Dictionary of coefficients for the model.

    Notes
    -----
    While this model is not described in ref [1], it is a simple extension
    of the logic described in the section on conjugate prior models.

    References
    ----------
    [1] Murphy, Kevin P. "Machine Learning: A Probabilistic Perspective."

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
        random_state: Union[int, np.random.Generator, None] = None,
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

        for group, arr in groupby_array(X[:, 0], y, by=X[:, 0]):
            key = group[0].item()

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

    def decay(self, X: NDArray[Any], *, decay_rate: Optional[float] = None) -> None:
        """
        Decay the prior by a factor of `learning_rate`.
        """
        if not hasattr(self, "coef_"):
            self._initialize_prior()

        if decay_rate is None:
            decay_rate = self.learning_rate

        for x in X:
            self.coef_[x.item()] *= decay_rate


class NormalRegressor(BaseEstimator, RegressorMixin):
    """
    A Bayesian linear regression model that assumes a Gaussian noise distribution.

    Parameters
    ----------
    alpha : float
        The prior for the precision of the weights. Weights are assumed to be
        Gaussian distributed with mean 0 and precision `alpha`.
    beta : float
        The prior for the precision of the noise.
    learning_rate : float, default=1.0
        The learning rate for the model. This transforms the model into a recursive
        Bayesian estimator, specifically a Kalman filter.
    random_state : int, np.random.Generator, default=None
        The random state for the model. If an int is passed, it is used to
        seed the numpy random number generator.

    Attributes
    ----------
    coef_ : NDArray[np.float_]
        The coefficients of the model.
    cov_inv_ : NDArray[np.float_]
        The inverse of the covariance matrix of the model.

    Notes
    -----
    This model implements the "known variance" version of the Bayesian linear
    formulation described in Chapter 7 of ref [1].

    References
    ----------
    [1] Murphy, Kevin P. "Machine Learning: A Probabilistic Perspective."

    Examples
    --------

    This regressor can be used in the same way as any other scikit-learn linear
    regressor.

    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> model = NormalRegressor(alpha=0.1, beta=1, random_state=0)
    >>> model.fit(X, y)
    NormalRegressor(alpha=0.1, beta=1, random_state=0)

    >>> model.predict(X)
    array([0.84615385, 1.69230769, 2.53846154, 3.38461538, 4.23076923])

    Unlike the intercept-only conjugate prior models in this package, this model
    learns a coefficient for each feature. The coefficients are stored in the
    `coef_` attribute.

    >>> model.coef_
    array([0.84615385])

    For compatibility with the `Bandit` class, this model also has a `partial_fit`
    method that updates the model using a single data point or a batch of data.

    >>> model.partial_fit(X, y)
    NormalRegressor(alpha=0.1, beta=1, random_state=0)
    >>> model.predict(X)
    array([0.91666667, 1.83333333, 2.75      , 3.66666667, 4.58333333])

    Futhermore, this model also has a `sample` method that samples from the
    posterior distribution of the coefficients.

    >>> model.sample(X)
    array([[0.92814421, 1.85628843, 2.78443264, 3.71257685, 4.64072107]])


    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        *,
        learning_rate: float = 1.0,
        random_state: Union[int, np.random.Generator, None] = None,
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
        X, y = check_X_y(
            X,
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float_,
        )

        self._initialize_prior(X)

        self._fit_helper(X, y)

        return self

    def _initialize_prior(self, X: NDArray[Any]) -> None:
        if isinstance(self.random_state, int) or self.random_state is None:
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        self.n_features_ = X.shape[1]
        self.coef_ = np.zeros(self.n_features_)
        self.cov_inv_ = np.eye(self.n_features_) / self.alpha

    @cached_property
    def cov_(self) -> Covariance:
        """
        The covariance matrix of the model.
        """
        cov = solve(
            self.cov_inv_, np.eye(self.n_features_), check_finite=False, assume_a="pos"
        )
        return Covariance.from_cholesky(cholesky(cov, lower=True))

    def _fit_helper(self, X: NDArray[Any], y: NDArray[Any]):
        # Apply the learning rate to the new data, if there are multiple observations
        # in the batch. This is done to ensure that one-at-a-time updates are
        # equivalent to batch updates.
        if len(X) > 1:
            obs_decays = np.flip(
                np.power(np.sqrt(self.learning_rate), np.arange(len(X)))
            )
            X = X * obs_decays[:, np.newaxis]
            y = y * obs_decays

        prior_decay = self.learning_rate ** len(X)

        # Calculate the posterior covariance
        cov_inv = prior_decay * self.cov_inv_ + self.beta * X.T @ X
        # Calculate the posterior mean
        coef = solve(
            cov_inv,
            prior_decay * self.cov_inv_ @ self.coef_ + self.beta * X.T @ y,
            check_finite=False,
            assume_a="pos",
        )

        self.cov_inv_ = cov_inv
        # Delete the cached covariance matrix, since it is no longer valid
        if hasattr(self, "cov_"):
            del self.cov_
        self.coef_ = coef

    def partial_fit(self, X: NDArray[Any], y: NDArray[Any]):
        """
        Update the model using X as training data and y as target values.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            return self.fit(X, y)

        X_fit, y = check_X_y(
            X,
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float_,
        )

        self._fit_helper(X_fit, y)
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict class for X.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X)

        X_pred = check_array(X, copy=True, ensure_2d=True)

        return X_pred @ self.coef_

    def sample(self, X: NDArray[Any], size: int = 1) -> NDArray[np.float64]:
        """
        Sample from the model posterior at X.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X)

        rv_gen = partial(
            multivariate_normal.rvs, size=size, random_state=self.random_state_
        )

        samples = np.atleast_1d(rv_gen(self.coef_, self.cov_))  # type: ignore

        return np.atleast_2d(samples @ X.T)  # type: ignore

    def decay(self, X: NDArray[Any], *, decay_rate: Optional[float] = None) -> None:
        """
        Decay the prior by a factor of `learning_rate`.
        """
        if not hasattr(self, "coef_"):
            self._initialize_prior(X)

        if decay_rate is None:
            decay_rate = self.learning_rate

        prior_decay = decay_rate ** len(X)

        # Decay the prior without making an update. Because we're only
        # increasing the prior variance, we do not need to update the
        # mean.
        cov_inv = prior_decay * self.cov_inv_

        self.cov_inv_ = cov_inv


class NormalInverseGammaRegressor(NormalRegressor):
    """
    Bayesian linear regression with unknown variance.

    Default prior values correspond to ridge regression with alpha = 1.

    Parameters
    ----------
    mu : ArrayLike, default=0
        Prior mean of the weights. If a scalar, the prior is assumed to be a
        vector with the given value in each entry. If a vector, the prior is
        assumed to be a vector with one entry for each column of X.
    lam : ArrayLike, default=1
        Prior covariance of the weights. If a scalar, the prior is assumed to
        be a diagonal matrix with the given value on the diagonal. If a vector,
        the prior is assumed to be a diagonal matrix with one entry for each
        column of X. If a matrix, the prior is assumed to be a
        full covariance matrix.
    a : float, default=0.1
        Prior shape parameter of the variance.
    b : float, default=0.1
        Prior rate parameter of the variance.
    learning_rate : float, default=1.0
        Learning rate for the model. This transforms the model into a
        recursive Bayesian estimator, specifically a Kalman filter.
    random_state : int, np.random.Generator, or None, default=None
        Random state for the model.

    Attributes
    ----------
    coef_ : NDArray[Any]
        Posterior mean of the weights.
    cov_inv_ : NDArray[Any]
        Posterior inverse covariance of the weights.
    n_features_ : int
        Number of features in the model.
    a_ : float
        Posterior shape parameter of the variance.
    b_ : float
        Posterior rate parameter of the variance.

    Notes
    -----
    This model implements the "unknown variance" version of the Bayesian linear
    formulation described in Chapter 7 of ref [1].

    References
    ----------
    [1] Murphy, Kevin P. "Machine Learning: A Probabilistic Perspective."

    Examples
    --------

    This model can be used in the same way as the `NormalRegressor`
    model.

    >>> from sklearn.datasets import make_regression
    >>> X, y, coef = make_regression(n_samples=30, n_features=2,
    ...                              coef=True, random_state=1)
    >>> coef
    array([34.8898342, 75.0942434])

    >>> est = NormalInverseGammaRegressor()
    >>> est.fit(X, y)
    NormalInverseGammaRegressor()
    >>> est.coef_
    array([32.89089478, 71.16073032])

    For compatibility with this library, this model also implements a `partial_fit`
    method for online learning.

    >>> est = NormalInverseGammaRegressor(random_state=1)
    >>> for x_, y_ in zip(X, y):
    ...     est = est.partial_fit(x_.reshape(1, -1), np.array([y_]))
    >>> est.coef_
    array([32.89089478, 71.16073032])

    Furthermore, this model implements a `sample` method for sampling from the
    posterior distribution. Because the variance is unknown, the samples are
    drawn from the marginal posterior distribution of the weights, which is a
    multivariate t distribution.

    >>> est.sample(X[[0]], size=5)
    array([[15.01030526],
           [14.64281737],
           [15.21457505],
           [14.1703107 ],
           [14.57089036]])

    """

    def __init__(
        self,
        *,
        mu: ArrayLike = 0.0,
        lam: ArrayLike = 1.0,
        a=0.1,
        b=0.1,
        learning_rate=1.0,
        random_state: Union[int, np.random.Generator, None] = None,
    ):
        self.mu = mu
        self.lam = lam
        self.a = a
        self.b = b
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _initialize_prior(self, X: NDArray[Any]) -> None:
        if isinstance(self.random_state, int) or self.random_state is None:
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        self.n_features_ = X.shape[1]
        if np.isscalar(self.mu):
            self.coef_ = np.full(self.n_features_, self.mu, dtype=np.float_)
        elif cast(NDArray, self.mu).ndim == 1:
            self.coef_ = cast(NDArray[np.float_], self.mu)
        else:
            raise ValueError("The prior mean must be a scalar or vector.")

        if np.isscalar(self.lam):
            self.cov_inv_ = cast(np.float_, self.lam) * np.eye(self.n_features_)
        elif cast(NDArray[np.float_], self.lam).ndim == 1:
            self.cov_inv_ = np.diag(self.lam)
        elif cast(NDArray[np.float_], self.lam).ndim == 2:
            self.cov_inv_ = cast(NDArray[np.float_], self.lam)
        else:
            raise ValueError(
                "The prior covariance must be a scalar, vector, or matrix."
            )

        self.a_ = self.a
        self.b_ = self.b

    def _fit_helper(self, X: NDArray[Any], y: NDArray[Any]):
        # Apply the learning rate to the new data, if there are multiple observations
        # in the batch. This is done to ensure that one-at-a-time updates are
        # equivalent to batch updates.
        if len(X) > 1:
            obs_decays = np.flip(
                np.power(np.sqrt(self.learning_rate), np.arange(len(X)))
            )
            X = X * obs_decays[:, np.newaxis]
            y = y * obs_decays

        else:
            obs_decays = np.array([1.0])

        prior_decay = self.learning_rate ** len(X)

        # Update the inverse covariance matrix
        V_n = prior_decay * self.cov_inv_ + X.T @ X

        # Update the mean vector. Keeping track of the precision matrix
        # ensures we only need to do one LAPACK call.
        m_n = solve(
            V_n,
            prior_decay * self.cov_inv_ @ self.coef_ + X.T @ y,
            check_finite=False,
            assume_a="pos",
        )

        # Update the shape and rate parameters of the variance
        a_n = prior_decay * self.a_ + 0.5 * (obs_decays**2).sum()

        b_n = prior_decay * self.b_ + 0.5 * (
            y.T @ y
            + prior_decay * self.coef_.T @ self.cov_inv_ @ self.coef_
            - m_n.T @ V_n @ m_n
        )

        # Posteriors become priors for the next batch
        self.cov_inv_ = V_n
        # Delete the cached shape_ property so it is recalculated
        if hasattr(self, "shape_"):
            del self.shape_
        self.coef_ = m_n
        self.a_ = a_n
        self.b_ = b_n

    @cached_property
    def shape_(self) -> Covariance:
        extra_var = self.b_ / self.a_ * np.eye(self.n_features_)
        shape = solve(self.cov_inv_, extra_var, check_finite=False, assume_a="pos")
        return Covariance.from_cholesky(cholesky(shape, lower=True))

    def sample(self, X: NDArray[Any], size: int = 1) -> NDArray[np.float64]:
        """
        Sample from the coefficient marginal posterior at X. This is equivalent to
        sampling from a multivariate t distribution with the posterior mean and
        covariance, and degrees of freedom equal to 2 * a.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X)

        df = 2 * self.a_

        # Reimplement multivariate_t.rvs because scipy doesn't support
        # Covariance objects

        dim, loc, _, df = multivariate_t._process_parameters(
            self.coef_, self.shape_.covariance, df
        )

        x = self.random_state_.chisquare(df, size=size) / df

        z = multivariate_normal.rvs(
            np.zeros(dim),
            self.shape_,
            size=size,
            random_state=self.random_state_,
        )
        samples = loc + z / np.sqrt(x)[..., None]
        samples = _squeeze_output(samples)

        # End of multivariate_t.rvs implementation

        if self.n_features_ == 1:
            samples = np.expand_dims(samples, -1)

        return np.atleast_2d(samples @ X.T)  # type: ignore

    def decay(self, X: NDArray[Any], *, decay_rate: Optional[float] = None) -> None:
        """
        Decay the prior by a factor of `learning_rate`. This is equivalent to
        applying the learning rate to the prior, and then ignoring the data.
        It does not change the mean of the coefficient marginal posterior, but
        it does increase the variance.
        """

        if not hasattr(self, "coef_"):
            self._initialize_prior(X)

        if decay_rate is None:
            decay_rate = self.learning_rate

        prior_decay = decay_rate ** len(X)

        # decay only increases the variance, so we only need to update the
        # inverse covariance matrix, a_, and b_
        V_n = prior_decay * self.cov_inv_

        a_n = prior_decay * self.a_

        b_n = prior_decay * self.b_

        self.cov_inv_ = V_n
        self.a_ = a_n
        self.b_ = b_n
