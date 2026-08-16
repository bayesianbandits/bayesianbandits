from __future__ import annotations

from collections import defaultdict
from functools import cached_property, partial, wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.linalg.blas import dgemv, dsymv  # type: ignore
from scipy.sparse import csc_array, diags, eye, issparse
from scipy.special import expit
from scipy.stats import (
    Covariance,
    dirichlet,
    gamma,
    multivariate_normal,
)
from scipy.stats._multivariate import _squeeze_output
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin  # type: ignore
from sklearn.utils.validation import (
    NotFittedError,
    check_array,  # type: ignore
    check_is_fitted,
    check_X_y,  # type: ignore
)
from typing_extensions import Concatenate, ParamSpec, Self

from . import _support_covariance
from ._blas_helpers import (
    affine_lower_factor,
    compute_eta_dense,
    dense_matvec,
    lower_predictive_sqrt,
    standard_normal_f,
    update_precision_dense,
)
from ._gaussian import (
    LaplaceApproximator,
    LinkFunction,
    PosteriorApproximator,
    compute_effective_weights,
)
from ._np_utils import groupby_array
from ._sparse_bayesian_linear_regression import (
    DenseFactor,
    PrecisionFactor,
    SparseFactor,
    create_sparse_factor,
    multivariate_normal_sample_from_precision,
    multivariate_t_sample_from_precision,
    scale_factor,
)

Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")
SelfType = TypeVar("SelfType", bound="NormalRegressor | BayesianGLM")


class DirichletClassifier(BaseEstimator, ClassifierMixin):
    """
    Intercept-only Dirichlet-Multinomial classifier.

    Maintains a separate Dirichlet posterior over class probabilities
    for each unique value of the first feature. Supports sample
    weights (for importance-weighted updates in adversarial bandit
    algorithms) and online learning via ``partial_fit``.

    Parameters
    ----------
    alphas : dict of {int or str: float}
        Prior concentration parameters for each class. Keys define
        the set of classes; values are the initial Dirichlet
        :math:`\\alpha_k`. A uniform prior (e.g. ``{0: 1, 1: 1}``)
        encodes no prior preference.
    learning_rate : float, default=1.0
        Decay rate for the concentration parameters. Values less
        than 1 geometrically shrink the posterior on each call to
        ``decay``, increasing uncertainty over time. This converts
        the model into a forgetting estimator suitable for restless
        bandit problems.
    random_state : int, np.random.Generator, or None, default=None
        Controls the random number generator for ``sample``. Pass an
        int for reproducible results across calls.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels derived from the keys of ``alphas``.
    n_classes_ : int
        Number of classes.
    n_features_ : int
        Number of features seen during ``fit``. Must be 1.
    prior_ : ndarray of shape (n_classes,)
        Prior concentration parameters (values of ``alphas``).
    known_alphas_ : dict of {int or str: ndarray of shape (n_classes,)}
        Posterior concentration parameters for each observed feature
        value. Unseen feature values default to the prior.

    See Also
    --------
    GammaRegressor : Bayesian regression for positive continuous
        outcomes.
    NormalRegressor : Bayesian linear regression for real-valued
        outcomes.

    Notes
    -----
    This model implements the Dirichlet-Multinomial conjugate model
    described in Chapter 3 of [1]_. The posterior update is:

    .. math::

        \\alpha_k^{\\text{post}} = \\alpha_k^{\\text{prior}}
        + \\sum_{i=1}^{N} w_i \\, \\mathbb{1}[y_i = k]

    where :math:`w_i` are sample weights (defaulting to 1).

    The predictive distribution for class probabilities is:

    .. math::

        \\mathbb{E}[\\theta_k] = \\frac{\\alpha_k}{\\sum_j \\alpha_j}

    When ``learning_rate < 1``, calling ``decay`` scales all
    concentration parameters by the decay rate, uniformly increasing
    posterior uncertainty.

    References
    ----------
    .. [1] Murphy, Kevin P. "Machine Learning: A Probabilistic
       Perspective." MIT Press, 2012.

    Examples
    --------
    Basic classification with a uniform prior:

    >>> from bayesianbandits import DirichletClassifier
    >>> X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(-1, 1)
    >>> y = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
    >>> clf = DirichletClassifier({1: 1, 2: 1, 3: 1}, random_state=0)
    >>> clf.fit(X, y)
    DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)

    Using sample weights for importance-weighted updates:

    >>> weights = np.array([2.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 2.0])
    >>> clf.fit(X, y, sample_weight=weights)
    DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)

    Online learning with ``partial_fit``:

    >>> clf.partial_fit(X, y, sample_weight=weights)
    DirichletClassifier(alphas={1: 1, 2: 1, 3: 1}, random_state=0)
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

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """
        Fit the model from scratch, resetting the prior.

        Initializes the Dirichlet prior from ``alphas`` and updates the
        posterior concentration parameters using the observed class
        counts (optionally weighted).

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training data. Only the first column is used; each unique
            value indexes a separate Dirichlet posterior.
        y : array-like of shape (n_samples,)
            Class labels. Must be a subset of the keys in ``alphas``.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples
            are given weight 1.0.

        Returns
        -------
        self : DirichletClassifier
            Fitted estimator.

        See Also
        --------
        partial_fit : Incremental update without resetting the prior.
        """
        X, y = check_X_y(X, y, copy=True, ensure_2d=True)

        self._initialize_prior()

        y_encoded: NDArray[np.int_] = (y[:, np.newaxis] == self.classes_).astype(int)

        self.n_features_ = X.shape[1]

        if self.n_features_ > 1:
            raise NotImplementedError("Only one feature supported")

        self._fit_helper(X, y_encoded, sample_weight)

        return self

    def _initialize_prior(self) -> None:
        if isinstance(self.random_state, int):
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        self.classes_ = np.array(list(self.alphas.keys()))
        self.n_classes_ = len(self.classes_)
        self.prior_ = np.array(list(self.alphas.values()), dtype=np.float64)

        self.known_alphas_: Dict[Any, NDArray[np.float64]] = defaultdict(
            self._return_prior
        )

    def _return_prior(self) -> NDArray[np.float64]:
        return self.prior_

    def partial_fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ):
        """
        Incrementally update the model with new observations.

        Uses the current posterior as the prior for the new update.
        If the model has not been fitted yet, this is equivalent to
        calling ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training data. Only the first column is used; each unique
            value indexes a separate Dirichlet posterior.
        y : array-like of shape (n_samples,)
            Class labels. Must be a subset of the keys in ``alphas``.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples
            are given weight 1.0.

        Returns
        -------
        self : DirichletClassifier
            Updated estimator.

        See Also
        --------
        fit : Fit from scratch, resetting the prior.
        """
        try:
            check_is_fitted(self, "n_features_")
        except NotFittedError:
            return self.fit(X, y, sample_weight)

        X_fit, y = check_X_y(X, y, copy=True, ensure_2d=True)
        y = (y[:, np.newaxis] == self.classes_).astype(int)

        self._fit_helper(X_fit, y, sample_weight)
        return self

    def _fit_helper(
        self, X: NDArray[Any], y: NDArray[Any], sample_weight: Optional[NDArray[Any]]
    ):
        # Handle sample weights
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"sample_weight.shape[0]={sample_weight.shape[0]} should be "
                    f"equal to X.shape[0]={X.shape[0]}"
                )

        # Group X values, y, and sample weights together
        for group, arr, weights in groupby_array(X[:, 0], y, sample_weight, by=X[:, 0]):
            key = group[0].item()

            # Apply sample weights to the observations
            weighted_arr = arr * weights[:, np.newaxis]

            # Stack with prior
            vals = np.vstack((self.known_alphas_[key], weighted_arr))

            # Apply learning rate decay
            decay_idx = np.flip(np.arange(len(vals)))
            posterior = vals * (self.learning_rate**decay_idx)[:, np.newaxis]
            self.known_alphas_[key] = posterior.sum(axis=0)

    def predict_proba(self, X: NDArray[Any]) -> Any:
        """
        Predict class probabilities using the posterior mean.

        Computes the expected class probabilities under the Dirichlet
        posterior:
        :math:`\\mathbb{E}[\\theta_k] = \\alpha_k / \\sum_j \\alpha_j`.

        If the model has not been fitted, returns the prior mean.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input features. Each unique value of ``X[:, 0]`` indexes a
            separate Dirichlet posterior.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities. Each row sums to 1.

        See Also
        --------
        predict : Return the most likely class label.
        sample : Draw from the Dirichlet posterior.
        """
        try:
            check_is_fitted(self, "n_features_")
        except NotFittedError:
            self._initialize_prior()

        X_pred = check_array(X, copy=True, ensure_2d=True)

        alphas = np.vstack(list(self.known_alphas_[x.item()] for x in X_pred))
        return alphas / alphas.sum(axis=1)[:, np.newaxis]

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict the most likely class for each sample.

        Returns the class with the highest posterior mean probability
        :math:`\\arg\\max_k \\alpha_k`.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input features. Each unique value of ``X[:, 0]`` indexes a
            separate Dirichlet posterior.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.

        See Also
        --------
        predict_proba : Return full probability vectors.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def sample(self, X: NDArray[Any], size: int = 1) -> NDArray[np.float64]:
        """
        Sample class probability vectors from the Dirichlet posterior.

        For each input, draws from
        :math:`\\text{Dir}(\\alpha_1, \\ldots, \\alpha_K)` where the
        :math:`\\alpha_k` are the posterior concentration parameters
        for that input's group. If the model has not been fitted,
        samples from the prior.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input features. Each unique value of ``X[:, 0]`` indexes a
            separate Dirichlet posterior.
        size : int, default=1
            Number of independent draws from each posterior.

        Returns
        -------
        samples : ndarray of shape (size, n_samples, n_classes)
            Sampled probability vectors. Each sample along the last
            axis sums to 1.

        See Also
        --------
        predict_proba : Point estimate using the posterior mean.
        """
        try:
            check_is_fitted(self, "n_features_")
        except NotFittedError:
            self._initialize_prior()

        alphas = list(self.known_alphas_[x.item()] for x in X)
        return np.stack(
            list(dirichlet.rvs(alpha, size, self.random_state_) for alpha in alphas),
        ).transpose(1, 0, 2)

    def decay(self, X: NDArray[Any], *, decay_rate: Optional[float] = None) -> None:
        """
        Decay concentration parameters to increase uncertainty.

        Scales the posterior concentration parameters
        :math:`\\alpha_k \\leftarrow \\gamma \\, \\alpha_k` for each
        group present in ``X``. This uniformly increases posterior
        variance, allowing the model to adapt to non-stationary
        environments.

        Has no effect if the model has not been fitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input features. Each unique value of ``X[:, 0]`` identifies
            a group whose concentration parameters are decayed.
        decay_rate : float, default=None
            Multiplicative decay factor :math:`\\gamma` in (0, 1].
            If None, uses ``self.learning_rate``.

        See Also
        --------
        partial_fit : Update the model with new observations.
        """
        if not hasattr(self, "known_alphas_"):
            self._initialize_prior()

        if decay_rate is None:
            decay_rate = self.learning_rate

        for x in X:
            self.known_alphas_[x.item()] *= decay_rate


class GammaRegressor(BaseEstimator, RegressorMixin):
    """
    Intercept-only Gamma-Poisson conjugate regression model.

    Maintains a separate Gamma posterior over the rate parameter
    :math:`\\lambda` for each unique value of the first feature.
    Designed for modeling count data where the rate may differ
    across groups. Supports sample weights (for
    importance-weighted updates in adversarial bandit algorithms)
    and online learning via ``partial_fit``.

    Parameters
    ----------
    alpha : float
        Prior shape parameter of the Gamma distribution. The prior
        is :math:`\\lambda \\sim \\text{Gamma}(\\alpha, \\beta)`.
        Larger values concentrate the prior more tightly around the
        prior mean :math:`\\alpha / \\beta`.
    beta : float
        Prior rate (inverse scale) parameter. Together with
        ``alpha``, determines the prior mean
        :math:`\\mathbb{E}[\\lambda] = \\alpha / \\beta` and prior
        variance :math:`\\text{Var}[\\lambda] = \\alpha / \\beta^2`.
    learning_rate : float, default=1.0
        Decay rate for the posterior parameters. Values less than 1
        geometrically shrink both :math:`\\alpha` and :math:`\\beta`
        on each call to ``decay``, increasing posterior variance
        while preserving the mean. This converts the model into a
        forgetting estimator suitable for restless bandit problems.
    random_state : int, np.random.Generator, or None, default=None
        Controls the random number generator for ``sample``. Pass an
        int for reproducible results across calls.

    Attributes
    ----------
    coef_ : dict of {int or float: ndarray of shape (2,)}
        Posterior parameters ``[alpha, beta]`` for each observed
        feature value. Unseen feature values default to the prior.
    n_features_ : int
        Number of features seen during ``fit``. Must be 1.
    prior_ : ndarray of shape (2,)
        Prior parameters ``[alpha, beta]``.

    See Also
    --------
    DirichletClassifier : Bayesian classification for categorical
        outcomes.
    NormalRegressor : Bayesian linear regression for real-valued
        outcomes.

    Notes
    -----
    This model implements the Gamma-Poisson conjugate model. Given
    observations :math:`y_i` with sample weights :math:`w_i`, the
    posterior update is [1]_:

    .. math::

        \\alpha^{\\text{post}} = \\alpha^{\\text{prior}}
        + \\sum_{i=1}^{N} w_i \\, y_i, \\qquad
        \\beta^{\\text{post}} = \\beta^{\\text{prior}}
        + \\sum_{i=1}^{N} w_i

    The posterior mean rate is
    :math:`\\mathbb{E}[\\lambda] = \\alpha / \\beta` and the
    posterior variance is
    :math:`\\text{Var}[\\lambda] = \\alpha / \\beta^2`.

    When ``learning_rate < 1``, calling ``decay`` scales both
    :math:`\\alpha` and :math:`\\beta` equally, so the posterior
    mean is preserved but the variance increases.

    References
    ----------
    .. [1] Murphy, Kevin P. "Machine Learning: A Probabilistic
       Perspective." MIT Press, 2012.

    Examples
    --------
    Basic rate estimation:

    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> model = GammaRegressor(alpha=1, beta=1, random_state=0)
    >>> model.fit(X, y)
    GammaRegressor(alpha=1, beta=1, random_state=0)

    Using sample weights for importance-weighted updates:

    >>> weights = np.array([1.0, 2.0, 1.0, 0.5, 1.5])
    >>> model.fit(X, y, sample_weight=weights)
    GammaRegressor(alpha=1, beta=1, random_state=0)

    Online learning with ``partial_fit``:

    >>> model.partial_fit(X, y, sample_weight=weights)
    GammaRegressor(alpha=1, beta=1, random_state=0)
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

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """
        Fit the model from scratch, resetting the prior.

        Initializes the Gamma prior from ``alpha`` and ``beta`` and
        updates the posterior parameters using the observed counts
        (optionally weighted).

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training data. Only the first column is used; each unique
            value indexes a separate Gamma posterior.
        y : array-like of shape (n_samples,)
            Target values (non-negative integer counts).
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples
            are given weight 1.0.

        Returns
        -------
        self : GammaRegressor
            Fitted estimator.

        See Also
        --------
        partial_fit : Incremental update without resetting the prior.
        """
        X, y = check_X_y(X, y, copy=True, ensure_2d=True)

        self._initialize_prior()

        y_encoded: NDArray[np.int_] = y.astype(int)

        self.n_features_ = X.shape[1]

        if self.n_features_ > 1:
            raise NotImplementedError("Only one feature supported")

        self._fit_helper(X, y_encoded, sample_weight)

        return self

    def _initialize_prior(self) -> None:
        if isinstance(self.random_state, int):
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        self.prior_ = np.array([self.alpha, self.beta], dtype=np.float64)

        self.coef_: Dict[Any, NDArray[np.float64]] = defaultdict(self._return_prior)

    def _return_prior(self) -> NDArray[np.float64]:
        return self.prior_

    def _fit_helper(
        self, X: NDArray[Any], y: NDArray[Any], sample_weight: Optional[NDArray[Any]]
    ):
        # Handle sample weights
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"sample_weight.shape[0]={sample_weight.shape[0]} should be "
                    f"equal to X.shape[0]={X.shape[0]}"
                )

        # Group X values, y, and sample weights together
        for group, arr, weights in groupby_array(X[:, 0], y, sample_weight, by=X[:, 0]):
            key = group[0].item()

            # The update is computed by stacking the prior with the weighted data
            # For gamma-poisson: alpha increases by weighted counts, beta by weights
            weighted_counts = arr * weights
            weighted_data = np.column_stack((weighted_counts, weights))

            vals = np.vstack((self.coef_[key], weighted_data))

            # Calculate the decay index for nonstationary models
            decay_idx = np.flip(np.arange(len(vals)))
            # Calculate the posterior
            posterior = vals * (self.learning_rate**decay_idx)[:, np.newaxis]
            # Calculate the coefficient
            self.coef_[key] = posterior.sum(axis=0)

    def partial_fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ):
        """
        Incrementally update the model with new observations.

        Uses the current posterior as the prior for the new update.
        If the model has not been fitted yet, this is equivalent to
        calling ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training data. Only the first column is used; each unique
            value indexes a separate Gamma posterior.
        y : array-like of shape (n_samples,)
            Target values (non-negative integer counts).
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples
            are given weight 1.0.

        Returns
        -------
        self : GammaRegressor
            Updated estimator.

        See Also
        --------
        fit : Fit from scratch, resetting the prior.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            return self.fit(X, y, sample_weight)

        X_fit, y = check_X_y(X, y, copy=True, ensure_2d=True)
        y_encoded: NDArray[np.int_] = y.astype(int)

        self._fit_helper(X_fit, y_encoded, sample_weight)
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict the posterior mean rate for each sample.

        Returns the mean of the Gamma posterior
        :math:`\\mathbb{E}[\\lambda] = \\alpha / \\beta`.

        If the model has not been fitted, returns the prior mean.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input features. Each unique value of ``X[:, 0]`` indexes a
            separate Gamma posterior.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted rates (posterior means).

        See Also
        --------
        sample : Draw from the Gamma posterior.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior()

        X_pred = check_array(X, copy=True, ensure_2d=True)

        shape_params = np.vstack(list(self.coef_[x.item()] for x in X_pred))
        return shape_params[:, 0] / shape_params[:, 1]

    def sample(self, X: NDArray[Any], size: int = 1) -> NDArray[np.float64]:
        """
        Sample rates from the Gamma posterior.

        For each input, draws from
        :math:`\\text{Gamma}(\\alpha, \\beta)` where :math:`\\alpha`
        and :math:`\\beta` are the posterior shape and rate parameters
        for that input's group. If the model has not been fitted,
        samples from the prior.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input features. Each unique value of ``X[:, 0]`` indexes a
            separate Gamma posterior.
        size : int, default=1
            Number of independent draws from each posterior.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Sampled rates from the Gamma posterior.

        See Also
        --------
        predict : Point estimate using the posterior mean.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior()

        shape_params = list(self.coef_[x.item()] for x in X)

        rv_gen = partial(gamma.rvs, size=size, random_state=self.random_state_)

        return np.stack(
            list(rv_gen(alpha, scale=1 / beta) for alpha, beta in shape_params),
        ).T

    def decay(self, X: NDArray[Any], *, decay_rate: Optional[float] = None) -> None:
        """
        Decay posterior parameters to increase uncertainty.

        Scales both shape and rate by the decay factor:
        :math:`\\alpha \\leftarrow \\gamma \\alpha,\\;
        \\beta \\leftarrow \\gamma \\beta`. Because both parameters
        are scaled equally, the posterior mean
        :math:`\\alpha / \\beta` is preserved but the variance
        :math:`\\alpha / \\beta^2` increases.

        Has no effect if the model has not been fitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input features. Each unique value of ``X[:, 0]`` identifies
            a group whose parameters are decayed.
        decay_rate : float, default=None
            Multiplicative decay factor :math:`\\gamma` in (0, 1].
            If None, uses ``self.learning_rate``.

        See Also
        --------
        partial_fit : Update the model with new observations.
        """
        if not hasattr(self, "coef_"):
            self._initialize_prior()

        if decay_rate is None:
            decay_rate = self.learning_rate

        for x in X:
            self.coef_[x.item()] *= decay_rate


def _invalidate_cached_properties(
    func: Callable[Concatenate[SelfType, Params], ReturnType],  # type: ignore
) -> Callable[Concatenate[SelfType, Params], ReturnType]:
    @wraps(func)
    def wrapper(self, *args: Params.args, **kwargs: Params.kwargs) -> ReturnType:
        for attr in ("shape_", "cov_"):
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        return func(self, *args, **kwargs)

    return wrapper


# Cap on the dense (n_features, block) scratch densified from a sparse
# X in _marginal_predictive_sd: ~2M float64 elements = 16 MB per array.
_MARGINAL_SD_BLOCK_ELEMS = 2**21


def _marginal_predictive_sd(
    factor: PrecisionFactor, X: Union[NDArray[Any], csc_array]
) -> NDArray[np.float64]:
    """Per-row predictive standard deviations :math:`\\sqrt{x_i^T \\Lambda^{-1} x_i}`.

    Computes ``B = factor.half_solve(X.T)`` -- one triangular solve per
    prediction row against the cached precision factor, so ``B.T @ B``
    equals :math:`X \\Lambda^{-1} X^T` -- and takes column norms.
    Neither :math:`\\Lambda^{-1}` nor any :math:`n \\times n` matrix is
    ever formed; the cost is linear in the number of prediction rows.
    A sparse ``X`` is densified for the solve in row blocks, keeping the
    dense scratch bounded regardless of the number of prediction rows.

    When ``X`` is sparse and touches fewer distinct columns than it has
    rows, the support-covariance route is exact and cheaper -- it pays
    one solve per distinct column instead of one per row -- so it is
    used instead. See :mod:`bayesianbandits._support_covariance`.
    """
    # Dispatch on the actual type of X: sparse models accept dense X too
    # (check_array's accept_sparse only *permits* sparse input)
    if not issparse(X):
        rhs = np.asarray(X, dtype=np.float64).T
        B = np.asarray(factor.half_solve(rhs), dtype=np.float64)
        return cast(NDArray[np.float64], np.sqrt(np.einsum("ij,ij->j", B, B)))
    assert X.shape is not None  # for the type checker
    n_rows, n_features = X.shape
    support_draw = _support_covariance.build(factor, X, n_features, budget=n_rows)
    if support_draw is not None:
        return support_draw.sd()
    block = max(1, _MARGINAL_SD_BLOCK_ELEMS // max(1, n_features))
    # Rows are CSC's minor axis, so slicing rows per chunk would scan all
    # of X's nnz each chunk; slice from a CSR copy when chunking
    X_rows: Any = cast(csc_array, X).tocsr() if n_rows > block else X
    var = np.empty(n_rows, dtype=np.float64)
    for start in range(0, n_rows, block):
        stop = min(n_rows, start + block)
        rhs = np.asarray(X_rows[start:stop].T.toarray(), dtype=np.float64)
        B = np.asarray(factor.half_solve(rhs), dtype=np.float64)
        var[start:stop] = np.einsum("ij,ij->j", B, B)
    return cast(NDArray[np.float64], np.sqrt(var))


def _validated_marginal_mean_sd(
    est: Any, X: Union[NDArray[Any], csc_array]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shared ``sample_marginal`` preamble for the linear/GLM estimators:
    validate ``X``, initialize the prior if the model is unfitted, and
    return the per-row predictive mean and standard deviation of the
    linear predictor ``X @ w``."""
    X_pred = check_array(
        X, copy=False, ensure_2d=True, accept_sparse="csc" if est.sparse else False
    )
    try:
        check_is_fitted(est, "coef_")
    except NotFittedError:
        est._initialize_prior(X_pred)
    mean = np.asarray(X_pred @ est.coef_, dtype=np.float64).ravel()
    sd = _marginal_predictive_sd(est._precision_factor, X_pred)
    return mean, sd


def _reward_space_is_cheaper(
    n: int,
    d: int,
    size: int,
    sparse: bool,
    precision_nnz: int,
    block_size: Optional[int] = None,
) -> bool:
    """Is reward-space sampling cheaper than weight-space for this call?

    Weight-space draws ``size`` weight vectors from the ``d``-dimensional
    posterior and projects them through ``X``.  Reward-space computes an
    exact square root of the predictive covariance
    :math:`X \\Lambda^{-1} X^T` once (``n`` triangular half-solves plus a
    QR) and draws directly in reward space, so its per-draw cost is
    independent of ``d``.

    The decision rests on the one quantity both paths pay in the same
    currency: **triangular solves against the cached factor**.
    Reward-space performs ``n`` of them, weight-space ``size``.  Both are
    exactly known integers, so comparing them needs no calibration.
    Requiring ``2n <= size`` rather than ``n < size`` leaves the other
    half of the weight-space solve budget to cover reward-space's
    remaining work, which the guards below then check fits.

    With ``k`` rows per QR block (``k = n`` in full mode), that remaining
    work is ``n·k·d`` for the QR and ``n·k·size`` for the draws:

    - Dense.  Weight-space spends ``d²·size`` colorizing, so the draw
      term fits when ``2·n·k <= d²``.  This is what stops the path being
      taken at ``n ≈ d``, where the two do the same work but
      reward-space does it in many small, latency-bound calls rather
      than one large one.  ``n <= d`` additionally keeps reward-space's
      ``n·size`` normals under weight-space's ``d·size``.
    - Sparse.  A solve costs ``nnz``, not ``d²``, so the draw term fits
      when ``2·n·k <= nnz``.  Sparse also densifies a ``(d, k)`` scratch
      for the half-solves, which is the only allocation here that can
      exceed the factor itself, so it is held to the same element budget
      the marginal path uses.  That budget, not the cost model, is what
      declines large ``n`` at high ``d``; ``block_size`` is the way to
      take those calls.
    """
    k = n if block_size is None else block_size
    if 2 * n > size:
        return False
    if sparse:
        return k * d <= _MARGINAL_SD_BLOCK_ELEMS and 2 * n * k <= precision_nnz
    return n <= d and 2 * n * k <= d * d


def _predictive_cholesky_from_factor(
    factor: PrecisionFactor,
    coef: NDArray[np.float64],
    X: Union[NDArray[Any], csc_array],
    block_size: Optional[int] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Predictive mean and lower-triangular square root(s) of :math:`X \\Lambda^{-1} X^T`.

    Computes ``B = factor.half_solve(X.T)`` -- one triangular solve per
    prediction row against the cached factor, so ``B.T @ B`` equals the
    predictive covariance without :math:`\\Lambda^{-1}` or the covariance
    itself ever being formed -- and takes the ``R`` factor of its QR
    decomposition: ``R.T @ R = B.T @ B``, making ``R.T`` an exact lower
    triangular square root even when ``X`` has linearly dependent rows
    (e.g. repeated contexts) -- unlike a Cholesky of the explicit
    covariance, which would need regularization there and would square
    the condition number.

    With ``block_size=k``, rows are treated as consecutive groups of
    ``k`` and the QR is batched per group, returning one ``(k, k)``
    factor per group with shape ``(n_blocks, k, k)``: exact joint
    structure within each group, none across groups. This is the shape
    a caller wants when rows group into independent decision units
    (e.g. one block of arm rows per context). Blocked mode processes
    rows in chunks so the dense half-solve scratch stays within the
    same ``_MARGINAL_SD_BLOCK_ELEMS`` budget ``_marginal_predictive_sd``
    uses, regardless of the total row count. Full mode has no such cap:
    a sparse ``X`` is densified transposed all at once, so the scratch
    is a dense ``(n_features, n_rows)`` array -- pass ``block_size`` to
    bound memory when both dimensions are large.
    """

    def half_solve_rows(rows: slice) -> NDArray[np.float64]:
        # Dispatch on the actual type: sparse models accept dense X too
        # (check_array's accept_sparse only *permits* sparse input)
        X_rows = X_row_sliceable[rows]
        if issparse(X_rows):
            rhs = np.asarray(X_rows.T.toarray(), dtype=np.float64)
        else:
            rhs = np.asarray(X_rows, dtype=np.float64).T
        return np.asarray(factor.half_solve(rhs), dtype=np.float64)

    # Sparse products never enter a BLAS thread pool, so only the dense
    # case needs routing away from numpy.
    if issparse(X):
        mean = np.asarray(X @ coef, dtype=np.float64).ravel()
    else:
        mean = dense_matvec(X, coef)
    n = mean.shape[0]
    X_row_sliceable: Any = X

    if block_size is None:
        B = half_solve_rows(slice(0, n)).reshape(-1, n)
        return mean, lower_predictive_sqrt(B, n)

    if block_size <= 0 or n % block_size:
        raise ValueError(
            f"block_size={block_size} must be positive and divide the "
            f"number of prediction rows ({n})"
        )
    n_blocks = n // block_size
    # Only the small (block_size, block_size) factors survive, so bound
    # the dense (n_features, chunk_rows) half-solve scratch by chunking
    # over consecutive blocks
    chunk_blocks = max(1, _MARGINAL_SD_BLOCK_ELEMS // max(1, X.shape[1] * block_size))
    if issparse(X) and n_blocks > chunk_blocks:
        # Rows are CSC's minor axis, so slicing rows per chunk would scan
        # all of X's nnz each chunk; slice from a CSR copy when chunking
        X_row_sliceable = cast(csc_array, X).tocsr()
    L = np.empty((n_blocks, block_size, block_size))
    for start in range(0, n_blocks, chunk_blocks):
        stop = min(start + chunk_blocks, n_blocks)
        B = half_solve_rows(slice(start * block_size, stop * block_size))
        B = B.reshape(-1, (stop - start) * block_size)
        blocks = B.reshape(B.shape[0], stop - start, block_size).transpose(1, 0, 2)
        # One factorization per block rather than numpy's batched QR: the
        # batched call is a single trip into numpy's OpenBLAS, and the
        # round trip out of scipy's costs more than this loop does. See
        # ``lower_predictive_sqrt``.
        for offset, block in enumerate(blocks):
            L[start + offset] = lower_predictive_sqrt(block, block_size)
    return mean, L


def _blocked_colorize(
    L: NDArray[np.float64], z: NDArray[np.floating[Any]]
) -> NDArray[np.float64]:
    """Apply per-block lower factors: ``(L_b @ z_{s,b})`` for every draw.

    ``L`` has shape ``(n_blocks, k, k)``, ``z`` has shape
    ``(n_blocks, size, k)``; returns shape ``(size, n_blocks * k)``. The
    result is written straight into a C-order ``(size, n_blocks, k)``
    array, so neither the draw tensor nor the result is copied through
    a transposed intermediate.
    """
    n_blocks, size, k = z.shape
    out = np.empty((size, n_blocks, k))
    np.matmul(z, L.transpose(0, 2, 1), out=out.transpose(1, 0, 2))
    # explicit column count: reshape(size, -1) is ambiguous when size == 0
    return out.reshape(size, n_blocks * k)


class _RewardSpacePredictiveMixin:
    """Reward-space sampling support shared by the conjugate linear models.

    Hosts the predictive-Cholesky construction, the flop-model gate, and
    the shared ``sample``/``sample_reward_space`` validation preamble
    over the ``coef_``/``_precision_factor``/``sparse`` contract that
    :class:`NormalRegressor` and :class:`BayesianGLM` share.
    """

    if TYPE_CHECKING:
        coef_: NDArray[np.float64]
        cov_inv_: Union[NDArray[np.float64], csc_array]
        n_features_: int
        sparse: bool

        @property
        def _precision_factor(self) -> PrecisionFactor: ...

        def _initialize_prior(self, X: Union[NDArray[Any], csc_array]) -> None: ...

    def _predictive_cholesky(
        self, X: Union[NDArray[Any], csc_array], block_size: Optional[int] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predictive mean and lower square root(s) of the linear
        predictor's covariance :math:`X \\Lambda^{-1} X^T`. Assumes ``X``
        is already validated and the model is fitted (or the prior
        initialized). For the NIG subclass this is the *unscaled* factor:
        the ``b/a`` scaling is applied by ``sample_reward_space``."""
        return _predictive_cholesky_from_factor(
            self._precision_factor, self.coef_, X, block_size
        )

    def _use_reward_space(
        self, n: int, size: int, block_size: Optional[int] = None
    ) -> bool:
        """Flop-count test: is reward-space sampling cheaper for this call?"""
        if not hasattr(self, "n_features_"):
            # Unfitted: no cached factor or dimensions to reason about
            # yet, and the prior predictive is cheap either way
            return False
        nnz = cast(csc_array, self.cov_inv_).nnz if self.sparse else 0
        return _reward_space_is_cheaper(
            n, self.n_features_, size, self.sparse, nnz, block_size
        )

    def _validated_for_sampling(
        self, X: Union[NDArray[Any], csc_array]
    ) -> Union[NDArray[Any], csc_array]:
        """Shared ``sample``/``sample_reward_space`` preamble: validate
        ``X`` and initialize the prior if the model is unfitted."""
        X_pred = check_array(
            X, copy=True, ensure_2d=True, accept_sparse="csc" if self.sparse else False
        )
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X_pred)
        return X_pred


class NormalRegressor(_RewardSpacePredictiveMixin, BaseEstimator, RegressorMixin):
    """
    Bayesian linear regression with known noise variance.

    Places a Gaussian prior on the weight vector and performs exact
    conjugate updates. Supports both dense and sparse feature matrices,
    online learning via ``partial_fit``, and non-stationary environments
    via ``decay``.

    Parameters
    ----------
    alpha : float
        Prior precision for the weights. The prior is
        :math:`w \\sim \\mathcal{N}(0, \\alpha^{-1} I)`. Higher values
        give stronger regularization toward zero.
    beta : float
        Known noise precision. The likelihood is
        :math:`y \\mid x, w \\sim \\mathcal{N}(x^T w, \\beta^{-1})`.
    learning_rate : float, default=1.0
        Decay rate for the posterior precision on each call to
        ``decay``. Values less than 1 geometrically shrink the
        precision matrix, increasing posterior uncertainty over time.
        This converts the model into a forgetting estimator suitable
        for restless bandit problems.
    sparse : bool, default=False
        If True, use sparse matrix operations for the precision
        matrix. Input ``X`` must be a ``scipy.sparse.csc_array``.
        When CHOLMOD is available (via ``scikit-sparse``), it is used
        for efficient Cholesky factorization; otherwise falls back to
        UMFPACK (``scikit-umfpack``) or SuperLU.
    random_state : int, np.random.Generator, or None, default=None
        Controls the random number generator for ``sample``. Pass an
        int for reproducible results across calls.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Posterior mean of the weight vector.
    cov_inv_ : ndarray of shape (n_features, n_features) or \
scipy.sparse.csc_array
        Posterior precision matrix (inverse covariance).
    n_features_ : int
        Number of features seen during ``fit``.

    See Also
    --------
    NormalInverseGammaRegressor : Bayesian linear regression with unknown
        noise variance (marginal posterior is a multivariate t).
    EmpiricalBayesNormalRegressor : Automatic hyperparameter tuning via
        evidence maximization.
    BayesianGLM : Bayesian GLM for non-Gaussian likelihoods.

    Notes
    -----
    This model implements the "known variance" Bayesian linear regression
    formulation described in Chapter 7 of [1]_. The posterior is:

    .. math::

        \\Lambda_n = \\gamma^n \\Lambda_0 + \\beta X^T W X, \\qquad
        \\mu_n = \\Lambda_n^{-1}
        (\\gamma^n \\Lambda_0 \\mu_0 + \\beta X^T W y)

    where :math:`\\gamma` is the learning rate (1.0 for standard
    Bayesian update) and :math:`W` is a diagonal matrix of effective
    sample weights incorporating both user-supplied weights and
    learning-rate decay.

    When ``learning_rate < 1``, calling ``decay`` scales the precision
    matrix by :math:`\\gamma^n`, uniformly increasing posterior
    uncertainty while preserving the mean.

    References
    ----------
    .. [1] Murphy, Kevin P. "Machine Learning: A Probabilistic
       Perspective." MIT Press, 2012.

    Examples
    --------
    Basic linear regression:

    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> model = NormalRegressor(alpha=0.1, beta=1, random_state=0)
    >>> model.fit(X, y)
    NormalRegressor(alpha=0.1, beta=1, random_state=0)
    >>> model.predict(X)
    array([0.99818512, 1.99637024, 2.99455535, 3.99274047, 4.99092559])

    The posterior mean weights are stored in ``coef_``:

    >>> model.coef_
    array([0.99818512])

    Online learning with ``partial_fit``:

    >>> model.partial_fit(X, y)
    NormalRegressor(alpha=0.1, beta=1, random_state=0)
    >>> model.predict(X)
    array([0.99909173, 1.99818347, 2.9972752 , 3.99636694, 4.99545867])

    Sampling from the posterior predictive:

    >>> model.sample(X)
    array([[1.0110742 , 2.02214839, 3.03322259, 4.04429678, 5.05537098]])
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        *,
        learning_rate: float = 1.0,
        sparse: bool = False,
        random_state: Union[int, np.random.Generator, None] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.sparse = sparse
        self.random_state = random_state

    @_invalidate_cached_properties
    def __getstate__(self) -> Any:
        # Exclude cached C extension objects that cannot be pickled
        state = super().__getstate__()  # type: ignore
        state.pop("_precision_factor", None)
        return state

    def fit(
        self,
        X_fit: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """
        Fit the model from scratch, resetting the prior.

        Initializes the prior
        :math:`w \\sim \\mathcal{N}(0, \\alpha^{-1} I)` and computes
        the exact posterior. Any previously learned parameters are
        discarded.

        Parameters
        ----------
        X_fit : array-like of shape (n_samples, n_features)
            Training data. Must be a ``scipy.sparse.csc_array`` when
            ``sparse=True``.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples
            are given equal weight.

        Returns
        -------
        self : NormalRegressor
            Fitted estimator.

        See Also
        --------
        partial_fit : Incremental update without resetting the prior.
        """
        X_fit, y = check_X_y(
            X_fit,  # type: ignore
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        self._initialize_prior(X_fit)
        self._fit_helper(X_fit, y, sample_weight)
        return self

    def _initialize_prior(self, X: Union[NDArray[Any], csc_array]) -> None:
        if isinstance(self.random_state, int) or self.random_state is None:
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        assert X.shape is not None  # for the type checker
        self.n_features_ = X.shape[1]
        self.coef_ = np.zeros(self.n_features_)
        if self.sparse:
            self.cov_inv_ = csc_array(eye(self.n_features_, format="csc")) * self.alpha
        else:
            self.cov_inv_ = np.eye(self.n_features_) * self.alpha

    @cached_property
    def _precision_factor(self) -> PrecisionFactor:
        """Factorization of the precision matrix (cached).

        Returns a ``DenseFactor`` (dense) or ``SparseFactor`` (sparse).
        Lazily computed on first access; eagerly set by ``_fit_helper``
        when the factorization is a free byproduct of the solve.
        Invalidated by ``_invalidate_cached_properties``.
        """
        if self.sparse:
            assert isinstance(self.cov_inv_, csc_array)
            return create_sparse_factor(self.cov_inv_)
        else:
            cho = cho_factor(self.cov_inv_, lower=False, check_finite=False)
            return DenseFactor(_U=cho[0], _n_features=cho[0].shape[0])

    @cached_property
    def cov_(self) -> Union[Covariance, SparseFactor]:
        """Posterior covariance matrix (cached, lazily computed).

        Returns a ``scipy.stats.Covariance`` object (dense) or a
        ``SparseFactor`` (sparse) wrapping the Cholesky factorization.
        Automatically invalidated when the model is updated via
        ``fit``, ``partial_fit``, or ``decay``.

        .. warning::

           For dense models, this is an :math:`O(p^3)` computation
           with :math:`O(p^2)` memory.
        """
        factor = self._precision_factor
        if self.sparse:
            assert isinstance(factor, SparseFactor)
            return factor
        else:
            assert isinstance(factor, DenseFactor)
            cov = factor.solve(np.eye(self.n_features_))
        return Covariance.from_cholesky(cholesky(cov, lower=True))

    @_invalidate_cached_properties
    def _fit_helper(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ):
        if self.sparse:
            X = csc_array(X)

        assert X.shape is not None  # for the type checker

        # Handle sample weights
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"sample_weight.shape[0]={sample_weight.shape[0]} should be "
                    f"equal to X.shape[0]={X.shape[0]}"
                )

        # Apply the learning rate decay to get effective weights
        effective_weights = compute_effective_weights(
            X.shape[0], sample_weight, self.learning_rate
        )

        assert X.shape is not None  # for the type checker
        prior_decay = self.learning_rate ** X.shape[0]

        # Apply weights to X and y
        if self.sparse:
            # Element-wise scaling avoids constructing a diagonal matrix.
            # Absorb beta into weights so X_weighted^T @ X_weighted = beta * X^T W X
            assert isinstance(X, csc_array)
            w_sqrt = np.sqrt(self.beta * effective_weights)
            X_weighted = X.multiply(w_sqrt.reshape(-1, 1)).tocsc()
            cov_inv = cast(
                csc_array,
                prior_decay * self.cov_inv_ + X_weighted.T @ X_weighted,
            )
        else:
            # For dense matrices, use broadcasting for efficiency
            w_sqrt = np.sqrt(effective_weights)
            X_weighted = X * w_sqrt[:, np.newaxis]
            # Fused X^T W X + prior via dsyrk (upper triangle only)
            prior_scaled = np.asfortranarray(prior_decay * self.cov_inv_)
            cov_inv = update_precision_dense(self.beta, X_weighted, prior_scaled)

        # Apply weights to y for the linear term
        y_weighted = y * effective_weights

        if self.sparse:
            # Scale vectors instead of sparse matrices to avoid copies
            eta = self.cov_inv_ @ (prior_decay * self.coef_) + X.T @ (
                self.beta * y_weighted
            )
            eta = cast(NDArray[np.float64], eta)
            assert isinstance(cov_inv, csc_array)
            factor: PrecisionFactor = create_sparse_factor(cov_inv)
            coef = factor.solve(eta)
            self._precision_factor = factor
        else:
            # eta = prior_decay * cov_inv_ @ coef_ + beta * X^T @ y_weighted
            eta = compute_eta_dense(
                prior_decay, self.cov_inv_, self.coef_, self.beta, X, y_weighted
            )
            # Cache the Cholesky factor for reuse in cov_/sample
            cho = cho_factor(cov_inv, lower=False, check_finite=False)
            self._precision_factor = DenseFactor(_U=cho[0], _n_features=cho[0].shape[0])
            coef = cho_solve(cho, eta, check_finite=False)

        self.cov_inv_ = cov_inv
        self.coef_ = coef

    def partial_fit(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """
        Incrementally update the posterior with new data.

        Uses the current posterior as the prior for the new update,
        decayed by ``learning_rate``. If the model has not been
        fitted, this is equivalent to calling ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be a ``scipy.sparse.csc_array`` when
            ``sparse=True``.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples
            are given equal weight.

        Returns
        -------
        self : NormalRegressor
            Updated estimator.

        See Also
        --------
        fit : Fit from scratch, resetting the prior.
        decay : Increase uncertainty without observing new data.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            return self.fit(X, y, sample_weight)

        X_fit, y = check_X_y(
            X,  # type: ignore (scipy is migrating to numpy-like types)
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        self._fit_helper(X_fit, y, sample_weight)
        return self

    def predict(self, X: Union[NDArray[Any], csc_array]) -> NDArray[Any]:
        """
        Predict target values using the posterior mean.

        Computes :math:`X \\hat{w}` where :math:`\\hat{w}` is the
        posterior mean of the weight vector.

        If the model has not been fitted, the prior mean (zero) is
        used, returning all zeros.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Must be a ``scipy.sparse.csc_array`` when
            ``sparse=True``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.

        See Also
        --------
        sample : Draw from the posterior predictive distribution.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X)

        X_pred = check_array(
            X, copy=True, ensure_2d=True, accept_sparse="csc" if self.sparse else False
        )

        return X_pred @ self.coef_

    def sample(
        self, X: Union[NDArray[Any], csc_array], size: int = 1
    ) -> NDArray[np.float64]:
        """
        Sample from the posterior predictive distribution.

        Draws weight vectors from the posterior
        :math:`w \\sim \\mathcal{N}(\\hat{w}, \\Lambda^{-1})` and
        computes :math:`X w` for each draw. This marginalizes over
        parameter uncertainty but not observation noise.

        If the model has not been fitted, samples are drawn from the
        prior predictive distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Must be a ``scipy.sparse.csc_array`` when
            ``sparse=True``.
        size : int, default=1
            Number of posterior samples to draw.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Predicted values for each posterior draw.

        See Also
        --------
        predict : Point predictions using the posterior mean.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X)

        X_sample = check_array(
            X, copy=False, ensure_2d=True, accept_sparse="csc" if self.sparse else False
        )

        # Exact joint draws via the support covariance when X is sparse
        # and touches fewer distinct columns than the number of draws
        # requested: |U| solves instead of one per draw. Same
        # distribution, different consumption of the random stream.
        support_draw = _support_covariance.build(
            self._precision_factor, X_sample, X_sample.shape[1], budget=size
        )
        if support_draw is not None:
            mean = np.asarray(X_sample @ self.coef_, dtype=np.float64).ravel()
            return cast(
                NDArray[np.float64],
                mean + support_draw.joint(size, self.random_state_),
            )

        samples = np.atleast_2d(
            multivariate_normal_sample_from_precision(
                self.coef_,
                self._precision_factor,
                size=size,
                random_state=self.random_state_,
            )
        )

        return samples @ X_sample.T  # type: ignore

    def sample_marginal(
        self, X: Union[NDArray[Any], csc_array], size: int = 1
    ) -> NDArray[np.float64]:
        """
        Sample iid draws from each row's marginal posterior predictive.

        Each prediction row's draws come from its exact marginal
        posterior predictive
        :math:`\\mathcal{N}(x_i^T \\hat{w},\\; x_i^T \\Lambda^{-1} x_i)`.
        Unlike ``sample`` -- whose rows within one draw share a weight
        vector and are therefore correlated -- draws are independent
        across rows, so only per-row statistics (means, quantiles,
        variances) are meaningful across the returned rows.

        For those statistics the result is exact and much cheaper than
        ``sample`` when many draws are needed: one triangular solve per
        row against the cached precision factor plus univariate draws,
        with per-draw cost independent of the number of features.

        If the model has not been fitted, samples are drawn from the
        prior predictive distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. May be dense, or a ``scipy.sparse.csc_array``
            when ``sparse=True``.
        size : int, default=1
            Number of marginal samples to draw per row.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Independent marginal draws for each row.

        See Also
        --------
        sample : Joint draws whose rows share a weight draw.
        predict : Point predictions using the posterior mean.
        """
        mean, sd = _validated_marginal_mean_sd(self, X)
        z = self.random_state_.standard_normal((size, mean.shape[0]))
        return cast(NDArray[np.float64], mean + sd * z)

    def sample_reward_space(
        self,
        X: Union[NDArray[Any], csc_array],
        size: int = 1,
        *,
        block_size: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Sample jointly from the posterior predictive, drawing in reward space.

        Distributionally identical to :meth:`sample` -- draws from
        :math:`X w \\sim \\mathcal{N}(X \\hat{w}, X \\Lambda^{-1} X^T)`
        -- but computed by factoring the ``n``-row predictive covariance
        once (one triangular half-solve per prediction row against the
        cached precision factor, then a QR) and drawing directly in
        reward space. Per-draw cost is then independent of the number of
        features, so this is much cheaper than ``sample`` when ``size``
        is large relative to the number of rows, and much more expensive
        in the opposite regime. Neither :math:`\\Lambda^{-1}` nor the
        covariance is ever formed, and linearly dependent rows (e.g.
        repeated contexts) are represented exactly. Note that without
        ``block_size``, a sparse model densifies the transposed
        prediction rows all at once -- a dense
        ``(n_features, n_samples)`` scratch array -- so pass
        ``block_size`` to bound memory when both dimensions are large.

        With ``block_size=k``, consecutive groups of ``k`` rows are
        factored and drawn independently: draws are joint within each
        group and independent across groups (unlike ``sample``, where
        all rows of one draw share a weight vector). Use this when rows
        group into independent decision units -- e.g. one block of arm
        rows per context -- and no cross-group statistic is consumed;
        the per-group joint distribution is exact.

        If the model has not been fitted, samples are drawn from the
        prior predictive distribution.

        .. note::

           Only the sampling *distribution* is guaranteed to match
           ``sample``, not the bitstream: the two methods consume
           different amounts of randomness, so the same seed yields
           different (equally distributed) realizations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. May be dense, or a ``scipy.sparse.csc_array``
            when ``sparse=True``.
        size : int, default=1
            Number of posterior samples to draw.
        block_size : int, optional
            If given, must divide ``n_samples``; rows are drawn jointly
            within consecutive blocks of this many rows and
            independently across blocks.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Predicted values for each posterior draw.

        See Also
        --------
        sample : Joint draws in weight space (cheaper for small ``size``).
        sample_marginal : iid per-row draws for per-row statistics.
        """
        X_sample = self._validated_for_sampling(X)
        mean, L = self._predictive_cholesky(X_sample, block_size)
        if block_size is None:
            z = standard_normal_f(self.random_state_, size, mean.shape[0])
            return affine_lower_factor(mean, z, L)
        z = self.random_state_.standard_normal(
            (mean.shape[0] // block_size, size, block_size)
        )
        return cast(NDArray[np.float64], mean + _blocked_colorize(L, z))

    @_invalidate_cached_properties
    def decay(
        self,
        X: Union[NDArray[Any], csc_array],
        *,
        decay_rate: Optional[float] = None,
    ) -> None:
        """
        Decay the posterior precision to increase uncertainty.

        Scales the precision matrix by :math:`\\gamma^n`, where
        :math:`\\gamma` is the decay rate and :math:`n` is the number
        of rows in ``X``. The posterior mean is unchanged.

        Has no effect if the model has not been fitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Used only for its number of rows ``n_samples``, which
            determines the exponent of the decay factor.
        decay_rate : float, default=None
            Decay factor :math:`\\gamma` in (0, 1]. If None, uses
            ``self.learning_rate``.

        See Also
        --------
        partial_fit : Update the model with new observations.
        """
        # If the model has not been fit, there is no prior to decay
        if not hasattr(self, "coef_"):
            return

        if decay_rate is None:
            decay_rate = self.learning_rate

        assert X.shape is not None  # for the type checker
        prior_decay = decay_rate ** X.shape[0]

        # Decay the prior without making an update. Because we're only
        # increasing the prior variance, we do not need to update the
        # mean.
        self.cov_inv_ = prior_decay * self.cov_inv_
        if "_precision_factor" in self.__dict__:
            if self.sparse:
                factor = self._precision_factor
                assert isinstance(factor, SparseFactor)
                self._precision_factor = scale_factor(factor, prior_decay)
            else:
                del self._precision_factor


class NormalInverseGammaRegressor(NormalRegressor):
    """
    Bayesian linear regression with unknown noise variance.

    Extends :class:`NormalRegressor` by placing a conjugate
    Normal-Inverse-Gamma (NIG) prior on the weights and noise
    variance jointly. Because the noise variance is integrated out
    analytically, the marginal posterior over the weights is a
    multivariate t distribution, producing heavier-tailed and more
    robust uncertainty estimates than the known-variance model.

    Parameters
    ----------
    mu : float or array-like of shape (n_features,), default=0.0
        Prior mean of the weights. A scalar is broadcast to all
        features.
    lam : float, array-like of shape (n_features,), or \
array-like of shape (n_features, n_features), default=1.0
        Prior precision (inverse covariance) of the weights. A
        scalar gives :math:`\\lambda I`; a vector gives
        :math:`\\text{diag}(\\lambda)`; a matrix is used directly.
    a : float, default=0.1
        Prior shape parameter of the Inverse-Gamma distribution on
        the noise variance :math:`\\sigma^2`. The prior is
        :math:`\\sigma^2 \\sim \\text{IG}(a, b)`.
    b : float, default=0.1
        Prior rate parameter of the Inverse-Gamma distribution.
        The prior mean of the noise variance is :math:`b / (a - 1)`
        for :math:`a > 1`.
    learning_rate : float, default=1.0
        Decay rate for sequential updates. Values less than 1
        geometrically shrink the precision and Inverse-Gamma
        parameters on each call to ``decay``, enabling adaptation
        to non-stationary environments.
    sparse : bool, default=False
        If True, use sparse matrix operations for the precision
        matrix. Input ``X`` must be a ``scipy.sparse.csc_array``.
        When CHOLMOD is available (via ``scikit-sparse``), it is
        used for efficient Cholesky factorization; otherwise falls
        back to UMFPACK or SuperLU.
    random_state : int, np.random.Generator, or None, default=None
        Controls the random number generator for ``sample``. Pass an
        int for reproducible results across calls.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Posterior mean of the weight vector.
    cov_inv_ : ndarray of shape (n_features, n_features) or \
scipy.sparse.csc_array
        Posterior precision matrix of the weights (conditioned on
        :math:`\\sigma^2`).
    a_ : float
        Posterior shape parameter of the Inverse-Gamma distribution.
    b_ : float
        Posterior rate parameter of the Inverse-Gamma distribution.
    n_features_ : int
        Number of features seen during ``fit``.

    See Also
    --------
    NormalRegressor : Known-variance variant (Gaussian posterior on
        weights).
    EmpiricalBayesNormalRegressor : Known-variance with empirical
        Bayes tuning of ``alpha`` and ``beta``.
    BayesianGLM : Bayesian GLM for non-Gaussian likelihoods.

    Notes
    -----
    This model implements the "unknown variance" Bayesian linear
    regression formulation described in Chapter 7 of [1]_. The
    joint prior is:

    .. math::

        w \\mid \\sigma^2 \\sim
        \\mathcal{N}(\\mu_0,\\; \\sigma^2 \\Lambda_0^{-1}), \\qquad
        \\sigma^2 \\sim \\text{IG}(a_0, b_0)

    After observing data :math:`(X, y)`, the posterior parameters
    are updated as:

    .. math::

        \\Lambda_n &= \\Lambda_0 + X^T X \\\\
        \\mu_n &= \\Lambda_n^{-1}(\\Lambda_0 \\mu_0 + X^T y) \\\\
        a_n &= a_0 + \\tfrac{N}{2} \\\\
        b_n &= b_0 + \\tfrac{1}{2}(y^T y + \\mu_0^T \\Lambda_0 \\mu_0
        - \\mu_n^T \\Lambda_n \\mu_n)

    The marginal posterior of the weights (integrating out
    :math:`\\sigma^2`) is a multivariate t distribution with
    :math:`2 a_n` degrees of freedom, location :math:`\\mu_n`, and
    shape :math:`(b_n / a_n) \\Lambda_n^{-1}`.

    References
    ----------
    .. [1] Murphy, Kevin P. "Machine Learning: A Probabilistic
       Perspective." MIT Press, 2012.

    Examples
    --------
    Batch fitting:

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

    Online learning with ``partial_fit``:

    >>> est = NormalInverseGammaRegressor(random_state=1)
    >>> for x_, y_ in zip(X, y):
    ...     est = est.partial_fit(x_.reshape(1, -1), np.array([y_]))
    >>> est.coef_
    array([32.89089478, 71.16073032])

    Sampling from the marginal posterior predictive (multivariate t):

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
        lam: Union[ArrayLike, csc_array] = 1.0,
        a: float = 0.1,
        b: float = 0.1,
        learning_rate: float = 1.0,
        sparse: bool = False,
        random_state: Union[int, np.random.Generator, None] = None,
    ):
        self.mu = mu
        self.lam = lam
        self.a = a
        self.b = b
        self.learning_rate = learning_rate
        self.sparse = sparse
        self.random_state = random_state

    @_invalidate_cached_properties
    def __getstate__(self):
        # Delete the cached covariance matrix, since it likely contains C
        # objects that cannot be pickled
        return super().__getstate__()

    def _initialize_prior(self, X: Union[NDArray[Any], csc_array]) -> None:
        if isinstance(self.random_state, int) or self.random_state is None:
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        assert X.shape is not None  # for the type checker
        self.n_features_ = X.shape[1]
        if np.isscalar(self.mu):
            self.coef_ = np.full(self.n_features_, self.mu, dtype=np.float64)
        elif cast(NDArray[np.float64], self.mu).ndim == 1:
            self.coef_ = cast(NDArray[np.float64], self.mu)
        else:
            raise ValueError("The prior mean must be a scalar or vector.")

        if self.sparse:
            if np.isscalar(self.lam):
                self.cov_inv_ = (
                    csc_array(eye(self.n_features_, format="csc")) * self.lam
                )
            elif cast(NDArray[np.float64], self.lam).ndim == 1:
                self.cov_inv_ = csc_array(diags(self.lam, format="csc"))
            elif cast(NDArray[np.float64], self.lam).ndim == 2:
                self.cov_inv_ = csc_array(self.lam)
            else:
                raise ValueError(
                    "The prior covariance must be a scalar, vector, or matrix."
                )

        else:
            if np.isscalar(self.lam):
                self.cov_inv_ = cast(np.float64, self.lam) * np.eye(self.n_features_)
            elif cast(NDArray[np.float64], self.lam).ndim == 1:
                self.cov_inv_ = np.diag(self.lam)  # type: ignore
            elif cast(NDArray[np.float64], self.lam).ndim == 2:
                self.cov_inv_ = cast(NDArray[np.float64], self.lam)
            else:
                raise ValueError(
                    "The prior covariance must be a scalar, vector, or matrix."
                )

        self.a_ = self.a
        self.b_ = self.b

    @_invalidate_cached_properties
    def _fit_helper(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ):
        if self.sparse:
            X = csc_array(X)

        assert X.shape is not None  # for the type checker

        # Handle sample weights
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"sample_weight.shape[0]={sample_weight.shape[0]} should be "
                    f"equal to X.shape[0]={X.shape[0]}"
                )

        # Apply the learning rate decay to get effective weights
        effective_weights = compute_effective_weights(
            X.shape[0], sample_weight, self.learning_rate
        )

        assert X.shape is not None  # for the type checker
        prior_decay = self.learning_rate ** X.shape[0]

        # Update the inverse covariance matrix with weights
        if self.sparse:
            # Element-wise scaling avoids constructing a diagonal matrix
            assert isinstance(X, csc_array)
            w_sqrt = np.sqrt(effective_weights)
            X_weighted = X.multiply(w_sqrt.reshape(-1, 1)).tocsc()
            V_n = prior_decay * self.cov_inv_ + X_weighted.T @ X_weighted
        else:
            w_sqrt = np.sqrt(effective_weights)
            X_weighted = X * w_sqrt[:, np.newaxis]
            # Fused X^T W X + prior via dsyrk (upper triangle only)
            prior_scaled = np.asfortranarray(prior_decay * self.cov_inv_)
            V_n = update_precision_dense(1.0, X_weighted, prior_scaled)

        # Apply weights to y for the linear term
        y_weighted = y * effective_weights

        prior_quad = 0.0  # set below in both branches
        if self.sparse:
            # Compute prior term and extract prior_quad before adding
            # the data term, mirroring the dense path optimisation.
            prior_cov_coef = self.cov_inv_ @ (prior_decay * self.coef_)
            prior_quad = float(self.coef_.dot(prior_cov_coef))
            eta = prior_cov_coef + X.T @ y_weighted
            eta = cast(NDArray[np.float64], eta)
            assert isinstance(V_n, csc_array)
            factor: PrecisionFactor = create_sparse_factor(V_n)
            m_n = factor.solve(eta)
            self._precision_factor = factor
        else:
            # Inline compute_eta_dense to reuse the intermediate dsymv
            # result for prior_quad, saving one O(p²) matvec.
            prior_cov_coef = dsymv(prior_decay, self.cov_inv_, self.coef_)
            prior_quad = self.coef_.dot(prior_cov_coef)
            # eta = prior_decay * cov_inv @ coef + X^T @ y_weighted
            eta = dgemv(
                1.0,
                X,
                y_weighted,
                trans=1,
                beta=1.0,
                y=prior_cov_coef,
                overwrite_y=True,
            )
            # Cache the Cholesky factor for reuse in shape_/sample
            cho = cho_factor(V_n, lower=False, check_finite=False)
            self._precision_factor = DenseFactor(_U=cho[0], _n_features=cho[0].shape[0])
            m_n = cho_solve(cho, eta, check_finite=False)

        # Update the shape and rate parameters of the variance
        # For a_n: sum of effective weights
        a_n = prior_decay * self.a_ + 0.5 * effective_weights.sum()

        # For b_n: weighted residual sum of squares
        # Use matvec + dot to compute quadratic forms efficiently
        weighted_y_squared = y.T @ (y * effective_weights)
        # prior_quad already computed in both sparse and dense eta blocks above.
        posterior_quad = m_n.dot(eta)
        b_n = prior_decay * self.b_ + 0.5 * (
            weighted_y_squared + prior_quad - posterior_quad
        )

        # Posteriors become priors for the next batch
        self.cov_inv_ = V_n
        self.coef_ = m_n
        self.a_ = a_n
        self.b_ = b_n

    @cached_property
    def shape_(self) -> PrecisionFactor:
        """Precision of the shape matrix for the multivariate t posterior.

        The shape covariance is (b/a)·Λ⁻¹, so the shape precision is
        (a/b)·Λ.  For dense models this is represented as a DenseFactor
        with U_scaled = √(a/b)·U (zero extra factorizations); for sparse
        models it wraps the existing SparseFactor via ``scale_factor``.
        """
        if self.sparse:
            factor = self._precision_factor
            assert isinstance(factor, SparseFactor)
            return scale_factor(factor, float(self.a_ / self.b_))
        else:
            factor = self._precision_factor
            assert isinstance(factor, DenseFactor)
            scale = np.sqrt(self.a_ / self.b_)
            U_scaled = np.triu(scale * factor._U)
            return DenseFactor(_U=U_scaled, _n_features=factor._n_features)

    def sample(
        self, X: Union[NDArray[Any], csc_array], size: int = 1
    ) -> NDArray[np.float64]:
        """
        Sample predicted values from the marginal posterior predictive.

        Draws weight vectors from the marginal posterior, which is a
        multivariate t distribution with :math:`2 a_n` degrees of
        freedom, location :math:`\\mu_n`, and shape
        :math:`(b_n / a_n) \\Lambda_n^{-1}`. The noise variance is
        integrated out analytically, producing heavier tails than the
        Gaussian posterior of :class:`NormalRegressor`.

        If the model has not been fitted, samples are drawn from the
        prior predictive distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Must be a ``scipy.sparse.csc_array`` when
            ``sparse=True``.
        size : int, default=1
            Number of posterior samples to draw.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Predicted values for each posterior draw.

        See Also
        --------
        predict : Point predictions using the posterior mean.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X)

        X_sample = check_array(
            X, copy=False, ensure_2d=True, accept_sparse="csc" if self.sparse else False
        )
        df = 2 * self.a_

        # Exact joint draws via the support covariance, built from
        # ``shape_`` so S already carries the (b/a) factor. The
        # chi-square mixing is a per-draw scalar, so it commutes with
        # the linear map and the reduction stays exact.
        support_draw = _support_covariance.build(
            self.shape_, X_sample, X_sample.shape[1], budget=size
        )
        if support_draw is not None:
            mean = np.asarray(X_sample @ self.coef_, dtype=np.float64).ravel()
            z = support_draw.joint(size, self.random_state_)
            x = self.random_state_.chisquare(df, size) / df
            return cast(
                NDArray[np.float64],
                mean + z / np.sqrt(x)[..., None],
            )

        # Sample from multivariate t via precision parameterization
        # shape_ returns a PrecisionFactor (DenseFactor or SparseFactor)
        # whose precision is (a/b)·Λ, i.e. shape cov = (b/a)·Λ⁻¹
        samples = multivariate_t_sample_from_precision(
            self.coef_,
            self.shape_,
            df,
            size,
            self.random_state_,  # type: ignore[arg-type]
        )

        if self.n_features_ == 1:
            samples = np.expand_dims(samples, -1)

        return np.atleast_2d(samples @ X_sample.T)  # type: ignore

    def sample_marginal(
        self, X: Union[NDArray[Any], csc_array], size: int = 1
    ) -> NDArray[np.float64]:
        """
        Sample iid draws from each row's marginal posterior predictive.

        Each row's marginal is a Student t with :math:`2 a_n` degrees
        of freedom, location :math:`x_i^T \\mu_n`, and scale
        :math:`\\sqrt{(b_n / a_n)\\, x_i^T \\Lambda_n^{-1} x_i}`. Each
        (draw, row) cell mixes its own chi-square variable, so draws
        are fully independent across rows -- see
        :meth:`NormalRegressor.sample_marginal` for the contrast with
        ``sample``.

        If the model has not been fitted, samples are drawn from the
        prior predictive distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. May be dense, or a ``scipy.sparse.csc_array``
            when ``sparse=True``.
        size : int, default=1
            Number of marginal samples to draw per row.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Independent marginal draws for each row.

        See Also
        --------
        sample : Joint draws whose rows share a weight draw.
        predict : Point predictions using the posterior mean.
        """
        mean, sd = _validated_marginal_mean_sd(self, X)
        df = 2.0 * self.a_
        z = self.random_state_.standard_normal((size, mean.shape[0]))
        # one chi-square per (draw, row) cell: rows must be fully
        # independent, not merely marginally exact
        g = self.random_state_.chisquare(df, size=(size, mean.shape[0]))
        # the (b/a) scale factor and the df/g mixing fold into one
        # per-cell scale
        scale = np.sqrt((self.b_ / self.a_) * df / g)
        return cast(NDArray[np.float64], mean + (sd * z) * scale)

    def sample_reward_space(
        self,
        X: Union[NDArray[Any], csc_array],
        size: int = 1,
        *,
        block_size: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Sample jointly from the posterior predictive, drawing in reward space.

        Distributionally identical to :meth:`sample` -- the ``n``-row
        multivariate t with :math:`2 a_n` degrees of freedom, location
        :math:`X \\mu_n`, and shape :math:`(b_n / a_n) X \\Lambda_n^{-1} X^T`
        -- but drawn via the exact square root of the shape matrix, with
        one shared chi-square mixing variable per draw. See
        :meth:`NormalRegressor.sample_reward_space` for the cost model
        and the bitstream caveat.

        With ``block_size=k``, consecutive groups of ``k`` rows are
        drawn independently, each with its own chi-square mixing
        variable per draw: every group's joint t distribution is exact,
        and groups are fully independent (whereas ``sample`` shares one
        noise-scale draw across all rows). Use only when no cross-group
        statistic is consumed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. May be dense, or a ``scipy.sparse.csc_array``
            when ``sparse=True``.
        size : int, default=1
            Number of posterior samples to draw.
        block_size : int, optional
            If given, must divide ``n_samples``; rows are drawn jointly
            within consecutive blocks of this many rows and
            independently across blocks.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Predicted values for each posterior draw.

        See Also
        --------
        sample : Joint draws in weight space (cheaper for small ``size``).
        sample_marginal : iid per-row draws for per-row statistics.
        """
        X_sample = self._validated_for_sampling(X)
        mean, L = self._predictive_cholesky(X_sample, block_size)
        df = 2.0 * self.a_
        if block_size is None:
            z = standard_normal_f(self.random_state_, size, mean.shape[0])
            g = self.random_state_.chisquare(df, size=size)
            # t draw: location + sqrt(df/g) · (shape-chol @ z), with the
            # (b/a) shape scaling (shape cov = (b/a) · X Λ⁻¹ Xᵀ) folded
            # into the per-draw scale. Scaling the draws rather than the
            # product is equivalent (both scale whole rows) and lets the
            # mean be accumulated inside the same GEMM.
            scale = np.sqrt((self.b_ / self.a_) * df / g)
            z *= scale[:, np.newaxis]
            return affine_lower_factor(mean, z, L)
        n_blocks = mean.shape[0] // block_size
        z = self.random_state_.standard_normal((n_blocks, size, block_size))
        # one chi-square per (draw, block): joint t within a block,
        # independence across blocks
        g = self.random_state_.chisquare(df, size=(size, n_blocks))
        scale = np.sqrt((self.b_ / self.a_) * df / g)
        y = _blocked_colorize(L, z * scale.T[:, :, np.newaxis])
        return cast(NDArray[np.float64], mean + y)

    @_invalidate_cached_properties
    def decay(
        self,
        X: Union[NDArray[Any], csc_array],
        *,
        decay_rate: Optional[float] = None,
    ) -> None:
        """
        Decay precision and variance parameters to increase uncertainty.

        Applies exponential forgetting to the precision matrix and
        the Inverse-Gamma parameters:

        .. math::

            \\Lambda \\leftarrow \\gamma^n \\Lambda, \\quad
            a \\leftarrow \\gamma^n a, \\quad
            b \\leftarrow \\gamma^n b

        The posterior mean is unchanged, but the marginal t
        distribution widens (fewer degrees of freedom and higher
        scale), reflecting greater uncertainty.

        Has no effect if the model has not been fitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Used only for its number of rows ``n_samples``, which
            determines the exponent of the decay factor.
        decay_rate : float, default=None
            Decay factor :math:`\\gamma` in (0, 1]. If None, uses
            ``self.learning_rate``.

        See Also
        --------
        partial_fit : Update the model with new observations.
        """
        # If the model has not been fit, there is no prior to decay
        if not hasattr(self, "coef_"):
            return

        if decay_rate is None:
            decay_rate = self.learning_rate

        assert X.shape is not None  # for the type checker
        prior_decay = decay_rate ** X.shape[0]

        # decay only increases the variance, so we only need to update the
        # inverse covariance matrix, a_, and b_
        self.cov_inv_ = prior_decay * self.cov_inv_
        self.a_ = prior_decay * self.a_
        self.b_ = prior_decay * self.b_
        if "_precision_factor" in self.__dict__:
            if self.sparse:
                factor = self._precision_factor
                assert isinstance(factor, SparseFactor)
                self._precision_factor = scale_factor(factor, prior_decay)
            else:
                del self._precision_factor


def multivariate_t_sample_from_covariance(
    loc: Optional[NDArray[np.float64]],
    shape: Covariance,
    df: float = 1,
    size: Union[int, tuple[int, ...]] = 1,
    random_state: Union[int, np.random.Generator, None] = None,
):
    """
    Sample from a multivariate t distribution with the given mean and covariance.

    Parameters
    ----------
    loc : NDArray[np.float64]
        Mean of the distribution.
    shape : Covariance
        Covariance of the distribution.
    df : float, default=1
        Degrees of freedom of the distribution.
    size : int, default=1
        Number of samples to draw.
    random_state : int, np.random.Generator, or None, default=None
        Random state for the model.

    Returns
    -------
    samples : NDArray[np.float64]
        Samples from the distribution.

    Notes
    -----
    This function is a reimplementation of `scipy.stats.multivariate_t.rvs` that
    uses a `Covariance` object instead of a covariance matrix.
    """
    rng = np.random.default_rng(random_state)

    x = rng.chisquare(df, size=size) / df

    z = multivariate_normal.rvs(
        0,
        shape,  # type: ignore
        size=size,  # type: ignore[arg-type]  # scipy stubs say int; tuples work
        random_state=rng,
    )
    # rvs squeezes singleton axes (size=1 or 1-dimensional shape) and
    # returns a single draw for empty sizes; restore ``(*size, dim)`` so
    # the chi-square scaling broadcasts per draw. ``size`` may be an int
    # or a tuple, so shape the result after ``x`` rather than ``size``.
    dim = shape.shape[-1]
    if np.prod(np.shape(x), dtype=int) == 0:
        z = np.zeros(np.shape(x) + (dim,))
    else:
        z = np.asarray(z).reshape(np.shape(x) + (dim,))
    if loc is None:
        loc = np.zeros_like(z)
    samples = loc + z / np.sqrt(x)[..., None]
    return _squeeze_output(samples)


class BayesianGLM(_RewardSpacePredictiveMixin, BaseEstimator, RegressorMixin):
    """
    Bayesian Generalized Linear Model with Laplace approximation.

    Extends Bayesian linear regression to non-Gaussian likelihoods (binary
    and count data) using a configurable posterior approximation method.
    The default uses Laplace approximation via iteratively reweighted least
    squares (IRLS). Supports both dense and sparse feature matrices, online
    learning via ``partial_fit``, and non-stationary environments via
    ``decay``.

    Parameters
    ----------
    alpha : float, default=1.0
        Prior precision for the weights. The prior is
        :math:`w \\sim \\mathcal{N}(0, \\alpha^{-1} I)`. Higher values
        give stronger regularization toward zero.
    link : {'logit', 'log'}, default='logit'
        Link function relating the linear predictor to the mean of the
        response distribution:

        - ``'logit'``: Inverse logit (sigmoid). Use for binary outcomes
          (Bernoulli likelihood).
        - ``'log'``: Exponential. Use for count outcomes (Poisson
          likelihood).
    learning_rate : float, default=1.0
        Decay rate for the posterior precision on each call to ``decay``.
        Values less than 1 geometrically shrink the precision matrix,
        increasing posterior uncertainty over time. This converts the
        model into a forgetting (non-stationary) estimator suitable for
        restless bandit problems.
    approximator : PosteriorApproximator, default=LaplaceApproximator()
        Strategy object for approximating the posterior. The default
        ``LaplaceApproximator`` performs 5 IRLS iterations per update.
        For fast online updates, use ``LaplaceApproximator(n_iter=1)``.
        For batch convergence, increase ``n_iter`` and set a ``tol``.
    sparse : bool, default=False
        If True, use sparse matrix operations for the precision matrix.
        Input ``X`` must be a ``scipy.sparse.csc_array``. When CHOLMOD
        is available (via ``scikit-sparse``), it is used for efficient
        Cholesky factorization; otherwise falls back to SuperLU.
    random_state : int, np.random.Generator, or None, default=None
        Controls the random number generator for ``sample``. Pass an
        int for reproducible results across calls.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Posterior mean of the weight vector (MAP estimate).
    cov_inv_ : ndarray of shape (n_features, n_features) or \
scipy.sparse.csc_array
        Posterior precision matrix (inverse covariance). Dense when
        ``sparse=False``, sparse CSC when ``sparse=True``.
    n_features_ : int
        Number of features seen during ``fit``.

    See Also
    --------
    NormalRegressor : Bayesian linear regression with known noise variance.
    NormalInverseGammaRegressor : Bayesian linear regression with unknown
        noise variance.
    LaplaceApproximator : The default posterior approximation strategy.

    Notes
    -----
    The model places a Gaussian prior on the weight vector:

    .. math::

        w \\sim \\mathcal{N}(0, \\alpha^{-1} I)

    For the logit link (Bernoulli likelihood):

    .. math::

        p(y=1 \\mid x) = \\sigma(x^T w)

    For the log link (Poisson likelihood):

    .. math::

        \\mathbb{E}[y \\mid x] = \\exp(x^T w)

    Since these likelihoods are not conjugate to the Gaussian prior, the
    posterior is approximated using the Laplace approximation. This finds
    the MAP estimate :math:`\\hat{w}` and approximates the posterior as:

    .. math::

        p(w \\mid \\mathcal{D}) \\approx
        \\mathcal{N}(\\hat{w}, \\Lambda^{-1})

    where :math:`\\Lambda = \\alpha I + X^T W X` is the posterior precision
    and :math:`W` is the diagonal matrix of IRLS weights. See Chapter 8
    of [1]_ for details.

    When ``learning_rate < 1``, calling ``decay`` scales the precision
    matrix by :math:`\\gamma^n` where :math:`\\gamma` is the learning rate
    and :math:`n` is the number of samples. This uniformly increases
    posterior uncertainty, allowing the model to adapt to non-stationary
    environments.

    References
    ----------
    .. [1] Murphy, Kevin P. "Machine Learning: A Probabilistic
       Perspective." MIT Press, 2012.

    Examples
    --------
    Binary classification with logistic regression:

    >>> from bayesianbandits import BayesianGLM
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([0, 0, 1, 1])  # Binary outcomes
    >>> model = BayesianGLM(alpha=1.0, link='logit')
    >>> model.fit(X, y)
    BayesianGLM()
    >>> model.predict(X)  # Returns probabilities
    array([0.56154064, 0.62124455, 0.67748791, 0.72902237])

    Count regression with Poisson:

    >>> y_counts = np.array([1, 2, 5, 8])  # Count data
    >>> model = BayesianGLM(alpha=1.0, link='log')
    >>> model.fit(X, y_counts)
    BayesianGLM(link='log')
    >>> model.predict(X)  # Returns expected counts
    array([1.72636481, 2.98033545, 5.14514623, 8.88239939])

    Poisson rate modeling with varying exposure (e.g., different observation
    periods). For Poisson with log link, fitting the rate ``y / exposure``
    with ``sample_weight=exposure`` is mathematically equivalent to using
    an offset of ``log(exposure)`` in the linear predictor:

    >>> exposure = np.array([1, 2, 5, 10])
    >>> y_counts = np.array([2, 5, 12, 30])
    >>> model = BayesianGLM(alpha=1.0, link='log')
    >>> model.fit(X, y_counts / exposure, sample_weight=exposure)
    BayesianGLM(link='log')
    >>> model.predict(X) * exposure  # Scale predicted rates by exposure
    array([ 1.32755877,  3.52482459, 11.69852952, 31.06097099])

    Online learning with fast single-iteration updates:

    >>> from bayesianbandits import LaplaceApproximator
    >>> approximator = LaplaceApproximator(n_iter=1)  # Fast online updates
    >>> model = BayesianGLM(alpha=1.0, link='logit', approximator=approximator)
    >>> stream = [(np.array([[1]]), [0]), (np.array([[2]]), [0]),
    ...           (np.array([[3]]), [1]), (np.array([[4]]), [1])]
    >>> for X_batch, y_batch in stream:
    ...     model = model.partial_fit(X_batch, y_batch)
    >>> model.predict(np.array([[1], [2], [3], [4]]))
    array([0.59667319, 0.68637901, 0.76402365, 0.82728258])

    Batch convergence with tighter tolerance:

    >>> approximator = LaplaceApproximator(n_iter=500, tol=1e-6)
    >>> model = BayesianGLM(alpha=0.1, approximator=approximator)
    >>> model = model.fit(X, y)  # Iterates until convergence
    >>> model.predict(X)
    array([0.57027497, 0.63782727, 0.70034041, 0.75618799])
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        link: LinkFunction = "logit",
        learning_rate: float = 1.0,
        approximator: Optional[PosteriorApproximator] = None,
        sparse: bool = False,
        random_state: Union[int, np.random.Generator, None] = None,
    ) -> None:
        self.alpha = alpha
        self.link: LinkFunction = link
        self.learning_rate = learning_rate
        self.approximator = approximator
        self.sparse = sparse
        self.random_state = random_state

    def _initialize_prior(self, X: Union[NDArray[Any], csc_array]) -> None:
        """Initialize prior distribution."""
        if isinstance(self.random_state, int) or self.random_state is None:
            self.random_state_ = np.random.default_rng(self.random_state)
        else:
            self.random_state_ = self.random_state

        assert X.shape is not None
        self.n_features_ = X.shape[1]
        self.coef_ = np.zeros(self.n_features_)

        if self.sparse:
            self.cov_inv_ = csc_array(eye(self.n_features_, format="csc")) * self.alpha
        else:
            self.cov_inv_ = cast(
                NDArray[np.float64], np.eye(self.n_features_) * self.alpha
            )

        # A cached factor from a previous fit would be stale against
        # the freshly-reset cov_inv_; drop it.
        self.__dict__.pop("_precision_factor", None)

        # Initialize approximator if not provided
        if self.approximator is None:
            self.approximator_ = LaplaceApproximator()
        else:
            self.approximator_ = self.approximator

    @_invalidate_cached_properties
    def __getstate__(self) -> Any:
        # Exclude cached C extension objects that cannot be pickled
        state = super().__getstate__()  # type: ignore
        state.pop("_precision_factor", None)
        return state

    @cached_property
    def _precision_factor(self) -> PrecisionFactor:
        """Factorization of the precision matrix (cached)."""
        if self.sparse:
            assert isinstance(self.cov_inv_, csc_array)
            return create_sparse_factor(self.cov_inv_)
        else:
            cho = cho_factor(self.cov_inv_, lower=False, check_finite=False)
            return DenseFactor(_U=cho[0], _n_features=cho[0].shape[0])

    @cached_property
    def cov_(self) -> Union[Covariance, SparseFactor]:
        """Posterior covariance matrix (cached, lazily computed).

        Returns a ``scipy.stats.Covariance`` object (dense) or a
        ``SparseFactor`` (sparse) that wraps the Cholesky factorization
        of the covariance. Automatically invalidated when the model is
        updated via ``fit``, ``partial_fit``, or ``decay``.

        .. warning::

           For dense models, this is an :math:`O(p^3)` computation with
           :math:`O(p^2)` memory. For high-dimensional problems, prefer
           ``sparse=True`` or avoid accessing this property directly.
        """
        factor = self._precision_factor
        if self.sparse:
            assert isinstance(factor, SparseFactor)
            return factor
        else:
            assert isinstance(factor, DenseFactor)
            cov = factor.solve(np.eye(self.n_features_))
            return Covariance.from_cholesky(cholesky(cov, lower=True))

    @_invalidate_cached_properties
    def _fit_helper(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> None:
        """Update posterior using the configured approximation method."""
        # For sparse partial_fit, hand the cached prior factor to the
        # approximator so it can skip redundant factorization work.
        prior_factor: Optional[Any] = (
            self.__dict__.get("_precision_factor") if self.sparse else None
        )

        posterior = self.approximator_.update_posterior(
            X,
            y,
            self.coef_,
            self.cov_inv_,  # type: ignore
            link=self.link,
            sample_weight=sample_weight,
            learning_rate=self.learning_rate,
            sparse=self.sparse,
            prior_factor=prior_factor,
        )
        self.coef_ = posterior.mean
        self.cov_inv_ = posterior.precision
        if posterior.factor is not None:
            if self.sparse:
                self._precision_factor = posterior.factor
            else:
                cho = posterior.factor
                self._precision_factor = DenseFactor(
                    _U=cho[0], _n_features=cho[0].shape[0]
                )

    def fit(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """
        Fit the model from scratch, resetting the prior.

        Initializes the prior :math:`w \\sim \\mathcal{N}(0, \\alpha^{-1} I)`
        and computes the posterior using the configured approximation method.
        Any previously learned parameters are discarded.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be a ``scipy.sparse.csc_array`` when
            ``sparse=True``.
        y : array-like of shape (n_samples,)
            Target values. For ``link='logit'``, values should be 0 or 1.
            For ``link='log'``, values should be non-negative counts.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples are
            given equal weight.

        Returns
        -------
        self : BayesianGLM
            Fitted estimator.

        See Also
        --------
        partial_fit : Incremental update without resetting the prior.
        """
        X, y = check_X_y(
            X,  # type: ignore
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        self._initialize_prior(X)
        self._fit_helper(X, y, sample_weight)
        return self

    def partial_fit(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """
        Incrementally update the model with new data.

        Uses the current posterior as the prior for the new update. If the
        model has not been fitted yet, this is equivalent to calling
        ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be a ``scipy.sparse.csc_array`` when
            ``sparse=True``.
        y : array-like of shape (n_samples,)
            Target values. For ``link='logit'``, values should be 0 or 1.
            For ``link='log'``, values should be non-negative counts.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples are
            given equal weight.

        Returns
        -------
        self : BayesianGLM
            Updated estimator.

        See Also
        --------
        fit : Fit from scratch, resetting the prior.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            return self.fit(X, y, sample_weight)

        X, y = check_X_y(
            X,  # type: ignore
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        self._fit_helper(X, y, sample_weight)
        return self

    def predict(self, X: Union[NDArray[Any], csc_array]) -> NDArray[Any]:
        """
        Predict mean of the response distribution for each sample.

        Computes the inverse link applied to the linear predictor
        :math:`g^{-1}(X \\hat{w})`, where :math:`\\hat{w}` is the
        posterior mean.

        If the model has not been fitted, the prior mean (zero) is used,
        returning 0.5 for logit link and 1.0 for log link.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict. Must be a ``scipy.sparse.csc_array``
            when ``sparse=True``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values. For ``link='logit'``, probabilities in
            [0, 1]. For ``link='log'``, expected counts (positive reals).

        See Also
        --------
        sample : Draw from the posterior predictive distribution.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X)

        X_pred = check_array(
            X, copy=True, ensure_2d=True, accept_sparse="csc" if self.sparse else False
        )

        eta = X_pred @ self.coef_

        return self._inverse_link(eta)

    def _inverse_link(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the inverse link elementwise, mapping the linear
        predictor to the response scale."""
        if self.link == "logit":
            return cast(NDArray[np.float64], expit(eta))
        elif self.link == "log":
            return cast(NDArray[np.float64], np.exp(np.clip(eta, -700, 700)))
        else:
            raise ValueError(f"Unknown link: {self.link}")

    def sample(
        self, X: Union[NDArray[Any], csc_array], size: int = 1
    ) -> NDArray[np.float64]:
        """
        Sample from the posterior predictive distribution.

        Draws weight vectors from the posterior
        :math:`w \\sim \\mathcal{N}(\\hat{w}, \\Lambda^{-1})` and computes
        :math:`g^{-1}(X w)` for each draw. This marginalizes over
        parameter uncertainty, producing samples from the posterior
        predictive distribution.

        If the model has not been fitted, samples are drawn from the
        prior predictive distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Must be a ``scipy.sparse.csc_array`` when
            ``sparse=True``.
        size : int, default=1
            Number of posterior samples to draw.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Predicted values for each posterior sample. For
            ``link='logit'``, probabilities in [0, 1]. For
            ``link='log'``, expected counts (positive reals).

        See Also
        --------
        predict : Point predictions using the posterior mean.
        """
        try:
            check_is_fitted(self, "coef_")
        except NotFittedError:
            self._initialize_prior(X)

        X_sample = check_array(
            X, copy=False, ensure_2d=True, accept_sparse="csc" if self.sparse else False
        )

        # The support covariance gives the linear predictor's exact joint
        # law directly; the inverse link is applied elementwise either
        # way, so it composes unchanged.
        support_draw = _support_covariance.build(
            self._precision_factor, X_sample, X_sample.shape[1], budget=size
        )
        if support_draw is not None:
            mean = np.asarray(X_sample @ self.coef_, dtype=np.float64).ravel()
            eta = mean + support_draw.joint(size, self.random_state_)
            return self._inverse_link(cast(NDArray[np.float64], eta))

        param_samples = np.atleast_2d(
            multivariate_normal_sample_from_precision(
                self.coef_,
                self._precision_factor,
                size=size,
                random_state=self.random_state_,
            )
        )

        # Vectorized: (size, n_features) @ (n_features, n_samples) -> (size, n_samples)
        eta = param_samples @ X_sample.T

        return self._inverse_link(cast(NDArray[np.float64], eta))

    def sample_marginal(
        self, X: Union[NDArray[Any], csc_array], size: int = 1
    ) -> NDArray[np.float64]:
        """
        Sample iid draws from each row's marginal posterior predictive.

        Draws each row's linear predictor from its exact marginal
        :math:`\\eta_i \\sim \\mathcal{N}(x_i^T \\hat{w},\\;
        x_i^T \\Lambda^{-1} x_i)` and applies the inverse link
        elementwise. Draws are independent across rows -- see
        :meth:`NormalRegressor.sample_marginal` for the contrast with
        ``sample``.

        If the model has not been fitted, samples are drawn from the
        prior predictive distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. May be dense, or a ``scipy.sparse.csc_array``
            when ``sparse=True``.
        size : int, default=1
            Number of marginal samples to draw per row.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Independent marginal draws for each row, on the response
            scale of the link.

        See Also
        --------
        sample : Joint draws whose rows share a weight draw.
        predict : Point predictions using the posterior mean.
        """
        mean, sd = _validated_marginal_mean_sd(self, X)
        z = self.random_state_.standard_normal((size, mean.shape[0]))
        return self._inverse_link(mean + sd * z)

    def sample_reward_space(
        self,
        X: Union[NDArray[Any], csc_array],
        size: int = 1,
        *,
        block_size: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Sample jointly from the posterior predictive, drawing in eta space.

        Distributionally identical to :meth:`sample` -- draws linear
        predictors :math:`\\eta \\sim \\mathcal{N}(X \\hat{w},
        X \\Lambda^{-1} X^T)` and applies the inverse link elementwise
        -- but the :math:`\\eta` draws come directly from the ``n``-row
        predictive Gaussian via the exact square root of its covariance.
        See :meth:`NormalRegressor.sample_reward_space` for the cost
        model, the ``block_size`` semantics, and the bitstream caveat.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. May be dense, or a ``scipy.sparse.csc_array``
            when ``sparse=True``.
        size : int, default=1
            Number of posterior samples to draw.
        block_size : int, optional
            If given, must divide ``n_samples``; rows are drawn jointly
            within consecutive blocks of this many rows and
            independently across blocks.

        Returns
        -------
        samples : ndarray of shape (size, n_samples)
            Predicted values for each posterior sample, on the response
            scale of the link.

        See Also
        --------
        sample : Joint draws in weight space (cheaper for small ``size``).
        sample_marginal : iid per-row draws for per-row statistics.
        """
        X_sample = self._validated_for_sampling(X)
        mean, L = self._predictive_cholesky(X_sample, block_size)
        if block_size is None:
            z = standard_normal_f(self.random_state_, size, mean.shape[0])
            eta = affine_lower_factor(mean, z, L)
        else:
            z = self.random_state_.standard_normal(
                (mean.shape[0] // block_size, size, block_size)
            )
            eta = mean + _blocked_colorize(L, z)
        return self._inverse_link(cast(NDArray[np.float64], eta))

    @_invalidate_cached_properties
    def decay(
        self,
        X: Union[NDArray[Any], csc_array],
        *,
        decay_rate: Optional[float] = None,
    ) -> None:
        """
        Decay the posterior precision to increase uncertainty.

        Scales the precision matrix by :math:`\\gamma^n`, where
        :math:`\\gamma` is the decay rate and :math:`n` is the number
        of rows in ``X``. This uniformly increases posterior variance
        while leaving the posterior mean unchanged, allowing the model
        to adapt to non-stationary environments.

        Has no effect if the model has not been fitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Used only for its number of rows ``n_samples``, which
            determines the exponent of the decay factor.
        decay_rate : float, default=None
            Decay factor per sample. If None, uses the model's
            ``learning_rate``. Values less than 1 increase uncertainty;
            a value of 1 has no effect.

        See Also
        --------
        partial_fit : Update the model with new observations.
        """
        if not hasattr(self, "coef_"):
            return

        if decay_rate is None:
            decay_rate = self.learning_rate

        assert X.shape is not None
        prior_decay = decay_rate ** X.shape[0]

        self.cov_inv_ = prior_decay * self.cov_inv_
        if "_precision_factor" in self.__dict__:
            if self.sparse:
                factor = self._precision_factor
                assert isinstance(factor, SparseFactor)
                self._precision_factor = scale_factor(factor, prior_decay)
            else:
                del self._precision_factor
