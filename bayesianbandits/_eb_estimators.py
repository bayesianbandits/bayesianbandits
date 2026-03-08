"""Empirical Bayes estimators via MacKay's evidence maximization."""

from __future__ import annotations

import math
from typing import Any, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array
from sklearn.utils.validation import check_X_y
from typing_extensions import Self

from ._empirical_bayes import (
    accumulate_sufficient_stats,
    mackay_update_normal,
    mackay_update_normal_online,
)
from ._estimators import NormalRegressor, _invalidate_cached_properties
from ._sparse_bayesian_linear_regression import create_sparse_factor


class _EmpiricalBayesMixin:
    """Mixin providing the empirical Bayes fit loop.

    Subclasses must implement ``_eb_mackay_step(X, y) -> float``
    which runs one MacKay update (mutating hyperparams) and returns
    log evidence.
    """

    def _eb_mackay_step(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
    ) -> float:
        raise NotImplementedError

    def _eb_fit(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> None:
        prev_evidence = -math.inf
        n_eb_iter: int = self.n_eb_iter  # type: ignore[attr-defined]
        eb_tol: float = self.eb_tol  # type: ignore[attr-defined]

        converged = False
        iterations = 0

        for i in range(n_eb_iter):
            self._initialize_prior(X)  # type: ignore[attr-defined]
            self._fit_helper(X, y, sample_weight)  # type: ignore[attr-defined]
            log_ev = self._eb_mackay_step(X, y)
            iterations = i + 1

            if abs(log_ev - prev_evidence) < eb_tol:
                converged = True
                break
            prev_evidence = log_ev

        # Final refit with converged hyperparams
        self._initialize_prior(X)  # type: ignore[attr-defined]
        self._fit_helper(X, y, sample_weight)  # type: ignore[attr-defined]

        self.log_evidence_ = log_ev if n_eb_iter > 0 else -math.inf  # type: ignore[possibly-undefined]
        self.n_eb_iterations_ = iterations
        self.eb_converged_ = converged


class EmpiricalBayesNormalRegressor(_EmpiricalBayesMixin, NormalRegressor):
    """NormalRegressor with automatic hyperparameter tuning via evidence maximization.

    Uses MacKay's update rules to iteratively optimize the prior precision
    (alpha) and noise precision (beta) by maximizing the marginal likelihood.

    When ``learning_rate < 1``, exponential forgetting is applied to the
    precision matrix.  To prevent the prior contribution from collapsing to
    zero under repeated decay, *stabilized forgetting* re-injects a fixed
    prior floor after each decay step, following [1]_.

    References
    ----------
    .. [1] Kulhavy, R. & Zarrop, M. B. (1993). "On a general concept of
       forgetting." *International Journal of Control*, 58(4), 905-924.

    Parameters
    ----------
    alpha : float
        Initial prior precision of the weights.
    beta : float
        Initial noise precision.
    n_eb_iter : int, default=10
        Maximum number of empirical Bayes iterations during ``fit``.
    eb_tol : float, default=1e-4
        Convergence tolerance on log evidence change.
    learning_rate : float, default=1.0
        Learning rate for recursive Bayesian updates.
    sparse : bool, default=False
        Whether to use sparse precision matrices.
    random_state : int, Generator, or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    log_evidence_ : float
        Log marginal likelihood at convergence.
    n_eb_iterations_ : int
        Number of EB iterations performed.
    eb_converged_ : bool
        Whether the EB loop converged within tolerance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        *,
        n_eb_iter: int = 10,
        eb_tol: float = 1e-4,
        learning_rate: float = 1.0,
        sparse: bool = False,
        random_state: Union[int, np.random.Generator, None] = None,
    ) -> None:
        super().__init__(
            alpha=alpha,
            beta=beta,
            learning_rate=learning_rate,
            sparse=sparse,
            random_state=random_state,
        )
        self.n_eb_iter = n_eb_iter
        self.eb_tol = eb_tol

    def _eb_mackay_step(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
    ) -> float:
        alpha_new, beta_new, log_ev = mackay_update_normal(
            self.coef_,
            self.cov_inv_,
            X,
            y,
            self.alpha,
            self.beta,
            sparse=self.sparse,
            factor=getattr(self, "_factor", None),
            rng=self.random_state_,
        )
        self.alpha = alpha_new
        self.beta = beta_new
        return log_ev

    def _eb_mackay_step_online(self) -> None:
        """MacKay step using accumulated sufficient statistics for beta."""
        alpha_new, beta_new = mackay_update_normal_online(
            self.coef_,
            self.cov_inv_,
            self.alpha,
            self.beta,
            self._prior_scalar,
            self._effective_n,
            self._eff_yTy,
            self._eff_XTy,
            sparse=self.sparse,
            factor=getattr(self, "_factor", None),
            rng=self.random_state_,
        )
        self.alpha = alpha_new
        self.beta = beta_new

    def _accumulate_stats(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        prior_decay: float,
    ) -> None:
        """Update decayed sufficient statistics for the beta update."""
        self._effective_n, self._eff_yTy, self._eff_XTy = accumulate_sufficient_stats(
            self._effective_n,
            self._eff_yTy,
            self._eff_XTy,
            X,
            y,
            prior_decay,
        )

    def fit(
        self,
        X_fit: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        X_fit, y = check_X_y(
            X_fit,  # type: ignore
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        if self.n_eb_iter > 0:
            self._eb_fit(X_fit, y, sample_weight)
        else:
            self._initialize_prior(X_fit)
            self._fit_helper(X_fit, y, sample_weight)

        # After fit, the precision matrix is consistent with self.alpha.
        self._prior_scalar = self.alpha

        # Initialize sufficient statistics from the fit data.
        self._effective_n = float(y.shape[0])
        self._eff_yTy = float(y @ y)
        if isinstance(X_fit, csc_array):
            self._eff_XTy: NDArray[np.float64] = np.asarray(
                X_fit.T @ y, dtype=np.float64
            ).ravel()
        else:
            self._eff_XTy = np.asarray(X_fit.T @ y, dtype=np.float64)

        return self

    @_invalidate_cached_properties
    def _correct_precision(
        self,
        alpha_old: float,
        beta_old: float,
    ) -> None:
        """Correct the precision matrix after a MacKay hyperparameter update.

        The precision matrix decomposes as prior + data:
            Λ = _prior_scalar·I + data_component

        After MacKay changes α and β, rescale both components so the
        matrix is consistent with the new hyperparameters.
        """
        alpha_new = self.alpha
        beta_new = self.beta

        if alpha_new == alpha_old and beta_new == beta_old:
            return

        prior_scalar = self._prior_scalar
        beta_ratio = beta_new / beta_old
        new_prior_scalar = prior_scalar * (alpha_new / alpha_old)
        diag_correction = new_prior_scalar - beta_ratio * prior_scalar

        if self.sparse:
            cov_inv = cast(csc_array, self.cov_inv_)
            cov_inv *= beta_ratio
            if diag_correction != 0.0:
                cov_inv.setdiag(cov_inv.diagonal() + diag_correction)
            self.cov_inv_ = cov_inv
        else:
            self.cov_inv_ *= beta_ratio
            if diag_correction != 0.0:
                diag_idx = np.diag_indices_from(self.cov_inv_)
                self.cov_inv_[diag_idx] += diag_correction

        self._prior_scalar = new_prior_scalar

    def partial_fit(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        had_prior_scalar = hasattr(self, "_prior_scalar")

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)  # type: ignore[arg-type]
        prior_decay = self.learning_rate**n_samples

        if had_prior_scalar:
            # Stabilized forgetting: decay the tracked prior contribution but
            # re-inject (1 - prior_decay) * alpha so it converges to alpha
            # instead of decaying to zero.  This ensures the EB-tuned prior
            # always regularizes new or dormant coefficients.
            self._prior_scalar = (
                prior_decay * self._prior_scalar + (1 - prior_decay) * self.alpha
            )

        # The stabilized re-injection amount that must be added to the
        # precision diagonal after the base class applies uniform decay.
        prior_reinjection = (1 - prior_decay) * self.alpha if had_prior_scalar else 0.0

        alpha_old = self.alpha
        beta_old = self.beta

        result = super().partial_fit(X, y, sample_weight)

        # Add the stabilized prior re-injection to the precision diagonal.
        # _fit_helper applied uniform decay (prior_decay * cov_inv_), so
        # the prior diagonal is prior_decay * old_prior_scalar.  We need
        # to add back (1 - prior_decay) * alpha to match _prior_scalar.
        if prior_reinjection != 0.0:
            if self.sparse:
                cov_inv = cast(csc_array, self.cov_inv_)
                cov_inv.setdiag(cov_inv.diagonal() + prior_reinjection)
                self.cov_inv_ = cov_inv
            else:
                diag_idx = np.diag_indices_from(self.cov_inv_)
                self.cov_inv_[diag_idx] += prior_reinjection
            if hasattr(self, "_factor"):
                del self._factor
                if self.sparse:
                    self._factor = create_sparse_factor(cast(csc_array, self.cov_inv_))

        if not had_prior_scalar:
            if hasattr(self, "_prior_scalar"):
                # fit() was called by super().partial_fit() and already
                # set _prior_scalar, sufficient stats, and ran the EB loop.
                return result

            # sample() previously called _initialize_prior (setting coef_),
            # so super().partial_fit() did an incremental update instead of
            # calling fit(). Initialize EB state for the first time.
            #
            # After the incremental update the precision matrix is:
            #   cov_inv_ = prior_decay * alpha_old * I + beta_old * X^T X
            # so the prior contribution to the diagonal is prior_decay * alpha_old.
            self._prior_scalar = prior_decay * alpha_old

        X_fit, y = check_X_y(
            X,  # type: ignore
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        if not had_prior_scalar:
            # First observation — initialize sufficient statistics.
            self._effective_n = float(y.shape[0])
            self._eff_yTy = float(y @ y)
            if isinstance(X_fit, csc_array):
                self._eff_XTy = np.asarray(X_fit.T @ y).ravel()
            else:
                self._eff_XTy = X_fit.T @ y
        else:
            # Accumulate sufficient statistics for the beta update.
            self._accumulate_stats(X_fit, y, prior_decay)

        self._eb_mackay_step_online()
        self._correct_precision(alpha_old, beta_old)

        return result

    def decay(
        self,
        X: Union[NDArray[Any], csc_array],
        *,
        decay_rate: Optional[float] = None,
    ) -> None:
        """Decay the precision matrix with stabilized prior re-injection.

        Uses stabilized forgetting (Kulhavy & Zarrop, 1993) so that the
        prior precision converges to ``alpha`` instead of decaying to zero.
        """
        if not hasattr(self, "coef_"):
            return

        if decay_rate is None:
            decay_rate = self.learning_rate

        assert X.shape is not None
        prior_decay = decay_rate ** X.shape[0]

        prior_reinjection = 0.0
        if hasattr(self, "_prior_scalar"):
            self._prior_scalar = (
                prior_decay * self._prior_scalar + (1 - prior_decay) * self.alpha
            )
            prior_reinjection = (1 - prior_decay) * self.alpha
        if hasattr(self, "_effective_n"):
            self._effective_n *= prior_decay
            self._eff_yTy *= prior_decay
            self._eff_XTy *= prior_decay

        # Base class applies uniform decay: cov_inv_ *= prior_decay
        super().decay(X, decay_rate=decay_rate)

        # Add back the stabilized prior re-injection to the diagonal.
        if prior_reinjection != 0.0:
            if self.sparse:
                cov_inv = cast(csc_array, self.cov_inv_)
                cov_inv.setdiag(cov_inv.diagonal() + prior_reinjection)
                self.cov_inv_ = cov_inv
            else:
                diag_idx = np.diag_indices_from(self.cov_inv_)
                self.cov_inv_[diag_idx] += prior_reinjection
            if hasattr(self, "_factor"):
                del self._factor
                if self.sparse:
                    self._factor = create_sparse_factor(cast(csc_array, self.cov_inv_))
