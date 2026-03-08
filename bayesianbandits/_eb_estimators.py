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
    trace_method : str, default="auto"
        Method for computing tr(Λ⁻¹) in the MacKay update.
        ``"auto"`` uses Takahashi recursion (exact) for sparse and
        Cholesky for dense.  ``"diagonal"`` uses the fast O(p)
        approximation tr(Λ⁻¹) ≈ Σ 1/Λᵢᵢ.

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
        trace_method: str = "auto",
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
        self.trace_method = trace_method

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
            trace_method=self.trace_method,
        )
        self.alpha = alpha_new
        self.beta = beta_new
        return log_ev

    def _eb_mackay_step_online(self) -> None:
        """MacKay step using accumulated sufficient statistics for beta."""
        alpha_new, beta_new, factor = mackay_update_normal_online(
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
            trace_method=self.trace_method,
        )
        self.alpha = alpha_new
        self.beta = beta_new
        if factor is not None:
            self._factor = factor

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

        After correction, eagerly refactorizes so the factor is ready
        for ``sample()`` without an extra factorization.
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

    def _reinject_prior(self, prior_reinjection: float) -> None:
        """Add stabilized prior re-injection to the precision diagonal.

        After exponential decay the prior contribution shrinks toward zero.
        This adds back ``prior_reinjection`` to every diagonal entry so that
        the prior converges to ``alpha`` instead (Kulhavy & Zarrop, 1993).
        """
        if prior_reinjection == 0.0:
            return
        if self.sparse:
            cov_inv = cast(csc_array, self.cov_inv_)
            cov_inv.setdiag(cov_inv.diagonal() + prior_reinjection)
            self.cov_inv_ = cov_inv
        else:
            diag_idx = np.diag_indices_from(self.cov_inv_)
            self.cov_inv_[diag_idx] += prior_reinjection
        if hasattr(self, "_factor"):
            del self._factor

    @_invalidate_cached_properties
    def _fit_helper(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> None:  # type: ignore[override]
        """Override to fold prior reinjection into precision construction.

        When ``_pending_reinjection > 0``, adds ``(1 - γ) · α · I`` to
        the precision during the decay + data update.  This means the
        factorization for the linear solve already reflects the
        reinjected prior — no separate ``_reinject_prior`` call or
        refactorization needed.
        """
        reinjection = getattr(self, "_pending_reinjection", 0.0)
        if reinjection == 0.0:
            # No reinjection needed — delegate to base class.
            super()._fit_helper(X, y, sample_weight)
            return

        # -- Below: base class logic with reinjection folded in --

        if self.sparse:
            X = csc_array(X)

        assert X.shape is not None

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        from ._estimators import compute_effective_weights

        effective_weights = compute_effective_weights(
            X.shape[0], sample_weight, self.learning_rate
        )

        prior_decay = self.learning_rate ** X.shape[0]

        if self.sparse:
            from scipy.sparse import diags as sp_diags

            W_sqrt = sp_diags(np.sqrt(effective_weights), format="csc")
            X_weighted = W_sqrt @ X
            cov_inv = cast(
                csc_array,
                prior_decay * self.cov_inv_ + self.beta * (X_weighted.T @ X_weighted),
            )
            # Fold in the prior reinjection
            cov_inv.setdiag(cov_inv.diagonal() + reinjection)
        else:
            X_weighted = X * np.sqrt(effective_weights)[:, np.newaxis]
            cov_inv = prior_decay * self.cov_inv_ + self.beta * (
                X_weighted.T @ X_weighted
            )
            cov_inv = cast(NDArray[np.float64], cov_inv)
            diag_idx = np.diag_indices_from(cov_inv)
            cov_inv[diag_idx] += reinjection

        y_weighted = y * effective_weights
        eta = prior_decay * self.cov_inv_ @ self.coef_ + self.beta * X.T @ y_weighted
        eta = cast(NDArray[np.float64], eta)

        if self.sparse:
            assert isinstance(cov_inv, csc_array)
            factor = create_sparse_factor(cov_inv)
            coef = factor.solve(eta)
            self._factor = factor
        else:
            from scipy.linalg import solve

            coef = solve(cov_inv, eta, check_finite=False, assume_a="pos")

        self.cov_inv_ = cov_inv
        self.coef_ = coef

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

        # Store reinjection amount for _fit_helper to fold in.
        # When > 0, _fit_helper adds this to the precision diagonal during
        # construction, avoiding a separate _reinject_prior + refactorization.
        self._pending_reinjection = (
            (1 - prior_decay) * self.alpha if had_prior_scalar else 0.0
        )

        alpha_old = self.alpha
        beta_old = self.beta

        result = super().partial_fit(X, y, sample_weight)

        self._pending_reinjection = 0.0

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

        self._reinject_prior(prior_reinjection)
