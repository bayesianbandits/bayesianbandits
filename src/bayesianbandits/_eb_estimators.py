"""Empirical Bayes estimators via evidence maximization.

Provides MacKay's update rules for Normal regression and for
Laplace-approximated GLMs, Minka's fixed-point iteration for
Dirichlet-Multinomial classification, and Minka's EM for Gamma-Poisson
rate estimation.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_array
from sklearn.utils.validation import check_X_y
from typing_extensions import Self

from ._blas_helpers import compute_eta_dense, dgemv, dsymv, update_precision_dense
from ._empirical_bayes import (
    accumulate_sufficient_stats,
    glm_log_likelihood,
    mackay_update_glm,
    mackay_update_normal_online,
    minka_update_dirichlet_multinomial,
    negbin_update_gamma_poisson,
)
from ._estimators import (
    BayesianGLM,
    DirichletClassifier,
    GammaRegressor,
    NormalRegressor,
    _invalidate_cached_properties,
    compute_effective_weights,
)
from ._gaussian import LaplaceApproximator, LinkFunction, PosteriorApproximator
from ._np_utils import groupby_array
from ._sparse_bayesian_linear_regression import (
    DenseFactor,
    PrecisionFactor,
    scale_factor,
)


class _StabilizedPriorMixin:
    """Precision-diagonal bookkeeping shared by the EB estimators whose
    posterior precision is ``_prior_scalar · I + data``.

    Hosts the in-place diagonal shift and the factor-drop that a shift
    forces; the estimators own ``_prior_scalar`` itself and decide when
    to re-inject.
    """

    sparse: bool
    cov_inv_: Any
    _factor_hint: Any

    def _drop_factor(self) -> None:
        """Drop the cached factor, keeping it as the hint ``_sparse_factor``
        refactorizes from: a diagonal shift leaves the pattern alone."""
        factor = self.__dict__.pop("_precision_factor")
        if self.sparse:
            self._factor_hint = factor

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
            self._shift_diagonal(cov_inv, prior_reinjection)
            self.cov_inv_ = cov_inv
        else:
            diag_idx = np.diag_indices_from(self.cov_inv_)
            self.cov_inv_[diag_idx] += prior_reinjection
        if "_precision_factor" in self.__dict__:
            self._drop_factor()

    def _shift_diagonal(self, cov_inv: csc_array, shift: float) -> None:
        """``cov_inv += shift * I`` in place.

        The diagonal is always stored (the prior is ``alpha * I``), so
        its positions, found once per pattern by running ``diagonal()``
        over entry numbers and cached against the index array's
        identity, make this a gather-add; ``setdiag`` relocates it on
        every call.
        """
        if shift == 0.0:
            return
        cached = self.__dict__.get("_diag_pos")
        if cached is None or cached[0] is not cov_inv.indices:
            values = cov_inv.data
            cov_inv.data = np.arange(1, values.size + 1, dtype=np.float64)
            pos = cov_inv.diagonal().astype(np.intp) - 1
            cov_inv.data = values
            if np.any(pos < 0):  # missing diagonal entry: let scipy insert it
                cov_inv.setdiag(cov_inv.diagonal() + shift)
                return
            cached = (cov_inv.indices, pos)
            self._diag_pos = cached
        cov_inv.data[cached[1]] += shift


class EmpiricalBayesNormalRegressor(_StabilizedPriorMixin, NormalRegressor):
    """Bayesian linear regression with empirical Bayes hyperparameter tuning.

    Extends :class:`NormalRegressor` with automatic optimization of the
    prior precision (``alpha``) and noise precision (``beta``) via
    MacKay's evidence maximization framework [1]_. During ``fit``, the
    hyperparameters are iteratively updated to maximize the log marginal
    likelihood. During ``partial_fit``, a single online MacKay step is
    performed using accumulated sufficient statistics.

    When ``learning_rate < 1``, exponential forgetting is applied to the
    precision matrix. To prevent the prior contribution from collapsing
    to zero under repeated decay, *stabilized forgetting* [2]_ re-injects
    a fixed prior floor after each decay step.

    Parameters
    ----------
    alpha : float, default=1.0
        Initial prior precision of the weights. The prior is
        :math:`w \\sim \\mathcal{N}(0, \\alpha^{-1} I)`. Updated
        automatically during fitting.
    beta : float, default=1.0
        Initial noise precision. The likelihood is
        :math:`y \\mid x, w \\sim \\mathcal{N}(x^T w, \\beta^{-1})`.
        Updated automatically during fitting.
    n_eb_iter : int, default=10
        Maximum number of empirical Bayes iterations during ``fit``.
        Each iteration re-fits the posterior and runs one MacKay update.
        Set to 0 to disable EB tuning during ``fit``.
    eb_tol : float, default=1e-4
        Convergence tolerance on the change in log marginal likelihood
        between successive EB iterations.
    learning_rate : float, default=1.0
        Decay rate for sequential updates. Values less than 1
        geometrically shrink the precision matrix on each call to
        ``decay`` or ``partial_fit``, enabling adaptation to
        non-stationary environments.
    sparse : bool, default=False
        If True, use sparse matrix operations for the precision matrix.
        Input ``X`` must be a ``scipy.sparse.csc_array``. When CHOLMOD
        is available (via ``scikit-sparse``), it is used for efficient
        Cholesky factorization; otherwise falls back to SuperLU.
    random_state : int, np.random.Generator, or None, default=None
        Controls the random number generator for ``sample``. Pass an
        int for reproducible results across calls.
    trace_method : {'auto', 'diagonal'}, default='auto'
        Method for computing :math:`\\operatorname{tr}(\\Lambda^{-1})`
        in the MacKay update for ``alpha``:

        - ``'auto'``: Uses Takahashi recursion (exact, exploits
          sparsity) for sparse matrices, Cholesky for dense.
        - ``'diagonal'``: Fast :math:`O(p)` approximation
          :math:`\\operatorname{tr}(\\Lambda^{-1}) \\approx
          \\sum_i 1/\\Lambda_{ii}`.

    Attributes
    ----------
    log_evidence_ : float
        Log marginal likelihood at the most recent MacKay step --
        the last iteration of ``fit``, then refreshed by every
        ``partial_fit`` -- or ``-inf`` if ``n_eb_iter=0``. Under
        forgetting this is the evidence of the *decayed* data.
    n_eb_iterations_ : int
        Number of EB iterations performed during the last ``fit``.
    eb_converged_ : bool
        Whether the EB loop converged within ``eb_tol`` during
        the last ``fit``.
    eb_updates_rejected_ : int
        Number of MacKay updates declined by the ill-conditioning
        guardrail since the last ``fit``, counting both ``fit``'s own
        iterations and every ``partial_fit`` since. A nonzero and
        growing count means the hyperparameters are pinned and the
        estimator is no longer maximizing the evidence -- see the
        ``beta``/``alpha`` guardrail in :mod:`._empirical_bayes`.

    See Also
    --------
    NormalRegressor : Base estimator without empirical Bayes tuning.
    BayesianGLM : Bayesian GLM for non-Gaussian likelihoods (logistic,
        Poisson).

    Notes
    -----
    **Evidence maximization (MacKay's update rules)**

    Given the posterior precision
    :math:`\\Lambda = \\alpha I + \\beta X^T X` and MAP estimate
    :math:`\\hat{w}`, the effective number of well-determined
    parameters is:

    .. math::

        \\gamma = p - \\alpha \\operatorname{tr}(\\Lambda^{-1})

    The hyperparameters are updated as [1]_:

    .. math::

        \\alpha_{\\text{new}} = \\frac{\\gamma}{\\hat{w}^T \\hat{w}},
        \\qquad
        \\beta_{\\text{new}} = \\frac{N - \\gamma}{
        \\| y - X \\hat{w} \\|^2}

    **Stabilized forgetting**

    Under exponential decay with rate :math:`\\gamma < 1`, the prior
    contribution to the precision diagonal decays as
    :math:`\\gamma^t \\alpha \\to 0`. Stabilized forgetting [2]_
    re-injects :math:`(1 - \\gamma^n) \\alpha` after each decay step,
    so the prior contribution converges to :math:`\\alpha` instead of
    zero:

    .. math::

        s_{t+n} = \\gamma^n s_t + (1 - \\gamma^n) \\alpha

    where :math:`s_t` tracks the cumulative prior scalar on the
    diagonal.

    References
    ----------
    .. [1] MacKay, D. J. C. (1992). "Bayesian Interpolation."
       *Neural Computation*, 4(3), 415-447.

    .. [2] Kulhavy, R. & Zarrop, M. B. (1993). "On a general concept of
       forgetting." *International Journal of Control*, 58(4), 905-924.

    Examples
    --------
    Batch fitting with automatic hyperparameter tuning:

    >>> import numpy as np
    >>> from bayesianbandits import EmpiricalBayesNormalRegressor
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 3))
    >>> w_true = np.array([1.0, -0.5, 0.0])
    >>> y = X @ w_true + 0.1 * rng.standard_normal(100)
    >>> model = EmpiricalBayesNormalRegressor(random_state=42)
    >>> _ = model.fit(X, y)
    >>> model.eb_converged_
    True
    >>> np.round(model.coef_, 2)  # Close to [1.0, -0.5, 0.0]
    array([ 1.  , -0.49, -0.02])

    Online learning with forgetting for non-stationary environments:

    >>> model = EmpiricalBayesNormalRegressor(
    ...     learning_rate=0.99, random_state=42
    ... )
    >>> for i in range(10):  # doctest: +SKIP
    ...     X_batch = rng.standard_normal((5, 3))
    ...     y_batch = X_batch @ w_true + 0.1 * rng.standard_normal(5)
    ...     model.partial_fit(X_batch, y_batch)
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

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Posterior mean, written by every fit like the plain attribute
        it replaces -- with one solve deferred behind it.

        When a MacKay step shifts the precision diagonal,
        :meth:`_correct_precision` has to re-solve the mean against the
        rescaled precision, and that solve needs a factorization the
        cheap factor updates do not cover.  The next ``partial_fit``
        factors its own updated precision anyway and needs only
        ``Λ·coef_`` from this one -- which is the information vector the
        solve was against.  So the solve is left pending here and paid
        only when something actually reads the mean: ``predict``,
        ``sample`` and ``decay`` all reach it through their fitted
        checks, and ``partial_fit`` hands the pending vector straight to
        :meth:`_fit_helper` instead.
        """
        eta = self.__dict__.pop("_pending_eta", None)
        if eta is not None:
            self.__dict__["_coef"] = self._precision_factor.solve(eta)
        try:
            return self.__dict__["_coef"]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute 'coef_'"
            ) from None

    @coef_.setter
    def coef_(self, value: NDArray[np.float64]) -> None:
        self.__dict__.pop("_pending_eta", None)
        self.__dict__["_coef"] = value

    def _eb_mackay_step_online(self) -> None:
        """MacKay step using accumulated sufficient statistics for beta.

        Records the log evidence and whether the guardrail declined the
        update, so a stalled EB loop is visible rather than silent --
        the Dirichlet and Gamma estimators already keep their evidence
        across ``partial_fit``.
        """
        update = mackay_update_normal_online(
            self.coef_,
            self.cov_inv_,
            self.alpha,
            self.beta,
            self._prior_scalar,
            self._effective_n,
            self._eff_yTy,
            self._eff_XTy,
            factor=self._precision_factor,
            trace_method=self.trace_method,
        )
        self.alpha = update.alpha
        self.beta = update.beta
        self.log_evidence_ = update.log_evidence
        if update.rejected:
            self.eb_updates_rejected_ += 1

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
        """
        Fit the model with empirical Bayes hyperparameter tuning.

        Iteratively optimizes ``alpha`` (prior precision) and ``beta``
        (noise precision) by maximizing the log marginal likelihood
        using MacKay's evidence framework, then performs a final
        posterior update with the converged hyperparameters.

        Parameters
        ----------
        X_fit : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample. If None, all samples
            are given weight 1.0.

        Returns
        -------
        self : EmpiricalBayesNormalRegressor
            Fitted estimator with tuned ``alpha`` and ``beta``.

        See Also
        --------
        partial_fit : Incremental update with one online MacKay step.
        """
        X_fit, y = check_X_y(
            X_fit,  # type: ignore
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        prior_decay = self.learning_rate ** y.shape[0]

        self.eb_updates_rejected_ = 0

        if self.n_eb_iter > 0:
            # Sufficient stats are constant during batch fitting (no decay)
            effective_n = float(y.shape[0])
            eff_yTy = float(y @ y)
            if isinstance(X_fit, csc_array):
                eff_XTy = np.asarray(X_fit.T @ y, dtype=np.float64).ravel()
            else:
                eff_XTy = np.asarray(X_fit.T @ y, dtype=np.float64)

            prev_evidence = -math.inf
            converged = False
            iterations = 0
            log_ev = -math.inf

            for i in range(self.n_eb_iter):
                self._initialize_prior(X_fit)
                self._fit_helper(X_fit, y, sample_weight)
                # After a fresh fit: Λ = prior_decay·α·I + β·XᵀWX
                self._prior_scalar = prior_decay * self.alpha

                update = mackay_update_normal_online(
                    self.coef_,
                    self.cov_inv_,
                    self.alpha,
                    self.beta,
                    self._prior_scalar,
                    effective_n,
                    eff_yTy,
                    eff_XTy,
                    factor=self._precision_factor,
                    trace_method=self.trace_method,
                )
                self.alpha = update.alpha
                self.beta = update.beta
                log_ev = update.log_evidence
                if update.rejected:
                    self.eb_updates_rejected_ += 1
                iterations = i + 1

                if abs(log_ev - prev_evidence) < self.eb_tol:
                    converged = True
                    break
                prev_evidence = log_ev

            # Final refit with converged hyperparams
            self._initialize_prior(X_fit)
            self._fit_helper(X_fit, y, sample_weight)

            self.log_evidence_ = log_ev
            self.n_eb_iterations_ = iterations
            self.eb_converged_ = converged
        else:
            self._initialize_prior(X_fit)
            self._fit_helper(X_fit, y, sample_weight)
            self.log_evidence_ = -math.inf
            self.n_eb_iterations_ = 0
            self.eb_converged_ = False

        # The prior's contribution to the diagonal is the decayed alpha,
        # not alpha itself -- _fit_helper scales the prior by prior_decay.
        self._prior_scalar = prior_decay * self.alpha

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
        matrix is consistent with the new hyperparameters, then re-solve
        the posterior mean against the rescaled matrix.

        The mean is ``Λ⁻¹·η`` with ``η = β·Xᵀy`` (decayed): the prior has
        zero mean, so it contributes nothing to the information vector.
        Leaving ``coef_`` as it was would make the next ``partial_fit``
        read ``Λ_new·θ_old`` as the prior information, silently
        re-applying the old shrinkage and the old β. Since
        ``Λ_old·θ_old = η_old`` exactly, the information under the new
        noise precision is ``η_new = (β_new/β_old)·Λ_old·θ_old`` and the
        corrected mean is ``Λ_new⁻¹·η_new``.

        A pure rescale (``diag_correction == 0``, i.e. α and β moved by
        the same ratio) is absorbed by the cached factorization, the way
        ``decay`` absorbs its own, and leaves the mean unchanged
        (``Λ_new`` and ``η_new`` share the ratio). A diagonal shift is a
        rank-p change that no cheap factor update covers, so there the
        factor is dropped and the solve is deferred to whoever reads
        ``coef_`` -- ``partial_fit`` needs only ``η_new`` and skips it.
        """
        alpha_new = self.alpha
        beta_new = self.beta

        if alpha_new == alpha_old and beta_new == beta_old:
            return

        prior_scalar = self._prior_scalar
        beta_ratio = beta_new / beta_old
        new_prior_scalar = prior_scalar * (alpha_new / alpha_old)
        diag_correction = new_prior_scalar - beta_ratio * prior_scalar

        if diag_correction == 0.0:
            # Pure rescale: Λ and η share the ratio, so the mean is unchanged.
            self._scale_precision(beta_ratio)
            self._prior_scalar = new_prior_scalar
            if "_precision_factor" in self.__dict__:
                self._precision_factor = scale_factor(
                    self._precision_factor, beta_ratio
                )
            return

        # Λ_old·θ_old, read before Λ is touched. The dense matrix is
        # upper-triangle-only (dsyrk), which dsymv handles.
        if self.sparse:
            eta_old = np.asarray(
                cast(csc_array, self.cov_inv_) @ self.coef_, dtype=np.float64
            )
        else:
            eta_old = dsymv(1.0, self.cov_inv_, self.coef_)
        eta_new = beta_ratio * eta_old

        self._scale_precision(beta_ratio)
        if self.sparse:
            cov_inv = cast(csc_array, self.cov_inv_)
            self._shift_diagonal(cov_inv, diag_correction)
            self.cov_inv_ = cov_inv
        else:
            diag_idx = np.diag_indices_from(self.cov_inv_)
            self.cov_inv_[diag_idx] += diag_correction

        self._prior_scalar = new_prior_scalar
        if "_precision_factor" in self.__dict__:
            self._drop_factor()
        # Leave the solve pending: see the ``coef_`` property.  Written
        # past the setter, which clears exactly this.
        self.__dict__["_pending_eta"] = eta_new

    def _scale_precision(self, ratio: float) -> None:
        """Multiply ``cov_inv_`` in place by ``ratio``."""
        if self.sparse:
            cov_inv = cast(csc_array, self.cov_inv_)
            cov_inv *= ratio
            self.cov_inv_ = cov_inv
        else:
            self.cov_inv_ *= ratio

    @_invalidate_cached_properties
    def _fit_helper(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> None:  # type: ignore[override]
        """Override to fold prior reinjection into precision construction,
        and to take the prior information vector from ``partial_fit``.

        When ``_pending_reinjection > 0``, adds ``(1 - γ) · α · I`` to
        the precision during the decay + data update.  This means the
        factorization for the linear solve already reflects the
        reinjected prior — no separate ``_reinject_prior`` call or
        refactorization needed.

        When ``_pending_prior_eta`` is set, a MacKay correction left the
        posterior mean unsolved (see the ``coef_`` property) and handed
        the information vector here instead.  ``Λ·coef_`` is exactly
        that vector, so using it skips both the deferred solve and the
        matvec that would undo it.
        """
        reinjection = getattr(self, "_pending_reinjection", 0.0)
        pending_eta = getattr(self, "_pending_prior_eta", None)
        # Consumed here, so partial_fit can tell an update that raised
        # before this point from one that used the vector.
        self._pending_prior_eta = None
        if reinjection == 0.0 and pending_eta is None:
            # Nothing to fold in — delegate to base class.
            super()._fit_helper(X, y, sample_weight)
            return

        # -- Below: base class logic with reinjection folded in --

        if self.sparse:
            X = csc_array(X)

        assert X.shape is not None

        from ._estimators import compute_effective_weights

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        effective_weights = compute_effective_weights(
            X.shape[0], sample_weight, self.learning_rate
        )

        prior_decay = self.learning_rate ** X.shape[0]

        y_weighted = y * effective_weights

        if self.sparse:
            # Element-wise scaling; absorb beta into weights
            assert isinstance(X, csc_array)
            w_sqrt = np.sqrt(self.beta * effective_weights)
            X_weighted = X.multiply(w_sqrt.reshape(-1, 1)).tocsc()
            prior = cast(csc_array, self.cov_inv_)
            if prior_decay != 1.0:
                # New data array, not in-place: leaves cov_inv_ untouched if the solve below fails.
                prior = csc_array(
                    (prior.data * prior_decay, prior.indices, prior.indptr),
                    shape=prior.shape,
                )
            prior_eta = (
                prior @ self.coef_ if pending_eta is None else prior_decay * pending_eta
            )
            cov_inv = cast(csc_array, prior + X_weighted.T @ X_weighted)
            self._shift_diagonal(cov_inv, reinjection)
            eta = cast(NDArray[np.float64], prior_eta + X.T @ (self.beta * y_weighted))
            factor: PrecisionFactor = self._sparse_factor(cov_inv)
            coef = factor.solve(eta)
            self._precision_factor = factor
        else:
            w_sqrt = np.sqrt(effective_weights)
            X_weighted = X * w_sqrt[:, np.newaxis]
            if pending_eta is None:
                eta = compute_eta_dense(
                    prior_decay, self.cov_inv_, self.coef_, self.beta, X, y_weighted
                )
            else:
                # As compute_eta_dense, but dgemv accumulates onto the already-done dsymv term.
                eta = dgemv(
                    self.beta,
                    X,
                    y_weighted,
                    trans=1,
                    beta=1.0,
                    y=prior_decay * pending_eta,
                    overwrite_y=True,
                )
            # Copy, not a view: dsyrk below writes this buffer, so aliasing cov_inv_
            # would corrupt it before cho_factor gets a chance to reject the update.
            prior_scaled = np.array(self.cov_inv_, order="F", copy=True)
            if prior_decay != 1.0:
                prior_scaled *= prior_decay
            if reinjection != 0.0:
                diag_idx = np.diag_indices_from(prior_scaled)
                prior_scaled[diag_idx] += reinjection
            # Fused X^T W X + prior via dsyrk (upper triangle only)
            cov_inv = update_precision_dense(self.beta, X_weighted, prior_scaled)
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
        Incrementally update the posterior and retune hyperparameters.

        Performs a recursive Bayesian update (delegating to
        ``NormalRegressor.partial_fit``), then runs one online MacKay
        step to adjust ``alpha`` and ``beta`` using accumulated
        sufficient statistics. The precision matrix is corrected to
        reflect the updated hyperparameters.

        When ``learning_rate < 1``, stabilized forgetting (Kulhavy &
        Zarrop 1993) re-injects ``(1 - γ^n)·α`` into the precision
        diagonal so that the prior contribution converges to ``alpha``
        instead of decaying to zero.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample. If None, all samples
            are given weight 1.0.

        Returns
        -------
        self : EmpiricalBayesNormalRegressor
            Updated estimator with retuned hyperparameters.

        See Also
        --------
        fit : Fit from scratch with full EB iteration loop.
        decay : Increase uncertainty without observing new data.
        """
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

        # Hand any solve the last MacKay correction left pending to
        # _fit_helper as the prior information vector -- it is Λ·coef_
        # exactly.  Taken out of __dict__ here, before super()'s
        # check_is_fitted would read coef_ and force the solve.
        self._pending_prior_eta = self.__dict__.pop("_pending_eta", None)

        alpha_old = self.alpha
        beta_old = self.beta

        try:
            result = super().partial_fit(X, y, sample_weight)
        finally:
            self._pending_reinjection = 0.0
            if self._pending_prior_eta is not None:
                # The update raised before _fit_helper consumed it; hand
                # it back so coef_ stays solvable and the correction is
                # not silently dropped.
                self.__dict__["_pending_eta"] = self._pending_prior_eta
                self._pending_prior_eta = None

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
            # fit() never ran on this path, so the EB reporting state that
            # _eb_mackay_step_online maintains has to start here.
            self.eb_updates_rejected_ = 0
            self.n_eb_iterations_ = 0
            self.eb_converged_ = False

        X_fit, y = check_X_y(
            X,  # type: ignore
            y,
            copy=False,
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
        """
        Decay the precision matrix with stabilized prior re-injection.

        Applies exponential forgetting ``Λ_new = γ^n · Λ_old`` and
        then re-injects ``(1 - γ^n)·α`` onto the diagonal, following
        stabilized forgetting (Kulhavy & Zarrop 1993). This ensures
        the prior contribution to the precision converges to ``alpha``
        rather than decaying to zero under repeated decay steps.

        Sufficient statistics (``_effective_n``, ``_eff_yTy``,
        ``_eff_XTy``) used for the online MacKay beta update are
        also decayed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Used only to determine the number of time steps ``n`` for
            the decay exponent ``γ^n``. The actual feature values are
            not used.
        decay_rate : float, default=None
            Decay factor :math:`\\gamma` in (0, 1]. If None, uses
            ``self.learning_rate``.

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


class EmpiricalBayesGLM(_StabilizedPriorMixin, BayesianGLM):
    """Bayesian GLM with empirical Bayes tuning of the prior precision.

    Extends :class:`BayesianGLM` with automatic optimization of the
    prior precision ``alpha`` via MacKay's evidence framework [1]_
    applied to the Laplace approximation. The GLM posterior is already
    a Gaussian centred on the MAP with the Hessian as precision, so
    MacKay's update slots in directly: with
    :math:`\\gamma = p - s\\,\\operatorname{tr}(\\Lambda^{-1})` (``s`` the
    prior's contribution to the diagonal of :math:`\\Lambda`),
    :math:`\\alpha_{\\text{new}} = \\gamma / \\|\\theta_{\\text{MAP}}\\|^2`.

    During ``fit``, IRLS and MacKay steps alternate until the Laplace
    log evidence converges. During ``partial_fit``, one MacKay step is
    taken on the current Laplace approximation; the resulting change to
    ``alpha`` is folded into the next update's factorization rather
    than costing one of its own, so between updates the posterior is
    the coherent one under the previous ``alpha``.

    MacKay's update treats the data Hessian as fixed in ``alpha``,
    while the Laplace evidence also depends on ``alpha`` through the
    curvature at the mode, so the fixed point lies very near but not
    exactly at the evidence maximum (relative discrepancies of order
    1e-6 in practice), and the evidence can tick down by that much in
    the final EB iterations.

    When ``learning_rate < 1``, *stabilized forgetting* [2]_ re-injects
    ``(1 - γⁿ)·alpha`` onto the precision diagonal after each decay so
    the prior's contribution converges to ``alpha`` instead of
    vanishing, which keeps ``alpha`` tuning load-bearing indefinitely.

    Parameters
    ----------
    alpha : float, default=1.0
        Initial prior precision. Updated automatically during fitting.
    link : {'logit', 'log'}, default='logit'
        Link function; see :class:`BayesianGLM`.
    n_eb_iter : int, default=10
        Maximum number of EB iterations during ``fit``. Each iteration
        re-runs the posterior approximation from the prior and takes
        one MacKay step. Set to 0 to disable EB tuning during ``fit``.
    eb_tol : float, default=1e-4
        Convergence tolerance on the change in log evidence between
        successive EB iterations.
    learning_rate : float, default=1.0
        Decay rate for sequential updates; see :class:`BayesianGLM`.
    approximator : PosteriorApproximator, optional
        Posterior approximation strategy. Defaults to
        ``LaplaceApproximator(n_iter=25, tol=1e-6)`` rather than the
        base class's 5 fixed iterations: the evidence is only a Laplace
        evidence when the Hessian is taken at the mode, so ``fit``
        needs IRLS to actually converge. With an ``RVGAApproximator``
        the precision is an expected rather than observed curvature and
        ``log_evidence_`` is a heuristic; the ``alpha`` update is still
        well-defined. A custom approximator must accept the
        ``prior_floor``, ``prior_shift`` and ``coef_init`` keywords of
        :class:`PosteriorApproximator`: the first two are how this
        estimator applies stabilized forgetting and MacKay's change to
        ``alpha`` without a factorization of its own, the last is the
        warm start between EB iterations of ``fit``.
    sparse : bool, default=False
        Use sparse precision matrices; see :class:`BayesianGLM`.
    random_state : int, np.random.Generator, or None, default=None
        Controls the random number generator for ``sample``.
    trace_method : {'auto', 'diagonal'}, default='auto'
        Method for :math:`\\operatorname{tr}(\\Lambda^{-1})` in the
        MacKay update; see :class:`EmpiricalBayesNormalRegressor`.

    Attributes
    ----------
    log_evidence_ : float
        Laplace log evidence at the most recent MacKay step, or
        ``-inf`` if ``n_eb_iter=0``. After ``fit`` it is exact for the
        fitted data. Under ``partial_fit`` the log-likelihood term is a
        decayed running sum of each batch's log-likelihood at the MAP
        right after that batch, since earlier batches are not
        re-evaluated at later coefficients.
    n_eb_iterations_ : int
        Number of EB iterations performed during the last ``fit``.
    eb_converged_ : bool
        Whether the EB loop converged within ``eb_tol`` during the
        last ``fit``.
    eb_updates_rejected_ : int
        Number of MacKay updates declined by the ill-conditioning
        guardrail since the last ``fit``; see
        :class:`EmpiricalBayesNormalRegressor`.

    See Also
    --------
    BayesianGLM : Base estimator without empirical Bayes tuning.
    EmpiricalBayesNormalRegressor : The Gaussian-likelihood analogue.

    References
    ----------
    .. [1] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    .. [2] Kulhavý, R. and Zarrop, M. B. (1993). "On a general concept
       of forgetting", International Journal of Control 58(4), 905-924.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        link: LinkFunction = "logit",
        n_eb_iter: int = 10,
        eb_tol: float = 1e-4,
        learning_rate: float = 1.0,
        approximator: Optional[PosteriorApproximator] = None,
        sparse: bool = False,
        random_state: Union[int, np.random.Generator, None] = None,
        trace_method: str = "auto",
    ) -> None:
        super().__init__(
            alpha=alpha,
            link=link,
            learning_rate=learning_rate,
            approximator=approximator,
            sparse=sparse,
            random_state=random_state,
        )
        self.n_eb_iter = n_eb_iter
        self.eb_tol = eb_tol
        self.trace_method = trace_method

    def _initialize_prior(self, X: Union[NDArray[Any], csc_array]) -> None:
        super()._initialize_prior(X)
        if self.approximator is None:
            self.approximator_ = LaplaceApproximator(n_iter=25, tol=1e-6)

    def _restart_from_prior(self, X: Union[NDArray[Any], csc_array]) -> None:
        """Reset to the prior ``alpha·I`` for the next EB iteration, but
        keep what the previous iteration learned about the problem: its
        sparse factor (the pattern of ``alpha·I + H`` never changes, so
        the next IRLS refactorizes numerically instead of re-analysing)
        and its mode (a warm start one or two Newton steps from the new
        one, instead of starting from zero)."""
        warm_start = self.__dict__.get("coef_")
        hint = self.__dict__.pop("_precision_factor", None)
        self._initialize_prior(X)
        if warm_start is not None and warm_start.shape == self.coef_.shape:
            self._warm_start = warm_start
        if hint is not None and self.sparse:
            self._factor_hint = hint

    @_invalidate_cached_properties
    def _fit_helper(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> None:
        """Base-class update with the stabilized-forgetting floor and any
        deferred alpha correction passed through, so both are folded
        into the approximator's own factorization rather than costing
        a second one."""
        prior_factor: Optional[Any] = self._take_factor_hint() if self.sparse else None
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
            prior_floor=getattr(self, "_pending_floor", 0.0),
            prior_shift=getattr(self, "_pending_shift", 0.0),
            coef_init=self.__dict__.pop("_warm_start", None),
        )
        self._pending_shift = 0.0
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
        elif prior_factor is not None:
            self._factor_hint = prior_factor

    def _log_likelihood(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]],
    ) -> float:
        """Log-likelihood at ``coef_`` under the same effective row
        weights (sample weight times within-batch decay) the posterior
        update used."""
        weights = compute_effective_weights(
            y.shape[0], sample_weight, self.learning_rate
        )
        return glm_log_likelihood(X, y, self.coef_, self.link, weights)

    def _eb_mackay_step(self) -> None:
        update = mackay_update_glm(
            self.coef_,
            self.cov_inv_,
            self.alpha,
            self._prior_scalar - self._pending_shift,
            self._effective_n,
            self._eff_loglik,
            factor=self._precision_factor,
            trace_method=self.trace_method,
        )
        self.alpha = update.alpha
        self.log_evidence_ = update.log_evidence
        if update.rejected:
            self.eb_updates_rejected_ += 1

    def fit(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """
        Fit the model with empirical Bayes tuning of ``alpha``.

        Alternates the posterior approximation (IRLS from the prior
        ``alpha·I``) with MacKay updates of ``alpha`` until the Laplace
        log evidence changes by less than ``eb_tol``, then refits with
        the converged ``alpha``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values; 0/1 for ``link='logit'``, counts for
            ``link='log'``.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample.

        Returns
        -------
        self : EmpiricalBayesGLM

        See Also
        --------
        partial_fit : Incremental update with one online MacKay step.
        """
        X_fit, y = check_X_y(
            X,  # type: ignore
            y,
            copy=True,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        prior_decay = self.learning_rate ** y.shape[0]
        self.eb_updates_rejected_ = 0
        self._pending_floor = 0.0
        self._pending_shift = 0.0
        self._effective_n = float(y.shape[0])

        if self.n_eb_iter > 0:
            prev_evidence = -math.inf
            converged = False
            iterations = 0
            log_ev = -math.inf

            for i in range(self.n_eb_iter):
                self._restart_from_prior(X_fit)
                self._fit_helper(X_fit, y, sample_weight)
                # After a fresh fit: Λ = prior_decay·α·I + H_data
                self._prior_scalar = prior_decay * self.alpha
                self._eff_loglik = self._log_likelihood(X_fit, y, sample_weight)

                self._eb_mackay_step()
                log_ev = self.log_evidence_
                iterations = i + 1

                if abs(log_ev - prev_evidence) < self.eb_tol:
                    converged = True
                    break
                prev_evidence = log_ev

            self._restart_from_prior(X_fit)
            self._fit_helper(X_fit, y, sample_weight)

            self.log_evidence_ = log_ev
            self.n_eb_iterations_ = iterations
            self.eb_converged_ = converged
        else:
            self._initialize_prior(X_fit)
            self._fit_helper(X_fit, y, sample_weight)
            self.log_evidence_ = -math.inf
            self.n_eb_iterations_ = 0
            self.eb_converged_ = False

        self._prior_scalar = prior_decay * self.alpha
        self._eff_loglik = self._log_likelihood(X_fit, y, sample_weight)
        return self

    def _correct_precision(self, alpha_old: float) -> None:
        """Book the prior's rescale to the new alpha, to be applied by the
        next update.

        ``Λ = _prior_scalar·I + H_data``; only the prior part moves, by
        the ratio ``alpha / alpha_old``. That is a diagonal shift, which
        no cheap factor update covers, so rather than refactorizing here
        the shift is deferred to the next ``partial_fit`` or ``decay``,
        where a factorization happens anyway. Until then ``coef_`` and
        ``cov_inv_`` stay the coherent posterior under ``alpha_old``.

        The deferral is exact. Around the mode the log-likelihood's
        quadratic model has information ``H·θ_old + g = Λ_old·θ_old``
        (the gradient at the mode is ``alpha_old·θ_old``), and the next
        update reads exactly that from ``(cov_inv_, coef_)`` as the
        prior's eta while the shift lands on the precision alone: the
        same update as re-solving ``Λ_new⁻¹·Λ_old·θ_old`` now.
        ``_prior_scalar`` tracks the logical prior contribution, so it
        runs ahead of the stored diagonal by ``_pending_shift``.
        """
        if self.alpha == alpha_old:
            return
        ratio = self.alpha / alpha_old
        self._pending_shift += (ratio - 1.0) * self._prior_scalar
        self._prior_scalar *= ratio

    def partial_fit(
        self,
        X: Union[NDArray[Any], csc_array],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """
        Incrementally update the posterior and retune ``alpha``.

        Runs the base-class update from the current posterior (with
        stabilized re-injection of ``(1 - γⁿ)·alpha`` when
        ``learning_rate < 1``, and any prior rescale the previous
        MacKay step left pending), then takes one MacKay step on the
        resulting Laplace approximation. The step's change to ``alpha``
        reaches the precision at the next update, so between updates
        the posterior is the one under the previous ``alpha``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample.

        Returns
        -------
        self : EmpiricalBayesGLM

        See Also
        --------
        fit : Fit from scratch with the full EB loop.
        decay : Increase uncertainty without observing new data.
        """
        had_prior_scalar = hasattr(self, "_prior_scalar")

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)  # type: ignore[arg-type]
        prior_decay = self.learning_rate**n_samples

        if had_prior_scalar:
            self._prior_scalar = (
                prior_decay * self._prior_scalar + (1 - prior_decay) * self.alpha
            )
        self._pending_floor = self.alpha if had_prior_scalar else 0.0

        alpha_old = self.alpha

        result = super().partial_fit(X, y, sample_weight)

        self._pending_floor = 0.0

        if not had_prior_scalar:
            if hasattr(self, "_prior_scalar"):
                # fit() ran inside super().partial_fit() and did the EB loop.
                return result

            # sample()/predict() previously called _initialize_prior, so the
            # base class took the incremental path instead of fit(). The
            # precision is prior_decay·alpha_old·I + H_data; start the EB
            # state here.
            self._prior_scalar = prior_decay * alpha_old
            self._pending_shift = 0.0
            self.eb_updates_rejected_ = 0
            self.n_eb_iterations_ = 0
            self.eb_converged_ = False
            self._effective_n = 0.0
            self._eff_loglik = 0.0

        X_fit, y = check_X_y(
            X,  # type: ignore
            y,
            copy=False,
            ensure_2d=True,
            dtype=np.float64,
            accept_sparse="csc" if self.sparse else False,
        )

        self._effective_n = prior_decay * self._effective_n + float(y.shape[0])
        self._eff_loglik = prior_decay * self._eff_loglik + self._log_likelihood(
            X_fit, y, sample_weight
        )

        self._eb_mackay_step()
        self._correct_precision(alpha_old)

        return result

    def decay(
        self,
        X: Union[NDArray[Any], csc_array],
        *,
        decay_rate: Optional[float] = None,
    ) -> None:
        """
        Decay the precision matrix with stabilized prior re-injection.

        Applies ``Λ_new = γⁿ·Λ_old`` and re-injects ``(1 - γⁿ)·alpha``
        onto the diagonal (Kulhavý & Zarrop 1993), so the prior's
        contribution converges to ``alpha`` rather than zero. The
        running statistics behind ``log_evidence_`` decay alongside.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Used only for its number of rows ``n``.
        decay_rate : float, default=None
            Decay factor in (0, 1]. If None, uses ``learning_rate``.
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
            # The pending alpha correction is part of the prior, so it
            # decays with it; the diagonal is being shifted anyway.
            prior_reinjection = (
                1 - prior_decay
            ) * self.alpha + prior_decay * self._pending_shift
            self._pending_shift = 0.0
        if hasattr(self, "_effective_n"):
            self._effective_n *= prior_decay
            self._eff_loglik *= prior_decay

        super().decay(X, decay_rate=decay_rate)

        self._reinject_prior(prior_reinjection)


class EmpiricalBayesDirichletClassifier(DirichletClassifier):
    """Dirichlet-Multinomial classifier with empirical Bayes prior tuning.

    Extends :class:`DirichletClassifier` with automatic optimization of
    the Dirichlet prior concentration parameters via Minka's fixed-point
    iteration [1]_ for the Dirichlet-Multinomial marginal likelihood.

    During ``fit``, the prior is iteratively updated to maximize the
    marginal likelihood across all groups. During ``partial_fit``, a
    single Minka step is performed using the current group posteriors.

    The prior is decomposed as ``α_k = s · m_k`` where ``s`` is the
    scalar concentration (prior strength) and ``m_k`` is the base
    measure (prior shape). The ``m`` and ``s`` updates are alternated
    for faster convergence.

    When ``learning_rate < 1``, exponential forgetting is applied.
    Stabilized forgetting re-injects the EB-tuned prior after each
    decay step, ensuring the prior contribution converges to the tuned
    value rather than zero.

    Parameters
    ----------
    alphas : dict of {int or str: float}
        Initial prior concentration parameters for each class.
    n_eb_iter : int, default=10
        Maximum number of EB iterations during ``fit``.
    eb_tol : float, default=1e-4
        Convergence tolerance on change in log marginal likelihood.
    learning_rate : float, default=1.0
        Decay rate for sequential updates.
    random_state : int, np.random.Generator, or None, default=None
        Controls RNG for ``sample``.

    Attributes
    ----------
    log_evidence_ : float
        Log marginal likelihood at convergence.
    n_eb_iterations_ : int
        Number of EB iterations in last ``fit``.
    eb_converged_ : bool
        Whether the EB loop converged within ``eb_tol``.

    See Also
    --------
    DirichletClassifier : Base estimator without empirical Bayes tuning.
    EmpiricalBayesNormalRegressor : EB tuning for Normal regression.

    Notes
    -----
    **Minka's fixed-point iteration**

    Given G groups with effective counts ``c_gk`` (derived from the
    difference between posterior and prior concentrations), the
    per-component update is [1]_:

    .. math::

        \\alpha_k^{\\text{new}} = \\alpha_k \\cdot
        \\frac{\\sum_g [\\psi(c_{gk} + \\alpha_k) - \\psi(\\alpha_k)]}
             {\\sum_g [\\psi(c_g + \\alpha_0) - \\psi(\\alpha_0)]}

    where :math:`\\psi` is the digamma function,
    :math:`\\alpha_0 = \\sum_k \\alpha_k`, and
    :math:`c_g = \\sum_k c_{gk}`.

    **Stabilized forgetting**

    Every time ``learning_rate`` :math:`\\gamma < 1` causes decay
    (both in ``partial_fit`` and in ``decay``), the prior is
    re-injected:

    .. math::

        \\boldsymbol{\\alpha}_g \\leftarrow
        \\gamma^n \\boldsymbol{\\alpha}_g
        + (1 - \\gamma^n) \\boldsymbol{\\alpha}^{\\text{prior}}

    This ensures effective counts are
    :math:`\\gamma^n \\cdot \\text{counts}` and the prior
    contribution never vanishes.

    References
    ----------
    .. [1] Minka, T. P. (2000; revised 2003, 2012). "Estimating a
       Dirichlet distribution." Technical report, MIT.

    Examples
    --------
    Basic classification with EB-tuned prior:

    >>> import numpy as np
    >>> from bayesianbandits import EmpiricalBayesDirichletClassifier
    >>> X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(-1, 1)
    >>> y = np.array([1, 1, 2, 2, 2, 1, 1, 1, 2])
    >>> clf = EmpiricalBayesDirichletClassifier(
    ...     {1: 1, 2: 1}, random_state=0
    ... )
    >>> clf.fit(X, y)
    EmpiricalBayesDirichletClassifier(alphas={1: ..., 2: ...}, random_state=0)

    Misspecified priors converge to the true shape. Here the true
    DGP is ``Dir(3, 1)`` (75/25 split) but the initial prior
    heavily favors class 2:

    >>> rng = np.random.default_rng(42)
    >>> true_alpha = np.array([3.0, 1.0])
    >>> clf = EmpiricalBayesDirichletClassifier(
    ...     {1: 1.0, 2: 5.0}, random_state=0
    ... )
    >>> for g in range(200):  # doctest: +SKIP
    ...     theta = rng.dirichlet(true_alpha)
    ...     obs = rng.choice([1, 2], size=rng.poisson(10) + 1, p=theta)
    ...     clf.partial_fit(np.full((len(obs), 1), g), obs)
    >>> # Base measure recovers true shape: m ≈ [0.75, 0.25]
    """

    def __init__(
        self,
        alphas: dict[Union[int, str], float],
        *,
        n_eb_iter: int = 10,
        eb_tol: float = 1e-4,
        learning_rate: float = 1.0,
        random_state: Union[int, np.random.Generator, None] = None,
    ) -> None:
        super().__init__(
            alphas=alphas,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.n_eb_iter = n_eb_iter
        self.eb_tol = eb_tol

    def _fit_helper(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]],
    ) -> None:
        """Override to add stabilized prior re-injection after decay.

        The base class ``_fit_helper`` decays the existing posterior by
        ``learning_rate ** n_obs`` when stacking with new observations,
        but does not re-inject the prior. Without re-injection the prior
        contribution decays toward zero, breaking the invariant
        ``known_alphas_[g] - prior_ = decayed_counts``.
        """
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"sample_weight.shape[0]={sample_weight.shape[0]} should be "
                    f"equal to X.shape[0]={X.shape[0]}"
                )

        for group, arr, weights in groupby_array(X[:, 0], y, sample_weight, by=X[:, 0]):
            key = group[0].item()

            weighted_arr = arr * weights[:, np.newaxis]

            vals = np.vstack((self.known_alphas_[key], weighted_arr))

            decay_idx = np.flip(np.arange(len(vals)))
            posterior = vals * (self.learning_rate**decay_idx)[:, np.newaxis]

            # Stabilized forgetting: the prior was decayed by lr^n_obs,
            # reinject (1 - lr^n_obs) * prior so the prior contribution
            # converges to prior_ instead of zero.
            n_obs = len(arr)
            reinjection = (1 - self.learning_rate**n_obs) * self.prior_

            self.known_alphas_[key] = posterior.sum(axis=0) + reinjection

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """Fit with empirical Bayes hyperparameter tuning.

        Iteratively optimizes the Dirichlet prior concentration
        parameters by maximizing the Dirichlet-Multinomial marginal
        likelihood using Minka's fixed-point iteration, then performs
        a final posterior update with the converged prior.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample.

        Returns
        -------
        self : EmpiricalBayesDirichletClassifier
            Fitted estimator with tuned prior.
        """
        X, y = check_X_y(X, y, copy=True, ensure_2d=True)

        if self.n_eb_iter > 0:
            y_encoded = (y[:, np.newaxis] == np.array(list(self.alphas.keys()))).astype(
                int
            )

            # Fit once to get the counts, then iterate the prior.
            # For the Dirichlet-Multinomial, the counts are fixed data,
            # so we only need to refit the posterior once at the end with
            # the converged prior — unlike Normal regression where the
            # posterior depends on the prior precision.
            self._initialize_prior()
            self.n_features_ = X.shape[1]
            self._fit_helper(X, y_encoded, sample_weight)

            # Run Minka iterations on the fixed counts with convergence
            new_prior, log_ev, converged = minka_update_dirichlet_multinomial(
                dict(self.known_alphas_),
                self.prior_,
                n_iter=self.n_eb_iter,
                tol=self.eb_tol,
            )

            for idx, cls_key in enumerate(self.classes_):
                self.alphas[cls_key] = float(new_prior[idx])

            # Refit with converged prior
            self._initialize_prior()
            self.n_features_ = X.shape[1]
            self._fit_helper(X, y_encoded, sample_weight)

            self.log_evidence_ = log_ev
            self.n_eb_iterations_ = self.n_eb_iter
            self.eb_converged_ = converged
        else:
            super().fit(X, y, sample_weight)
            self.log_evidence_ = -math.inf
            self.n_eb_iterations_ = 0
            self.eb_converged_ = False

        return self

    def partial_fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """Incrementally update and retune hyperparameters.

        Performs a posterior update (via the base class), then runs
        one Minka fixed-point step to adjust the prior. All group
        posteriors are corrected to reflect the new prior.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels.
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample.

        Returns
        -------
        self : EmpiricalBayesDirichletClassifier
            Updated estimator with retuned prior.
        """
        result = super().partial_fit(X, y, sample_weight)

        if hasattr(self, "known_alphas_") and len(self.known_alphas_) > 0:
            old_prior = self.prior_.copy()

            new_prior, log_ev, _ = minka_update_dirichlet_multinomial(
                dict(self.known_alphas_),
                self.prior_,
                n_iter=1,
            )

            # Correct all group posteriors: shift by the prior change
            prior_delta = new_prior - old_prior
            for key in list(self.known_alphas_.keys()):
                self.known_alphas_[key] = np.asarray(
                    self.known_alphas_[key] + prior_delta, dtype=np.float64
                )

            # Update prior_ and alphas dict
            self.prior_ = new_prior
            for idx, cls_key in enumerate(self.classes_):
                self.alphas[cls_key] = float(new_prior[idx])

            self.log_evidence_ = log_ev

        return result

    def decay(
        self,
        X: NDArray[Any],
        *,
        decay_rate: Optional[float] = None,
    ) -> None:
        """Decay with stabilized prior re-injection.

        Applies exponential forgetting and re-injects the EB-tuned
        prior so that the prior contribution converges to ``prior_``
        rather than zero. This ensures that effective counts
        (``known_alphas - prior``) remain correct after decay.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Used to identify which groups to decay.
        decay_rate : float, default=None
            Decay factor in (0, 1]. If None, uses ``self.learning_rate``.
        """
        if not hasattr(self, "known_alphas_"):
            self._initialize_prior()

        if decay_rate is None:
            decay_rate = self.learning_rate

        for x in X:
            key = x.item()
            # Stabilized forgetting: decay + re-inject prior
            self.known_alphas_[key] = np.asarray(
                decay_rate * self.known_alphas_[key] + (1 - decay_rate) * self.prior_,
                dtype=np.float64,
            )


class EmpiricalBayesGammaRegressor(GammaRegressor):
    """Gamma-Poisson regressor with empirical Bayes prior tuning.

    Extends :class:`GammaRegressor` with automatic optimization of
    the Gamma prior parameters ``(alpha, beta)`` via the Negative
    Binomial marginal likelihood, using Minka's EM algorithm [1]_.

    During ``fit``, the prior is iteratively updated to maximize the
    marginal likelihood across all groups. During ``partial_fit``, a
    single EM step is performed using the current group posteriors.

    The EM treats per-group rates as latent Gamma variables. The
    E-step computes posterior moments; the M-step solves a Gamma MLE
    for the shared prior, using the generalized Newton method for the
    shape parameter [1]_.

    When ``learning_rate < 1``, exponential forgetting is applied.
    Stabilized forgetting re-injects the EB-tuned prior after each
    decay step, ensuring the prior contribution converges to the tuned
    value rather than zero.

    Parameters
    ----------
    alpha : float
        Initial Gamma shape parameter.
    beta : float
        Initial Gamma rate parameter.
    n_eb_iter : int, default=10
        Maximum number of EB iterations during ``fit``.
    eb_tol : float, default=1e-4
        Convergence tolerance on change in log marginal likelihood.
    learning_rate : float, default=1.0
        Decay rate for sequential updates.
    random_state : int, np.random.Generator, or None, default=None
        Controls RNG for ``sample``.

    Attributes
    ----------
    log_evidence_ : float
        Log marginal likelihood at convergence.
    n_eb_iterations_ : int
        Number of EB iterations in last ``fit``.
    eb_converged_ : bool
        Whether the EB loop converged within ``eb_tol``.

    See Also
    --------
    GammaRegressor : Base estimator without empirical Bayes tuning.
    EmpiricalBayesDirichletClassifier : EB tuning for Dirichlet classification.
    EmpiricalBayesNormalRegressor : EB tuning for Normal regression.

    Notes
    -----
    **Negative Binomial marginal likelihood**

    The Gamma-Poisson marginal for group :math:`g` with effective
    count :math:`c_g` and exposure :math:`n_g` is:

    .. math::

        \\log p(c_g \\mid \\alpha, \\beta)
        = \\log\\Gamma(c_g + \\alpha) - \\log\\Gamma(\\alpha)
          - \\log\\Gamma(c_g + 1)
          + \\alpha \\log\\frac{\\beta}{\\beta + n_g}
          + c_g \\log\\frac{n_g}{\\beta + n_g}

    **EM update (Minka 2002, section 2.1)**

    E-step: posterior moments of the latent rate :math:`\\lambda_g`:

    .. math::

        \\mathbb{E}[\\lambda_g] &= \\frac{\\alpha + c_g}{\\beta + n_g} \\\\
        \\mathbb{E}[\\log \\lambda_g] &= \\psi(\\alpha + c_g)
          - \\log(\\beta + n_g)

    M-step: Gamma MLE on expected sufficient statistics. The rate
    update is closed-form given the shape:

    .. math::

        \\beta = \\frac{\\alpha}{\\bar{\\mathbb{E}}[\\lambda]}

    The shape uses the generalized Newton method from [1]_:

    .. math::

        \\frac{1}{\\alpha^{\\text{new}}} = \\frac{1}{\\alpha}
        + \\frac{\\log\\bar{\\mathbb{E}}[\\lambda]
          - \\bar{\\mathbb{E}}[\\log\\lambda]
          - \\log\\alpha + \\psi(\\alpha)}
             {\\alpha^2 (1/\\alpha - \\psi'(\\alpha))}

    **Stabilized forgetting**

    Every time ``learning_rate`` :math:`\\gamma < 1` causes decay,
    the prior is re-injected:

    .. math::

        \\mathbf{p}_g \\leftarrow
        \\gamma^n \\mathbf{p}_g
        + (1 - \\gamma^n) \\mathbf{p}^{\\text{prior}}

    where :math:`\\mathbf{p}_g = (\\alpha_g, \\beta_g)` are the
    posterior parameters. This ensures effective counts and exposures
    are correctly decayed and the prior contribution never vanishes.

    References
    ----------
    .. [1] Minka, T. P. (2002). "Estimating a Gamma distribution."
       Microsoft Research Technical Report.

    Examples
    --------
    Basic rate estimation with EB-tuned prior:

    >>> import numpy as np
    >>> from bayesianbandits import EmpiricalBayesGammaRegressor
    >>> rng = np.random.default_rng(42)
    >>> X = np.repeat(np.arange(1, 6), 30).reshape(-1, 1)
    >>> y = rng.poisson(3.0, size=150)
    >>> model = EmpiricalBayesGammaRegressor(
    ...     alpha=1.0, beta=1.0, random_state=0
    ... )
    >>> model.fit(X, y)
    EmpiricalBayesGammaRegressor(alpha=..., beta=..., random_state=0)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        *,
        n_eb_iter: int = 10,
        eb_tol: float = 1e-4,
        learning_rate: float = 1.0,
        random_state: Union[int, np.random.Generator, None] = None,
    ) -> None:
        super().__init__(
            alpha=alpha,
            beta=beta,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.n_eb_iter = n_eb_iter
        self.eb_tol = eb_tol

    def _fit_helper(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]],
    ) -> None:
        """Override to add stabilized prior re-injection after decay.

        The base class ``_fit_helper`` decays the existing posterior by
        ``learning_rate ** n_obs`` when stacking with new observations,
        but does not re-inject the prior. Without re-injection the prior
        contribution decays toward zero, breaking the invariant
        ``coef_[g] - prior_ = decayed_counts``.
        """
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    f"sample_weight.shape[0]={sample_weight.shape[0]} should be "
                    f"equal to X.shape[0]={X.shape[0]}"
                )

        lr = self.learning_rate
        no_decay = lr == 1.0

        for group, arr, weights in groupby_array(X[:, 0], y, sample_weight, by=X[:, 0]):
            key = group[0].item()

            weighted_counts = arr * weights
            weighted_data = np.column_stack((weighted_counts, weights))

            if no_decay:
                self.coef_[key] = self.coef_[key] + weighted_data.sum(axis=0)
            else:
                n_obs = len(arr)
                decay_factor = lr**n_obs
                data_weights = lr ** np.arange(n_obs - 1, -1, -1)
                data_contribution = data_weights @ weighted_data
                reinjection = (1 - decay_factor) * self.prior_
                self.coef_[key] = (
                    decay_factor * self.coef_[key] + data_contribution + reinjection
                )

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """Fit with empirical Bayes hyperparameter tuning.

        Iteratively optimizes the Gamma prior parameters by maximizing
        the Negative Binomial marginal likelihood using Minka's EM
        algorithm, then performs a final posterior update with the
        converged prior.
        """
        X, y = check_X_y(X, y, copy=True, ensure_2d=True)

        if self.n_eb_iter > 0:
            y_encoded: NDArray[np.int_] = y.astype(int)

            # Fit once to get the counts, then iterate the prior.
            self._initialize_prior()
            self.n_features_ = X.shape[1]
            self._fit_helper(X, y_encoded, sample_weight)

            # Run EM iterations on the fixed counts
            new_prior, log_ev, converged = negbin_update_gamma_poisson(
                self.coef_,
                self.prior_,
                n_iter=self.n_eb_iter,
                tol=self.eb_tol,
            )

            self.alpha = float(new_prior[0])
            self.beta = float(new_prior[1])

            # Refit with converged prior
            self._initialize_prior()
            self.n_features_ = X.shape[1]
            self._fit_helper(X, y_encoded, sample_weight)

            self.log_evidence_ = log_ev
            self.n_eb_iterations_ = self.n_eb_iter
            self.eb_converged_ = converged
        else:
            super().fit(X, y, sample_weight)
            self.log_evidence_ = -math.inf
            self.n_eb_iterations_ = 0
            self.eb_converged_ = False

        return self

    def partial_fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        sample_weight: Optional[NDArray[Any]] = None,
    ) -> Self:
        """Incrementally update and retune hyperparameters.

        Performs a posterior update (via the base class), then runs
        one EM step to adjust the prior. All group posteriors are
        corrected to reflect the new prior.
        """
        result = super().partial_fit(X, y, sample_weight)

        if hasattr(self, "coef_") and len(self.coef_) > 0:
            old_prior = self.prior_.copy()

            new_prior, log_ev, _ = negbin_update_gamma_poisson(
                self.coef_,
                self.prior_,
                n_iter=1,
            )

            prior_delta = new_prior - old_prior
            for key in list(self.coef_.keys()):
                self.coef_[key] = np.asarray(
                    self.coef_[key] + prior_delta, dtype=np.float64
                )

            self.prior_ = new_prior
            self.alpha = float(new_prior[0])
            self.beta = float(new_prior[1])

            self.log_evidence_ = log_ev

        return result

    def decay(
        self,
        X: NDArray[Any],
        *,
        decay_rate: Optional[float] = None,
    ) -> None:
        """Decay with stabilized prior re-injection.

        Applies exponential forgetting and re-injects the EB-tuned
        prior so that the prior contribution converges to ``prior_``
        rather than zero.
        """
        if not hasattr(self, "coef_"):
            self._initialize_prior()

        if decay_rate is None:
            decay_rate = self.learning_rate

        for x in X:
            key = x.item()
            self.coef_[key] = np.asarray(
                decay_rate * self.coef_[key] + (1 - decay_rate) * self.prior_,
                dtype=np.float64,
            )
