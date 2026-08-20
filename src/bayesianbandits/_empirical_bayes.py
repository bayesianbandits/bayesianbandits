"""Pure helper functions for empirical Bayes via MacKay's update rules.

Provides:
- logdet dispatch (dense/sparse)
- Factorization stats (logdet + trace of inverse from a single factorization)
- MacKay update rules for Normal, NIG, and GLM models (each returns log evidence)
- Minka's fixed-point iteration for Dirichlet-Multinomial (returns log evidence)
- Online MacKay update with sufficient statistics for streaming settings
"""

from __future__ import annotations

import math
from typing import Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, issparse
from scipy.special import digamma, gammaln, polygamma

from ._blas_helpers import dsymv
from ._sparse_bayesian_linear_regression import (
    PrecisionFactor,
)

_LOG_2PI = math.log(2.0 * math.pi)

# Guardrails for MacKay hyperparameter updates.
# In the underdetermined regime (p >> n), MacKay can drive α→0 and β→∞.
# The precision matrix α·I + β·X^T X then has smallest eigenvalue ≈ α
# (in the null space of X) while the largest is ≈ β·λ_max(X^T X), giving
# a condition number of β/α · λ_max.  When this exceeds ~1/eps ≈ 1e16,
# Cholesky fails numerically even though the matrix is theoretically PD.
#
# We reject the update entirely (keep current α, β) when β/α would
# produce such ill-conditioning.  Once enough data arrives (n ≥ p),
# the updates become well-behaved and start flowing normally.
_MAX_BETA_ALPHA_RATIO = 1e10


def _diagonal_trace_approx(
    precision: Union[NDArray[np.float64], csc_array],
) -> float:
    """Approximate tr(Λ⁻¹) ≈ Σ 1/Λᵢᵢ.

    Fast O(p) approximation valid when the precision matrix is strongly
    diagonally dominant.
    """
    if issparse(precision):
        diag = np.asarray(precision.diagonal(), dtype=np.float64)
    else:
        assert isinstance(precision, np.ndarray)
        diag = np.diag(precision).astype(np.float64)
    return float(np.sum(1.0 / diag))


def _factorization_stats(
    precision: Union[NDArray[np.float64], csc_array],
    factor: PrecisionFactor,
    trace_method: str = "auto",
) -> tuple[float, float]:
    """Return (logdet, trace_inverse) from a factorization.

    Parameters
    ----------
    precision : dense or sparse matrix
    factor : pre-computed factorization (DenseFactor or SparseFactor)
    trace_method : ``"auto"`` takes the exact trace from the factor
        (Takahashi recursion for sparse, the Cholesky inverse for
        dense).  ``"diagonal"`` uses the fast O(p) approximation
        tr(Λ⁻¹) ≈ Σ 1/Λᵢᵢ, the one case that reads the precision matrix
        rather than the factorization of it.
    """
    if trace_method == "diagonal":
        return factor.logdet(), _diagonal_trace_approx(precision)
    return factor.logdet(), factor.trace_inv()


def accumulate_sufficient_stats(
    effective_n: float,
    eff_yTy: float,
    eff_XTy: NDArray[np.float64],
    X: Union[NDArray[np.float64], csc_array],
    y: NDArray[np.float64],
    prior_decay: float,
) -> tuple[float, float, NDArray[np.float64]]:
    """Update decayed sufficient statistics with new observations.

    Maintains running (decayed) totals of:
    - effective_n: effective number of observations
    - eff_yTy: yᵀy (scalar)
    - eff_XTy: Xᵀy (vector)

    Parameters
    ----------
    effective_n, eff_yTy, eff_XTy : current sufficient statistics
    X : design matrix for new observations
    y : response vector for new observations
    prior_decay : decay factor to apply before accumulating

    Returns
    -------
    (effective_n, eff_yTy, eff_XTy) : updated sufficient statistics
    """
    effective_n = effective_n * prior_decay + y.shape[0]
    eff_yTy = eff_yTy * prior_decay + float(y @ y)
    if issparse(X):
        eff_XTy = np.asarray(
            eff_XTy * prior_decay + np.asarray(X.T @ y).ravel(), dtype=np.float64
        )
    else:
        eff_XTy = np.asarray(eff_XTy * prior_decay + X.T @ y, dtype=np.float64)
    return effective_n, eff_yTy, eff_XTy


def mackay_update_normal_online(
    mu_n: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    alpha: float,
    beta: float,
    prior_scalar: float,
    effective_n: float,
    eff_yTy: float,
    eff_XTy: NDArray[np.float64],
    factor: PrecisionFactor,
    trace_method: str = "auto",
) -> tuple[float, float, float]:
    """MacKay update using accumulated sufficient statistics for beta.

    Uses decayed sufficient statistics (effective_n, yᵀy, Xᵀy) and
    recovers XᵀX from the precision matrix:

        XᵀX_decayed = (Λ − prior_scalar·I) / β

    to compute RSS under the current posterior mean:

        RSS = yᵀy − 2·mᵀ·Xᵀy + mᵀ·XᵀX·m

    Parameters
    ----------
    mu_n : posterior mean
    precision : posterior precision matrix Λ
    alpha, beta : current hyperparameters
    prior_scalar : cumulative (decayed) prior contribution to the
        diagonal of Λ
    effective_n : decayed effective sample size
    eff_yTy : decayed yᵀy
    eff_XTy : decayed Xᵀy
    factor : pre-computed factorization
    trace_method : method for computing tr(Λ⁻¹)

    Returns
    -------
    (alpha_new, beta_new, log_evidence)
    """
    p = cast(tuple[int, int], precision.shape)[0]
    mu_norm_sq = float(mu_n @ mu_n)

    ld, tr_inv = _factorization_stats(precision, factor, trace_method)

    # gamma is the effective number of well-determined parameters,
    # bounded by (0, min(effective_n, p)].
    _EPS = 1e-8
    gamma = float(np.clip(p - alpha * tr_inv, _EPS, min(effective_n, p)))

    # Alpha update.
    alpha_new = gamma / mu_norm_sq if mu_norm_sq > 0 else alpha

    # Beta update from sufficient statistics.
    # XᵀX_decayed = (Λ − prior_scalar·I) / β
    m = mu_n
    mTXTy = float(m @ eff_XTy)

    # dsymv reads only the upper triangle (safe for upper-triangle-only
    # dense precision matrices produced by dsyrk).
    if issparse(precision):
        prec_m = precision @ m
    else:
        prec_m = dsymv(1.0, precision, m)
    XTX_m = prec_m - prior_scalar * m
    mTXTXm = float(m @ XTX_m) / beta

    rss = eff_yTy - 2.0 * mTXTy + mTXTXm

    denom = effective_n - gamma
    if rss > 0 and denom > 0:
        beta_new = denom / rss
    else:
        beta_new = beta

    # Reject pathological updates.
    if beta_new / alpha_new > _MAX_BETA_ALPHA_RATIO:
        alpha_new = alpha
        beta_new = beta

    log_ev = float(
        0.5 * p * math.log(alpha)
        + 0.5 * effective_n * math.log(beta)
        - 0.5 * ld
        - 0.5 * (beta * rss + alpha * mu_norm_sq)
        - 0.5 * effective_n * _LOG_2PI
    )

    return alpha_new, beta_new, log_ev


def _dirichlet_multinomial_log_evidence(
    counts: NDArray[np.floating],
    count_totals: NDArray[np.floating],
    alpha: NDArray[np.floating],
) -> float:
    """Compute Dirichlet-Multinomial marginal log-likelihood.

    Parameters
    ----------
    counts : ndarray of shape (G, K)
        Effective counts per group and class.
    count_totals : ndarray of shape (G,)
        Total counts per group (``counts.sum(axis=1)``).
    alpha : ndarray of shape (K,)
        Current Dirichlet concentration parameters.

    Returns
    -------
    float
        Log marginal likelihood ``Σ_g [logΓ(α₀) - logΓ(c_g + α₀)
        + Σ_k (logΓ(c_gk + α_k) - logΓ(α_k))]``.
    """
    s = float(alpha.sum())
    return float(
        np.sum(
            gammaln(s)
            - gammaln(count_totals + s)
            + np.sum(
                gammaln(counts + alpha[np.newaxis, :]) - gammaln(alpha[np.newaxis, :]),
                axis=1,
            )
        )
    )


def minka_update_dirichlet_multinomial(
    posterior_alphas: dict,
    prior: NDArray[np.float64],
    n_iter: int = 1,
    tol: float = 0.0,
) -> tuple[NDArray[np.float64], float, bool]:
    """Minka's fixed-point iteration for Dirichlet-Multinomial EB.

    Given G groups with posterior concentration parameters and the current
    prior, performs ``n_iter`` fixed-point updates to the prior that
    maximize the Dirichlet-Multinomial marginal likelihood [1]_.

    Uses the (s, m) decomposition: ``α_k = s · m_k`` where ``s = Σ α_k``
    is the scalar concentration and ``m_k = α_k / s`` is the base measure.
    The m and s updates are alternated for faster convergence.

    Parameters
    ----------
    posterior_alphas : dict mapping group keys to posterior alpha arrays
        Each value is shape (K,) where K is the number of classes.
    prior : ndarray of shape (K,)
        Current prior concentration parameters.
    n_iter : int, default=1
        Maximum number of fixed-point iterations to perform.
    tol : float, default=0.0
        Early stopping tolerance on change in log evidence between
        successive iterations. Set to 0 to always run all iterations.

    Returns
    -------
    alpha_new : ndarray of shape (K,)
        Updated prior concentration parameters.
    log_evidence : float
        Dirichlet-Multinomial marginal log-likelihood at the new alpha.
    converged : bool
        Whether the iteration converged within ``tol``.

    References
    ----------
    .. [1] Minka, T. P. (2000; revised 2003, 2012). "Estimating a
       Dirichlet distribution." Technical report, MIT.
    """
    if not posterior_alphas:
        return prior.copy(), -math.inf, True

    # Stack all posterior alphas: shape (G, K)
    alpha_matrix = np.array(list(posterior_alphas.values()), dtype=np.float64)
    # Effective counts: shape (G, K)
    counts = alpha_matrix - prior[np.newaxis, :]
    # Total counts per group: shape (G,)
    count_totals = counts.sum(axis=1)

    # Filter out groups with non-positive total counts (no useful data).
    valid = count_totals > 0
    if not np.any(valid):
        log_ev = _dirichlet_multinomial_log_evidence(counts, count_totals, prior)
        return prior.copy(), log_ev, True
    counts = counts[valid]
    count_totals = count_totals[valid]

    alpha = prior.copy()
    _EPS = 1e-8
    prev_ev = -math.inf
    converged = False

    for _ in range(n_iter):
        s = float(alpha.sum())

        # --- m update (fix s, update per-component α_k) ---
        # Numerator per k: Σ_g [ψ(c_gk + α_k) - ψ(α_k)]  shape (K,)
        numerator_k = (
            digamma(counts + alpha[np.newaxis, :]) - digamma(alpha[np.newaxis, :])
        ).sum(axis=0)
        # Denominator (shared): Σ_g [ψ(c_g + s) - ψ(s)]  scalar
        denominator = float((digamma(count_totals + s) - digamma(s)).sum())

        if abs(denominator) < _EPS:
            converged = True
            break

        # Multiplicative update for each α_k, then extract new m
        alpha = np.maximum(alpha * numerator_k / denominator, _EPS)
        m = alpha / alpha.sum()

        # --- s update (fix m, update scalar concentration) ---
        # Numerator: Σ_g Σ_k m_k [ψ(c_gk + s·m_k) - ψ(s·m_k)]
        s_alpha = s * m
        s_numerator = float(
            np.sum(
                m[np.newaxis, :]
                * (
                    digamma(counts + s_alpha[np.newaxis, :])
                    - digamma(s_alpha[np.newaxis, :])
                )
            )
        )
        # Denominator: Σ_g [ψ(c_g + s) - ψ(s)]  (same as above but at new s)
        s_denominator = float((digamma(count_totals + s) - digamma(s)).sum())

        s_new = max(s * s_numerator / s_denominator, _EPS)
        alpha = np.maximum(s_new * m, _EPS)

        # Check convergence
        if tol > 0:
            ev = _dirichlet_multinomial_log_evidence(counts, count_totals, alpha)
            if abs(ev - prev_ev) < tol:
                converged = True
                prev_ev = ev
                break
            prev_ev = ev

    log_ev = _dirichlet_multinomial_log_evidence(counts, count_totals, alpha)
    return alpha, log_ev, converged


def _negbin_log_evidence(
    counts: NDArray[np.floating],
    exposures: NDArray[np.floating],
    alpha: float,
    beta: float,
) -> float:
    """Compute Negative Binomial marginal log-likelihood.

    The Gamma-Poisson marginal for G groups, each with effective
    count ``c_g`` and exposure ``n_g``, under a shared
    Gamma(alpha, beta) prior.

    Parameters
    ----------
    counts : ndarray of shape (G,)
        Effective counts per group (sum of weighted observations).
    exposures : ndarray of shape (G,)
        Effective exposures per group (sum of weights).
    alpha : float
        Gamma shape parameter.
    beta : float
        Gamma rate parameter.

    Returns
    -------
    float
        Log marginal likelihood
        ``Σ_g [logΓ(c_g + α) - logΓ(α) - logΓ(c_g + 1)
        + α·log(β/(β+n_g)) + c_g·log(n_g/(β+n_g))]``.
    """
    log_beta_plus_n = np.log(beta + exposures)
    log_p = math.log(beta) - log_beta_plus_n
    # Guard against log(0) when exposure is 0; use np.log1p to avoid
    # eager evaluation of log(0) inside np.where.
    safe_exp = np.maximum(exposures, 1e-300)
    log_q = np.where(exposures > 0, np.log(safe_exp) - log_beta_plus_n, 0.0)
    return float(
        np.sum(
            gammaln(counts + alpha)
            - gammaln(alpha)
            - gammaln(counts + 1)
            + alpha * log_p
            + counts * log_q
        )
    )


def negbin_update_gamma_poisson(
    posterior_params: dict,
    prior: NDArray[np.float64],
    n_iter: int = 1,
    tol: float = 0.0,
) -> tuple[NDArray[np.float64], float, bool]:
    """EM-based update for Gamma-Poisson empirical Bayes.

    Given G groups with posterior Gamma parameters and the current
    prior, performs ``n_iter`` EM steps to maximize the Negative
    Binomial marginal likelihood [1]_.

    Each EM iteration treats the per-group rate as a latent variable:

    - **E-step**: compute ``E[λ_g]`` and ``E[log λ_g]`` under the
      posterior ``Gamma(α + c_g, β + n_g)``.
    - **M-step**: solve the Gamma MLE for ``(α, β)`` using the
      expected sufficient statistics. The shape update uses Minka's
      generalized Newton method [1]_ (section 1); the rate has a
      closed-form conditional on the shape.

    Parameters
    ----------
    posterior_params : dict mapping group keys to ndarray of shape (2,)
        Each value is ``[alpha_post, beta_post]`` for that group.
    prior : ndarray of shape (2,)
        Current prior ``[alpha, beta]``.
    n_iter : int, default=1
        Maximum number of EM iterations to perform.
    tol : float, default=0.0
        Early stopping tolerance on change in log evidence between
        successive iterations. Set to 0 to always run all iterations.

    Returns
    -------
    prior_new : ndarray of shape (2,)
        Updated prior ``[alpha_new, beta_new]``.
    log_evidence : float
        Negative Binomial marginal log-likelihood at the new prior.
    converged : bool
        Whether the iteration converged within ``tol``.

    References
    ----------
    .. [1] Minka, T. P. (2002). "Estimating a Gamma distribution."
       Microsoft Research Technical Report.
    """
    if not posterior_params:
        return prior.copy(), -math.inf, True

    param_matrix = np.array(list(posterior_params.values()), dtype=np.float64)
    counts = param_matrix[:, 0] - prior[0]
    exposures = param_matrix[:, 1] - prior[1]

    valid = exposures > 0
    if not np.any(valid):
        log_ev = _negbin_log_evidence(counts, exposures, prior[0], prior[1])
        return prior.copy(), log_ev, True
    counts = counts[valid]
    exposures = exposures[valid]

    alpha = float(prior[0])
    beta = float(prior[1])
    _EPS = 1e-8
    prev_ev = -math.inf
    converged = False

    for _ in range(n_iter):
        # --- E-step ---
        E_lambda = (counts + alpha) / (exposures + beta)
        E_log_lambda = digamma(counts + alpha) - np.log(exposures + beta)

        mean_E_lambda = float(np.mean(E_lambda))
        mean_E_log_lambda = float(np.mean(E_log_lambda))

        # --- M-step: beta update (closed-form given alpha) ---
        if mean_E_lambda > _EPS:
            beta = alpha / mean_E_lambda
        beta = max(beta, _EPS)

        # --- M-step: alpha update (Minka's generalized Newton) ---
        target = math.log(max(mean_E_lambda, _EPS)) - mean_E_log_lambda
        if target > _EPS:
            for _ in range(5):
                inv_alpha = 1.0 / alpha
                numerator = (math.log(alpha) - float(digamma(alpha))) - target
                trigamma = float(polygamma(1, alpha))
                denom = alpha * alpha * (inv_alpha - trigamma)
                if abs(denom) < _EPS:
                    break
                inv_alpha_new = inv_alpha + numerator / denom
                if inv_alpha_new > _EPS:
                    alpha = 1.0 / inv_alpha_new
                alpha = max(alpha, _EPS)

        if tol > 0:
            ev = _negbin_log_evidence(counts, exposures, alpha, beta)
            if abs(ev - prev_ev) < tol:
                converged = True
                prev_ev = ev
                break
            prev_ev = ev

    log_ev = (
        prev_ev
        if prev_ev > -math.inf
        else _negbin_log_evidence(counts, exposures, alpha, beta)
    )
    return np.array([alpha, beta], dtype=np.float64), log_ev, converged
