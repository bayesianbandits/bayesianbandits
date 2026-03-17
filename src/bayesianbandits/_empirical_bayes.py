"""Pure helper functions for empirical Bayes via MacKay's update rules.

Provides:
- logdet dispatch (dense/sparse)
- Factorization stats (logdet + trace of inverse from a single factorization)
- MacKay update rules for Normal, NIG, and GLM models (each returns log evidence)
- Online MacKay update with sufficient statistics for streaming settings
"""

from __future__ import annotations

import math
from typing import Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, issparse
from scipy.special import gammaln

from ._blas_helpers import dsymv
from ._sparse_bayesian_linear_regression import (
    DenseFactor,
    PrecisionFactor,
    SparseFactor,
    create_sparse_factor,
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


def _takahashi_diagonal(L_csc: csc_array) -> NDArray[np.float64]:
    """Compute diag((LL')⁻¹) via Takahashi recursion (selected inversion).

    Given a lower triangular CSC matrix L where LL' = A, computes the
    diagonal of A⁻¹ exactly using backward recursion through L's sparsity
    structure.  Cost is O(Σⱼ nⱼ²) where nⱼ is the number of sub-diagonal
    entries in column j — the same order as the Cholesky factorization
    itself.

    Delegates to a Cython implementation for performance.

    Parameters
    ----------
    L_csc : csc_array
        Lower triangular Cholesky factor in CSC format.

    Returns
    -------
    NDArray[np.float64]
        Diagonal of (LL')⁻¹, length p.

    References
    ----------
    .. [1] Takahashi, K., Fagan, J., & Chin, M.-S. (1973). "Formation of
       a sparse bus impedance matrix and its application to short circuit
       study." 8th PICA Conference Proceedings.
    """
    from ._takahashi import takahashi_diagonal as _cy_impl

    p = cast(tuple[int, int], L_csc.shape)[0]
    if not L_csc.has_sorted_indices:
        L_csc = L_csc.copy()
        L_csc.sort_indices()
    return _cy_impl(
        L_csc.data,
        L_csc.indices.astype(np.int32),
        L_csc.indptr.astype(np.int32),
        p,
    )


def _takahashi_trace(factor: SparseFactor) -> float:
    """Compute tr(Λ⁻¹) exactly via Takahashi recursion.

    Extracts the Cholesky factor L from the SparseFactor, runs Takahashi
    diagonal recursion, and returns sum(diag((LL')⁻¹)).

    The trace is permutation-invariant, so the result is correct regardless
    of fill-reducing permutation order.
    """
    L = factor.get_L_csc()
    Z_diag = _takahashi_diagonal(L)
    return float(np.sum(Z_diag))


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


def logdet(
    precision: Union[NDArray[np.float64], csc_array],
    factor: Optional[SparseFactor] = None,
    sparse: bool = False,
) -> float:
    """Compute log-determinant of a precision matrix.

    Parameters
    ----------
    precision : dense or sparse matrix
        The precision (or any SPD) matrix.
    factor : SparseFactor, optional
        Pre-computed sparse factorization. If None and sparse=True,
        a new factorization is created.
    sparse : bool
        Whether to use sparse computation.

    Returns
    -------
    float
        log|precision|
    """
    if sparse:
        if factor is None:
            if not issparse(precision):
                raise TypeError("precision must be sparse when sparse=True")
            factor = create_sparse_factor(cast(csc_array, precision))
        return factor.logdet()
    else:
        assert isinstance(precision, np.ndarray)
        sign, ld = np.linalg.slogdet(precision)
        if sign <= 0:
            raise ValueError("Precision matrix is not positive definite")
        return float(ld)


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
    trace_method : ``"auto"`` uses Takahashi recursion for sparse and
        Cholesky for dense.  ``"diagonal"`` uses the fast O(p)
        approximation tr(Λ⁻¹) ≈ Σ 1/Λᵢᵢ.
    """
    if trace_method == "diagonal":
        tr_inv = _diagonal_trace_approx(precision)
        if isinstance(factor, DenseFactor):
            ld = factor.logdet()
        else:
            ld = factor.logdet()
        return ld, tr_inv

    # trace_method == "auto" (default)
    if isinstance(factor, DenseFactor):
        ld = factor.logdet()
        tr_inv = factor.trace_inv()
    else:
        ld = factor.logdet()
        tr_inv = _takahashi_trace(factor)
    return ld, tr_inv


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


def mackay_update_nig(
    mu_n: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    alpha: float,
    an: float,
    bn: float,
    a0: float,
    b0: float,
    factor: PrecisionFactor,
    n_obs: int = 0,
    trace_method: str = "auto",
) -> tuple[float, float]:
    """MacKay update for NormalInverseGammaRegressor.

    Returns (α_new, log_evidence) using the direct stationary-point formula:
        σ² = bₙ/aₙ  (posterior mean of noise variance)
        α_new = p·σ² / (‖μₙ‖² + σ²·tr(Λₙ⁻¹))

    log p(y|X,α,a₀,b₀) = logΓ(aₙ) - logΓ(a₀) + a₀·log(b₀) - aₙ·log(bₙ)
                         + ½·p·log(α) - ½·log|Λₙ| - n/2·log(2π)

    Parameters
    ----------
    mu_n : posterior mean
    precision : posterior precision matrix
    alpha : prior precision scalar
    an, bn : posterior IG parameters
    a0, b0 : prior IG parameters (needed for evidence)
    sparse, factor : sparse computation parameters
    n_obs : number of observations (needed for evidence)

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.2 — known-variance case; NIG extension follows analogously.
    .. [2] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    """
    p = cast(tuple[int, int], precision.shape)[0]
    sigma_sq = bn / an
    mu_norm_sq = float(mu_n @ mu_n)

    ld, tr_inv = _factorization_stats(precision, factor, trace_method)

    denom = mu_norm_sq + sigma_sq * tr_inv
    alpha_new = float(p * sigma_sq / denom) if denom > 0 else alpha

    if b0 > 0 and bn > 0 and alpha > 0:
        log_ev = float(
            gammaln(an)
            - gammaln(a0)
            + a0 * math.log(b0)
            - an * math.log(bn)
            + 0.5 * p * math.log(alpha)
            - 0.5 * ld
            - 0.5 * n_obs * _LOG_2PI
        )
    else:
        log_ev = -math.inf

    return alpha_new, log_ev


def mackay_update_glm(
    theta_MAP: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    alpha: float,
    X: Union[NDArray[np.float64], csc_array],
    y: NDArray[np.float64],
    link: str,
    factor: PrecisionFactor,
    trace_method: str = "auto",
) -> tuple[float, float]:
    """MacKay update for BayesianGLM.

    Returns (α_new, log_evidence) where:
        γ = p − α·tr(H⁻¹)
        α_new = γ / ‖θ_MAP‖²

    log p(y|X,α) ≈ ℓ(θ_MAP) + log p(θ_MAP|α) + p/2·log(2π) - ½·log|H|

    Parameters
    ----------
    theta_MAP : MAP estimate
    precision : Hessian of negative log-posterior (αI + Hessian of NLL)
    alpha : prior precision scalar
    X : design matrix
    y : response vector
    link : link function ("logit" or "log")
    sparse, factor : sparse computation parameters

    References
    ----------
    .. [1] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    .. [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.2, §4.4.1.
    """
    p = cast(tuple[int, int], precision.shape)[0]
    theta_norm_sq = float(theta_MAP @ theta_MAP)

    ld, tr_inv = _factorization_stats(precision, factor, trace_method)

    gamma = float(p - alpha * tr_inv)
    alpha_new = gamma / theta_norm_sq if theta_norm_sq > 0 else alpha

    # Log-likelihood at MAP
    eta = X @ theta_MAP
    if link == "logit":
        log_lik = float(np.sum(y * eta - np.logaddexp(0, eta)))
    elif link == "log":
        log_lik = float(np.sum(y * eta - np.exp(eta) - gammaln(y + 1)))
    else:
        raise ValueError(f"Unknown link function: {link}")

    # Log-prior: p/2·log(α/(2π)) - α/2·‖θ‖²
    log_prior = float(
        0.5 * p * (math.log(alpha) - _LOG_2PI) - 0.5 * alpha * theta_norm_sq
    )

    # Laplace correction: + p/2·log(2π) - ½·log|H|
    laplace_correction = float(0.5 * p * _LOG_2PI - 0.5 * ld)

    log_ev = float(log_lik + log_prior + laplace_correction)

    return alpha_new, log_ev
