"""Pure helper functions for empirical Bayes via MacKay's update rules.

Provides:
- logdet dispatch (dense/sparse)
- Trace of inverse (exact dense, Hutchinson sparse)
- Effective number of parameters
- MacKay update rules for Normal, NIG, and GLM models (each returns log evidence)
"""

from __future__ import annotations

import math
from typing import Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from scipy.sparse import csc_array, issparse
from scipy.special import gammaln

from ._sparse_bayesian_linear_regression import (
    SparseFactor,
    create_sparse_factor,
)

_LOG_2PI = math.log(2.0 * math.pi)


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


def _dense_cholesky_stats(
    precision: NDArray[np.float64],
) -> tuple[float, float]:
    """Returns (logdet, trace_of_inverse) from one Cholesky.

    Math: logdet = 2·sum(log(diag(L))), tr(A^-1) = ||L^-1||^2_F.
    """
    L = np.linalg.cholesky(precision)
    ld = 2.0 * float(np.sum(np.log(np.diag(L))))
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)
    tr_inv = float(np.sum(L_inv**2))
    return ld, tr_inv


def _factorization_stats(
    precision: Union[NDArray[np.float64], csc_array],
    factor: Optional[SparseFactor],
    sparse: bool,
    n_probes: int,
    rng: Optional[np.random.Generator],
) -> tuple[float, float]:
    """Return (logdet, trace_inverse) from a single factorization."""
    if sparse:
        if factor is None:
            if not issparse(precision):
                raise TypeError("precision must be sparse when sparse=True")
            factor = create_sparse_factor(cast(csc_array, precision))
        p = cast(tuple[int, int], precision.shape)[0]
        ld = factor.logdet()
        tr_inv = _hutchinson_trace(factor, p, n_probes, rng)
    else:
        assert isinstance(precision, np.ndarray)
        ld, tr_inv = _dense_cholesky_stats(precision)
    return ld, tr_inv


def _hutchinson_trace(
    factor: SparseFactor,
    p: int,
    n_probes: int,
    rng: Optional[np.random.Generator],
) -> float:
    """Batched Hutchinson trace estimator: tr(A^-1) via multi-RHS solve."""
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.choice(np.array([-1.0, 1.0]), size=(p, n_probes))
    V = factor.solve(Z)
    return float(np.sum(Z * V) / n_probes)


def trace_of_inverse(
    precision: Union[NDArray[np.float64], csc_array],
    factor: Optional[SparseFactor] = None,
    sparse: bool = False,
    n_probes: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute tr(precision^{-1}).

    Dense: exact via Cholesky.
    Sparse: Hutchinson's stochastic trace estimator using Rademacher probes.

    Parameters
    ----------
    precision : dense or sparse matrix
        SPD precision matrix.
    factor : SparseFactor, optional
        Pre-computed sparse factorization.
    sparse : bool
        Whether to use sparse computation.
    n_probes : int
        Number of Rademacher probe vectors for Hutchinson estimator.
    rng : np.random.Generator, optional
        Random number generator for probe vectors.

    Returns
    -------
    float
        tr(precision^{-1})

    References
    ----------
    .. [1] Hutchinson, M. F. (1990). "A stochastic estimator of the trace
       of the influence matrix for Laplacian smoothing splines",
       Communications in Statistics — Simulation and Computation 19(2).
    """
    if sparse:
        if factor is None:
            if not issparse(precision):
                raise TypeError("precision must be sparse when sparse=True")
            factor = create_sparse_factor(cast(csc_array, precision))
        p = cast(tuple[int, int], precision.shape)[0]
        return _hutchinson_trace(factor, p, n_probes, rng)
    else:
        assert isinstance(precision, np.ndarray)
        _, tr_inv = _dense_cholesky_stats(precision)
        return tr_inv


def effective_num_parameters(
    alpha: float,
    precision: Union[NDArray[np.float64], csc_array],
    factor: Optional[SparseFactor] = None,
    sparse: bool = False,
    n_probes: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute effective number of parameters: γ = p − α·tr(Λₙ⁻¹).

    Parameters
    ----------
    alpha : float
        Prior precision scalar.
    precision : dense or sparse matrix
        Posterior precision matrix Λₙ.
    factor : SparseFactor, optional
        Pre-computed sparse factorization.
    sparse : bool
        Whether to use sparse computation.
    n_probes : int
        Number of probes for Hutchinson estimator (sparse only).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    float
        γ, the effective number of parameters.

    References
    ----------
    .. [1] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    .. [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.2, Eq 3.91.
    """
    p = cast(tuple[int, int], precision.shape)[0]
    tr_inv = trace_of_inverse(
        precision, factor=factor, sparse=sparse, n_probes=n_probes, rng=rng
    )
    return float(p - alpha * tr_inv)


# ---------------------------------------------------------------------------
# MacKay update rules (M-step) — each returns updated hyperparams + log evidence
# ---------------------------------------------------------------------------


def mackay_update_normal(
    mu_n: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    X: Union[NDArray[np.float64], csc_array],
    y: NDArray[np.float64],
    alpha: float,
    beta: float,
    sparse: bool = False,
    factor: Optional[SparseFactor] = None,
    n_probes: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """MacKay update for NormalRegressor.

    Returns (α_new, β_new, log_evidence) where:
        γ = p − α·tr(Λₙ⁻¹)
        α_new = γ / ‖μₙ‖²
        β_new = (n − γ) / ‖y − Xμₙ‖²

    log p(y|X,α,β) = p/2·log(α) + n/2·log(β) - ½·log|Λₙ|
                     - ½[β‖y-Xμₙ‖² + α‖μₙ‖²] - n/2·log(2π)

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.1–§3.5.2, Eqs 3.86, 3.92, 3.95.
    .. [2] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    """
    n = y.shape[0]
    p = cast(tuple[int, int], precision.shape)[0]

    residual = y - X @ mu_n
    rss = float(residual @ residual)
    mu_norm_sq = float(mu_n @ mu_n)

    ld, tr_inv = _factorization_stats(precision, factor, sparse, n_probes, rng)

    gamma = float(p - alpha * tr_inv)
    alpha_new = gamma / mu_norm_sq if mu_norm_sq > 0 else alpha
    denom = n - gamma
    beta_new = denom / rss if rss > 0 and denom > 0 else beta

    log_ev = float(
        0.5 * p * math.log(alpha)
        + 0.5 * n * math.log(beta)
        - 0.5 * ld
        - 0.5 * (beta * rss + alpha * mu_norm_sq)
        - 0.5 * n * _LOG_2PI
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
    sparse: bool = False,
    factor: Optional[SparseFactor] = None,
    n_probes: int = 20,
    rng: Optional[np.random.Generator] = None,
    n_obs: int = 0,
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
    sparse, factor, n_probes, rng : sparse computation parameters
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

    ld, tr_inv = _factorization_stats(precision, factor, sparse, n_probes, rng)

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
    sparse: bool = False,
    factor: Optional[SparseFactor] = None,
    n_probes: int = 20,
    rng: Optional[np.random.Generator] = None,
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
    sparse, factor, n_probes, rng : sparse computation parameters

    References
    ----------
    .. [1] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    .. [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.2, §4.4.1.
    """
    p = cast(tuple[int, int], precision.shape)[0]
    theta_norm_sq = float(theta_MAP @ theta_MAP)

    ld, tr_inv = _factorization_stats(precision, factor, sparse, n_probes, rng)

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
        0.5 * p * (math.log(alpha) - _LOG_2PI)
        - 0.5 * alpha * theta_norm_sq
    )

    # Laplace correction: + p/2·log(2π) - ½·log|H|
    laplace_correction = float(0.5 * p * _LOG_2PI - 0.5 * ld)

    log_ev = float(log_lik + log_prior + laplace_correction)

    return alpha_new, log_ev
