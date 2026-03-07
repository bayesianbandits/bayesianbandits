"""Pure helper functions for empirical Bayes via MacKay's update rules.

Provides:
- logdet dispatch (dense/sparse)
- Trace of inverse (exact dense, Hutchinson sparse)
- Effective number of parameters
- MacKay update rules for Normal, NIG, and GLM models
- Log evidence functions for monitoring
"""

from __future__ import annotations

from typing import Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve as scipy_solve
from scipy.sparse import csc_array, issparse
from scipy.special import gammaln

from ._sparse_bayesian_linear_regression import (
    SparseFactor,
    create_sparse_factor,
)


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


def trace_of_inverse(
    precision: Union[NDArray[np.float64], csc_array],
    factor: Optional[SparseFactor] = None,
    sparse: bool = False,
    n_probes: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute tr(precision^{-1}).

    Dense: exact via solving precision @ X = I.
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
        if rng is None:
            rng = np.random.default_rng()
        p = cast(tuple[int, int], precision.shape)[0]
        # Hutchinson: tr(A^{-1}) ≈ (1/m) Σ zᵢᵀ A^{-1} zᵢ
        total = 0.0
        for _ in range(n_probes):
            z = rng.choice(np.array([-1.0, 1.0]), size=p)
            total += float(z @ factor.solve(z))
        return total / n_probes
    else:
        assert isinstance(precision, np.ndarray)
        p = precision.shape[0]
        inv_prec = scipy_solve(
            precision, np.eye(p), check_finite=False, assume_a="pos"
        )
        return float(np.trace(inv_prec))


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
# MacKay update rules (M-step)
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
) -> tuple[float, float]:
    """MacKay update for NormalRegressor.

    Returns (α_new, β_new) where:
        γ = p − α·tr(Λₙ⁻¹)
        α_new = γ / ‖μₙ‖²
        β_new = (n − γ) / ‖y − Xμₙ‖²

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.2, Eqs 3.92, 3.95.
    .. [2] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    """
    gamma = effective_num_parameters(
        alpha, precision, factor=factor, sparse=sparse, n_probes=n_probes, rng=rng
    )
    mu_norm_sq = float(mu_n @ mu_n)
    alpha_new = gamma / mu_norm_sq if mu_norm_sq > 0 else alpha

    n = y.shape[0]
    residual = y - X @ mu_n
    rss = float(residual @ residual)
    denom = n - gamma
    beta_new = denom / rss if rss > 0 and denom > 0 else beta

    return alpha_new, beta_new


def mackay_update_nig(
    mu_n: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    alpha: float,
    an: float,
    bn: float,
    sparse: bool = False,
    factor: Optional[SparseFactor] = None,
    n_probes: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """MacKay update for NormalInverseGammaRegressor.

    Returns α_new using the direct stationary-point formula:
        σ² = bₙ/aₙ  (posterior mean of noise variance)
        α_new = p·σ² / (‖μₙ‖² + σ²·tr(Λₙ⁻¹))

    In the NIG model the prior is w|σ² ~ N(0, σ²/α · I), so both the prior
    precision (α/σ²)I and the posterior precision Λₙ/σ² scale identically
    with σ². Setting ∂/∂α log p(y|α) = 0 and solving yields the formula
    above. This is the direct stationary-point form, which converges in one
    step and avoids dependence on the previous α.

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.2 — known-variance case; NIG extension follows analogously.
    .. [2] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    """
    sigma_sq = bn / an
    tr_inv = trace_of_inverse(
        precision, factor=factor, sparse=sparse, n_probes=n_probes, rng=rng
    )
    p = cast(tuple[int, int], precision.shape)[0]
    mu_norm_sq = float(mu_n @ mu_n)
    denom = mu_norm_sq + sigma_sq * tr_inv
    return float(p * sigma_sq / denom) if denom > 0 else alpha


def mackay_update_glm(
    theta_MAP: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    alpha: float,
    sparse: bool = False,
    factor: Optional[SparseFactor] = None,
    n_probes: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """MacKay update for BayesianGLM.

    Returns α_new where:
        γ = p − α·tr(H⁻¹)
        α_new = γ / ‖θ_MAP‖²

    H is the Hessian of the negative log-posterior (i.e. αI + Hessian of
    negative log-likelihood), used as the precision in a Laplace
    approximation.

    References
    ----------
    .. [1] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447.
    .. [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.2, Eq 3.92 — same γ-based update applies to the Laplace case.
    """
    gamma = effective_num_parameters(
        alpha, precision, factor=factor, sparse=sparse, n_probes=n_probes, rng=rng
    )
    theta_norm_sq = float(theta_MAP @ theta_MAP)
    return gamma / theta_norm_sq if theta_norm_sq > 0 else alpha


# ---------------------------------------------------------------------------
# Log evidence functions (monitoring, uses logdet)
# ---------------------------------------------------------------------------


def log_evidence_normal(
    X: Union[NDArray[np.float64], csc_array],
    y: NDArray[np.float64],
    mu_n: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    alpha: float,
    beta: float,
    sparse: bool = False,
    factor: Optional[SparseFactor] = None,
) -> float:
    """Log marginal likelihood for NormalRegressor.

    log p(y|X,α,β) = p/2·log(α) + n/2·log(β) - ½·log|Λₙ|
                     - ½[β‖y-Xμₙ‖² + α‖μₙ‖²] - n/2·log(2π)

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.1, Eq 3.86.
    """
    n = y.shape[0]
    p = cast(tuple[int, int], precision.shape)[0]

    ld = logdet(precision, factor=factor, sparse=sparse)
    residual = y - X @ mu_n
    rss = float(residual @ residual)
    mu_norm_sq = float(mu_n @ mu_n)

    return float(
        0.5 * p * np.log(alpha)
        + 0.5 * n * np.log(beta)
        - 0.5 * ld
        - 0.5 * (beta * rss + alpha * mu_norm_sq)
        - 0.5 * n * np.log(2.0 * np.pi)
    )


def log_evidence_nig(
    X: Union[NDArray[np.float64], csc_array],
    y: NDArray[np.float64],
    mu_n: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    alpha: float,
    a0: float,
    b0: float,
    an: float,
    bn: float,
    sparse: bool = False,
    factor: Optional[SparseFactor] = None,
) -> float:
    """Log marginal likelihood for NormalInverseGammaRegressor.

    log p(y|X,α,a₀,b₀) = logΓ(aₙ) - logΓ(a₀) + a₀·log(b₀) - aₙ·log(bₙ)
                         + ½·p·log(α) - ½·log|Λₙ| - n/2·log(2π)

    Obtained by integrating out both w and σ² from the joint
    p(y,w,σ²|X,α,a₀,b₀). The (2π)^{-n/2} normalizing constant arises
    because the prior's (2π)^{-p/2} cancels with (2π)^{p/2} from the
    Gaussian integral over w.

    References
    ----------
    .. [1] Murphy, K. P. (2007). "Conjugate Bayesian analysis of the
       Gaussian distribution", Technical Report.
    .. [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §3.5.1 — known-variance case; NIG extension integrates over σ².
    """
    n = y.shape[0]
    p = cast(tuple[int, int], precision.shape)[0]

    ld = logdet(precision, factor=factor, sparse=sparse)

    return float(
        gammaln(an)
        - gammaln(a0)
        + a0 * np.log(b0)
        - an * np.log(bn)
        + 0.5 * p * np.log(alpha)
        - 0.5 * ld
        - 0.5 * n * np.log(2.0 * np.pi)
    )


def log_evidence_glm_laplace(
    X: Union[NDArray[np.float64], csc_array],
    y: NDArray[np.float64],
    theta_MAP: NDArray[np.float64],
    precision: Union[NDArray[np.float64], csc_array],
    alpha: float,
    link: str,
    sparse: bool = False,
    factor: Optional[SparseFactor] = None,
) -> float:
    """Log marginal likelihood for BayesianGLM (Laplace approximation).

    log p(y|X,α) ≈ ℓ(θ_MAP) + log p(θ_MAP|α) + p/2·log(2π) - ½·log|H|

    where ℓ(θ_MAP) is the log-likelihood at the MAP estimate and
    log p(θ_MAP|α) = p/2·log(α/(2π)) - α/2·‖θ_MAP‖² is the log-prior.

    References
    ----------
    .. [1] MacKay, D. J. C. (1992). "Bayesian Interpolation",
       Neural Computation 4(3), 415-447, §4.
    .. [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning,
       §4.4.1 — Laplace approximation for model evidence.
    """
    from scipy.special import expit as _expit

    p = cast(tuple[int, int], precision.shape)[0]
    eta = X @ theta_MAP

    # Log-likelihood at MAP
    if link == "logit":
        mu = _expit(eta)
        log_lik = float(
            np.sum(y * np.log(mu + 1e-15) + (1 - y) * np.log(1 - mu + 1e-15))
        )
    elif link == "log":
        log_lik = float(np.sum(y * eta - np.exp(eta) - gammaln(y + 1)))
    else:
        raise ValueError(f"Unknown link function: {link}")

    # Log-prior: p/2·log(α/(2π)) - α/2·‖θ‖²
    log_prior = float(
        0.5 * p * np.log(alpha / (2.0 * np.pi))
        - 0.5 * alpha * float(theta_MAP @ theta_MAP)
    )

    # Laplace correction: + p/2·log(2π) - ½·log|H|
    ld = logdet(precision, factor=factor, sparse=sparse)
    laplace_correction = float(0.5 * p * np.log(2.0 * np.pi) - 0.5 * ld)

    return float(log_lik + log_prior + laplace_correction)
