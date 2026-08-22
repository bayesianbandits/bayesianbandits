import functools
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional, Protocol, Tuple, Union, cast

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.linalg.blas import dgemv, dsymv, dsyrk
from scipy.sparse import csc_array, eye
from scipy.sparse import issparse as sp_issparse
from scipy.special import expit

from ._memory import MemoryUsageMixin

# Type aliases
ArrayType = Union[NDArray[np.float64], csc_array]
LinkFunction = Literal["logit", "log"]


# Named tuples for return types
class LinkOutput(NamedTuple):
    """Output from link function evaluation."""

    mu: NDArray[np.float64]  # Mean parameter
    d_mu_d_eta: NDArray[np.float64]  # Derivative of mean w.r.t. linear predictor


class GLMWeights(NamedTuple):
    """Weights and working response for GLM."""

    W: NDArray[np.float64]  # Diagonal weight matrix
    z: NDArray[np.float64]  # Working response (pseudo-targets)


class GaussianPosterior(NamedTuple):
    """Gaussian posterior parameters."""

    mean: ArrayType  # Posterior mean
    precision: ArrayType  # Posterior precision matrix
    factor: Optional[Any] = None  # Sparse factor (CHOLMOD/SuperLU), if available


def compute_effective_weights(
    n_samples: int, sample_weight: Optional[NDArray[np.float64]], learning_rate: float
) -> NDArray[np.float64]:
    """Apply learning rate decay to sample weights."""
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight.shape[0]={sample_weight.shape[0]} should be "
                f"equal to n_samples={n_samples}"
            )

    if n_samples > 1:
        decay_factors = np.flip(np.power(learning_rate, np.arange(n_samples)))
        return cast(NDArray[np.float64], sample_weight * decay_factors)
    else:
        return sample_weight


def logit_link_and_derivative(
    eta: NDArray[np.float64],
) -> LinkOutput:
    """Logit link function and its derivative."""
    mu = expit(eta)
    d_mu_d_eta = mu * (1.0 - mu)
    return LinkOutput(mu, d_mu_d_eta)


def log_link_and_derivative(
    eta: NDArray[np.float64],
) -> LinkOutput:
    """Log link function and its derivative."""
    eta_safe = np.clip(eta, -700, 700)
    mu = np.exp(eta_safe)
    d_mu_d_eta = mu
    return LinkOutput(mu, d_mu_d_eta)


def compute_glm_weights_and_working_response(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    d_mu_d_eta: NDArray[np.float64],
    eta: NDArray[np.float64],
    sample_weight: Optional[NDArray[np.float64]] = None,
) -> GLMWeights:
    """
    Compute GLM weights and working response for natural gradient update.

    This transforms a GLM optimization into weighted least squares by:
    - W: diagonal weights (Fisher information)
    - z: linearized targets (working response)

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Observed responses
    mu : array-like of shape (n_samples,)
        Current mean estimates g^{-1}(eta)
    d_mu_d_eta : array-like of shape (n_samples,)
        Derivative of mean w.r.t. linear predictor
    eta : array-like of shape (n_samples,)
        Current linear predictor values
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights to apply

    Returns
    -------
    W : array-like of shape (n_samples,)
        Diagonal of weight matrix
    z : array-like of shape (n_samples,)
        Working response values
    """
    residual = y - mu
    np.maximum(d_mu_d_eta, 1e-10, out=d_mu_d_eta)
    residual /= d_mu_d_eta
    z = eta + residual

    W = d_mu_d_eta if sample_weight is None else d_mu_d_eta * sample_weight
    return GLMWeights(W, z)  # type: ignore


def _eval_link(link: LinkFunction, eta: NDArray[np.float64]) -> LinkOutput:
    """Evaluate link function and derivative."""
    if link == "logit":
        return logit_link_and_derivative(eta)
    elif link == "log":
        return log_link_and_derivative(eta)
    else:
        raise ValueError(f"Unknown link function: {link}")


def _prior_diag_shift(
    prior_decay: float, prior_floor: float, prior_shift: float
) -> float:
    """Total shift to the decayed prior precision's diagonal.

    ``(1 - γⁿ)·prior_floor`` is the stabilized-forgetting re-injection
    (Kulhavy & Zarrop 1993); ``γⁿ·prior_shift`` is a shift the caller
    owed the prior *before* decay (an empirical-Bayes change to the
    prior precision, deferred to this update), so it decays with the
    rest of the prior.
    """
    return (1.0 - prior_decay) * prior_floor + prior_decay * prior_shift


def _stabilized_prior_dense(
    prior_prec_F: NDArray[Any],
    prior_precision: NDArray[Any],
    prior_decay: float,
    prior_floor: float,
    prior_shift: float,
) -> NDArray[Any]:
    """Add :func:`_prior_diag_shift` to the diagonal of the decayed prior
    precision.

    Only the precision is shifted, never the prior's eta term, so the
    re-injected prior is centered at zero: it shrinks dormant
    coefficients rather than pinning them where they are.  Without
    decay ``prior_prec_F`` may be the caller's own array; it is copied
    before being touched.
    """
    shift = _prior_diag_shift(prior_decay, prior_floor, prior_shift)
    if shift == 0.0:
        return prior_prec_F
    if np.shares_memory(prior_prec_F, prior_precision):
        prior_prec_F = prior_prec_F.copy(order="F")
    prior_prec_F[np.diag_indices_from(prior_prec_F)] += shift
    return prior_prec_F


def _stabilized_prior_sparse(
    prior_precision_scaled: csc_array,
    prior_decay: float,
    prior_floor: float,
    prior_shift: float,
) -> csc_array:
    """Sparse counterpart of :func:`_stabilized_prior_dense`.

    Returns the input object itself when there is nothing to add, so
    callers can detect by identity whether a cached factor is stale.
    """
    shift = _prior_diag_shift(prior_decay, prior_floor, prior_shift)
    if shift == 0.0:
        return prior_precision_scaled
    assert prior_precision_scaled.shape is not None
    n = prior_precision_scaled.shape[0]
    return csc_array(prior_precision_scaled + shift * eye(n, format="csc"))


def _irls_dense(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
    *,
    link: LinkFunction,
    effective_weights: NDArray[np.float64],
    prior_decay: float,
    prior_floor: float,
    prior_shift: float,
    n_iter: int,
    tol: float,
    coef_init: Optional[NDArray[np.float64]] = None,
) -> GaussianPosterior:
    """Dense IRLS loop with pre-allocated buffers and fused BLAS calls."""
    n_samples, n_features = X.shape

    no_decay = prior_decay == 1.0
    prior_precision_scaled = (
        prior_precision if no_decay else prior_decay * prior_precision
    )
    prior_eta_scaled = dsymv(prior_decay, prior_precision, prior_mean)

    # F-order copy of prior precision for dsyrk accumulation buffer
    prior_prec_F = _stabilized_prior_dense(
        np.asfortranarray(prior_precision_scaled),
        prior_precision,
        prior_decay,
        prior_floor,
        prior_shift,
    )

    X_weighted = np.empty_like(X)
    W_sqrt_buf = np.empty(n_samples, dtype=np.float64)
    Wz_buf = np.empty(n_samples, dtype=np.float64)
    diff_buf = np.empty(n_features, dtype=np.float64)
    precision_buf = np.empty_like(prior_prec_F)
    eta_buf = np.empty_like(prior_eta_scaled)

    coef = prior_mean.copy() if coef_init is None else coef_init.copy()
    coef_old = coef
    posterior_precision = prior_prec_F

    for iteration in range(n_iter):
        if iteration > 0 and n_iter > 1:
            coef_old = coef.copy()

        eta = cast(NDArray[np.float64], X @ coef)
        link_out = _eval_link(link, eta)
        glm_weights = compute_glm_weights_and_working_response(
            y, link_out.mu, link_out.d_mu_d_eta, eta, effective_weights
        )

        # Fused X^T W X + prior via dsyrk(beta=1, c=prior_copy)
        np.sqrt(glm_weights.W, out=W_sqrt_buf)
        np.multiply(X, W_sqrt_buf[:, np.newaxis], out=X_weighted)
        np.copyto(precision_buf, prior_prec_F)
        posterior_precision = dsyrk(
            1.0, X_weighted, trans=1, beta=1.0, c=precision_buf, overwrite_c=True
        )

        # Fused X^T @ (W*z) + prior_eta via dgemv(beta=1, y=prior_copy)
        np.multiply(glm_weights.W, glm_weights.z, out=Wz_buf)
        np.copyto(eta_buf, prior_eta_scaled)
        posterior_eta = dgemv(
            1.0, X, Wz_buf, trans=1, beta=1.0, y=eta_buf, overwrite_y=True
        )

        cho = cho_factor(posterior_precision, lower=False, check_finite=False)
        coef = cho_solve(cho, posterior_eta, check_finite=False)

        if iteration > 0 and n_iter > 1:
            np.subtract(coef, coef_old, out=diff_buf)
            np.abs(diff_buf, out=diff_buf)
            if diff_buf.max() < tol:
                break

    return GaussianPosterior(coef, cast(NDArray[np.float64], posterior_precision), cho)


def _irls_sparse(
    X: csc_array,
    y: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_precision: csc_array,
    *,
    link: LinkFunction,
    effective_weights: NDArray[np.float64],
    prior_decay: float,
    prior_floor: float,
    prior_shift: float,
    n_iter: int,
    tol: float,
    prior_factor: Optional[Any] = None,
    coef_init: Optional[NDArray[np.float64]] = None,
) -> GaussianPosterior:
    """Sparse IRLS loop using CHOLMOD/SuperLU factorization.

    ``prior_factor``, when given, seeds the posterior's factorization:
    ``refactorize`` reuses its symbolic analysis whenever the posterior
    keeps the prior's sparsity pattern (every update after the first
    touching a given feature set), and falls back to a fresh one when
    the pattern grew.
    """
    from ._sparse_bayesian_linear_regression import create_sparse_factor

    no_decay = prior_decay == 1.0
    prior_precision_scaled = (
        prior_precision if no_decay else prior_decay * prior_precision
    )
    prior_eta_scaled = (
        prior_precision @ prior_mean
        if no_decay
        else prior_decay * (prior_precision @ prior_mean)
    )
    prior_precision_scaled = _stabilized_prior_sparse(
        prior_precision_scaled, prior_decay, prior_floor, prior_shift
    )

    coef = prior_mean.copy() if coef_init is None else coef_init.copy()
    coef_old = coef
    sparse_factor = prior_factor
    posterior_precision = prior_precision

    for iteration in range(n_iter):
        if iteration > 0 and n_iter > 1:
            coef_old = coef.copy()

        eta = cast(NDArray[np.float64], X @ coef)
        link_out = _eval_link(link, eta)
        glm_weights = compute_glm_weights_and_working_response(
            y, link_out.mu, link_out.d_mu_d_eta, eta, effective_weights
        )

        # X^T W X via element-wise row scaling (avoids diags construction)
        XW = X.multiply(glm_weights.W.reshape(-1, 1)).tocsc()
        posterior_precision = csc_array(X.T @ XW)
        posterior_precision += prior_precision_scaled

        likelihood_eta = X.T @ (glm_weights.W * glm_weights.z)
        posterior_eta = prior_eta_scaled + likelihood_eta

        if sparse_factor is None:
            sparse_factor = create_sparse_factor(posterior_precision)
        else:
            sparse_factor = sparse_factor.refactorize(posterior_precision)
        coef = sparse_factor.solve(posterior_eta)

        if iteration > 0 and n_iter > 1:
            coef_change = np.max(np.abs(coef - coef_old))
            if coef_change < tol:
                break

    return GaussianPosterior(coef, posterior_precision, sparse_factor)


def update_gaussian_posterior_laplace(
    X: ArrayType,
    y: NDArray[np.float64],
    prior_mean: ArrayType,
    prior_precision: ArrayType,
    *,
    link: LinkFunction,
    sample_weight: Optional[NDArray[np.float64]] = None,
    learning_rate: float = 1.0,
    sparse: bool = False,
    prior_floor: float = 0.0,
    prior_shift: float = 0.0,
    n_iter: int = 3,
    tol: float = 1e-4,
    prior_factor: Optional[Any] = None,
    coef_init: Optional[NDArray[np.float64]] = None,
) -> GaussianPosterior:
    """
    Update Gaussian posterior using Laplace approximation (IRLS).

    This finds the Maximum A Posteriori (MAP) estimate and approximates
    the posterior as Gaussian with covariance equal to the inverse Hessian
    at the MAP point.

    Mathematical Details
    --------------------
    The true posterior for a GLM is:
        p(θ|y,X) ∝ p(y|X,θ) × p(θ)
                 ∝ ∏ᵢ p(yᵢ|xᵢ,θ) × N(θ|μ₀,Σ₀)

    This is intractable for non-Gaussian likelihoods. The Laplace
    approximation finds θ_MAP = argmax p(θ|y,X) and approximates:
        p(θ|y,X) ≈ N(θ|θ_MAP, H⁻¹)
    where H is the Hessian of -log p(θ|y,X) at θ_MAP.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix
    y : array-like of shape (n_samples,)
        Target values
    prior_mean : array-like of shape (n_features,)
        Prior mean
    prior_precision : array-like of shape (n_features, n_features)
        Prior precision matrix
    link : {'logit', 'log'}
        Link function
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    learning_rate : float
        Decay factor for prior contribution
    sparse : bool
        Whether to use sparse operations
    n_iter : int, default=3
        Maximum number of Newton (IRLS) iterations.
        - n_iter=1: Single step update (fast, may not converge)
        - n_iter>1: Iterate until convergence or max iterations
        Common values:
        - Online learning: 1 (one step from previous posterior)
        - Minibatch: 3-5 (good compromise)
        - Batch: 10-20 (full convergence)
    tol : float, default=1e-4
        Convergence tolerance for coefficient change. Only used if n_iter > 1.
        Convergence when: ||coef_new - coef_old||_∞ < tol

    Returns
    -------
    GaussianPosterior
        posterior.mean : MAP estimate
        posterior.precision : Hessian at MAP (approximate posterior precision)
    """
    assert X.shape is not None, "X must be a 2D array"
    n_samples = X.shape[0]

    # Fast path: no data means posterior == prior
    if n_samples == 0:
        return GaussianPosterior(prior_mean.copy(), prior_precision.copy(), None)

    effective_weights = compute_effective_weights(
        n_samples, sample_weight, learning_rate
    )
    prior_decay = learning_rate**n_samples

    if sparse:
        return _irls_sparse(
            csc_array(X),
            y,
            np.asarray(prior_mean),
            csc_array(prior_precision),
            link=link,
            effective_weights=effective_weights,
            prior_decay=prior_decay,
            prior_floor=prior_floor,
            prior_shift=prior_shift,
            n_iter=n_iter,
            coef_init=coef_init,
            tol=tol,
            prior_factor=prior_factor,
        )
    else:
        return _irls_dense(
            np.asarray(X),
            y,
            np.asarray(prior_mean),
            np.asarray(prior_precision),
            link=link,
            effective_weights=effective_weights,
            prior_decay=prior_decay,
            prior_floor=prior_floor,
            prior_shift=prior_shift,
            n_iter=n_iter,
            coef_init=coef_init,
            tol=tol,
        )


class PosteriorApproximator(Protocol):
    """Protocol for posterior approximation strategies.

    Implementations are passed to :class:`BayesianGLM` via the
    ``approximator`` parameter to control how the posterior is
    approximated for non-conjugate likelihoods.

    See Also
    --------
    LaplaceApproximator : Default implementation using IRLS.
    BayesianGLM : Estimator that uses this protocol.

    Notes
    -----
    ``prior_factor``, when provided, is a factorization of
    ``prior_precision`` handed over by the caller from the previous
    update. Implementations may solve against it, refactorize it in
    place for the posterior (the caller no longer holds it as the
    prior's factor), or ignore it entirely.

    ``prior_floor``, when nonzero, requests stabilized forgetting
    (Kulhavy & Zarrop 1993): after decaying the prior precision by
    ``γⁿ``, ``(1 - γⁿ)·prior_floor`` is added back to its diagonal so
    the prior's contribution converges to ``prior_floor·I`` instead of
    vanishing.  ``prior_shift`` is a further diagonal shift the caller
    owed the prior before decay (it is scaled by ``γⁿ`` with the rest of
    the prior): how an empirical-Bayes estimator applies a change to its
    prior precision without a factorization of its own.  Both shift the
    precision only, never the prior's eta term, so the shifted prior is
    centered at zero.

    ``coef_init``, when given, is the iteration's starting point in
    place of ``prior_mean``: a warm start for callers that refit the
    same data under a slightly changed prior.  It does not change the
    fixed point, only how many iterations reach it.
    """

    def update_posterior(
        self,
        X: ArrayType,
        y: NDArray[np.float64],
        prior_mean: ArrayType,
        prior_precision: ArrayType,
        link: LinkFunction,
        sample_weight: Optional[NDArray[np.float64]],
        learning_rate: float,
        sparse: bool,
        prior_factor: Optional[Any] = None,
        prior_floor: float = 0.0,
        prior_shift: float = 0.0,
        coef_init: Optional[NDArray[np.float64]] = None,
    ) -> GaussianPosterior: ...


@dataclass
class LaplaceApproximator(MemoryUsageMixin, PosteriorApproximator):
    """Laplace approximation via iteratively reweighted least squares (IRLS).

    Approximates the posterior of a Bayesian GLM by finding the MAP
    estimate :math:`\\hat{w}` and fitting a Gaussian centered at the
    mode with precision equal to the negative Hessian:

    .. math::

        p(w \\mid \\mathcal{D}) \\approx
        \\mathcal{N}\\bigl(\\hat{w},\\;
        (\\alpha I + X^T W X)^{-1}\\bigr)

    where :math:`W` is the diagonal matrix of IRLS weights (Fisher
    information).

    Parameters
    ----------
    n_iter : int, default=5
        Maximum number of Newton (IRLS) iterations per update.

        - ``1``: Single-step update from the current posterior. Fast
          and usually sufficient for online/streaming use where the
          posterior from the previous step is a good initialization.
        - ``3–5``: Good default for mini-batch updates.
        - ``>10``: Use for batch fitting when full convergence is
          needed (pair with a tight ``tol``).
    tol : float, default=1e-4
        Convergence tolerance on the coefficient change. Iteration
        stops when
        :math:`\\|w_{\\text{new}} - w_{\\text{old}}\\|_\\infty < \\text{tol}`.
        Only effective when ``n_iter > 1``.

    See Also
    --------
    BayesianGLM : Estimator that uses this approximation strategy.

    Examples
    --------
    Fast single-step updates for online learning:

    >>> from bayesianbandits import LaplaceApproximator, BayesianGLM
    >>> fast = LaplaceApproximator(n_iter=1)
    >>> model = BayesianGLM(approximator=fast)

    Tight convergence for batch fitting:

    >>> batch = LaplaceApproximator(n_iter=500, tol=1e-8)
    >>> model = BayesianGLM(approximator=batch)
    """

    n_iter: int = 5
    tol: float = 1e-4

    def update_posterior(
        self,
        X: ArrayType,
        y: NDArray[np.float64],
        prior_mean: ArrayType,
        prior_precision: ArrayType,
        link: LinkFunction,
        sample_weight: Optional[NDArray[np.float64]],
        learning_rate: float,
        sparse: bool,
        prior_factor: Optional[Any] = None,
        prior_floor: float = 0.0,
        prior_shift: float = 0.0,
        coef_init: Optional[NDArray[np.float64]] = None,
    ) -> GaussianPosterior:
        return update_gaussian_posterior_laplace(
            X,
            y,
            prior_mean,
            prior_precision,
            link=link,
            sample_weight=sample_weight,
            learning_rate=learning_rate,
            sparse=sparse,
            prior_floor=prior_floor,
            prior_shift=prior_shift,
            n_iter=self.n_iter,
            coef_init=coef_init,
            tol=self.tol,
            prior_factor=prior_factor,
        )


# ---------------------------------------------------------------------------
# R-VGA: Recursive Variational Gaussian Approximation
# (Lambert, Bonnabel, Bach 2022 -- Statistics and Computing)
# ---------------------------------------------------------------------------

_PROBIT_BETA = np.sqrt(8.0 / np.pi)


@functools.lru_cache(maxsize=8)
def _gh_nodes_cached(n: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Probabilist's Gauss-Hermite nodes/weights for N(0,1)."""
    z, w = hermegauss(n)
    w = w / np.sqrt(2.0 * np.pi)
    z = z.astype(np.float64)
    w = w.astype(np.float64)
    z.flags.writeable = False
    w.flags.writeable = False
    return z, w


def gh_expected_weights(
    m: NDArray[np.float64],
    v: NDArray[np.float64],
    link: LinkFunction,
    n_gh_nodes: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """E_q[h(eta)] and E_q[h'(eta)] via Gauss-Hermite quadrature.

    Parameters
    ----------
    m : (N,) predictive means  x_i^T mu
    v : (N,) predictive variances  x_i^T Sigma x_i
    link : 'logit' or 'log'
    n_gh_nodes : number of quadrature nodes

    Returns
    -------
    E_mu : (N,) expected mean response
    E_W  : (N,) expected Fisher weight
    """
    z_nodes, w_nodes = _gh_nodes_cached(n_gh_nodes)
    std = np.sqrt(np.maximum(v, 0.0))
    eta_grid = cast(NDArray[np.float64], m[:, None] + std[:, None] * z_nodes[None, :])
    link_out = _eval_link(link, eta_grid.ravel())
    mu_grid = link_out.mu.reshape(eta_grid.shape)
    dmu_grid = link_out.d_mu_d_eta.reshape(eta_grid.shape)
    E_mu = cast(NDArray[np.float64], mu_grid @ w_nodes)
    E_W = cast(NDArray[np.float64], dmu_grid @ w_nodes)
    return E_mu, E_W


def probit_expected_weights(
    m: NDArray[np.float64],
    v: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """E_q[sigma(eta)] and E_q[sigma'(eta)] via probit approximation.

    Only valid for logit link.  k = beta / sqrt(v + beta^2).
    """
    beta = _PROBIT_BETA
    k = beta / np.sqrt(v + beta**2)
    km = k * m
    E_mu = cast(NDArray[np.float64], expit(km))
    E_W = cast(NDArray[np.float64], k * E_mu * (1.0 - E_mu))
    return E_mu, E_W


def log_expected_weights(
    m: NDArray[np.float64],
    v: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """E_q[exp(eta)] and E_q[exp(eta)] for the log link, in closed form.

    For eta ~ N(m, v) the lognormal mean gives
    E[exp(eta)] = exp(m + v/2) exactly, and for the log link
    mu(eta) = mu'(eta) = exp(eta), so the expected response and the
    expected Fisher weight coincide.
    """
    exponent = np.clip(m + 0.5 * v, -700.0, 700.0)
    E_mu = cast(NDArray[np.float64], np.exp(exponent))
    return E_mu, E_mu.copy()


def _compute_predictive_variances_dense(
    cho: Tuple[NDArray[np.float64], bool],
    X: NDArray[np.float64],
    Z_buf: NDArray[np.float64],
    v_buf: NDArray[np.float64],
) -> NDArray[np.float64]:
    """v_i = x_i^T Lambda^{-1} x_i via Cholesky triangular solve.

    Reuses ``Z_buf`` (n_features x n_samples, F-ordered) as the solve
    scratch and ``v_buf`` (n_samples,) as the output, so the per-iteration
    R-VGA variance computation allocates nothing.
    """
    U = cho[0]
    np.copyto(Z_buf, X.T)
    # overwrite_b reuses Z_buf in place; squaring is also in place.
    Z = solve_triangular(
        U, Z_buf, lower=False, trans=1, check_finite=False, overwrite_b=True
    )
    np.multiply(Z, Z, out=Z)
    return cast(NDArray[np.float64], np.sum(Z, axis=0, out=v_buf))


def _precompute_gram_matrix(
    prior_precision_scaled: csc_array,
    X: csc_array,
    prior_factor: Optional[Any] = None,
) -> NDArray[np.float64]:
    """Compute G = X A^{-1} X^T  (n_samples × n_samples).

    When A is diagonal this is a single sparse matrix product.
    Otherwise a factor for A is needed: if *prior_factor* is supplied
    it is used directly (partial_fit path); otherwise one is created
    ad-hoc.  The n×n result is reused across every IRLS iteration
    via the Woodbury identity.
    """
    # Diagonal A: G = X diag(1/a) X^T — no solves
    nnz_per_col = np.diff(prior_precision_scaled.indptr)
    if np.max(nnz_per_col) <= 1:
        diag_inv = 1.0 / prior_precision_scaled.diagonal()
        X_scaled = X.multiply(diag_inv.reshape(1, -1))
        return np.asarray((X_scaled @ X.T).todense(), dtype=np.float64)

    # General sparse A: reuse passed factor or create one
    if prior_factor is None:
        from ._sparse_bayesian_linear_regression import create_sparse_factor

        prior_factor = create_sparse_factor(prior_precision_scaled)
    Z = prior_factor.solve(X.T)  # (p, n) — sparse RHS handled by CHOLMOD/SuperLU
    G = X @ Z
    if sp_issparse(G):
        G = G.toarray()
    return np.asarray(G, dtype=np.float64)


def _rvga_dense(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
    *,
    link: LinkFunction,
    effective_weights: NDArray[np.float64],
    prior_decay: float,
    prior_floor: float,
    prior_shift: float,
    n_iter: int,
    tol: float,
    n_gh_nodes: int,
    use_probit: bool,
    coef_init: Optional[NDArray[np.float64]] = None,
) -> GaussianPosterior:
    """Dense R-VGA loop with expected curvature."""
    n_samples, n_features = X.shape

    no_decay = prior_decay == 1.0
    prior_precision_scaled = (
        prior_precision if no_decay else prior_decay * prior_precision
    )
    prior_eta_scaled = dsymv(prior_decay, prior_precision, prior_mean)
    prior_prec_F = _stabilized_prior_dense(
        np.asfortranarray(prior_precision_scaled),
        prior_precision,
        prior_decay,
        prior_floor,
        prior_shift,
    )

    X_weighted = np.empty_like(X)
    m_buf = np.empty(n_samples, dtype=np.float64)
    W_sqrt_buf = np.empty(n_samples, dtype=np.float64)
    Wz_buf = np.empty(n_samples, dtype=np.float64)
    diff_buf = np.empty(n_features, dtype=np.float64)
    precision_buf = np.empty_like(prior_prec_F)
    eta_buf = np.empty_like(prior_eta_scaled)
    # Scratch for the per-iteration predictive-variance solve (x_i^T Λ^{-1} x_i).
    Z_buf = np.empty((n_features, n_samples), dtype=np.float64, order="F")
    v_buf = np.empty(n_samples, dtype=np.float64)

    coef = prior_mean.copy() if coef_init is None else coef_init.copy()
    posterior_precision = prior_prec_F
    cho = cho_factor(prior_prec_F, lower=False, check_finite=False)

    use_probit_path = use_probit and link == "logit"

    for iteration in range(n_iter):
        if iteration > 0:
            coef_old = coef.copy()

        np.dot(X, coef, out=m_buf)
        v = _compute_predictive_variances_dense(cho, X, Z_buf, v_buf)

        if link == "log":
            E_mu, E_W = log_expected_weights(m_buf, v)
        elif use_probit_path:
            E_mu, E_W = probit_expected_weights(m_buf, v)
        else:
            E_mu, E_W = gh_expected_weights(m_buf, v, link, n_gh_nodes)

        np.maximum(E_W, 1e-10, out=E_W)
        np.multiply(E_W, effective_weights, out=W_sqrt_buf)
        np.sqrt(W_sqrt_buf, out=W_sqrt_buf)
        np.multiply(X, W_sqrt_buf[:, np.newaxis], out=X_weighted)
        np.copyto(precision_buf, prior_prec_F)
        posterior_precision = dsyrk(
            1.0, X_weighted, trans=1, beta=1.0, c=precision_buf, overwrite_c=True
        )

        # Fused Wz = eff_w * (E_W * m + y - E_mu) avoids division by E_W
        np.multiply(E_W, m_buf, out=Wz_buf)
        np.add(Wz_buf, y, out=Wz_buf)
        np.subtract(Wz_buf, E_mu, out=Wz_buf)
        np.multiply(Wz_buf, effective_weights, out=Wz_buf)
        np.copyto(eta_buf, prior_eta_scaled)
        posterior_eta = dgemv(
            1.0, X, Wz_buf, trans=1, beta=1.0, y=eta_buf, overwrite_y=True
        )

        cho = cho_factor(posterior_precision, lower=False, check_finite=False)
        coef = cho_solve(cho, posterior_eta, check_finite=False)

        if iteration > 0:
            np.subtract(coef, coef_old, out=diff_buf)
            np.abs(diff_buf, out=diff_buf)
            if diff_buf.max() < tol:
                break

    return GaussianPosterior(coef, cast(NDArray[np.float64], posterior_precision), cho)


def _rvga_sparse(
    X: csc_array,
    y: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_precision: csc_array,
    *,
    link: LinkFunction,
    effective_weights: NDArray[np.float64],
    prior_decay: float,
    prior_floor: float,
    prior_shift: float,
    n_iter: int,
    tol: float,
    n_gh_nodes: int,
    use_probit: bool,
    prior_factor: Optional[Any] = None,
    coef_init: Optional[NDArray[np.float64]] = None,
) -> GaussianPosterior:
    """Sparse R-VGA loop with Woodbury predictive variances.

    Precomputes the n×n Gram matrix G = X A⁻¹ Xᵀ (A = scaled prior)
    once, then each IRLS iteration obtains predictive variances via
    the Woodbury identity in O(n³) instead of O(|A|³) or O(n·p).
    """
    from ._sparse_bayesian_linear_regression import create_sparse_factor

    n_samples = X.shape[0]

    no_decay = prior_decay == 1.0
    prior_precision_scaled = (
        prior_precision if no_decay else prior_decay * prior_precision
    )
    prior_eta_scaled = (
        prior_precision @ prior_mean
        if no_decay
        else prior_decay * (prior_precision @ prior_mean)
    )
    # prior_factor is for unscaled prior_precision; rescale to match
    if prior_factor is not None and not no_decay:
        from ._sparse_bayesian_linear_regression import scale_factor

        prior_factor = scale_factor(prior_factor, prior_decay)
    shifted = _stabilized_prior_sparse(
        prior_precision_scaled, prior_decay, prior_floor, prior_shift
    )
    if shifted is not prior_precision_scaled:
        # A diagonal shift is a rank-p change no cheap factor update
        # covers; the Gram matrix refactorizes from scratch.
        prior_precision_scaled = shifted
        prior_factor = None

    # Precompute G = X A^{-1} X^T (n×n, computed once).
    # Woodbury: Λ = A + X^T diag(W) X
    #   => X Λ^{-1} X^T = G - G (diag(1/W) + G)^{-1} G
    G = _precompute_gram_matrix(prior_precision_scaled, X, prior_factor=prior_factor)
    diag_G = np.diag(G).copy()

    coef = prior_mean.copy() if coef_init is None else coef_init.copy()
    # The Gram matrix was the last use of the prior's factor as such;
    # from here it seeds the posterior's (refactorize checks the pattern).
    sparse_factor = prior_factor
    use_probit_path = use_probit and link == "logit"
    W_prev: Optional[NDArray[np.float64]] = None

    for iteration in range(n_iter):
        if iteration > 0:
            coef_old = coef.copy()

        m = cast(NDArray[np.float64], X @ coef)

        # Predictive variances via Woodbury identity.
        # Iteration 0: Λ = A, so v = diag(G).
        # Iteration k: Λ = A + X^T diag(W_{k-1}) X, use Woodbury.
        if W_prev is None:
            v = diag_G
        else:
            M = G.copy()
            M.ravel()[:: n_samples + 1] += 1.0 / W_prev
            cho_M = cho_factor(M, lower=False, check_finite=False)
            # diag(G M^{-1} G) = colsum(V^2) where R^T V = G
            V = solve_triangular(cho_M[0], G, lower=False, trans=1, check_finite=False)
            v = diag_G - np.sum(V * V, axis=0)

        if link == "log":
            E_mu, E_W = log_expected_weights(m, v)
        elif use_probit_path:
            E_mu, E_W = probit_expected_weights(m, v)
        else:
            E_mu, E_W = gh_expected_weights(m, v, link, n_gh_nodes)

        np.maximum(E_W, 1e-10, out=E_W)
        W = E_W * effective_weights
        W_prev = W

        XW = X.multiply(W.reshape(-1, 1)).tocsc()
        posterior_precision = csc_array(X.T @ XW)
        posterior_precision += prior_precision_scaled

        # Fused Wz = eff_w * (E_W * m + y - E_mu) avoids division by E_W
        Wz = E_W * m + (y - E_mu)
        Wz *= effective_weights
        likelihood_eta = X.T @ Wz
        posterior_eta = prior_eta_scaled + likelihood_eta

        if sparse_factor is None:
            sparse_factor = create_sparse_factor(posterior_precision)
        else:
            sparse_factor = sparse_factor.refactorize(posterior_precision)
        coef = sparse_factor.solve(posterior_eta)

        if iteration > 0:
            coef_change = np.max(np.abs(coef - coef_old))
            if coef_change < tol:
                break

    return GaussianPosterior(coef, posterior_precision, sparse_factor)


def update_gaussian_posterior_rvga(
    X: ArrayType,
    y: NDArray[np.float64],
    prior_mean: ArrayType,
    prior_precision: ArrayType,
    *,
    link: LinkFunction,
    sample_weight: Optional[NDArray[np.float64]] = None,
    learning_rate: float = 1.0,
    sparse: bool = False,
    prior_floor: float = 0.0,
    prior_shift: float = 0.0,
    n_iter: int = 5,
    tol: float = 1e-4,
    n_gh_nodes: int = 20,
    use_probit: bool = True,
    prior_factor: Optional[Any] = None,
    batch_size: Optional[int] = None,
    coef_init: Optional[NDArray[np.float64]] = None,
) -> GaussianPosterior:
    """Update Gaussian posterior using R-VGA (expected curvature).

    Parameters
    ----------
    batch_size : int, optional
        Sparse path only. When set and ``n_samples > batch_size``, the
        batch is processed as sequential minibatch updates (R-VGA's
        native recursive mode), threading the posterior factor between
        chunks. Bounds the Gram-matrix allocation at
        ``8 * batch_size**2`` bytes instead of ``8 * n_samples**2``.
        Decay and sample weights compose exactly across chunks; the
        variational weights make the result mildly dependent on row
        order. Ignored when ``sparse=False``.

    References
    ----------
    Lambert, Bonnabel, Bach (2022). Statistics and Computing, 32, 10.
    """
    assert X.shape is not None, "X must be a 2D array"
    n_samples = X.shape[0]

    if n_samples == 0:
        return GaussianPosterior(prior_mean.copy(), prior_precision.copy(), None)

    if sparse and batch_size is not None and batch_size >= 1 and n_samples > batch_size:
        X_csr = csc_array(X).tocsr()
        posterior = GaussianPosterior(prior_mean, prior_precision, prior_factor)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            posterior = update_gaussian_posterior_rvga(
                csc_array(X_csr[start:end]),
                y[start:end],
                posterior.mean,
                posterior.precision,
                link=link,
                sample_weight=(
                    sample_weight[start:end] if sample_weight is not None else None
                ),
                learning_rate=learning_rate,
                sparse=True,
                prior_floor=prior_floor,
                prior_shift=prior_shift if start == 0 else 0.0,
                n_iter=n_iter,
                coef_init=coef_init if start == 0 else None,
                tol=tol,
                n_gh_nodes=n_gh_nodes,
                use_probit=use_probit,
                prior_factor=posterior.factor,
            )
        return posterior

    effective_weights = compute_effective_weights(
        n_samples, sample_weight, learning_rate
    )
    prior_decay = learning_rate**n_samples

    if sparse:
        return _rvga_sparse(
            csc_array(X),
            y,
            np.asarray(prior_mean),
            csc_array(prior_precision),
            link=link,
            effective_weights=effective_weights,
            prior_decay=prior_decay,
            prior_floor=prior_floor,
            prior_shift=prior_shift,
            n_iter=n_iter,
            coef_init=coef_init,
            tol=tol,
            n_gh_nodes=n_gh_nodes,
            use_probit=use_probit,
            prior_factor=prior_factor,
        )
    else:
        return _rvga_dense(
            np.asarray(X),
            y,
            np.asarray(prior_mean),
            np.asarray(prior_precision),
            link=link,
            effective_weights=effective_weights,
            prior_decay=prior_decay,
            prior_floor=prior_floor,
            prior_shift=prior_shift,
            n_iter=n_iter,
            coef_init=coef_init,
            tol=tol,
            n_gh_nodes=n_gh_nodes,
            use_probit=use_probit,
        )


@dataclass
class RVGAApproximator(MemoryUsageMixin, PosteriorApproximator):
    """R-VGA posterior approximation via expected curvature.

    Replaces Laplace's point-estimate curvature with expected curvature
    under the approximate posterior, correcting systematic bias for
    non-Gaussian likelihoods.

    Parameters
    ----------
    n_iter : int, default=5
        Maximum iterations per update.
    tol : float, default=1e-4
        Convergence tolerance on coefficient change.
    n_gh_nodes : int, default=20
        Number of Gauss-Hermite quadrature nodes. Only used for
        link='logit' with ``use_probit=False``; the log link uses the
        exact closed form :math:`E[\\exp(\\eta)] = \\exp(m + v/2)`.
    use_probit : bool, default=True
        If True and link='logit', use the analytical probit
        approximation instead of GH quadrature. Ignored for log link.
    batch_size : int or None, default=2048
        Sparse models only. Batches larger than this are processed as
        sequential minibatch updates (R-VGA's native recursive mode),
        bounding the predictive-variance Gram allocation at
        ``8 * batch_size**2`` bytes (~34 MB at the default) instead of
        growing quadratically with the batch. Sequential updates make
        the posterior mildly dependent on row order; set to None to
        force a single joint update regardless of batch size. Ignored
        for dense models, whose memory does not depend on batch size.

    References
    ----------
    Lambert, Bonnabel, Bach (2022). Statistics and Computing, 32, 10.

    See Also
    --------
    LaplaceApproximator : Default approximation using IRLS.
    BayesianGLM : Estimator that uses this protocol.

    Examples
    --------
    >>> from bayesianbandits import RVGAApproximator, BayesianGLM
    >>> model = BayesianGLM(approximator=RVGAApproximator())
    """

    n_iter: int = 5
    tol: float = 1e-4
    n_gh_nodes: int = 20
    use_probit: bool = True
    batch_size: Optional[int] = 2048

    def update_posterior(
        self,
        X: ArrayType,
        y: NDArray[np.float64],
        prior_mean: ArrayType,
        prior_precision: ArrayType,
        link: LinkFunction,
        sample_weight: Optional[NDArray[np.float64]],
        learning_rate: float,
        sparse: bool,
        prior_factor: Optional[Any] = None,
        prior_floor: float = 0.0,
        prior_shift: float = 0.0,
        coef_init: Optional[NDArray[np.float64]] = None,
    ) -> GaussianPosterior:
        return update_gaussian_posterior_rvga(
            X,
            y,
            prior_mean,
            prior_precision,
            link=link,
            sample_weight=sample_weight,
            learning_rate=learning_rate,
            sparse=sparse,
            prior_floor=prior_floor,
            prior_shift=prior_shift,
            n_iter=self.n_iter,
            coef_init=coef_init,
            tol=self.tol,
            n_gh_nodes=self.n_gh_nodes,
            use_probit=self.use_probit,
            prior_factor=prior_factor,
            batch_size=self.batch_size,
        )
