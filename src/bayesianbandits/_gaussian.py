from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional, Protocol, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dgemv, dsymv, dsyrk
from scipy.sparse import csc_array
from scipy.special import expit

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
    np.maximum(d_mu_d_eta, 1e-10, out=d_mu_d_eta)  # in-place
    residual /= d_mu_d_eta  # in-place
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


def _irls_dense(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
    *,
    link: LinkFunction,
    effective_weights: NDArray[np.float64],
    prior_decay: float,
    n_iter: int,
    tol: float,
) -> GaussianPosterior:
    """Dense IRLS loop with pre-allocated buffers and fused BLAS calls."""
    n_samples, n_features = X.shape

    # Precompute prior contributions
    no_decay = prior_decay == 1.0
    prior_precision_scaled = (
        prior_precision if no_decay else prior_decay * prior_precision
    )
    prior_eta_scaled = dsymv(prior_decay, prior_precision, prior_mean)

    # F-order copy of prior precision for dsyrk accumulation buffer
    prior_prec_F = np.asfortranarray(prior_precision_scaled)

    # Pre-allocate reusable buffers
    X_weighted = np.empty_like(X)
    W_sqrt_buf = np.empty(n_samples, dtype=np.float64)
    Wz_buf = np.empty(n_samples, dtype=np.float64)
    diff_buf = np.empty(n_features, dtype=np.float64)
    precision_buf = np.empty_like(prior_prec_F)
    eta_buf = np.empty_like(prior_eta_scaled)

    coef = prior_mean.copy()
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
    n_iter: int,
    tol: float,
) -> GaussianPosterior:
    """Sparse IRLS loop using CHOLMOD/SuperLU factorization."""
    from ._sparse_bayesian_linear_regression import create_sparse_factor

    # Precompute prior contributions
    no_decay = prior_decay == 1.0
    prior_precision_scaled = (
        prior_precision if no_decay else prior_decay * prior_precision
    )
    prior_eta_scaled = (
        prior_precision @ prior_mean
        if no_decay
        else prior_decay * (prior_precision @ prior_mean)
    )

    coef = prior_mean.copy()
    coef_old = coef
    sparse_factor = None
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
    n_iter: int = 3,
    tol: float = 1e-4,
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
            n_iter=n_iter,
            tol=tol,
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
            n_iter=n_iter,
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
    ) -> GaussianPosterior: ...


@dataclass
class LaplaceApproximator(PosteriorApproximator):
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
            n_iter=self.n_iter,
            tol=self.tol,
        )
