from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Protocol, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import expit

from ._sparse_bayesian_linear_regression import SparseSolver, solver

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


def solve_precision_weighted_mean(
    precision: Union[NDArray[np.float64], csc_array],
    eta: NDArray[np.float64],
    sparse: bool,
) -> NDArray[np.float64]:
    """Solve precision @ mu = eta for mu."""
    if sparse:
        if solver == SparseSolver.CHOLMOD:
            from sksparse.cholmod import cholesky as cholmod_cholesky

            return cholmod_cholesky(csc_matrix(precision))(eta)
        else:
            lu = splu(
                precision,
                # These two settings tell SuperLU that we're decomposing a Hermitian
                # positive-definite matrix, so we only want to pivot on the diagonal.
                # This preserves the sparsity of the matrix better than the default,
                # which allows for off-diagonal pivoting. See SuperLU User Guide
                # for more details.
                diag_pivot_thresh=0.0,
                permc_spec="MMD_AT_PLUS_A",
                options=dict(SymmetricMode=True),
            )
            return lu.solve(eta)
    else:
        from scipy.linalg import solve

        return solve(precision, eta, check_finite=False, assume_a="pos")


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

    Benefits of MAP/Laplace
    -----------------------
    - Fast: One Newton step per update (vs. quadrature integration)
    - Scalable: Handles batch updates naturally
    - Proven: Standard in production systems (Google, Meta, etc.)
    - Stable: Well-understood optimization problem
    - Natural parameters: Updates are additive

    Drawbacks of MAP/Laplace
    ------------------------
    - Point estimate: Only considers curvature at mode
    - Overconfident: Can underestimate uncertainty away from mode
    - No skewness: Forces symmetric Gaussian approximation
    - Mode ≠ Mean: For skewed posteriors, MAP is biased
    - Poor for multimodal: Only captures one mode

    When to Use
    -----------
    Good for:
    - Large datasets (CLT makes posterior more Gaussian)
    - Batch processing (sees all data at once)
    - Minibatch updates (fast convergence)
    - Well-specified models
    - When computational speed is critical


    Implementation Note
    -------------------
    We use one step of Newton's method (IRLS) starting from the prior mean.
    For well-specified problems, this is often sufficient. For difficult
    problems, multiple iterations might be needed (not implemented here).

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
    if sparse:
        X_sparse = csc_array(X)

    assert X.shape is not None, "X must be a 2D array"
    n_samples = X.shape[0]

    # Apply learning rate decay to sample weights
    effective_weights = compute_effective_weights(
        n_samples, sample_weight, learning_rate
    )
    prior_decay = learning_rate**n_samples

    # Precompute prior contributions (these don't change in the loop)
    prior_precision_scaled = prior_decay * prior_precision
    prior_eta_scaled = prior_decay * (prior_precision @ prior_mean)

    # Initialize at prior mean
    coef = prior_mean.copy()

    # IRLS iterations
    for iteration in range(n_iter):
        # Store old coefficients for convergence check
        if iteration > 0 and n_iter > 1:
            coef_old = coef.copy()

        # Compute current predictions
        eta = X @ coef

        # Get mean and derivative from link function
        if link == "logit":
            link_out = logit_link_and_derivative(eta)
        elif link == "log":
            link_out = log_link_and_derivative(eta)
        else:
            raise ValueError(f"Unknown link function: {link}")

        # Compute GLM weights and working response
        glm_weights = compute_glm_weights_and_working_response(
            y, link_out.mu, link_out.d_mu_d_eta, eta, effective_weights
        )

        # Likelihood contribution
        if sparse:
            from scipy.sparse import diags

            W_sqrt = diags(np.sqrt(glm_weights.W), format="csc")
            X_weighted = W_sqrt @ X_sparse  # type: ignore
            likelihood_precision = X_weighted.T @ X_weighted
            likelihood_eta = X_sparse.T @ (glm_weights.W * glm_weights.z)  # type: ignore
        else:
            X_weighted = X * np.sqrt(glm_weights.W)[:, np.newaxis]
            likelihood_precision = X_weighted.T @ X_weighted
            likelihood_eta = X.T @ (glm_weights.W * glm_weights.z)

        # Combine prior and likelihood (using precomputed prior terms)
        posterior_precision = prior_precision_scaled + likelihood_precision
        posterior_eta = prior_eta_scaled + likelihood_eta

        # Solve for new coefficients
        coef = solve_precision_weighted_mean(posterior_precision, posterior_eta, sparse)  # type: ignore

        # Check convergence if n_iter > 1
        if iteration > 0 and n_iter > 1:
            coef_change = np.max(np.abs(coef - coef_old))  # type: ignore
            if coef_change < tol:
                break

    return GaussianPosterior(coef, posterior_precision)  # type: ignore


class PosteriorApproximator(Protocol):
    """Base class for posterior approximation methods."""

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
    """Laplace approximation using IRLS.

    Parameters
    ----------
    n_iter : int, default=5
        Number of Newton iterations to perform.
        - 1: Single step update (fast, may not converge)
        - >1: Iterate until convergence or max iterations

    tol : float, default=1e-4
        Convergence tolerance for coefficient change.
        Only used if n_iter > 1.
        Convergence when: ||coef_new - coef_old||_∞ < tol

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
