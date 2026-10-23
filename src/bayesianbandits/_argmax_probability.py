"""Selection probabilities of a single-draw argmax: Thompson sampling's
propensity, :math:`P(r_a > r_b \\;\\forall b \\ne a)` under the posterior
of the arm scores. Both routines work in log space, so the tail
probabilities IPS divides by keep their relative precision."""

from typing import Any, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, log_ndtr, ndtri_exp, stdtr
from scipy.stats import chi2, norm, qmc

FloatArray = NDArray[np.floating[Any]]


class PredictiveMoments(NamedTuple):
    """A learner's posterior over its linear predictor on prediction rows:
    ``(n_rows,)`` means, ``(n_blocks, k, k)`` covariances jointly within
    blocks of ``k`` rows, and ``df`` when Student-t, else None."""

    mean: NDArray[np.float64]
    cov: NDArray[np.float64]
    df: Optional[float]


# Trapezoid grids in the arm's own standardized coordinate; the Student-t
# grid is u = sinh(x), to reach its polynomial tails
_U_NORMAL = np.linspace(-8.5, 8.5, 340).astype(np.float64)
_LOG_W_NORMAL = norm.logpdf(_U_NORMAL) + np.log(_U_NORMAL[1] - _U_NORMAL[0])
_X_T = np.linspace(-10.0, 10.0, 400)
_U_T = np.sinh(_X_T).astype(np.float64)
_LOG_JACOBIAN_T = np.log(np.cosh(_X_T)) + np.log(_X_T[1] - _X_T[0])


def independent_argmax_probabilities(
    mean: FloatArray, var: FloatArray, df: Optional[FloatArray] = None
) -> NDArray[np.float64]:
    """``(n, K)`` argmax probabilities for independent arm scores,
    :math:`\\int f_a(t) \\prod_{b \\ne a} F_b(t)\\,dt` by trapezoid.
    ``mean``, ``var`` and ``df`` are ``(n, K)``; ``df`` None is Gaussian."""
    sd = np.sqrt(var)
    u: NDArray[np.float64] = _U_NORMAL if df is None else _U_T
    out = np.empty(mean.shape)
    for a in range(mean.shape[1]):
        t = mean[:, a, None] + sd[:, a, None] * u
        z = (t[:, None, :] - mean[:, :, None]) / sd[:, :, None]  # (n, K, G)
        if df is None:
            log_cdf, log_w = log_ndtr(z), _LOG_W_NORMAL
        else:
            log_cdf = _log_stdtr(df[:, :, None], z)
            log_w = _log_t_pdf(df[:, a, None], u) + _LOG_JACOBIAN_T
        log_cdf[:, a] = 0.0
        out[:, a] = np.exp(log_w + log_cdf.sum(axis=1)).sum(axis=1)
    return out


def _log_stdtr(df: FloatArray, z: FloatArray) -> NDArray[np.float64]:
    """Student-t log-cdf from one ``stdtr`` call on the lower tail (the
    ``scipy.stats`` version is far slower); the normal where ``df`` is
    infinite."""
    finite = np.isfinite(df)
    tail = stdtr(np.where(finite, df, 1.0), -np.abs(z))
    with np.errstate(divide="ignore"):  # an underflowing tail weighs nothing
        log_cdf = np.where(z <= 0, np.log(tail), np.log1p(-tail))
    return np.where(finite, log_cdf, log_ndtr(z))


def _log_t_pdf(df: FloatArray, u: FloatArray) -> NDArray[np.float64]:
    """Student-t log-density; the normal where ``df`` is infinite."""
    finite = np.isfinite(df)
    d = np.where(finite, df, 1.0)
    log_pdf = (
        gammaln((d + 1) / 2)
        - gammaln(d / 2)
        - np.log(d * np.pi) / 2
        - (d + 1) / 2 * np.log1p(u**2 / d)
    )
    return np.where(finite, log_pdf, norm.logpdf(u))


def argmax_probabilities(
    mean: FloatArray,
    cov: FloatArray,
    df: Optional[FloatArray] = None,
    n_points: int = 512,
) -> NDArray[np.float64]:
    """``(n, K)`` argmax probabilities for jointly Gaussian arm scores,
    ``cov`` ``(n, K, K)`` of any rank, or multivariate t with ``df``
    ``(n,)``.

    Genz's sequential conditioning on the score differences ``L z > -m``:
    each row bounds ``z`` at its last nonzero column of ``L``, the rows
    ending at a step make an interval integrated exactly, and ``z`` is
    drawn from it for the rows still open, on Sobol points. Ties go to
    the lower index, as ``argmax`` breaks them."""
    n, n_arms = mean.shape
    if n_arms == 1:
        return np.ones((n, 1))
    n_dims = max(n_arms - 2 + (df is not None), 1)
    # seed=, not rng=: the project floor is scipy 1.14
    sobol = qmc.Sobol(n_dims, scramble=True, seed=0)  # pyright: ignore[reportCallIssue]
    w = sobol.random(n_points).T
    w = np.clip(w, 1e-300, 1.0 - 1e-16)
    s = np.ones((n, 1))  # scale of the zero-mean part, per point under t
    if df is not None:
        s = np.sqrt(df[:, None] / chi2.ppf(w[0], df[:, None]))
        w = w[1:]
    out = np.empty((n, n_arms))
    for a in range(n_arms):
        others = np.delete(np.arange(n_arms), a)
        D = -np.eye(n_arms)[others]
        D[:, a] = 1.0
        L, last = _directions(np.einsum("ij,njk,lk->nil", D, cov, D))
        ms = (mean @ D.T)[:, :, None] / s[:, None, :]  # (n, K-1, 1 or N)
        # rows without a stochastic part hold or fail outright
        fixed = (last < 0)[:, :, None] & (
            (ms < 0) | ((ms == 0) & (others < a)[:, None])
        )
        logp = np.where(fixed.any(axis=1), -np.inf, 0.0) + np.zeros(n_points)
        z = np.zeros((n_arms - 2, n, n_points))
        for i in range(n_arms - 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                acc = np.einsum("nj,jnp->np", L[:, i, :i], z[:i])
                lo = (-ms[:, i] - acc) / L[:, i, i, None]
            lo = np.where((last[:, i] == i)[:, None], lo, -np.inf)
            hi = np.full_like(lo, np.inf)
            folded = np.flatnonzero((last[:, i + 1 :] == i).any(axis=1))
            if folded.size:  # rank-deficient contexts: later rows end here too
                lo_f, hi_f = _folded_bounds(
                    L[folded], last[folded], ms[folded], z[:, folded], i
                )
                lo[folded] = np.maximum(lo[folded], lo_f)
                hi[folded] = hi_f
                logp_i, log_far, upper = _log_interval(lo, hi)
            elif np.isneginf(lo).all():
                continue  # a null direction
            else:  # one lower bound: the upper tail alone
                logp_i = log_ndtr(-lo)
                log_far = np.full_like(lo, -np.inf)
                upper = np.ones(lo.shape, dtype=np.bool_)
            logp += logp_i
            if i < n_arms - 2:
                z[i] = _draw_interval(logp_i, log_far, upper, w[i])
        out[:, a] = np.exp(logp).mean(axis=1)
    return out


def _directions(S: FloatArray) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Lower factor of each ``S`` with null pivots zeroed, and each row's
    last nonzero column (``-1`` for an all-zero row)."""
    k = S.shape[-1]
    scale = np.einsum("nii->n", S) / k
    scale = np.where(scale > 0, scale, 1.0)
    L = np.linalg.cholesky(S + (1e-12 * scale)[:, None, None] * np.eye(k))
    null = np.diagonal(L, axis1=1, axis2=2) ** 2 < 1e-8 * scale[:, None]
    L = np.where(null[:, None, :], 0.0, L)
    nonzero = L != 0.0
    last = np.where(nonzero.any(axis=2), (nonzero * np.arange(k)).max(axis=2), -1)
    return L, last


def _folded_bounds(
    L: FloatArray, last: NDArray[np.intp], ms: FloatArray, z: FloatArray, i: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(n, N)`` lower and upper bounds on ``z_i`` from the rows below
    ``i`` that end at column ``i``."""
    rows = slice(i + 1, None)
    active = last[:, rows] == i
    coef = L[:, rows, i]
    acc = np.einsum("nji,inp->njp", L[:, rows, :i], z[:i])
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = (-ms[:, rows] - acc) / coef[:, :, None]
    lo = np.where((active & (coef > 0))[:, :, None], bound, -np.inf).max(axis=1)
    hi = np.where((active & (coef < 0))[:, :, None], bound, np.inf).min(axis=1)
    return lo, hi


def _log_interval(
    lo: FloatArray, hi: FloatArray
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """``log(Phi(hi) - Phi(lo))`` from the upper tail when the interval
    sits above zero, with the far tail and which side for
    :func:`_draw_interval`."""
    upper = lo > 0
    log_near = np.where(upper, log_ndtr(-lo), log_ndtr(hi))
    log_far = np.where(upper, log_ndtr(-hi), log_ndtr(lo))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logp = log_near + np.log1p(-np.exp(log_far - log_near))
    return np.where((lo < hi) & np.isfinite(log_near), logp, -np.inf), log_far, upper


def _draw_interval(
    logp: FloatArray, log_far: FloatArray, upper: NDArray[np.bool_], w: FloatArray
) -> NDArray[np.float64]:
    """A standard normal truncated to the interval, by inverse cdf from
    ``w`` in the tail :func:`_log_interval` chose; dead points get a
    finite placeholder."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ndtri_exp(np.logaddexp(log_far, np.log(w) + logp))
    z = np.clip(np.nan_to_num(z), -40.0, 40.0)
    return np.where(upper, -z, z)
