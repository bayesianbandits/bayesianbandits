"""Selection probabilities of a single-draw argmax.

Thompson sampling plays the arm whose one posterior draw is largest, so
its propensity at a context is :math:`P(r_a > r_b \\;\\forall b \\ne a)`
under the posterior of the score vector ``r``: a function of the
posterior, not of the draw. Two routines compute it, both in log space
so the tail probabilities IPS divides by keep their relative precision.
"""

from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, log_ndtr, ndtri_exp, stdtr
from scipy.stats import chi2, norm, qmc

#: Sobol points for the sequential-conditioning integral
N_POINTS = 512


class PredictiveMoments(NamedTuple):
    """A learner's posterior over its linear predictor on prediction rows:
    ``mean`` ``(n_rows,)``, ``cov`` ``(n_blocks, k, k)`` jointly within
    each block of ``k`` consecutive rows, and ``df`` when the predictive
    is Student-t rather than Gaussian (``cov`` is then its shape)."""

    mean: NDArray[np.float64]
    cov: NDArray[np.float64]
    df: Optional[float]


#: Trapezoid grids for the one-dimensional integral in the arm's own
#: standardized coordinate: plain for the Gaussian, ``u = sinh(x)`` for
#: the Student-t so its polynomial tails are covered
_U_NORMAL = np.linspace(-8.5, 8.5, 340).astype(np.float64)
_LOG_W_NORMAL = norm.logpdf(_U_NORMAL) + np.log(_U_NORMAL[1] - _U_NORMAL[0])
_X_T = np.linspace(-10.0, 10.0, 400)
_U_T = np.sinh(_X_T).astype(np.float64)
_LOG_JACOBIAN_T = np.log(np.cosh(_X_T)) + np.log(_X_T[1] - _X_T[0])


def independent_argmax_probabilities(
    mean: NDArray[np.float64],
    var: NDArray[np.float64],
    df: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """``(n, K)`` argmax probabilities for independent arm scores.

    :math:`p_a = \\int f_a(t) \\prod_{b \\ne a} F_b(t)\\,dt` by
    trapezoid on a ``sinh`` grid in each arm's own standardized
    coordinate, with the log-cdfs summed over every arm and the arm's
    own subtracted. ``mean`` and ``var`` are ``(n, K)``; ``df`` of the
    same shape makes each score Student-t, else Gaussian.
    """
    sd = np.sqrt(var)
    u: NDArray[np.float64] = _U_NORMAL if df is None else _U_T
    out = np.empty(mean.shape)
    for a in range(mean.shape[1]):
        t = mean[:, a, None] + sd[:, a, None] * u  # (n, G)
        z = np.asarray(
            (t[:, None, :] - mean[:, :, None]) / sd[:, :, None], dtype=np.float64
        )  # (n, K, G)
        if df is None:
            log_cdf = log_ndtr(z)
            log_w = _LOG_W_NORMAL
        else:
            log_cdf = _log_stdtr(df[:, :, None], z)
            log_w = _log_t_pdf(df[:, a, None], u) + _LOG_JACOBIAN_T
        log_cdf[:, a] = 0.0
        out[:, a] = np.exp(log_w + log_cdf.sum(axis=1)).sum(axis=1)
    return out


def _log_stdtr(df: NDArray[np.float64], z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Student-t log-cdf: one ``stdtr`` call on the lower tail, read from
    whichever side keeps precision; the normal where ``df`` is infinite.
    (``scipy.stats.t.logcdf`` is far slower.) An underflowing lower tail
    is ``-inf``, which weighs nothing, as it should."""
    finite = np.isfinite(df)
    tail = stdtr(np.where(finite, df, 1.0), -np.abs(z))
    with np.errstate(divide="ignore"):
        log_cdf = np.where(z <= 0, np.log(tail), np.log1p(-tail))
    return np.where(finite, log_cdf, log_ndtr(z))


def _log_t_pdf(df: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Student-t log-density in closed form; the normal where ``df`` is
    infinite."""
    finite = np.isfinite(df)
    d = np.where(finite, df, 1.0)
    log_pdf = (
        gammaln((d + 1) / 2)
        - gammaln(d / 2)
        - 0.5 * np.log(d * np.pi)
        - (d + 1) / 2 * np.log1p(u**2 / d)
    )
    return np.where(finite, log_pdf, norm.logpdf(u))


def argmax_probabilities(
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
    df: Optional[NDArray[np.float64]] = None,
    n_points: int = N_POINTS,
) -> NDArray[np.float64]:
    """``(n, K)`` argmax probabilities for jointly Gaussian arm scores.

    ``mean`` is ``(n, K)``, ``cov`` ``(n, K, K)`` of any rank; ``df``
    ``(n,)`` makes each context's score vector multivariate t with that
    many degrees of freedom and ``cov`` its shape matrix.

    Genz's sequential conditioning on the ``K - 1`` score differences,
    ``L z > -m``: the directions are the columns of a Cholesky factor
    with the null pivots zeroed, each constraint row bounds ``z`` at its
    last nonzero column, and at every step the rows that end there give
    an interval whose probability is exact, from which ``z`` is drawn
    for the rows still open. Sobol points carry the draws, so a
    rank-deficient covariance costs smoothness, not accuracy: the
    integrand has kinks where the folded rows cross, never jumps. Ties
    go to the lower index, as ``argmax`` breaks them.
    """
    n, n_arms = mean.shape
    out = np.empty((n, n_arms))
    if n_arms == 1:
        out.fill(1.0)
        return out
    n_dims = n_arms - 2 + (df is not None)
    w = qmc.Sobol(max(n_dims, 1), scramble=True, rng=0).random(n_points).T
    w = np.clip(w, 1e-300, 1.0 - 1e-16)
    if df is None:
        s = np.ones((n, 1))
    else:
        s = np.sqrt(df[:, None] / chi2.ppf(w[0], df[:, None]))  # (n, N)
        w = w[1:]
    for a in range(n_arms):
        others = np.delete(np.arange(n_arms), a)
        m = mean[:, [a]] - mean[:, others]  # (n, K-1)
        cov_a = cov[:, [a]]
        S = (
            cov_a[:, :, [a]]
            - cov_a[:, :, others]
            - cov[:, others][:, :, [a]]
            + cov[:, others][:, :, others]
        )
        L, last = _directions(np.asarray(S, dtype=np.float64))
        ms = np.asarray(
            m[:, :, None] / s[:, None, :], dtype=np.float64
        )  # (n, K-1, 1 or N)
        # a row with no stochastic part holds or fails outright
        fixed = (last < 0)[:, :, None] & (
            (ms < 0) | ((ms == 0) & (others < a)[None, :, None])
        )
        logp = np.where(fixed.any(axis=1), -np.inf, 0.0)
        logp = np.broadcast_to(logp, (n, n_points)).copy()
        z = np.zeros((n_arms - 2, n, n_points))
        for i in range(n_arms - 1):
            # the row of this direction, one-sided unless folded rows join it
            own = last[:, i] == i
            coef = L[:, i, i, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                lo = (-ms[:, i] - np.einsum("nj,jnp->np", L[:, i, :i], z[:i])) / coef
            lo = np.where(own[:, None], lo, -np.inf)
            hi = np.full_like(lo, np.inf)
            extra = np.flatnonzero((last[:, i + 1 :] == i).any(axis=1))
            if extra.size:
                lo_x, hi_x = _folded_bounds(
                    L[extra], last[extra], ms[extra], z[:, extra], i
                )
                lo[extra] = np.maximum(lo[extra], lo_x)
                hi[extra] = hi_x
            if np.isneginf(lo).all():
                continue  # a null direction: no row ends here
            if extra.size:
                logp_i, log_far, upper = _log_interval(lo, hi)
            else:
                # one lower bound: the upper tail, nothing to subtract
                logp_i = log_ndtr(-lo)
                log_far = np.full_like(lo, -np.inf)
                upper = np.ones(lo.shape, dtype=np.bool_)
            logp += logp_i
            if i < n_arms - 2:
                z[i] = _draw_interval(logp_i, log_far, upper, w[i])
        out[:, a] = np.exp(logp).mean(axis=1)
    return out


def _folded_bounds(
    L: NDArray[np.float64],
    last: NDArray[np.intp],
    ms: NDArray[np.float64],
    z: NDArray[np.float64],
    i: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bounds on ``z_i`` from the rows below ``i`` that end at column
    ``i``, for the contexts that have any: ``(n, N)`` lower and upper."""
    rows = slice(i + 1, None)
    active = last[:, rows] == i
    coef = L[:, rows, i]
    acc = np.einsum("nji,inp->njp", L[:, rows, :i], z[:i])
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = (-ms[:, rows] - acc) / coef[:, :, None]
    lo = np.where((active & (coef > 0))[:, :, None], bound, -np.inf).max(axis=1)
    hi = np.where((active & (coef < 0))[:, :, None], bound, np.inf).min(axis=1)
    return lo, hi


def _directions(S: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
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


def _log_interval(
    lo: NDArray[np.float64], hi: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """``log(Phi(hi) - Phi(lo))``, computed in the upper tail when the
    interval sits above zero; also the log tail at the interval's far end
    and which tail, for :func:`_draw_interval`."""
    upper = lo > 0
    log_near = np.where(upper, log_ndtr(-lo), log_ndtr(hi))
    log_far = np.where(upper, log_ndtr(-hi), log_ndtr(lo))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logp = log_near + np.log1p(-np.exp(log_far - log_near))
    logp = np.where((lo < hi) & np.isfinite(log_near), logp, -np.inf)
    return logp, log_far, upper


def _draw_interval(
    logp: NDArray[np.float64],
    log_far: NDArray[np.float64],
    upper: NDArray[np.bool_],
    w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """A standard normal truncated to the interval, by inverse cdf from
    ``w`` in the tail :func:`_log_interval` chose. Dead points (``logp``
    of ``-inf``) get a finite placeholder; they contribute nothing."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ndtri_exp(np.logaddexp(log_far, np.log(w) + logp))
    z = np.clip(np.nan_to_num(z), -40.0, 40.0)
    return np.where(upper, -z, z)
