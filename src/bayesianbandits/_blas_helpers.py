"""Thin wrappers around BLAS routines used by the Bayesian regression estimators."""

from __future__ import annotations

from typing import Any, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.blas import dgemm, dgemv, dsymv, dsyrk  # type: ignore[attr-defined]
from scipy.linalg.lapack import dgeqrf, dgeqrf_lwork  # type: ignore[attr-defined]
from scipy.sparse import csc_array

__all__ = [
    "dgemm",
    "dgemv",
    "dsymv",
    "dsyrk",
    "update_precision_dense",
    "compute_eta_dense",
    "lower_predictive_sqrt",
    "dense_matvec",
    "standard_normal_f",
    "affine_lower_factor",
]

_Array = Union[NDArray[Any], csc_array]


def update_precision_dense(
    alpha: float,
    X_weighted: _Array,
    prior_scaled: _Array,
) -> NDArray[np.float64]:
    """Compute ``alpha * X_weighted.T @ X_weighted + prior_scaled`` in-place.

    ``prior_scaled`` must be F-contiguous and is overwritten.  Only the
    **upper triangle** of the result is meaningful — ``dsyrk`` does not
    fill the lower triangle.  All downstream consumers (``dsymv``,
    ``cho_factor(lower=False)``, and ``dsyrk`` itself) read only the
    upper triangle, so no symmetrisation is needed.
    """
    return dsyrk(alpha, X_weighted, trans=1, beta=1.0, c=prior_scaled, overwrite_c=True)


def compute_eta_dense(
    prior_decay: float,
    cov_inv: _Array,
    coef: _Array,
    alpha: float,
    X: _Array,
    y_weighted: _Array,
) -> NDArray[np.float64]:
    """Compute ``prior_decay * cov_inv @ coef + alpha * X.T @ y_weighted``.

    Fuses two BLAS calls (``dsymv`` then ``dgemv``) into one buffer
    with no extra allocation.
    """
    eta = dsymv(prior_decay, cov_inv, coef)
    return dgemv(alpha, X, y_weighted, trans=1, beta=1.0, y=eta, overwrite_y=True)


def lower_predictive_sqrt(B: NDArray[np.float64], n_rows: int) -> NDArray[np.float64]:
    """Lower-triangular ``L`` of shape ``(n_rows, n_rows)`` with ``L Lᵀ = Bᵀ B``.

    ``B`` has shape ``(n_features, n_rows)`` and is **overwritten**.

    Takes the ``R`` factor of ``B``'s QR: ``Rᵀ R = Bᵀ B`` exactly, with
    no need for ``Bᵀ B`` to be formed and no loss of the conditioning
    that squaring it would cost.  ``R`` is padded with zero rows when
    ``B`` has fewer rows than columns, which is what makes the result
    exact for rank-deficient input (repeated prediction rows) where a
    Cholesky of the explicit covariance would fail.

    Calls ``dgeqrf`` directly rather than ``numpy.linalg.qr``.  The two
    compute the same factor, but numpy and scipy bind *separate* copies
    of OpenBLAS, each with its own thread pool; alternating between them
    within one call parks and unparks both pools and dominates the
    runtime at every size we measured.  Every routine on this path is
    therefore taken from ``scipy.linalg``, so a caller that reaches here
    through a ``scipy`` triangular solve stays inside one pool.
    """
    lwork = int(dgeqrf_lwork(B.shape[0], B.shape[1])[0])
    qr, _tau, _work, _info = dgeqrf(B, lwork=lwork, overwrite_a=True)
    R = np.triu(qr[: min(qr.shape)])
    if R.shape[0] < n_rows:  # more prediction rows than features
        R = np.vstack([R, np.zeros((n_rows - R.shape[0], n_rows))])
    # Sign-flipping rows of R leaves Rᵀ R unchanged, so this only
    # canonicalizes the factor to a non-negative diagonal.
    signs = np.where(np.diagonal(R) < 0.0, -1.0, 1.0)
    return cast(NDArray[np.float64], np.asfortranarray(R.T * signs, dtype=np.float64))


def dense_matvec(A: NDArray[Any], x: NDArray[Any]) -> NDArray[np.float64]:
    """``A @ x`` for dense ``A``, through ``dgemv`` rather than ``numpy``.

    Trivial in flops, but it is the one ``numpy`` BLAS call left on the
    reward-space path, and a single round trip out of ``scipy``'s thread
    pool costs far more than the product itself (see
    :func:`lower_predictive_sqrt`).

    ``dgemv`` wants Fortran order.  A C-contiguous ``A`` is passed as its
    transpose with ``trans=1``, which is the same memory read the other
    way round, so neither layout is copied.
    """
    A2 = np.asarray(A, dtype=np.float64)
    x1 = np.asarray(x, dtype=np.float64).ravel()
    if A2.flags.c_contiguous and not A2.flags.f_contiguous:
        return cast(NDArray[np.float64], dgemv(1.0, A2.T, x1, trans=1))
    return cast(NDArray[np.float64], dgemv(1.0, A2, x1))


def standard_normal_f(
    rng: np.random.Generator, size: int, n_rows: int
) -> NDArray[np.float64]:
    """``(size, n_rows)`` standard normals in **Fortran** order.

    Drawing the transpose and viewing it back costs nothing and lets
    ``affine_lower_factor`` hand the buffer to ``dgemm`` without a copy.
    """
    return cast(NDArray[np.float64], rng.standard_normal((n_rows, size)).T)


def affine_lower_factor(
    mean: NDArray[np.float64], z: NDArray[np.float64], L: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``mean + z @ Lᵀ`` in one ``dgemm``, accumulating onto the mean.

    ``z`` should come from :func:`standard_normal_f` and ``L`` from
    :func:`lower_predictive_sqrt` so that both are already F-contiguous
    and ``dgemm`` runs without copying either operand.
    """
    out = np.empty((z.shape[0], L.shape[0]), dtype=np.float64, order="F")
    out[:] = mean
    return cast(
        NDArray[np.float64],
        dgemm(1.0, z, L, trans_b=1, beta=1.0, c=out, overwrite_c=True),
    )


def project_draws(
    weights: NDArray[np.float64], X: Union[NDArray[np.float64], csc_array]
) -> NDArray[np.float64]:
    """``weights @ X.T``, draw-contiguous: ``result.T`` is C-contiguous.

    The layout contract of the sampling entry points (see
    ``tests/test_sample_layout.py``): computing the transposed product
    gives the same inner products but lands each row's draws adjacent,
    so the agents' transposed reshapes for policies need no copy. Do not
    "simplify" back to ``weights @ X.T``. ``size == 1`` keeps the
    original BLAS call, pinning Thompson sampling's draw stream.
    """
    if weights.shape[0] == 1:
        return cast(NDArray[np.float64], weights @ X.T)
    return cast(NDArray[np.float64], (X @ weights.T).T)


#: Slab byte budget for :func:`draw_contiguous`: keeps a slab's source
#: block L2-resident while it is scattered out.
_COPY_SLAB_BYTES = 1 << 20


def draw_contiguous(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """``samples`` with a unit-stride draw (last) axis, copying if needed.

    Normalization at the policy boundary for learners that ignore the
    layout contract; conforming input passes through untouched. Copies
    slab-wise: numpy's own strided copy walks the draw axis outermost
    and runs ~3x slower on large tensors.
    """
    if samples.dtype == np.float64 and samples.strides[-1] == samples.itemsize:
        return samples
    n_leading = int(np.prod(samples.shape[:-1]))
    n_draws = samples.shape[-1]
    out = np.empty(samples.shape, dtype=np.float64)
    slab = max(1, min(n_draws, _COPY_SLAB_BYTES // max(1, n_leading * 8)))
    for start in range(0, n_draws, slab):
        stop = start + slab
        out[..., start:stop] = samples[..., start:stop]
    return out
