"""Forgetting rules for precision-based Bayesian recursive least squares.

Three strategies for computing a "forgotten" precision matrix from the
current precision, each addressing a limitation of the previous one:

- :class:`ExponentialForgetting` -- scalar decay, simplest but subject to
  covariance windup under non-uniform excitation.
- :class:`StabilizedForgetting` -- Kulhavy & Zarrop (1993) prior floor
  prevents collapse, but forgetting is isotropic.
- :class:`SiftForgetting` -- directional forgetting via SIFt-RLS
  (Lai & Bernstein 2024), forgets only in excited directions.

See ``docs/math/forgetting.rst`` for the full mathematical reference.

Each forgetting rule is a frozen dataclass callable with signature::

    (precision, X, y, lam) -> (R_bar, X_eff, y_eff) | None

where ``R_bar`` is the forgotten precision, ``(X_eff, y_eff)`` is the
effective batch for the RLS update, and ``None`` means "skip this batch."

The caller then does::

    R_new = R_bar + X_eff.T @ X_eff
    eta = R_bar @ theta + X_eff.T @ y_eff
    theta_new = solve(R_new, eta)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cholesky, eigh, lapack, solve_triangular
from scipy.sparse import csc_array

from bayesianbandits._blas_helpers import dsyrk

ArrayType = Union[NDArray[Any], csc_array]

ForgettingResult = tuple[ArrayType, NDArray[Any], NDArray[Any]]


class _SparseFilterResult(NamedTuple):
    """Extended result from sparse filter_batch, carrying precomputed metadata."""

    X_bar: csc_array
    y_bar: NDArray[Any]
    active_cols: NDArray[Any]
    X_bar_a: NDArray[Any]


def _scatter_to_csc(
    dense_block: NDArray[Any],
    row_indices: NDArray[Any],
    col_indices: NDArray[Any],
    shape: tuple[int, int],
) -> csc_array:
    """Build a CSC array by scattering a dense block into specified rows/cols.

    Places ``dense_block`` (r, k) at the intersection of ``row_indices``
    and ``col_indices`` in an otherwise-zero sparse matrix of ``shape``.
    """
    num_rows = len(row_indices)
    num_cols = len(col_indices)
    indptr = np.zeros(shape[1] + 1, dtype=np.int64)
    indptr[col_indices + 1] = num_rows
    np.cumsum(indptr, out=indptr)
    indices = np.tile(row_indices.astype(np.int32), num_cols)
    data = dense_block.T.ravel().copy()
    return csc_array((data, indices, indptr), shape=shape)


def filter_batch(
    X: ArrayType,
    y: NDArray[Any],
    eps: float,
) -> tuple[NDArray[Any], NDArray[Any]] | None:
    """Filter a minibatch via Gram eigendecomposition.

    Eigendecompose the Gram matrix, threshold eigenvalues below ``eps``,
    and return ``(X_bar, y_bar)`` that preserve the sufficient statistics
    ``X.T @ X`` and ``X.T @ y`` in the surviving subspace.

    For sparse X, the eigendecomposition is performed on the k0 x k0
    active-column Gram ``X_a.T @ X_a`` (where k0 is the number of
    nonzero columns), which is equivalent to the p x p Gram but
    avoids O(p^3) when p >> k0.

    Parameters
    ----------
    X : (p, n) design matrix, dense or scipy sparse
    y : (p,) target vector
    eps : eigenvalue threshold

    Returns
    -------
    (X_bar, y_bar) where X_bar is (q, n) dense and y_bar is (q,),
    or None if no eigenvalues survive thresholding (q=0).
    """
    if sparse.issparse(X):
        result = _filter_batch_sparse(X, y, eps)
        if result is None:
            return None
        return result.X_bar, result.y_bar
    return _filter_batch_dense(X, y, eps)


def _filter_batch_dense(
    X: NDArray[Any],
    y: NDArray[Any],
    eps: float,
) -> tuple[NDArray[Any], NDArray[Any]] | None:
    """filter_batch for dense X. Uses the p x p Gram."""
    gram = np.asarray(X @ X.T, dtype=np.float64)

    eigenvalues, eigenvectors = eigh(gram)

    mask = eigenvalues >= eps
    if not np.any(mask):
        return None

    U_q = eigenvectors[:, mask]  # (p, q)
    X_bar = np.asarray(U_q.T @ X, dtype=np.float64)  # (q, n)
    y_bar: NDArray[Any] = U_q.T @ y  # (q,)

    return X_bar, y_bar


def _filter_batch_sparse(
    X: csc_array,
    y: NDArray[Any],
    eps: float,
) -> _SparseFilterResult | None:
    """filter_batch for sparse X.

    Uses the k0 x k0 active-column Gram ``X_a.T @ X_a`` instead of the
    p x p Gram. The nonzero eigenvalues are identical (squared singular
    values of X), so the result is mathematically equivalent.

    Returns a (q, n) X_bar stored as a csc_array to avoid materializing
    a dense (q, n) matrix (which can be gigabytes when n ~ 1M).
    """
    # Find active (nonzero) columns -- must be CSC for column indptr
    X_csc = csc_array(X)
    active_cols = _active_cols_from_csc(X_csc)

    k0 = len(active_cols)
    if k0 == 0:
        return None

    # Extract the dense active submatrix (p, k0)
    X_a = _dense_active_submatrix(X_csc, active_cols)

    # Factor the k0 x k0 Gram to get X_bar_a such that
    # X_bar_a.T @ X_bar_a = gram_a (preserving sufficient statistics).
    #
    # Pivoted Cholesky (dpstrf) reveals the rank and handles both the
    # full-rank and rank-deficient cases, and is >20x faster than eigh.
    gram_a = X_a.T @ X_a  # (k0, k0)
    L, piv, rank, info = lapack.dpstrf(gram_a, lower=True, tol=eps)
    if rank == 0:
        return None

    # dpstrf gives P^T gram_a P = L[:,:r] @ L[:,:r]^T where L is (k0, k0)
    # lower triangular, only the first r=rank columns are meaningful, and
    # piv is the 1-based permutation.
    piv_idx = piv - 1  # 0-based, full k0 permutation
    inv_piv = np.argsort(piv_idx)

    # dpstrf stores the factor in the lower triangle only; the upper
    # triangle retains the original matrix data. Extract via tril.
    # gram_a = M @ M^T where M = tril(L)[:, :rank][inv_piv].
    L_r = np.tril(L[:, :rank])  # (k0, rank), lower trapezoidal
    X_bar_a = L_r[inv_piv].T.copy()  # (rank, k0)

    # y_bar: M @ y_bar = Xty_a. Multiply both sides by the permutation
    # to get L[:rank, :rank] @ y_bar = Xty_a[piv_idx[:rank]], which is
    # a lower triangular solve.
    Xty_a = X_a.T @ y  # (k0,)
    y_bar: NDArray[Any] = solve_triangular(
        L[:rank, :rank], Xty_a[piv_idx[:rank]], lower=True
    )  # (rank,)

    # Build (q, n) as csc_array directly -- only active columns are nonzero.
    q = rank
    n = X.shape[1]
    X_bar = _scatter_to_csc(X_bar_a, np.arange(q, dtype=np.int32), active_cols, (q, n))

    return _SparseFilterResult(X_bar, y_bar, active_cols, X_bar_a)


def _active_cols_from_csc(X_csc: csc_array) -> NDArray[Any]:
    """Return sorted indices of columns that contain at least one nonzero."""
    ip = X_csc.indptr
    return np.flatnonzero(ip[1:] > ip[:-1])


def _dense_active_submatrix(
    X_csc: csc_array, active_cols: NDArray[Any]
) -> NDArray[Any]:
    """Extract the dense submatrix of ``X_csc`` at ``active_cols``."""
    return np.asarray(
        X_csc[:, active_cols].toarray(),  # type: ignore[union-attr]
        dtype=np.float64,
    )


def _sift_downdate_dense(
    precision: NDArray[Any],
    X_bar: NDArray[Any],
    lam: float,
) -> NDArray[Any]:
    """SIFt forgetting step for dense precision.

    Computes ``R_bar = R - (1 - lam) * V^T V`` where ``V = L^{-1} w^T``
    and ``H = L L^T`` is the Cholesky factor of ``X_bar R X_bar^T``.

    Using Cholesky + triangular solve instead of ``solve(H, w.T)``
    ensures the correction ``V^T V`` is exactly symmetric, preventing
    floating-point asymmetry drift over many forgetting steps.
    """
    w = np.asarray(precision @ X_bar.T, dtype=np.float64)  # (n, q)
    H = X_bar @ w  # (q, q)
    L = cholesky(H, lower=True)  # H = L L^T
    V = solve_triangular(L, w.T, lower=True)  # (q, n);  V^T V = w H^{-1} w^T
    R_bar = precision.copy(order="F")
    return dsyrk(-(1 - lam), V, trans=1, beta=1.0, c=R_bar, overwrite_c=True)


def _sift_downdate_sparse(
    precision: csc_array,
    active_cols: NDArray[Any],
    X_bar_a: NDArray[Any],
    lam: float,
) -> csc_array:
    """SIFt forgetting step for sparse precision.

    The correction lives in the submatrix spanned by ``active_cols``
    and their neighbors in precision's sparsity graph. All linear
    algebra is done on that dense submatrix.

    Parameters
    ----------
    precision : (n, n) sparse precision matrix
    active_cols : column indices with nonzero entries in X_bar
    X_bar_a : (q, k0) dense active submatrix of X_bar
    lam : forgetting factor in (0, 1]
    """
    R_csc = csc_array(precision)

    # Slice columns first (cheap for CSC), then extract nz_rows and the
    # dense submatrix in one pass over the smaller (n, k0) slice.
    R_cols = R_csc[:, active_cols]  # (n, k0) CSC -- O(k0) column slice
    nz_rows = np.unique(R_cols.indices)

    # w = R[nz, active] @ X_bar_a.T  -- the only nonzero rows of
    # R @ X_bar.T, computed via dense matmul on the submatrix.
    R_nz_active = np.asarray(
        R_cols[nz_rows].toarray(),  # type: ignore[union-attr]
        dtype=np.float64,
    )  # (k, k0)
    w_sub = R_nz_active @ X_bar_a.T  # (k, q)

    # H = X_bar_a @ R[active, active] @ X_bar_a.T
    active_in_nz = np.searchsorted(nz_rows, active_cols)
    R_block = R_nz_active[active_in_nz]  # (k0, k0)
    H = X_bar_a @ R_block @ X_bar_a.T  # (q, q)

    L = cholesky(H, lower=True)
    V = solve_triangular(L, w_sub.T, lower=True)  # (q, k)
    correction_sub = V.T @ V  # (k, k) -- exactly symmetric

    # Pre-scale the correction and use addition to avoid an intermediate
    # sparse matrix from scalar multiplication.
    n = precision.shape[0]
    correction = _scatter_to_csc(-(1 - lam) * correction_sub, nz_rows, nz_rows, (n, n))
    return precision + correction


@dataclass(frozen=True)
class ExponentialForgetting:
    """Uniform scalar decay: ``R_bar = lam * R``.

    Equivalent to the predict step of a Kalman filter with random-walk
    process noise ``Q = (1 - lam) * Sigma``.  All eigenvalues of the
    precision are scaled equally.  Risk: covariance windup when
    excitation is non-uniform (unexcited eigenvalues → 0).
    """

    def __call__(
        self,
        precision: ArrayType,
        X: ArrayType,
        y: NDArray[Any],
        lam: float,
    ) -> ForgettingResult:
        return lam * precision, np.asarray(X), y


@dataclass(frozen=True)
class StabilizedForgetting:
    """Kulhavy-Zarrop stabilized forgetting [1]_.

    ``R_bar = lam * R + (1 - lam) * alpha * I``

    The prior floor ``(1 - lam) * alpha * I`` prevents precision from
    collapsing to zero under sustained forgetting.  The prior scalar
    converges to ``alpha`` under repeated application regardless of
    starting value.  Still isotropic: all directions decay equally.

    References
    ----------
    .. [1] Kulhavy, R. & Zarrop, M. B. (1993). "On a general concept of
       forgetting." *Int. J. Control*, 58(4), 905--924.
    """

    alpha: float

    def __call__(
        self,
        precision: ArrayType,
        X: ArrayType,
        y: NDArray[Any],
        lam: float,
    ) -> ForgettingResult:
        n = precision.shape[0]
        if sparse.issparse(precision):
            floor = (1 - lam) * self.alpha * sparse.eye(n, format="csc")
        else:
            floor = (1 - lam) * self.alpha * np.eye(n)
        return precision * lam + floor, np.asarray(X), y


@dataclass(frozen=True)
class SiftForgetting:
    """Directional forgetting via SIFt-RLS [1]_ [2]_.

    ``R_bar = R - (1 - lam) * R @ X_bar.T @ inv(X_bar @ R @ X_bar.T) @ X_bar @ R``

    Decomposes precision relative to the information subspace of the
    current batch and forgets only in excited directions.  Unexcited
    directions retain full precision.

    Key properties (from [2]_):

    - **Precision retention**: ``R_bar >= lam * R`` (Loewner order).
      Always retains at least as much precision as exponential forgetting.
    - **Eigenvalue floor**: ``lambda_min(R_k) >= min(eps / (1 - lam),
      lambda_min(R_0))`` after arbitrarily many forget-update cycles.
      No artificial prior injection needed.

    Parameters
    ----------
    eps : float
        Eigenvalue threshold for :func:`filter_batch`.  Eigenvalues of
        the batch Gram below this value are discarded.

    References
    ----------
    .. [1] Cao, L. & Schwartz, H. M. (2000). "A directional forgetting
       algorithm based on the decomposition of the information matrix."
       *Automatica*, 36(11), 1725--1731.
    .. [2] Lai, B. & Bernstein, D. S. (2024). "SIFt-RLS: Subspace of
       Information Forgetting Recursive Least Squares."
       *arXiv:2404.10844*.
    """

    eps: float

    def __call__(
        self,
        precision: ArrayType,
        X: ArrayType,
        y: NDArray[Any],
        lam: float,
    ) -> ForgettingResult | None:
        if sparse.issparse(X):
            result = _filter_batch_sparse(X, y, self.eps)
            if result is None:
                return None
            R_bar = _sift_downdate_sparse(
                precision, result.active_cols, result.X_bar_a, lam
            )
            return R_bar, result.X_bar, result.y_bar

        filtered = _filter_batch_dense(X, y, self.eps)
        if filtered is None:
            return None
        X_bar, y_bar = filtered

        if sparse.issparse(precision):
            # Dense X produces fully-dense X_bar, so the sparse downdate
            # would extract the entire precision matrix as dense anyway.
            # Go straight to the dense path.
            R_bar = _sift_downdate_dense(
                np.asarray(precision.toarray(), dtype=np.float64), X_bar, lam
            )
        else:
            R_bar = _sift_downdate_dense(precision, X_bar, lam)
        return R_bar, X_bar, y_bar
