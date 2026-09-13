"""Forgetting rules for precision-based Bayesian recursive least squares.

Four strategies for computing a "forgotten" precision matrix from the
current precision:

- :class:`ExponentialForgetting` -- scalar decay, simplest but subject to
  covariance windup under non-uniform excitation.
- :class:`StabilizedForgetting` -- Kulhavy & Zarrop (1993) prior floor
  prevents collapse, but forgetting is isotropic.
- :class:`SiftForgetting` -- directional forgetting via SIFt-RLS
  (Lai & Bernstein 2024), forgets only in excited directions; the
  correction is dense on the neighbourhood of the excited features.
- :class:`FeatureWiseForgetting` -- vector-type forgetting (Saelid & Foss
  1983) with the factors chosen from the batch support; forgets only the
  observed features and preserves the sparsity pattern.

See ``docs/math/forgetting.rst`` for the full mathematical reference.

Each rule is a frozen dataclass carrying its own forgetting factor
``rate``. Two events can forget:

- A **tick** of the clock, with no batch: ``rule.tick(precision, alpha=...,
  steps=n)`` returns the precision after ``n`` steps. Only the uniform
  rules (:class:`ExponentialForgetting`, :class:`StabilizedForgetting`)
  implement it; this is what an estimator's ``decay`` applies.
- An **update** on a batch: ``rule.update(precision, X, y, alpha=...)``
  returns ``(R_bar, X_eff, y_eff)`` or ``None``, where ``R_bar`` is the
  forgotten precision, ``(X_eff, y_eff)`` is the effective batch for the
  RLS update, and ``None`` means "skip this batch." Every rule implements
  it; the directional rules forget only along what the batch excites.

The caller then does::

    R_new = R_bar + X_eff.T @ X_eff
    eta = R_bar @ theta + X_eff.T @ y_eff
    theta_new = solve(R_new, eta)
"""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Protocol, Union, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cholesky, eigh, lapack, solve_triangular
from scipy.sparse import csc_array

from bayesianbandits import _blas_helpers as blas
from bayesianbandits._blas_helpers import dsyrk, fortran_view

ArrayType = Union[NDArray[Any], csc_array]

ForgettingResult = tuple[ArrayType, NDArray[Any], NDArray[Any]]


@runtime_checkable
class TickRule(Protocol):
    """A forgetting rule that can be applied without a batch: the clock ticked."""

    @property
    def rate(self) -> float: ...

    def tick(
        self, precision: ArrayType, *, alpha: Optional[float], steps: float = 1
    ) -> ArrayType: ...


@runtime_checkable
class UpdateRule(Protocol):
    """A forgetting rule applied to a batch before it is absorbed."""

    @property
    def rate(self) -> float: ...

    def update(
        self,
        precision: ArrayType,
        X: ArrayType,
        y: NDArray[Any],
        *,
        alpha: Optional[float],
    ) -> ForgettingResult | None: ...


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

    Only the upper triangle of ``precision`` is read (``dsymm``) and only
    the upper triangle of the result is written (``dsyrk``), the same
    convention as the dense fit paths, so no symmetrizing copy is needed.
    """
    RF, transposed = fortran_view(np.asarray(precision, dtype=np.float64))
    # RF's lower triangle is precision's upper when RF is the transpose.
    w = blas.dsymm(1.0, RF, np.asfortranarray(X_bar.T), lower=transposed)  # (n, q)
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
    """Uniform scalar decay: ``R_bar = rate * R``.

    Equivalent to the predict step of a Kalman filter with random-walk
    process noise ``Q = (1 - rate) * Sigma``.  All eigenvalues of the
    precision are scaled equally, the prior included.  Risk: covariance
    windup when excitation is non-uniform (unexcited eigenvalues → 0).

    Parameters
    ----------
    rate : float
        Forgetting factor in (0, 1], applied once per tick or per batch.
    """

    rate: float

    def tick(
        self, precision: ArrayType, *, alpha: Optional[float], steps: float = 1
    ) -> ArrayType:
        return (self.rate**steps) * precision

    def update(
        self,
        precision: ArrayType,
        X: ArrayType,
        y: NDArray[Any],
        *,
        alpha: Optional[float],
    ) -> ForgettingResult:
        return self.rate * precision, np.asarray(X), y


@dataclass(frozen=True)
class StabilizedForgetting:
    """Kulhavy-Zarrop stabilized forgetting.

    ``R_bar = rate * R + (1 - rate) * alpha * I``, from [1]_.

    The prior floor ``(1 - rate) * alpha * I`` prevents precision from
    collapsing to zero under sustained forgetting.  The prior scalar
    converges to ``alpha`` under repeated application regardless of
    starting value.  Still isotropic: all directions decay equally.

    Parameters
    ----------
    rate : float
        Forgetting factor in (0, 1], applied once per tick or per batch.
    alpha : float, optional
        The prior precision to floor at. ``None`` (the default) means the
        estimator's own ``alpha``.

    References
    ----------
    .. [1] Kulhavy, R. & Zarrop, M. B. (1993). "On a general concept of
       forgetting." *Int. J. Control*, 58(4), 905--924.
    """

    rate: float
    alpha: Optional[float] = None

    def floor(self, alpha: Optional[float]) -> float:
        """The prior precision this rule floors at, given the estimator's;
        ``None`` means the estimator has no scalar prior precision."""
        if self.alpha is not None:
            return self.alpha
        if alpha is None:
            raise TypeError(
                "StabilizedForgetting needs a scalar prior precision to floor "
                "at, and this estimator's prior is not a scalar; pass "
                "StabilizedForgetting(rate, alpha=...)."
            )
        return alpha

    def _apply(self, precision: ArrayType, lam: float, alpha: float) -> ArrayType:
        shift = (1 - lam) * alpha
        if sparse.issparse(precision):
            n = precision.shape[0]
            return precision * lam + shift * sparse.eye(n, format="csc")
        # Scale into a new array of the same layout and shift its diagonal
        # in place: no identity matrix, and a Fortran input stays Fortran,
        # which the dense fit paths rely on to read the right triangle.
        R = np.asarray(precision) * lam
        R[np.diag_indices_from(R)] += shift
        return R

    def tick(
        self, precision: ArrayType, *, alpha: Optional[float], steps: float = 1
    ) -> ArrayType:
        return self._apply(precision, self.rate**steps, self.floor(alpha))

    def update(
        self,
        precision: ArrayType,
        X: ArrayType,
        y: NDArray[Any],
        *,
        alpha: Optional[float],
    ) -> ForgettingResult:
        return self._apply(precision, self.rate, self.floor(alpha)), np.asarray(X), y


@dataclass(frozen=True)
class SiftForgetting:
    """Directional forgetting via SIFt-RLS.

    ``R_bar = R - (1 - rate) * R @ X_bar.T @ inv(X_bar @ R @ X_bar.T) @ X_bar @ R``,
    from [1]_ [2]_.

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
    rate : float
        Forgetting factor in (0, 1], applied once per batch along the
        directions the batch excites.
    eps : float, default=1e-10
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

    rate: float
    eps: float = 1e-10

    def update(
        self,
        precision: ArrayType,
        X: ArrayType,
        y: NDArray[Any],
        *,
        alpha: Optional[float],
    ) -> ForgettingResult | None:
        lam = self.rate
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


def _active_counts(X: ArrayType) -> NDArray[np.intp]:
    """Number of rows of ``X`` in which each feature is nonzero, shape ``(n,)``."""
    if sparse.issparse(X):
        X_csc = csc_array(X)
        assert X_csc.shape is not None
        n = X_csc.shape[1]
        col = np.repeat(np.arange(n), np.diff(X_csc.indptr))
        return np.bincount(col[X_csc.data != 0], minlength=n).astype(np.intp)
    return np.count_nonzero(np.asarray(X), axis=0).astype(np.intp)


@dataclass(frozen=True)
class FeatureWiseForgetting:
    """Vector-type forgetting with the factors chosen from the batch support.

    ``R_bar = D R D`` with ``D = diag(rate ** (m_i / 2))``, where ``m_i``
    is the number of rows of ``X`` in which feature ``i`` is nonzero.  A
    feature present in every row of an ``n``-row batch decays by
    ``rate ** n``, exactly as under :class:`ExponentialForgetting`; a
    feature absent from the batch is untouched.

    In covariance form this inflates the standard deviation of each
    observed coefficient by ``rate ** (-m_i / 2)`` and leaves every
    correlation and every unobserved coefficient's marginal unchanged.
    Because it only scales rows and columns, the sparsity pattern of
    ``R`` is preserved and the cost is one pass over its nonzeros.

    Per-parameter factors of this form are the vector variable
    forgetting factor of [1]_ [2]_ and the selective forgetting of [3]_;
    the ``D R D`` form is equation (12) of [4]_ (and of [5]_, which calls
    it ad hoc).  Choosing the factors from the batch support is what
    makes the rule directional for sparse designs.  It is a coordinate
    stretch rather than a Bayesian update: ``R - R_bar`` need not be
    positive semidefinite [6]_, so a combination of an observed feature
    with an unobserved one it is correlated with can come out tighter
    than before, by an amount that grows with that correlation.  For
    dense or strongly correlated regressors prefer :class:`SiftForgetting`.

    Parameters
    ----------
    rate : float
        Forgetting factor in (0, 1], applied once per row a feature appears
        in.

    References
    ----------
    .. [1] Saelid, S. & Foss, B. (1983). "Adaptive controllers with a
       vector variable forgetting factor." *Proc. 22nd IEEE CDC*, 1488--1494.
    .. [2] Saelid, S., Egeland, O. & Foss, B. (1985). "A solution to the
       blow-up problem in adaptive controllers." *Modeling, Identification
       and Control*, 6(1), 39--56.
    .. [3] Parkum, J. E., Poulsen, N. K. & Holst, J. (1992). "Recursive
       forgetting algorithms." *Int. J. Control*, 55(1), 109--128.
    .. [4] Fraccaroli, F., Peruffo, A. & Zorzi, M. (2015). "A new recursive
       least-squares method with multiple forgetting schemes."
       *arXiv:1503.07338*.
    .. [5] Vahidi, A., Stefanopoulou, A. & Peng, H. (2005). "Recursive least
       squares with forgetting for online estimation of vehicle mass and
       road grade: theory and experiments." *Vehicle System Dynamics*,
       43(1), 31--55.
    .. [6] Lai, B. & Bernstein, D. S. (2024). "Generalized forgetting
       recursive least squares: stability and robustness guarantees."
       *IEEE Trans. Automatic Control*. *arXiv:2308.04259*.
    """

    rate: float

    def update(
        self,
        precision: ArrayType,
        X: ArrayType,
        y: NDArray[Any],
        *,
        alpha: Optional[float],
    ) -> ForgettingResult:
        d = self.rate ** (_active_counts(X) / 2.0)
        if sparse.issparse(precision):
            R = csc_array(precision)
            assert R.shape is not None
            # D R D on the stored entries only: rows by d[i], columns by d[j].
            data = R.data * d[R.indices]
            data *= np.repeat(d, np.diff(R.indptr))
            R_bar: ArrayType = csc_array(
                (data, R.indices.copy(), R.indptr.copy()), shape=R.shape
            )
        else:
            R_bar = (d[:, np.newaxis] * np.asarray(precision)) * d[np.newaxis, :]
        # The batch is passed through untouched; a sparse X stays sparse, as
        # in the sparse SIFt path.
        X_eff = cast(NDArray[Any], X) if sparse.issparse(X) else np.asarray(X)
        return R_bar, X_eff, y


UniformRule = Union[ExponentialForgetting, StabilizedForgetting]
"""The rules that forget every direction alike: the only ones the grouped
conjugate models and the empirical Bayes estimators accept."""


def resolve_tick(
    forgetting: Any,
    *,
    steps: float,
    decay_rate: Optional[float],
    default: type = ExponentialForgetting,
    stacklevel: int = 3,
) -> tuple[TickRule, float, Any]:
    """Sort out the arguments of an estimator's ``decay``.

    Returns ``(rule, steps, legacy_X)``. ``rule`` is the tick rule to
    apply: ``forgetting`` itself, or ``default`` built from ``decay_rate``.
    A context array passed where the rule goes is the pre-rule calling
    convention; it is returned as ``legacy_X`` with ``steps`` set to its
    row count, so estimators that read the rows (the grouped conjugate
    models) can keep doing so.
    """
    legacy_X = None
    if isinstance(forgetting, numbers.Real) and not isinstance(forgetting, bool):
        raise TypeError(
            "decay() takes a forgetting rule, not a bare rate; pass "
            "decay_rate=... or a rule such as ExponentialForgetting(rate)."
        )
    if forgetting is not None and not isinstance(forgetting, TickRule):
        if isinstance(forgetting, UpdateRule):
            raise TypeError(
                f"{type(forgetting).__name__} forgets along a batch, so it "
                "belongs on the learner's forgetting= argument, not decay()."
            )
        warnings.warn(
            "Passing a context array to decay() is deprecated; pass "
            "steps=<number of ticks> and a forgetting rule or decay_rate.",
            FutureWarning,
            stacklevel=stacklevel,
        )
        legacy_X = forgetting
        forgetting = None
        steps = legacy_X.shape[0] if hasattr(legacy_X, "shape") else len(legacy_X)
    if forgetting is None:
        if decay_rate is None:
            raise TypeError(
                "decay() needs a forgetting rule such as "
                "StabilizedForgetting(0.95), or decay_rate=."
            )
        forgetting = default(decay_rate)
    elif decay_rate is not None:
        raise TypeError(
            "Pass either a forgetting rule or decay_rate, not both; the rule "
            "carries its own rate."
        )
    return forgetting, steps, legacy_X


def tick_groups(
    table: dict[Any, NDArray[np.float64]],
    forgetting: Any,
    *,
    steps: float,
    decay_rate: Optional[float],
    prior: NDArray[np.float64],
    default: type = ExponentialForgetting,
) -> None:
    """``decay`` for a grouped conjugate model: one parameter vector per
    group in ``table``, all sharing ``prior``. Scales every group by
    ``rate ** steps``, mixing ``prior`` back in under
    :class:`StabilizedForgetting`. A legacy context array ticks once per
    row, on that row's group."""
    rule, steps, legacy_X = resolve_tick(
        forgetting,
        steps=steps,
        decay_rate=decay_rate,
        default=default,
        stacklevel=4,
    )

    def tick(value: NDArray[np.float64], n: float) -> NDArray[np.float64]:
        gamma = rule.rate**n
        if isinstance(rule, StabilizedForgetting):
            floor = prior if rule.alpha is None else rule.alpha
            return np.asarray(gamma * value + (1 - gamma) * floor, dtype=np.float64)
        return np.asarray(gamma * value, dtype=np.float64)

    if legacy_X is not None:
        for x in legacy_X:
            table[x.item()] = tick(table[x.item()], 1)
        return
    for key in list(table):
        table[key] = tick(table[key], steps)


def uniform_batch(
    rule: Optional[Any], n: int, *, alpha: Optional[float]
) -> tuple[float, float]:
    """``(gamma, floor)`` for a uniform rule over an ``n``-row batch: each
    row is one step, so the prior is scaled by ``gamma = rate ** n`` and
    ``(1 - gamma) * floor`` is added back to its diagonal. ``(1.0, 0.0)``
    with no rule."""
    if rule is None:
        return 1.0, 0.0
    gamma = rule.rate**n
    if isinstance(rule, StabilizedForgetting):
        return gamma, rule.floor(alpha)
    return gamma, 0.0


def check_update_rule(
    rule: Any, *, estimator: str, uniform_only: bool = False, sparse: bool = False
) -> None:
    """Raise unless ``rule`` can be the ``forgetting`` of ``estimator``."""
    if rule is None:
        return
    if uniform_only:
        if not isinstance(rule, (ExponentialForgetting, StabilizedForgetting)):
            raise TypeError(
                f"{estimator} forgets uniformly: forgetting= takes "
                "ExponentialForgetting or StabilizedForgetting, not "
                f"{type(rule).__name__}."
            )
        return
    if not isinstance(rule, UpdateRule):
        raise TypeError(
            f"forgetting= takes a forgetting rule such as "
            f"ExponentialForgetting(0.99), not {rule!r}."
        )
    if sparse and isinstance(rule, SiftForgetting):
        raise TypeError(
            "SiftForgetting fills in a sparse precision matrix; use "
            "FeatureWiseForgetting on a sparse estimator."
        )
