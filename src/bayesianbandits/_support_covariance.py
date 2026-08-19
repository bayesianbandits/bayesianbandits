"""Support-covariance reduction for sparse posterior predictive draws.

A sparse design matrix ``X`` touches only a small set of columns ``U``.
The joint predictive law of ``X w`` for ``w ~ N(ŵ, Λ⁻¹)`` depends on
``Λ⁻¹`` only through the ``|U| x |U|`` principal submatrix
``S = (Λ⁻¹)_{U,U}``:

.. math::

    \\operatorname{Cov}(X w) = X \\Lambda^{-1} X^T = X_U S X_U^T

This is an identity, not an approximation: ``X = X_U E_U^T`` exactly,
where ``E_U`` selects the columns in ``U``.

``S`` costs ``|U|`` solves against the cached factor -- the stock
multi-RHS solve both CHOLMOD and SuperLU provide -- after which every
draw and every per-row standard deviation is dense linear algebra on an
``|U| x |U|`` matrix, with no further reference to the factor. The cost
is therefore ``O(|U| · nnz(L))`` with no dependence on the elimination
tree, unlike approaches that walk the factor's sparsity structure.

Because ``S`` is a principal submatrix of the symmetric positive
definite ``Λ⁻¹``, it is itself SPD for *any* ``X``, so its Cholesky
always exists. Reducing on the feature side rather than the row side is
what buys that: the ``n x n`` predictive covariance ``X_U S X_U^T`` is
singular whenever two rows of ``X`` coincide, while ``S`` never is.
Duplicate rows come out perfectly correlated within a draw, which is
what a shared weight vector implies.
"""

from typing import Any, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, issparse  # type: ignore

from ._sparse_bayesian_linear_regression import PrecisionFactor

_SCRATCH_ELEMS = 2**23
"""Cap on a single dense scratch buffer, in float64 elements (~64 MB).

Sizes both the ``(n_features, k)`` right-hand-side block used to build
``S`` and the ``(rows, |U|)`` densified design block. Unlike the
blocking in ``_marginal_predictive_sd``, changing this trades memory for
nothing: the blocks are independent GEMMs against the same small
factor, so neither the solve count nor the flop count moves.
"""


def support_of(X: csc_array) -> NDArray[np.intp]:
    """Column indices ``X`` touches, ascending.

    For CSC these are exactly the columns whose ``indptr`` range is
    non-empty, so no scan of the row indices is needed. Comparing
    adjacent entries rather than differencing them keeps the temporary a
    boolean array instead of an index-width one, which at large feature
    counts is several times cheaper.
    """
    indptr = X.indptr
    return cast(NDArray[np.intp], np.flatnonzero(indptr[1:] != indptr[:-1]))


def _compact_columns(X: csc_array, support: NDArray[np.intp]) -> csc_array:
    """``X[:, support]`` for a ``support`` that lists every non-empty column.

    Empty columns never advance ``indptr``, so dropping them is a
    re-slice of ``indptr`` alone -- ``data`` and ``indices`` are shared
    with ``X``, not copied.
    """
    n_rows = cast("tuple[int, int]", X.shape)[0]
    indptr = np.append(X.indptr[support], X.indptr[-1])
    return csc_array(
        (X.data, X.indices, indptr), shape=(n_rows, support.size), copy=False
    )


def support_covariance(
    factor: PrecisionFactor, support: NDArray[np.intp], n_features: int
) -> NDArray[np.float64]:
    """``S = (Λ⁻¹)_{U,U}`` via ``|U|`` solves against ``factor``.

    Solves ``Λ Y = E_U`` in column blocks and keeps the ``U`` rows of
    each result. Only the small ``|U| x |U|`` block is retained, so the
    ``(n_features, block)`` scratch is the peak allocation.
    """
    n_u = support.size
    block = int(np.clip(_SCRATCH_ELEMS // max(1, n_features), 1, n_u))
    S = np.empty((n_u, n_u), dtype=np.float64)
    rhs = np.zeros((n_features, block), dtype=np.float64)
    for start in range(0, n_u, block):
        stop = min(n_u, start + block)
        width = stop - start
        cols = np.arange(width)
        rhs[support[start:stop], cols] = 1.0
        # Every factor preserves a 2-D right-hand side, but reshaping
        # rather than trusting that keeps a single-column block safe.
        out = np.asarray(factor.solve(rhs[:, :width]), dtype=np.float64).reshape(
            n_features, width
        )
        S[:, start:stop] = out[support, :]
        rhs[support[start:stop], cols] = 0.0
    # Λ⁻¹ is symmetric; the solves are not bitwise so average the halves.
    return cast(NDArray[np.float64], 0.5 * (S + S.T))


def _factorize(S: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lower-triangular ``C`` with ``C Cᵀ = S``.

    ``S`` is SPD in exact arithmetic; the eigenvalue branch is a guard
    against a numerically indefinite result for a severely
    ill-conditioned precision, not an expected path.
    """
    try:
        return cast(NDArray[np.float64], np.linalg.cholesky(S))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(S)
        return cast(NDArray[np.float64], V * np.sqrt(np.clip(w, 0.0, None)))


class SupportDraw:
    """Exact joint predictive law of ``X w``, reduced to ``X``'s support."""

    __slots__ = ("_C", "_XU", "_n_rows")

    def __init__(self, C: NDArray[np.float64], XU: csc_array) -> None:
        self._C = C
        # CSR so row blocks densify contiguously; |U| is small so this
        # conversion is cheap and happens once.
        self._XU = XU.tocsr()
        self._n_rows = cast("tuple[int, int]", XU.shape)[0]

    def _row_block(self) -> int:
        return max(1, _SCRATCH_ELEMS // max(1, self._C.shape[0]))

    def joint(self, size: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """``size`` jointly-distributed draws, shape ``(size, n_rows)``.

        Rows within one draw share a weight vector, matching ``sample``.
        Projected transposed so the result is draw-contiguous (see
        :func:`~bayesianbandits._blas_helpers.project_draws`).
        """
        ZC = rng.standard_normal((size, self._C.shape[0])) @ self._C.T
        out = np.empty((self._n_rows, size), dtype=np.float64)
        block = self._row_block()
        for start in range(0, self._n_rows, block):
            stop = min(self._n_rows, start + block)
            out[start:stop] = (
                np.asarray(self._XU[start:stop].todense(), dtype=np.float64) @ ZC.T
            )
        return out.T

    def sd(self) -> NDArray[np.float64]:
        """Per-row predictive standard deviations, shape ``(n_rows,)``."""
        out = np.empty(self._n_rows, dtype=np.float64)
        block = self._row_block()
        for start in range(0, self._n_rows, block):
            stop = min(self._n_rows, start + block)
            T = np.asarray(self._XU[start:stop].todense(), dtype=np.float64) @ self._C
            out[start:stop] = np.sqrt(np.einsum("ij,ij->i", T, T))
        return out


def build(
    factor: PrecisionFactor,
    X: Union[NDArray[Any], csc_array],
    n_features: int,
    budget: int,
) -> Optional[SupportDraw]:
    """Build a :class:`SupportDraw`, or ``None`` to keep the caller's path.

    ``budget`` is the number of triangular solves the caller's existing
    path would perform: ``size`` for a joint draw (one solve per draw)
    and ``n_rows`` for per-row standard deviations (one half-solve per
    row). This route costs ``|U|`` solves, so it is taken only when
    ``|U| < budget`` -- a comparison of two exactly known integers, with
    nothing calibrated.

    Returns ``None`` for dense ``X`` (where ``|U| = n_features``, so the
    reduction buys nothing) and for an all-zero ``X``.
    """
    if not issparse(X):
        return None
    # ``|U| >= 1`` for any non-empty X, so a budget below 2 can never be
    # beaten. Checking that first keeps the Thompson-sampling hot path
    # (``size=1``) clear of the O(n_features) support scan below, which
    # is otherwise a measurable cost at large feature counts.
    if budget < 2:
        return None
    Xc = cast(csc_array, X)
    n_rows = cast("tuple[int, int]", Xc.shape)[0]
    nnz = int(Xc.indptr[-1])
    if nnz == 0:
        return None
    # A column holds at most ``n_rows`` entries, so ``|U| >= nnz / n_rows``.
    # When that bound alone already meets the budget the answer is known
    # without scanning: decline in O(1). This is exact whenever the rows
    # share a common support, which is the shape that reaches here most
    # often (one design row per arm).
    if -(-nnz // max(1, n_rows)) >= budget:
        return None
    support = support_of(Xc)
    if support.size >= budget:
        return None
    S = support_covariance(factor, support, n_features)
    return SupportDraw(_factorize(S), _compact_columns(Xc, support))
