"""Support-covariance reduction for sparse posterior predictive draws.

A sparse design matrix ``X`` touches only a small set of columns ``U``,
and ``X = X_U E_Uᵀ`` exactly, so the joint law of ``X w`` for
``w ~ N(ŵ, Λ⁻¹)`` depends on ``Λ⁻¹`` only through the principal
submatrix ``S = (Λ⁻¹)_{U,U}``:

.. math::

    \\operatorname{Cov}(X w) = X \\Lambda^{-1} X^T = X_U S X_U^T

``S`` costs ``|U|`` solves against the cached factor; every draw and
per-row standard deviation after that is dense linear algebra on an
``|U| x |U|`` matrix. ``S`` is SPD for any ``X`` (unlike the ``n x n``
predictive covariance, singular whenever rows repeat), so its Cholesky
always exists.
"""

from typing import Any, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, issparse  # type: ignore

from ._sparse_bayesian_linear_regression import PrecisionFactor

_SCRATCH_ELEMS = 2**23
"""Cap on a dense scratch buffer, in float64 elements (~64 MB): the
``(n_features, k)`` right-hand-side block that builds ``S`` and the
``(rows, |U|)`` densified design block."""


def support_of(X: csc_array) -> NDArray[np.intp]:
    """Column indices ``X`` touches, ascending: the columns whose CSC
    ``indptr`` range is non-empty, with no scan of ``indices``."""
    indptr = X.indptr
    return cast(NDArray[np.intp], np.flatnonzero(indptr[1:] != indptr[:-1]))


def _compact_columns(X: csc_array, support: NDArray[np.intp]) -> csc_array:
    """``X[:, support]`` when ``support`` lists every non-empty column:
    a re-slice of ``indptr`` alone, sharing ``data`` and ``indices``."""
    n_rows = cast("tuple[int, int]", X.shape)[0]
    indptr = np.append(X.indptr[support], X.indptr[-1])
    return csc_array(
        (X.data, X.indices, indptr), shape=(n_rows, support.size), copy=False
    )


def support_covariance(
    factor: PrecisionFactor, support: NDArray[np.intp], n_features: int
) -> NDArray[np.float64]:
    """``S = (Λ⁻¹)_{U,U}``: solve ``Λ Y = E_U`` in column blocks and keep
    the ``U`` rows of each result."""
    n_u = support.size
    block = int(np.clip(_SCRATCH_ELEMS // max(1, n_features), 1, n_u))
    S = np.empty((n_u, n_u), dtype=np.float64)
    rhs = np.zeros((n_features, block), dtype=np.float64)
    for start in range(0, n_u, block):
        stop = min(n_u, start + block)
        width = stop - start
        cols = np.arange(width)
        rhs[support[start:stop], cols] = 1.0
        # reshape rather than trust every backend to keep a 2-D RHS 2-D
        out = np.asarray(factor.solve(rhs[:, :width]), dtype=np.float64).reshape(
            n_features, width
        )
        S[:, start:stop] = out[support, :]
        rhs[support[start:stop], cols] = 0.0
    # Λ⁻¹ is symmetric; the solves are not bitwise so average the halves.
    return cast(NDArray[np.float64], 0.5 * (S + S.T))


def _factorize(S: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lower-triangular ``C`` with ``C Cᵀ = S``; the eigenvalue branch
    guards against a numerically indefinite ``S``."""
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
        # CSR so row blocks densify contiguously
        self._XU = XU.tocsr()
        self._n_rows = cast("tuple[int, int]", XU.shape)[0]

    def _row_block(self) -> int:
        return max(1, _SCRATCH_ELEMS // max(1, self._C.shape[0]))

    def joint(self, size: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """``size`` jointly-distributed draws, shape ``(size, n_rows)``.

        Rows within one draw share a weight vector, matching ``sample``.
        """
        ZC = rng.standard_normal((size, self._C.shape[0])) @ self._C.T
        out = np.empty((size, self._n_rows), dtype=np.float64)
        block = self._row_block()
        for start in range(0, self._n_rows, block):
            stop = min(self._n_rows, start + block)
            out[:, start:stop] = (
                ZC @ np.asarray(self._XU[start:stop].todense(), dtype=np.float64).T
            )
        return out

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

    ``budget`` is the number of solves the caller's path would perform
    (``size`` for a joint draw, ``n_rows`` for per-row standard
    deviations); this route costs ``|U|``, so it is taken only when
    ``|U| < budget``. Returns ``None`` for dense or all-zero ``X``.
    """
    if not issparse(X):
        return None
    # |U| >= 1, so budget < 2 can never be beaten; this keeps the
    # Thompson hot path (size=1) clear of the O(n_features) scan below.
    if budget < 2:
        return None
    Xc = cast(csc_array, X)
    n_rows = cast("tuple[int, int]", Xc.shape)[0]
    nnz = int(Xc.indptr[-1])
    if nnz == 0:
        return None
    # A column holds at most n_rows entries, so |U| >= nnz / n_rows:
    # decline in O(1) when that bound alone meets the budget.
    if -(-nnz // max(1, n_rows)) >= budget:
        return None
    support = support_of(Xc)
    if support.size >= budget:
        return None
    S = support_covariance(factor, support, n_features)
    return SupportDraw(_factorize(S), _compact_columns(Xc, support))
