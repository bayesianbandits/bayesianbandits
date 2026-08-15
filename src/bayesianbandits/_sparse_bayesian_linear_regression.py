import os
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve, solve_triangular  # type: ignore
from scipy.linalg import qr as scipy_qr  # type: ignore
from scipy.sparse import (  # type: ignore  # type: ignore
    csc_array,
    csc_matrix,
    csr_matrix,
    diags,  # type: ignore
    issparse,  # type: ignore
)
from scipy.sparse.linalg import (  # type: ignore
    splu,
    spsolve,
    spsolve_triangular,
    use_solver,
)
from scipy.stats._multivariate import _squeeze_output  # type: ignore

use_solver(useUmfpack=False)

try:
    from sksparse.cholmod import cho_factor as cholmod_cho_factor  # type: ignore

    use_cholmod = True
except ImportError:
    use_cholmod = False

if os.environ.get("BB_NO_SUITESPARSE", "0") == "1":
    use_cholmod = False


class SparseSolver(Enum):
    """Enum for sparse solvers."""

    SUPERLU = 0
    CHOLMOD = 1


if use_cholmod:
    solver = SparseSolver.CHOLMOD
else:
    solver = SparseSolver.SUPERLU

if TYPE_CHECKING:
    from sksparse.cholmod import cho_factor as cholmod_cho_factor  # type: ignore

    # This helps Pylance understand that solver can be either SparseSolver.SUPERLU or SparseSolver.CHOLMOD.
    # For some reason, without this cast, it thinks solver is always SparseSolver.SUPERLU.
    solver: Literal[SparseSolver.SUPERLU] | Literal[SparseSolver.CHOLMOD] = cast(
        Literal[SparseSolver.SUPERLU, SparseSolver.CHOLMOD], solver
    )  # type: ignore


@dataclass
class CholmodSparseFactor:
    """Wraps a CHOLMOD factor for solving and sampling."""

    _factor: Any  # sksparse.cholmod.Factor (C extension, no useful type)
    _precision: csc_array
    _inv_perm: NDArray[np.intp] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._inv_perm = np.argsort(self._factor.perm)

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return self._factor.solve(b)

    def colorize(self, z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Solve L^T x = z, undo permutation."""
        return self._factor.solve(z, system="Lt")[self._inv_perm]

    def half_solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Apply the transpose of the ``colorize`` operator: L^{-1} P b.

        If ``M`` is the colorize operator (``M M^T = Λ^{-1}``), this
        returns ``M^T b``, so ``half_solve(X.T).T @ half_solve(X.T)``
        equals ``X Λ^{-1} X^T`` -- a half-solve per column against the
        cached factor, without ever forming ``Λ^{-1}``.
        """
        return self._factor.solve(b[self._factor.perm], system="L")

    def logdet(self) -> float:
        """Log-determinant of the factored matrix via CHOLMOD."""
        return float(self._factor.logdet())

    def refactorize(self, precision: csc_array) -> "CholmodSparseFactor":
        """Numeric refactorization reusing the existing symbolic analysis."""
        self._factor.factorize(csc_matrix(precision))
        self._precision = precision
        return self

    def get_L_csc(self) -> csc_array:
        """Return the lower triangular Cholesky factor as CSC.

        The factor satisfies PΛP' = LL' where P is the fill-reducing
        permutation.  The returned L includes the permutation implicitly
        (rows/cols are in permuted order).
        """
        return csc_array(self._factor.L)

    @cached_property
    def _reach_solver(self) -> "_ReachHalfSolver":
        """Reach-limited half-solver; lazy so fit paths pay nothing."""
        return _ReachHalfSolver(self.get_L_csc(), self._inv_perm)


@dataclass
class SuperLUSparseFactor:
    """Wraps SuperLU decomposition for solving and sampling."""

    _L: csr_matrix
    _inv_perm: NDArray[np.intp]
    _precision: csc_array
    _Lt_csc: csc_array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._Lt_csc = csc_array(self._L.T)

    @cached_property
    def _perm(self) -> NDArray[np.intp]:
        """Inverse of ``_inv_perm``, computed lazily so that ``fit``/
        ``partial_fit`` (which never call ``half_solve``) pay no extra cost."""
        perm = np.empty_like(self._inv_perm)
        perm[self._inv_perm] = np.arange(self._inv_perm.size)
        return perm

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], spsolve(self._precision, b))

    def colorize(self, z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Solve L^T x = z, undo permutation."""
        return cast(
            NDArray[np.float64],
            spsolve_triangular(self._Lt_csc, z, lower=False)[self._inv_perm],
        )

    @cached_property
    def _half_solve_factor(self) -> tuple[csc_array, NDArray[np.float64]]:
        """Unit-diagonal ``L`` (CSC) and its inverse diagonal, cached
        because ``spsolve_triangular`` otherwise re-copies and re-scales
        the factor on every call (O(nnz) per solve). Lazy, like
        ``_perm``, so ``fit``/``partial_fit`` pay no extra cost."""
        invdiag = np.asarray(1.0 / self._L.diagonal(), dtype=np.float64)
        L_unit = csc_array(self._L @ diags(invdiag))
        return L_unit, invdiag

    def half_solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Apply the transpose of the ``colorize`` operator: L^{-1} P b.

        Same contract as :meth:`CholmodSparseFactor.half_solve`, against
        the cached triangular factor (no refactorization, unlike ``solve``).
        With ``L = L' D`` (``L'`` unit-diagonal), ``x = D^{-1} L'^{-1} b``;
        the cached ``L'`` lets ``spsolve_triangular`` skip its per-call
        copy and rescale of the factor (``b[perm]`` is a fresh array, so
        overwriting both operands is safe).
        """
        L_unit, invdiag = self._half_solve_factor
        y = spsolve_triangular(
            L_unit,
            b[self._perm],
            lower=True,
            unit_diagonal=True,
            overwrite_A=True,
            overwrite_b=True,
        )
        scale = invdiag if y.ndim == 1 else invdiag[:, None]
        return cast(NDArray[np.float64], y * scale)

    def logdet(self) -> float:
        """Log-determinant of the factored matrix.

        L already has D folded in (L = L_splu @ diag(sqrt(D))), so
        |P| = |L|^2  =>  log|P| = 2 * sum(log|diag(L)|).
        """
        return float(2.0 * np.sum(np.log(np.abs(self._L.diagonal()))))

    def refactorize(self, precision: csc_array) -> "SuperLUSparseFactor":
        """SuperLU has no symbolic reuse API; performs a full refactorization."""
        splu_ = splu(
            precision,
            diag_pivot_thresh=0,
            permc_spec="MMD_AT_PLUS_A",
            options=dict(SymmetricMode=True),
        )
        if (splu_.perm_r != splu_.perm_c).any():
            raise ValueError("Matrix must be symmetric")
        L = splu_.L.dot(diags(np.sqrt(splu_.U.diagonal())))
        return SuperLUSparseFactor(
            _L=L, _inv_perm=splu_.perm_r.copy(), _precision=precision
        )

    def get_L_csc(self) -> csc_array:
        """Return the lower triangular factor as CSC.

        The SuperLU factor already has D folded in: L = L_splu @ diag(√D).
        Rows are in permuted order (matching _Pr).
        """
        return csc_array(self._L)

    @cached_property
    def _reach_solver(self) -> "_ReachHalfSolver":
        """Reach-limited half-solver; lazy so fit paths pay nothing."""
        return _ReachHalfSolver(self.get_L_csc(), self._inv_perm)


class _ReachHalfSolver:
    """Reach-limited sparse triangular solves against a cached factor.

    Solving ``L y = b`` for a lower-triangular ``L`` and a *sparse*
    ``b`` only ever touches the nodes reachable from ``b``'s support in
    the graph of ``L`` (Gilbert-Peierls). When the precision factor is
    nearly diagonal -- the typical posterior of a high-dimensional
    sparse-feature model -- that reach is a few dozen nodes out of
    millions, so a half-solve for one prediction row costs
    O(reach) instead of O(n_features).

    Holds persistent O(n) scratch (a value workspace and a visit-stamp
    array) so per-solve allocation is O(reach).
    """

    def __init__(self, L: csc_array, landing: NDArray[np.intp]) -> None:
        L = csc_array(L)
        if not L.has_sorted_indices:
            L.sort_indices()
        n = cast(tuple[int, int], L.shape)[0]
        self._indptr = L.indptr
        self._indices = L.indices
        self._data = L.data
        self._n = n
        # original coordinate s enters the solve at row landing[s]
        self._landing = landing
        # Per-call budgets across all columns, sized to stay well
        # below one dense O(n) triangular solve -- past them the dense
        # path is faster and the caller falls back. Nodes bound the
        # per-node Python overhead; entries bound the vectorized work,
        # which hub columns can inflate far beyond the node count.
        self._call_budget = max(2048, n // 256)
        self._entry_budget = 2**19
        self._work = np.zeros(n, dtype=np.float64)
        self._stamp = np.zeros(n, dtype=np.int64)
        self._gen = 0
        # diagonal-first layout is what the solve loop assumes
        self._usable = bool(np.all(self._indices[self._indptr[:-1]] == np.arange(n)))

    def solve_support(
        self,
        nodes: NDArray[np.intp],
        vals: NDArray[np.float64],
        budget: int,
        entry_budget: int,
    ) -> Union[tuple[NDArray[np.intp], NDArray[np.float64], int], None]:
        """Solve ``L y = b`` for ``b`` given as (landing nodes, values).

        Returns ``(rows, values, entries_traversed)`` of the sparse
        result, or None when the reach exceeds ``budget`` nodes or
        ``entry_budget`` traversed entries (caller falls back to a
        dense solve). The neighbor scan is vectorized per node, so a
        hub column with thousands of entries costs one numpy
        operation, not one Python iteration per entry.
        """
        if not self._usable:
            return None
        indptr, indices, data = self._indptr, self._indices, self._data
        work, stamp = self._work, self._stamp
        self._gen += 1
        gen = self._gen
        reach: list[int] = []
        stack: list[int] = []
        for n in nodes:
            jn = int(n)
            if stamp[jn] != gen:
                stamp[jn] = gen
                reach.append(jn)
                stack.append(jn)
        entries = 0
        while stack:
            if len(reach) > budget or entries > entry_budget:
                return None
            j = stack.pop()
            nb = indices[indptr[j] + 1 : indptr[j + 1]]
            if nb.size:
                entries += nb.size
                fresh = nb[stamp[nb] != gen]
                if fresh.size:
                    stamp[fresh] = gen
                    fresh_list = fresh.tolist()
                    reach.extend(fresh_list)
                    stack.extend(fresh_list)
        # lower-triangular: ascending index order is a topological order
        reach_arr = np.sort(np.asarray(reach, dtype=np.intp))
        np.add.at(work, nodes, vals)
        for j in reach_arr:
            s, e = indptr[j], indptr[j + 1]
            xj = work[j] / data[s]
            work[j] = xj
            if xj != 0.0 and e > s + 1:
                work[indices[s + 1 : e]] -= data[s + 1 : e] * xj
        out = work[reach_arr].copy()
        work[reach_arr] = 0.0
        return reach_arr, out, entries


ConcreteFactor = CholmodSparseFactor | SuperLUSparseFactor


@dataclass
class ScaledSparseFactor:
    """A factor scaled by a scalar. Avoids refactorizing for scalar precision changes.

    If L L^T = P, then (s*P) has factor (√s * L).
    - solve(s*P, b) = (1/s) * solve(P, b)
    - colorize(s*P, z) = (1/√s) * colorize(P, z)
    - half_solve(s*P, b) = (1/√s) * half_solve(P, b)
    """

    _inner: ConcreteFactor
    _scale: float
    _precision: csc_array  # the inner's precision; only used for .shape

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self._inner.solve(b) / self._scale)

    def colorize(self, z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self._inner.colorize(z) / np.sqrt(self._scale))

    def half_solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return cast(
            NDArray[np.float64], self._inner.half_solve(b) / np.sqrt(self._scale)
        )

    def logdet(self) -> float:
        """Log-determinant: log|s*P| = p*log(s) + log|P|."""
        p = cast(tuple[int, int], self._precision.shape)[0]
        return float(p * np.log(self._scale) + self._inner.logdet())

    def refactorize(self, precision: csc_array) -> ConcreteFactor:
        """Unwrap and delegate to the inner factor."""
        return self._inner.refactorize(precision)

    def get_L_csc(self) -> csc_array:
        """Return the scaled lower triangular factor as CSC.

        If inner factor has LL' = P, then scaled factor has (√s·L)(√s·L)' = s·P.
        """
        L = self._inner.get_L_csc()
        return csc_array(L * np.sqrt(self._scale))


SparseFactor = CholmodSparseFactor | SuperLUSparseFactor | ScaledSparseFactor


@dataclass
class DenseFactor:
    """Wraps a LAPACK Cholesky factorization for solving and sampling.

    Stores the upper-triangular factor U where Λ = U^T U.
    ``colorize`` starts as a direct triangular solve (O(p²) per
    column) and only materializes U^{-1} (an O(p³) build) once the
    cumulative solved columns have cost as much as one build, so
    a factor that is colorized once or twice — the fit/sample loop
    of a bandit agent — never pays the O(p³) price.
    """

    _U: NDArray[np.float64]
    _n_features: int
    _colorize_cols: int = field(default=0, init=False, repr=False)

    @cached_property
    def _U_inv(self) -> NDArray[np.float64]:
        return solve_triangular(
            self._U, np.eye(self._n_features), lower=False, check_finite=False
        )

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return cho_solve((self._U, False), b, check_finite=False)

    def colorize(self, z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Compute U^{-1} z, producing samples from N(0, Λ^{-1}).

        Ski-rental amortization: solve directly until the cumulative
        colorized columns reach ``n_features`` (the cost of building
        U^{-1} once), then build and reuse the cached inverse. Total
        work is never more than twice the optimal strategy, and the
        one-shot case skips the O(p³) build entirely.
        """
        if "_U_inv" not in self.__dict__:
            self._colorize_cols += z.shape[1] if z.ndim == 2 else 1
            if self._colorize_cols < self._n_features:
                return cast(
                    NDArray[np.float64],
                    solve_triangular(self._U, z, lower=False, check_finite=False),
                )
        return cast(NDArray[np.float64], self._U_inv @ z)

    def half_solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Solve U^T x = b, the transpose of the ``colorize`` operator.

        Same contract as :meth:`CholmodSparseFactor.half_solve`; one
        triangular solve, without building ``U^{-1}``.
        """
        return cast(
            NDArray[np.float64],
            solve_triangular(self._U, b, trans=1, lower=False, check_finite=False),
        )

    def logdet(self) -> float:
        """log|Λ| = 2·sum(log(diag(U)))."""
        return 2.0 * float(np.sum(np.log(np.diag(self._U))))

    def trace_inv(self) -> float:
        """tr(Λ⁻¹) = ||U⁻¹||²_F."""
        return float(np.sum(self._U_inv**2))


PrecisionFactor = SparseFactor | DenseFactor


def scale_factor(factor: SparseFactor, scale: float) -> SparseFactor:
    """Scale a sparse factor by a scalar, composing rather than nesting."""
    if scale == 1.0:
        return factor
    if isinstance(factor, ScaledSparseFactor):
        return ScaledSparseFactor(
            _inner=factor._inner,
            _scale=factor._scale * scale,
            _precision=factor._precision,
        )
    return ScaledSparseFactor(_inner=factor, _scale=scale, _precision=factor._precision)


def create_sparse_factor(
    precision: csc_array, solver: Union[SparseSolver, None] = None
) -> SparseFactor:
    """Create a SparseFactor from a precision matrix."""
    if solver is None:
        solver = globals()["solver"]
    if not issparse(precision):
        raise TypeError("precision must be a sparse array")
    if solver == SparseSolver.CHOLMOD:
        return CholmodSparseFactor(
            _factor=cholmod_cho_factor(csc_matrix(precision)),
            _precision=precision,
        )
    else:
        splu_ = splu(
            precision,
            diag_pivot_thresh=0,
            permc_spec="MMD_AT_PLUS_A",
            options=dict(SymmetricMode=True),
        )
        if (splu_.perm_r != splu_.perm_c).any():
            raise ValueError("Matrix must be symmetric")
        L = splu_.L.dot(diags(np.sqrt(splu_.U.diagonal())))
        return SuperLUSparseFactor(
            _L=L, _inv_perm=splu_.perm_r.copy(), _precision=precision
        )


def multivariate_normal_sample_from_precision(
    mean: Union[csc_array, NDArray[np.float64], None],
    factor: PrecisionFactor,
    size: int = 1,
    random_state: Union[int, None, np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Sample from a multivariate normal distribution parameterized by a
    factored precision matrix.

    Works with both dense (``DenseFactor``) and sparse
    (``CholmodSparseFactor``, ``SuperLUSparseFactor``,
    ``ScaledSparseFactor``) precision factors via the common
    ``colorize`` interface.

    Parameters
    ----------
    mean : array_like or None
        Mean of the distribution. If None, the zero vector is used.
    factor : PrecisionFactor
        Factored precision matrix (dense or sparse).
    size : int, default=1
        Number of samples to draw.
    random_state : int, Generator, or None, default=None
        Random state for reproducibility.

    Returns
    -------
    out : ndarray of shape (size, n_features) or (n_features,) if size=1
        The drawn samples.
    """
    rng = np.random.default_rng(random_state)
    if isinstance(factor, DenseFactor):
        n_features = factor._n_features
    else:
        n_features = cast(tuple[int, int], factor._precision.shape)[0]
    _Z = rng.standard_normal((size, n_features))
    samples = factor.colorize(_Z.T).T
    if samples.shape[0] == 1:
        samples = samples[0]
    if mean is not None:
        samples += mean
    return samples


# Backwards-compatible alias
multivariate_normal_sample_from_sparse_precision = (
    multivariate_normal_sample_from_precision
)


# Cap on the dense (n_features, block) scratch used while building a
# predictive root: 2^24 float64 elements = 128 MB.
_PREDICTIVE_DRAWS_ELEMS = 2**24

# Largest candidate set for which a predictive root is built: the root
# is n_rows x n_rows and the Gram fallback pays an O(n_rows^3) eigh.
_PREDICTIVE_ROOT_MAX_ROWS = 4096


def sparse_predictive_operator(
    factor: PrecisionFactor,
    X: Any,
    entry_cols: Union[NDArray[np.intp], None] = None,
) -> Union[NDArray[np.float64], None]:
    """Compact predictive operator for a sparse prediction matrix.

    Returns a dense ``op`` of shape (k, n_rows) -- k the size of the
    union of the reach-limited half-solve supports -- satisfying
    ``op^T op = X Λ^{-1} X^T`` exactly, so ``z @ op`` with iid
    standard-normal ``z`` gives joint draws of the centered linear
    predictor. Built with one reach-limited triangular solve per row,
    touching O(reach) nodes instead of O(n_features): the fast path
    for a *fresh* sparse ``X`` against a warm posterior.

    Returns None when the factor has no reach solver, the reach is not
    small (nearly-dense factor -- the dense paths are better), or the
    candidate set is too large.
    """
    scale = 1.0
    if isinstance(factor, ScaledSparseFactor):
        scale = factor._scale
        factor = factor._inner
    if not isinstance(factor, (CholmodSparseFactor, SuperLUSparseFactor)):
        return None
    n_rows, n_features = X.shape
    if n_rows > _PREDICTIVE_ROOT_MAX_ROWS:
        return None
    solver = factor._reach_solver
    landing = solver._landing
    Xs = csc_array(X)
    nnz = Xs.data.shape[0]
    if nnz == 0:
        return np.zeros((0, n_rows), dtype=np.float64)
    # COO triplets without touching the O(n_features) indptr densely;
    # callers that already computed the entry columns pass them in so
    # the O(nnz log n_features) search over the wide indptr runs once
    # per decision.
    rows = Xs.indices
    if entry_cols is not None:
        cols = entry_cols
    else:
        cols = np.searchsorted(Xs.indptr, np.arange(nnz), side="right") - 1
    order = np.argsort(rows, kind="stable")
    rows_o, cols_o, vals_o = rows[order], cols[order], Xs.data[order]
    boundaries = np.flatnonzero(np.diff(rows_o)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [rows_o.shape[0]]))
    solved: list[tuple[int, NDArray[np.intp], NDArray[np.float64]]] = []
    budget = solver._call_budget
    entry_budget = solver._entry_budget
    for start, stop in zip(starts, stops):
        r = int(rows_o[start])
        nodes = landing[cols_o[start:stop]]
        result = solver.solve_support(
            nodes,
            np.asarray(vals_o[start:stop], dtype=np.float64),
            budget,
            entry_budget,
        )
        if result is None:
            return None
        budget -= result[0].shape[0]
        entry_budget -= result[2]
        if budget < 0 or entry_budget < 0:
            return None
        solved.append((r, result[0], result[1]))
    if solved:
        union = np.unique(np.concatenate([reach for _, reach, _ in solved]))
    else:
        union = np.empty(0, dtype=np.intp)
    op = np.zeros((union.shape[0], n_rows), dtype=np.float64)
    for r, reach, vals in solved:
        op[np.searchsorted(union, reach), r] = vals
    if scale != 1.0:
        op /= np.sqrt(scale)
    return op


def predictive_root(
    factor: PrecisionFactor,
    X: Any,
) -> Union[NDArray[np.float64], None]:
    """Square root of the predictive covariance of ``X @ w``.

    Returns ``R`` of shape (k, n_rows), k = min(n_rows, n_features),
    with ``R^T R = X Λ^{-1} X^T`` exactly: one half-solve against the
    cached factor gives ``B = M^T X^T`` (``M`` being the colorize
    operator, so ``B^T B = X Λ^{-1} X^T``), and the R factor of a QR
    decomposition of ``B`` is the root. Joint draws of ``X @ w`` for
    ``w ~ N(0, Λ^{-1})`` are then ``z @ R`` with iid standard-normal
    ``z`` -- O(n_rows*k) per draw, independent of n_features.

    When a sparse ``X`` is too wide to densify within the scratch cap,
    the n_rows x n_rows Gram matrix is accumulated in bounded column
    blocks instead and its eigendecomposition provides the root.

    Returns None when ``n_rows >= n_features`` (weight space is the
    smaller space -- sample there instead) or the candidate set exceeds
    ``_PREDICTIVE_ROOT_MAX_ROWS``.
    """
    n_rows, n_features = X.shape
    if n_rows >= n_features or n_rows > _PREDICTIVE_ROOT_MAX_ROWS:
        return None
    if issparse(X) and n_rows * n_features > _PREDICTIVE_DRAWS_ELEMS:
        return _predictive_root_gram(factor, X)
    Xt = X.T.toarray() if issparse(X) else np.asarray(X.T, dtype=np.float64)
    B = np.asarray(factor.half_solve(Xt), dtype=np.float64)
    # B may be (p,) for a single row; keep columns = rows of X
    if B.ndim == 1:
        B = B[:, None]
    # mode="r" skips forming Q entirely; B is scratch, safe to overwrite.
    # It returns R padded to B's full row count -- everything below the
    # triangle is zero, so only the top min(p, n_rows) rows carry the
    # square root.
    k = min(B.shape)
    return cast(
        NDArray[np.float64],
        scipy_qr(B, mode="r", overwrite_a=True, check_finite=False)[0][:k],
    )


def _predictive_root_gram(factor: PrecisionFactor, X: Any) -> NDArray[np.float64]:
    """Predictive root via the Gram matrix, with bounded dense scratch.

    Accumulates ``S = X Λ^{-1} X^T`` one column block at a time using
    ``Λ^{-1} b = M (M^T b)`` against the cached factor, so the dense
    scratch never exceeds the block cap regardless of n_features, then
    roots the n_rows x n_rows ``S`` by eigendecomposition (clipping
    the tiny negative eigenvalues roundoff can produce).
    """
    n_rows, n_features = X.shape
    block = max(1, _PREDICTIVE_DRAWS_ELEMS // n_features)
    S = np.empty((n_rows, n_rows), dtype=np.float64)
    for start in range(0, n_rows, block):
        stop = min(n_rows, start + block)
        rhs = np.asarray(X[start:stop].T.toarray(), dtype=np.float64)
        W = factor.colorize(factor.half_solve(rhs))  # Λ^{-1} X_block^T
        S[:, start:stop] = X @ W
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    np.clip(w, 0.0, None, out=w)
    return cast(NDArray[np.float64], np.sqrt(w)[:, None] * V.T)


def multivariate_t_sample_from_precision(
    loc: Union[csc_array, NDArray[np.float64], None],
    factor: PrecisionFactor,
    df: float = 1.0,
    size: int = 1,
    random_state: Union[int, None, np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Sample from a multivariate t distribution with mean loc, shape matrix
    shape, and degrees of freedom df.

    Parameters
    ----------
    loc : array_like
        Mean of the distribution.
    factor : PrecisionFactor
        Factored precision (inverse shape) matrix.
    df : int or float, optional
        Degrees of freedom of the distribution. Default is 1.
    size : int or tuple of ints, optional
        Given a shape of, for example, (m,n,k), m*n*k samples are generated,
        and packed in an m-by-n-by-k arrangement. Because each sample is
        N-dimensional, the output shape is (m,n,k,N). If no shape is specified,
        a single (N-D) sample is returned.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random.default_rng`.

    Returns
    -------
    out : ndarray
        The drawn samples, of shape size, if that was provided. If not, the
        shape is (N,).


    """
    rng = np.random.default_rng(random_state)

    x = rng.chisquare(df, size) / df

    z = multivariate_normal_sample_from_precision(
        mean=None,
        factor=factor,
        size=size,
        random_state=random_state,
    )
    if loc is None:
        loc = np.zeros_like(z)
    samples = cast(NDArray[np.float64], loc + z / np.sqrt(x)[..., None])
    samples = _squeeze_output(samples)

    return samples
