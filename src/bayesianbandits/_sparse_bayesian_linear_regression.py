import copy
import os
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cho_solve, solve_triangular  # type: ignore
from scipy.linalg.blas import dtrsm  # type: ignore[attr-defined]
from scipy.linalg.lapack import dlantr as _dlantr  # type: ignore[attr-defined]
from scipy.linalg.lapack import dtrtri  # type: ignore[attr-defined]
from scipy.sparse import (  # type: ignore  # type: ignore
    block_diag,
    coo_array,
    csc_array,
    csc_matrix,
    csr_matrix,
    diags,  # type: ignore
    issparse,  # type: ignore
)
from scipy.sparse.linalg import (  # type: ignore
    splu,
    spsolve_triangular,
    use_solver,
)

from ._memory import MemoryUsageMixin

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


def takahashi_diagonal(L_csc: csc_array) -> NDArray[np.float64]:
    """``diag((L Lᵀ)⁻¹)`` by Takahashi recursion (selected inversion).

    Given a lower-triangular CSC ``L`` with ``L Lᵀ = A``, computes the
    diagonal of ``A⁻¹`` exactly by backward recursion through ``L``'s
    sparsity structure. Cost is ``O(Σⱼ nⱼ²)`` in the sub-diagonal counts
    ``nⱼ``, the same order as the Cholesky that produced ``L``; columns
    sharing a sub-diagonal structure (a supernode) go through BLAS as one
    dense block, from four columns up, which is where the copy into that
    block starts to pay. Wide supernodes are cut into block columns and the
    dense ``Z_RR`` is formed one column panel at a time, so the scratch a
    block asks for is bounded by what it already occupies in ``L`` plus a
    fixed panel, however badly the factor fills in. How many threads the
    BLAS underneath uses is the caller's to set.

    Delegates to a Cython implementation. Lives beside the factors
    because it reads a Cholesky factor and nothing else; consumers want
    :meth:`CholmodSparseFactor.trace_inv`, not the factor itself.

    References
    ----------
    .. [1] Takahashi, K., Fagan, J., & Chin, M.-S. (1973). "Formation of
       a sparse bus impedance matrix and its application to short circuit
       study." 8th PICA Conference Proceedings.
    """
    from ._takahashi import takahashi_diagonal as _cy_impl

    p = cast(tuple[int, int], L_csc.shape)[0]
    if not L_csc.has_sorted_indices:
        L_csc = L_csc.copy()
        L_csc.sort_indices()
    return cast(
        NDArray[np.float64],
        _cy_impl(
            L_csc.data,
            L_csc.indices.astype(np.int32),
            L_csc.indptr.astype(np.int32),
            p,
        ),
    )


def trivial_columns(precision: csc_array) -> NDArray[np.intp]:
    """Features whose posterior is an independent scalar: columns storing
    a single positive diagonal entry that no other column references.

    ``Λ = αI + Σ x_t x_tᵀ`` puts an off-diagonal entry between two
    features only when some observation touched both, so a feature no
    observation has touched holds nothing but its diagonal -- exactly,
    not approximately -- and ``Λ`` is block diagonal between those
    features and the rest. Their part of ``Λ⁻¹`` is ``1/Λ_jj``, no
    factorization required.

    The test reads the stored pattern, so it holds for any matrix, not
    just ones built that way. Both sides are checked -- the column's own
    length, and how many entries anywhere point at its row -- so a
    coupling stored only in the other column (triangular storage) still
    counts. A stored explicit zero counts as structure, which errs toward
    factoring more, never less. A non-positive diagonal is left to the
    backends, whose non-SPD behaviour this must not change.
    """
    indptr, indices = precision.indptr, precision.indices
    n = cast("tuple[int, int]", precision.shape)[0]
    single = np.flatnonzero(indptr[1:] - indptr[:-1] == 1)
    if single.size == 0:
        return single
    first = indptr[single]
    keep = (indices[first] == single) & (precision.data[first] > 0)
    single = single[keep]
    # every entry pointing at a candidate row must be its own diagonal
    referenced = np.bincount(indices, minlength=n)[single]
    return cast(NDArray[np.intp], single[referenced == 1])


def _same_pattern(a: csc_array, b: csc_array) -> bool:
    """Whether two CSC matrices store the same sparsity pattern; a grown
    pattern is rejected in ``O(1)`` by size."""
    return bool(
        a.shape == b.shape
        and a.indices.size == b.indices.size
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
    )


def _canonical_csc(
    data: NDArray[Any],
    indices: NDArray[Any],
    indptr: NDArray[Any],
    shape: tuple[int, int],
) -> csc_array:
    """A CSC matrix from canonical arrays, flagged so nothing re-sorts them."""
    out = csc_array((data, indices, indptr), shape=shape)
    out.has_sorted_indices = True
    out.has_canonical_format = True
    return out


class _BlockPattern(NamedTuple):
    """A factored block's pattern as a gather from the whole matrix:
    ``block.data == precision.data[g]`` over fixed ``indices``/``indptr``."""

    g: NDArray[np.intp]
    indices: NDArray[Any]
    indptr: NDArray[Any]

    def gather(self, precision: csc_array) -> csc_array:
        m = self.indptr.size - 1
        return _canonical_csc(precision.data[self.g], self.indices, self.indptr, (m, m))


def _principal_block(
    precision: csc_array, observed: NDArray[np.intp]
) -> tuple[csc_array, _BlockPattern]:
    """``Λ[observed, observed]`` for the ``observed`` complement of
    :func:`trivial_columns`, with its pattern; those columns' entries all
    sit in observed rows, so this is a column selection and a row relabel.

    The relabel keeps the matrix's own index dtype: an ``intp`` table
    would widen the block's indices to int64, and CHOLMOD then builds an
    int64 factor whose *sparse* solve segfaults on an int32 right-hand
    side (the dense solve is indifferent)."""
    n = cast("tuple[int, int]", precision.shape)[0]
    indptr = precision.indptr
    starts = indptr[observed]
    lens = indptr[observed + 1] - starts
    block_indptr = np.zeros(observed.size + 1, dtype=indptr.dtype)
    np.cumsum(lens, out=block_indptr[1:])
    total = int(block_indptr[-1])
    g = np.repeat(starts - block_indptr[:-1], lens) + np.arange(total, dtype=np.intp)
    lut = np.empty(n, dtype=precision.indices.dtype)
    lut[observed] = np.arange(observed.size, dtype=lut.dtype)
    pattern = _BlockPattern(g, lut[precision.indices[g]], block_indptr)
    return pattern.gather(precision), pattern


def _block_of(
    precision: csc_array, observed: Union[NDArray[np.intp], slice]
) -> tuple[csc_array, Optional[_BlockPattern]]:
    """The block to factor: the whole matrix when nothing is trivial."""
    if isinstance(observed, slice):
        return precision, None
    return _principal_block(precision, observed)


def _trivial_diag(
    precision: csc_array, trivial: NDArray[np.intp]
) -> NDArray[np.float64]:
    """``Λ_jj`` over the trivial features (each stores only its diagonal)."""
    return np.asarray(precision.data[precision.indptr[trivial]], dtype=np.float64)


def _partition(
    precision: csc_array,
) -> tuple[Union[NDArray[np.intp], slice], NDArray[np.intp], NDArray[np.float64]]:
    """``(observed, trivial, trivial_diag)`` for a factor over ``precision``.

    ``observed`` is a full slice when nothing is trivial, so indexing
    the block's operands stays a view rather than an ``O(n)`` copy. An
    all-trivial matrix is factored whole as well: the block backends
    want at least one column, and it is the fresh-prior case, where the
    diagonal factorization is cheap.
    """
    trivial = trivial_columns(precision)
    n = cast("tuple[int, int]", precision.shape)[0]
    if trivial.size in (0, n):
        return slice(None), np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64)
    keep = np.ones(n, dtype=bool)
    keep[trivial] = False
    observed = np.flatnonzero(keep)
    return observed, trivial, _trivial_diag(precision, trivial)


def _column_scale(values: NDArray[np.float64], ndim: int) -> NDArray[np.float64]:
    """``values`` shaped to scale the rows of a 1-D or 2-D operand."""
    return values if ndim == 1 else values[:, np.newaxis]


def _unscale(out: Any, scale: float) -> Any:
    """``out / scale``, or ``out`` itself when the factor carries no scale.

    Every caller here divides something it has just built, so handing it
    back undivided is safe, and it drops a full pass over (and a second
    allocation of) an operand-sized array whenever ``scale`` is one --
    which is every factor but a decayed one and the Normal-Inverse-Gamma
    shape (see :func:`scale_factor`).
    """
    return out if scale == 1.0 else out / scale


def _merge(
    observed: Union[NDArray[np.intp], slice],
    trivial: NDArray[np.intp],
    block: NDArray[np.floating[Any]],
    trivial_values: NDArray[np.floating[Any]],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """A feature-ordered result from its block and trivial parts; the
    block itself when nothing is trivial (no copy)."""
    if trivial.size == 0:
        return cast(NDArray[np.float64], block)
    out = np.empty(shape, dtype=np.float64)
    out[observed] = block
    out[trivial] = trivial_values
    return out


class SparseHalfSolve(NamedTuple):
    """A half-solve against a sparse right-hand side, in two parts whose
    Gram add: ``Bᵀ B = blockᵀ block + trivialᵀ trivial``.

    ``block`` is the dense ``(n_factored, k)`` half-solve at the features
    the factor factors; ``trivial`` is sparse ``(t, k)``, one row per
    distinct never-observed feature the right-hand side touches
    (ascending), holding ``value / sqrt(Λ_jj)`` at that entry's column.
    ``t`` is bounded by the right-hand side's nonzeros, not by anything
    the factor knows, so it is never densified: callers read a half-solve
    through its Gram or draw with it (``blockᵀ z₁ + trivialᵀ z₂``), and
    both split across the parts.
    """

    block: NDArray[np.float64]
    trivial: csc_array


class _SparseRhs:
    """A sparse right-hand side split for the partitioned solve: its
    observed rows densified for the block solver, its trivial entries
    kept as triples.

    Neither backend is handed a sparse operand. CHOLMOD's sparse solve
    reads the operand's index arrays with the factor's own integer type,
    and a mismatch (a CSC row-indexed in scipy comes back with int64
    indices against an int32 factor) does not raise -- it returns the
    wrong answer, silently, or segfaults. Densifying the block rows
    sidesteps that entirely and lets the solve run as one BLAS-3 call.

    Entries are classified through :func:`_locate` on the COO triples,
    ``O(nnz log n)``, rather than by row-indexing the matrix, which
    scipy does in ``O(n)`` per call however few entries there are.
    """

    __slots__ = ("block", "trivial_rank", "trivial_col", "trivial_val", "shape")

    def __init__(
        self,
        b: Any,
        observed: Union[NDArray[np.intp], slice],
        trivial: NDArray[np.intp],
    ) -> None:
        c = coo_array(b)
        c.sum_duplicates()
        n, k = cast("tuple[int, int]", c.shape)
        row = np.asarray(c.row, dtype=np.intp)
        col = np.asarray(c.col, dtype=np.intp)
        val = np.asarray(c.data, dtype=np.float64)
        is_trivial, block_pos, rank = _locate(observed, trivial, row)
        m = n - trivial.size
        block = np.zeros((m, k), dtype=np.float64, order="F")
        block[block_pos, col[~is_trivial]] = val[~is_trivial]
        self.block = block
        self.trivial_rank = rank
        self.trivial_col = col[is_trivial]
        self.trivial_val = val[is_trivial]
        self.shape = (n, k)

    def solution(
        self,
        observed: Union[NDArray[np.intp], slice],
        trivial: NDArray[np.intp],
        trivial_diag: NDArray[np.float64],
        block: NDArray[np.float64],
    ) -> csc_array:
        """The solution as CSC over all features: the dense block solve
        at the observed rows, plus each trivial entry over its diagonal.

        Assembled straight into CSC -- every column holds the observed
        rows, already sorted, so the block is one column-major ravel
        with a tiled index -- never through a dense ``(n, k)``; the cost
        is ``O(m k)``, not ``O(n k)``.
        """
        n, k = self.shape
        m = block.shape[0]
        idx_dtype = np.int32 if n < np.iinfo(np.int32).max else np.intp
        rows = (
            np.arange(n, dtype=idx_dtype) if isinstance(observed, slice) else observed
        )
        out = csc_array(
            (
                np.asfortranarray(block).ravel(order="F"),
                np.tile(np.asarray(rows, dtype=idx_dtype), k),
                np.arange(0, m * k + 1, m, dtype=idx_dtype),
            ),
            shape=self.shape,
        )
        if self.trivial_val.size == 0:
            return out
        scaled = csc_array(
            (
                self.trivial_val / trivial_diag[self.trivial_rank],
                (trivial[self.trivial_rank], self.trivial_col),
            ),
            shape=self.shape,
        )
        return cast(csc_array, out + scaled)

    def compact_trivial(
        self, trivial_diag: NDArray[np.float64], scale: float = 1.0
    ) -> csc_array:
        """The trivial part of a half-solve, the sparse ``(t, k)`` of
        :class:`SparseHalfSolve`: one row per *distinct* trivial feature
        the right-hand side touches, ascending, holding
        ``value / sqrt(scale · Λ_jj)`` at that entry's column."""
        _, k = self.shape
        if self.trivial_val.size == 0:
            return csc_array((0, k), dtype=np.float64)
        distinct, inverse = np.unique(self.trivial_rank, return_inverse=True)
        values = self.trivial_val / np.sqrt(scale * trivial_diag[self.trivial_rank])
        return csc_array(
            (values, (inverse.ravel(), self.trivial_col)), shape=(distinct.size, k)
        )


def _stack(
    block: NDArray[np.floating[Any]], trivial_values: NDArray[np.floating[Any]]
) -> NDArray[np.float64]:
    """A factor-row-ordered result: block rows first, then the trivial
    features in ascending order; the block itself when nothing is trivial."""
    if trivial_values.shape[0] == 0:
        return cast(NDArray[np.float64], block)
    return cast(NDArray[np.float64], np.concatenate([block, trivial_values], axis=0))


def _locate(
    observed: Union[NDArray[np.intp], slice],
    trivial: NDArray[np.intp],
    features: NDArray[np.intp],
) -> tuple[NDArray[np.bool_], NDArray[np.intp], NDArray[np.intp]]:
    """Split ``features`` for :meth:`~CholmodSparseFactor.sample_at`:
    a trivial mask, the block positions of the rest, and the ranks of the
    trivial ones within the trivial set."""
    if trivial.size == 0:
        return np.zeros(features.size, dtype=bool), features, features[:0]
    rank = np.searchsorted(trivial, features)
    is_trivial = (rank < trivial.size) & (
        trivial[np.minimum(rank, trivial.size - 1)] == features
    )
    block_pos = np.searchsorted(cast(NDArray[np.intp], observed), features[~is_trivial])
    return is_trivial, block_pos, rank[is_trivial]


def _normals(rng: np.random.Generator, size: int, n: int) -> NDArray[np.float64]:
    """``(n, size)`` standard normals, drawn as ``(size, n)`` and
    transposed: the stream is that of ``size`` weight vectors drawn in
    turn, and the layout is column-major, which is what CHOLMOD's dense
    format and ``dtrsm`` take without a copy."""
    return cast(NDArray[np.float64], rng.standard_normal((size, n)).T)


def _scatter_draws(
    is_trivial: NDArray[np.bool_],
    block: NDArray[np.floating[Any]],
    rank: NDArray[np.intp],
    trivial_diag: NDArray[np.float64],
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Interleave the block's draws with fresh scalar draws for the
    trivial features asked for: one normal per *distinct* trivial
    feature, so a repeated feature repeats its value; the block's draws
    alone when none was asked for."""
    if not is_trivial.any():
        return cast(NDArray[np.float64], block)
    distinct, inverse = np.unique(rank, return_inverse=True)
    scalars = (
        _normals(rng, size, distinct.size)
        / np.sqrt(trivial_diag[distinct])[:, np.newaxis]
    )
    out = np.empty((is_trivial.size, size), dtype=np.float64)
    out[~is_trivial] = block
    out[is_trivial] = scalars[inverse]
    return out


def _stacked_L(L_block: csc_array, trivial_diag: NDArray[np.float64]) -> csc_array:
    """The factor's ``L`` over all features, block rows first: the block's
    ``L`` and ``sqrt`` of the trivial diagonal, block diagonal."""
    if trivial_diag.size == 0:
        return L_block
    return csc_array(block_diag([L_block, diags(np.sqrt(trivial_diag))], format="csc"))


@dataclass
class CholmodSparseFactor(MemoryUsageMixin):
    """A CHOLMOD factor over the observed block of a precision matrix, and
    the trivial diagonal beside it.

    ``_precision`` is the whole matrix; ``_factor`` covers only the
    ``_observed`` features (see :func:`trivial_columns`), and
    ``_trivial_diag`` holds ``Λ_jj`` for the ``_trivial`` rest, whose
    every operation is elementwise. Factor-row order, where a method's
    contract is in terms of it, is the block's own rows followed by the
    trivial features ascending.

    ``_scale`` makes this a factor of ``_scale · Λ`` without refactoring:
    ``decay`` multiplies the precision by a scalar, and the
    Normal-Inverse-Gamma shape precision is ``(a/b) · Λ``. If
    ``L Lᵀ = Λ`` then ``(√s L)(√s L)ᵀ = s Λ``, so every operation is the
    unscaled one times a power of ``s``; see :func:`scale_factor`.
    """

    _factor: Any  # sksparse.cholmod.Factor (C extension, no useful type)
    _precision: csc_array
    _observed: Union[NDArray[np.intp], slice]
    _trivial: NDArray[np.intp]
    _trivial_diag: NDArray[np.float64]
    _scale: float = 1.0
    _block: Optional[_BlockPattern] = None  # None: the block is the whole matrix

    @cached_property
    def _inv_perm(self) -> NDArray[np.intp]:
        """Inverse of CHOLMOD's fill-reducing permutation, for
        :meth:`sample_at`.

        Scattered in one O(n_features) pass rather than sorted: a
        permutation's inverse is where each entry points, which
        ``argsort`` recovers by comparison at O(n log n). Lazy on top of
        that, because the factor is rebuilt by every ``fit``/
        ``partial_fit`` and neither samples -- but a Thompson pull does,
        so in a pull-and-update loop this is paid fresh every
        round and never amortizes.
        """
        perm = self._factor.perm
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size, dtype=perm.dtype)
        return cast(NDArray[np.intp], inv)

    @property
    def solve_cost(self) -> int:
        """Cost of one solve against this factor, in stored entries.

        The routing gates weigh solve counts against the other work each
        sampling route does, so they need a per-solve price. It belongs
        to the factor rather than the estimator's precision matrix: the
        two agree today, but only the factor knows what it actually
        solves against -- the block, not the trivial diagonal.
        """
        return self._precision.nnz - self._trivial.size

    @property
    def n_factored(self) -> int:
        """Features the factor actually factors: the observed block, not
        the trivial diagonal beside it. It is the row count of the
        factor's dense operands, and so the unit in which callers should
        size scratch and chunk work -- ``n_features`` overstates both by
        the trivial count."""
        return cast("tuple[int, int]", self._precision.shape)[0] - self._trivial.size

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Solve ``Λ x = b``. A sparse ``b`` comes back sparse (the RVGA
        Gram matrix and the support covariance both solve against sparse
        right-hand sides); see :class:`_SparseRhs` for why the block
        rows are densified first."""
        if issparse(b):
            rhs = _SparseRhs(b, self._observed, self._trivial)
            block = np.asarray(self._factor.solve(rhs.block), dtype=np.float64)
            out = rhs.solution(self._observed, self._trivial, self._trivial_diag, block)
            return cast(NDArray[np.float64], _unscale(out, self._scale))
        block = np.asarray(self._factor.solve(b[self._observed]), dtype=np.float64)
        rest = b[self._trivial] / _column_scale(self._trivial_diag, b.ndim)
        out = _merge(self._observed, self._trivial, block, rest, b.shape)
        return cast(NDArray[np.float64], _unscale(out, self._scale))

    def sample_at(
        self,
        features: Optional[NDArray[np.intp]],
        size: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Zero-mean draws ``w ~ N(0, Λ⁻¹)`` at ``features``, shape
        ``(|features|, size)``; every feature when ``features`` is None.

        The factor draws its own normals, so it draws only what it
        needs: one per block feature, plus one per *distinct* trivial
        feature asked for -- never one per never-observed feature it was
        not asked about. A repeated feature gets the same draw, and the
        caller reads the result through the design matrix (``X_U @ G``),
        so two rows touching one feature share its value: the joint law
        of ``X w``, not an independent draw per cell.

        Block draws are ``L⁻ᵀ z`` in CHOLMOD's row order, un-permuted on
        the way out; a trivial feature's draw is ``z / sqrt(Λ_jj)``.
        """
        m = cast("tuple[int, int]", self._precision.shape)[0] - self._trivial.size
        root = np.sqrt(self._scale)
        if features is None:
            Z = _normals(rng, size, m + self._trivial.size)
            block = self._factor.solve(Z[:m], system="Lt")[self._inv_perm]
            rest = Z[m:] / np.sqrt(self._trivial_diag)[:, np.newaxis]
            return _unscale(
                _merge(self._observed, self._trivial, block, rest, Z.shape), root
            )
        is_trivial, block_pos, rank = _locate(self._observed, self._trivial, features)
        block = self._factor.solve(_normals(rng, size, m), system="Lt")[
            self._inv_perm[block_pos]
        ]
        return _unscale(
            _scatter_draws(is_trivial, block, rank, self._trivial_diag, size, rng), root
        )

    def half_solve(
        self, b: NDArray[np.floating[Any]]
    ) -> Union[NDArray[np.float64], SparseHalfSolve]:
        """Apply the transpose of the ``sample_at`` operator: L^{-1} P b.

        If ``M`` is the operator ``sample_at`` applies to its normals
        (``M M^T = Λ^{-1}``), this
        returns ``M^T b``, so ``half_solve(X.T).T @ half_solve(X.T)``
        equals ``X Λ^{-1} X^T`` -- a half-solve per column against the
        cached factor, without ever forming ``Λ^{-1}``. A dense ``b``
        comes back dense in factor-row order (block rows, then the
        trivial features ascending); a sparse ``b`` comes back as a
        :class:`SparseHalfSolve`, its trivial part never densified, which
        keeps the marginal and reward-space paths at ``O(n_factored)``
        per column plus ``O(nnz)`` whatever ``b`` touches.
        """
        root = np.sqrt(self._scale)
        if issparse(b):
            rhs = _SparseRhs(b, self._observed, self._trivial)
            block = self._factor.solve(rhs.block[self._factor.perm], system="L")
            return SparseHalfSolve(
                _unscale(block, root),
                rhs.compact_trivial(self._trivial_diag, self._scale),
            )
        block = self._factor.solve(b[self._observed][self._factor.perm], system="L")
        rest = b[self._trivial] / _column_scale(np.sqrt(self._trivial_diag), b.ndim)
        return _unscale(_stack(block, rest), root)

    @cached_property
    def _L_csc(self) -> csc_array:
        """The block's ``L`` as a sorted CSC, shared by :meth:`logdet` and
        :meth:`trace_inv`; dropped by :meth:`refactorize`."""
        L = csc_array(self._factor.L)
        if not L.has_sorted_indices:
            L.sort_indices()
        return L

    def logdet(self) -> float:
        """``2 Σ log L_jj`` (each column's first stored entry, in a sorted
        lower-triangular CSC), plus the trivial diagonal's, plus ``n log s``
        for the scale. CHOLMOD's ``logdet`` walks all of ``L`` for the same."""
        n = cast("tuple[int, int]", self._precision.shape)[0]
        L = self._L_csc
        block = 2.0 * np.sum(np.log(L.data[L.indptr[:-1]]))
        return float(
            block + np.sum(np.log(self._trivial_diag)) + n * np.log(self._scale)
        )

    def trace_inv(self) -> float:
        """``tr(Λ⁻¹)``, exactly: Takahashi recursion over the block's
        ``L`` plus ``Σ 1/Λ_jj`` over the trivial features.

        The trace is permutation-invariant, so the fill-reducing
        permutation needs no undoing. Computed here rather than by
        handing ``L`` to a caller: only the factor knows what it
        factored.
        """
        block = np.sum(takahashi_diagonal(self._L_csc))
        return float(block + np.sum(1.0 / self._trivial_diag)) / self._scale

    def refactorize(self, precision: csc_array) -> "CholmodSparseFactor":
        """A factor of ``precision``: numeric-only, in place, when the
        sparsity pattern matches (the symbolic analysis depends on nothing
        else); fresh otherwise, since a permutation chosen for a different
        pattern can fill in catastrophically. The scale is reset, and
        views :func:`scale_factor` made of this factor are invalid after.
        """
        if not _same_pattern(precision, self._precision):
            return create_sparse_factor(precision, SparseSolver.CHOLMOD)  # type: ignore[return-value]
        block = precision if self._block is None else self._block.gather(precision)
        self._factor.factorize(csc_matrix(block))
        self.__dict__.pop("_L_csc", None)
        self._precision = precision
        self._trivial_diag = _trivial_diag(precision, self._trivial)
        self._scale = 1.0
        return self

    def get_L_csc(self) -> csc_array:
        """Return the lower triangular Cholesky factor as CSC, in
        factor-row order: the block's ``L`` (its rows in CHOLMOD's
        fill-reducing permutation) and then ``sqrt`` of the trivial
        diagonal, so ``L Lᵀ`` is a symmetric permutation of ``s Λ``.
        """
        L = _stacked_L(csc_array(self._factor.L), self._trivial_diag)
        return L if self._scale == 1.0 else csc_array(L * np.sqrt(self._scale))


@dataclass
class SuperLUSparseFactor(MemoryUsageMixin):
    """A SuperLU decomposition over the observed block of a precision
    matrix, and the trivial diagonal beside it; see
    :class:`CholmodSparseFactor` for the layout."""

    _lu: Any  # scipy.sparse.linalg.SuperLU (C extension, no useful type)
    _inv_perm: NDArray[np.intp]
    _precision: csc_array
    _observed: Union[NDArray[np.intp], slice]
    _trivial: NDArray[np.intp]
    _trivial_diag: NDArray[np.float64]
    _scale: float = 1.0
    # refactorize() factors the block pre-permuted into the cached order;
    # _lu's own permutation is then the identity and _perm wraps its solves.
    _prepermuted: bool = False
    _block: Optional[_BlockPattern] = None  # None: the block is the whole matrix
    _permuted: Optional[_BlockPattern] = None  # the block in cached order

    @cached_property
    def _L(self) -> csc_matrix:
        """Lower triangular factor with ``D`` folded in, ``L Lᵀ = P Λ Pᵀ``.

        SuperLU's own ``L`` is unit-diagonal, with the pivots on ``U``'s
        diagonal; folding ``sqrt(D)`` in gives the symmetric factor the
        sampling operators need. Lazy, like everything else here that
        only sampling reaches --
        ``fit``/``partial_fit`` solve through :attr:`_lu` and never
        build it."""
        L = self._lu.L
        root = np.sqrt(self._lu.U.diagonal())
        data = L.data * np.repeat(root, np.diff(L.indptr))
        return csc_matrix((data, L.indices, L.indptr), shape=L.shape)

    @cached_property
    def _Lt(self) -> csr_matrix:
        """``Lᵀ`` for :meth:`sample_at`: ``L``'s CSC arrays read as CSR."""
        L = self._L
        return csr_matrix((L.data, L.indices, L.indptr), shape=L.shape)

    @cached_property
    def _perm(self) -> NDArray[np.intp]:
        """Inverse of ``_inv_perm``, computed lazily so that ``fit``/
        ``partial_fit`` (which never call ``half_solve``) pay no extra cost."""
        perm = np.empty_like(self._inv_perm)
        perm[self._inv_perm] = np.arange(self._inv_perm.size)
        return perm

    @property
    def solve_cost(self) -> int:
        """See :attr:`CholmodSparseFactor.solve_cost`."""
        return self._precision.nnz - self._trivial.size

    @property
    def n_factored(self) -> int:
        """See :attr:`CholmodSparseFactor.n_factored`."""
        return cast("tuple[int, int]", self._precision.shape)[0] - self._trivial.size

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Solve ``Λ x = b`` through the retained SuperLU object. A sparse
        ``b`` comes back sparse; see :class:`_SparseRhs`.

        ``SuperLU.solve`` runs both triangular sweeps and both
        permutations inside one ``gstrs`` call, where going through
        :meth:`half_solve` and :meth:`sample_at` pays two trips out to
        ``spsolve_triangular`` -- each of which rebuilds its own solver
        state -- to reach the same answer.
        """
        if issparse(b):
            rhs = _SparseRhs(b, self._observed, self._trivial)
            block = self._block_solve(rhs.block)
            out = rhs.solution(self._observed, self._trivial, self._trivial_diag, block)
            return cast(NDArray[np.float64], _unscale(out, self._scale))
        b = np.asarray(b, dtype=np.float64)
        block = self._block_solve(b[self._observed])
        rest = b[self._trivial] / _column_scale(self._trivial_diag, b.ndim)
        out = _merge(self._observed, self._trivial, block, rest, b.shape)
        return cast(NDArray[np.float64], _unscale(out, self._scale))

    def _block_solve(self, rhs: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """``Λ_block x = rhs`` through SuperLU, around the pre-permutation
        if there is one: ``x[perm]`` solves against ``rhs[perm]``."""
        if not self._prepermuted:
            return np.asarray(self._lu.solve(rhs), dtype=np.float64)
        perm = self._perm
        out = np.empty(rhs.shape, dtype=np.float64)
        out[perm] = self._lu.solve(np.ascontiguousarray(rhs[perm]))
        return out

    def sample_at(
        self,
        features: Optional[NDArray[np.intp]],
        size: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """See :meth:`CholmodSparseFactor.sample_at`."""
        m = cast("tuple[int, int]", self._precision.shape)[0] - self._trivial.size
        root = np.sqrt(self._scale)
        if features is None:
            Z = _normals(rng, size, m + self._trivial.size)
            block = spsolve_triangular(self._Lt, Z[:m], lower=False)[self._inv_perm]
            rest = Z[m:] / np.sqrt(self._trivial_diag)[:, np.newaxis]
            return _unscale(
                _merge(self._observed, self._trivial, block, rest, Z.shape), root
            )
        is_trivial, block_pos, rank = _locate(self._observed, self._trivial, features)
        block = spsolve_triangular(self._Lt, _normals(rng, size, m), lower=False)[
            self._inv_perm[block_pos]
        ]
        return _unscale(
            _scatter_draws(is_trivial, block, rank, self._trivial_diag, size, rng), root
        )

    @cached_property
    def _half_solve_factor(self) -> tuple[csc_array, NDArray[np.float64]]:
        """Unit-diagonal ``L`` (CSC) and its inverse diagonal, cached
        because ``spsolve_triangular`` otherwise re-copies and re-scales
        the factor on every call (O(nnz) per solve). Lazy, like
        ``_perm``, so ``fit``/``partial_fit`` pay no extra cost."""
        L = self._L
        invdiag = np.asarray(1.0 / L.diagonal(), dtype=np.float64)
        L_unit = csc_array(
            (L.data * np.repeat(invdiag, np.diff(L.indptr)), L.indices, L.indptr),
            shape=L.shape,
        )
        return L_unit, invdiag

    def half_solve(
        self, b: NDArray[np.floating[Any]]
    ) -> Union[NDArray[np.float64], SparseHalfSolve]:
        """Apply the transpose of the ``sample_at`` operator: L^{-1} P b.

        Same contract as :meth:`CholmodSparseFactor.half_solve`, against
        the cached triangular factor (no refactorization, unlike ``solve``).
        With ``L = L' D`` (``L'`` unit-diagonal), ``x = D^{-1} L'^{-1} b``;
        the cached ``L'`` lets ``spsolve_triangular`` skip its per-call
        copy and rescale of the factor (``b[perm]`` is a fresh array, so
        overwriting both operands is safe).
        """
        L_unit, invdiag = self._half_solve_factor
        root = np.sqrt(self._scale)
        rhs = _SparseRhs(b, self._observed, self._trivial) if issparse(b) else None
        operand = b[self._observed] if rhs is None else rhs.block
        y = spsolve_triangular(
            L_unit,
            operand[self._perm],
            lower=True,
            unit_diagonal=True,
            overwrite_A=True,
            overwrite_b=True,
        )
        y *= _column_scale(invdiag / root, y.ndim)
        if rhs is not None:
            return SparseHalfSolve(
                y, rhs.compact_trivial(self._trivial_diag, self._scale)
            )
        rest = b[self._trivial] / _column_scale(np.sqrt(self._trivial_diag), b.ndim)
        return _stack(y, _unscale(rest, root))

    def logdet(self) -> float:
        """Log-determinant of the factored matrix.

        ``U``'s diagonal holds the block's pivots ``D``, which ``_L``
        carries as ``sqrt(D)``: ``log|Λ_block| = 2 * sum(log|diag(_L)|)
        = sum(log|D|)``, reading the pivots straight off ``U`` rather
        than building ``_L``; plus the trivial diagonal's.
        """
        n = cast("tuple[int, int]", self._precision.shape)[0]
        block = np.sum(np.log(np.abs(self._lu.U.diagonal())))
        return float(
            block + np.sum(np.log(self._trivial_diag)) + n * np.log(self._scale)
        )

    def trace_inv(self) -> float:
        """See :meth:`CholmodSparseFactor.trace_inv`."""
        block = np.sum(takahashi_diagonal(csc_array(self._L)))
        return float(block + np.sum(1.0 / self._trivial_diag)) / self._scale

    def refactorize(self, precision: csc_array) -> "SuperLUSparseFactor":
        """A factor of ``precision``, reusing this one's fill-reducing
        ordering when the sparsity pattern matches: the block is permuted
        into the cached order and factored ``NATURAL``, which skips the
        ordering search and yields the same ``L``. Fresh otherwise.
        """
        if not _same_pattern(precision, self._precision):
            return _superlu_factor(precision)
        if self._permuted is None:
            self._permuted = self._permuted_pattern()
        lu = splu(
            self._permuted.gather(precision),
            diag_pivot_thresh=0,
            permc_spec="NATURAL",
            options=dict(SymmetricMode=True, Equil=False),
        )
        return SuperLUSparseFactor(
            _lu=lu,
            _inv_perm=self._inv_perm,
            _precision=precision,
            _observed=self._observed,
            _trivial=self._trivial,
            _trivial_diag=_trivial_diag(precision, self._trivial),
            _prepermuted=True,
            _block=self._block,
            _permuted=self._permuted,
        )

    def _permuted_pattern(self) -> _BlockPattern:
        """The pattern of ``Λ_block[perm][:, perm]`` as a gather from the
        whole matrix, assembled through COO (a counting sort in C)."""
        m = self.n_factored
        if self._block is None:
            src_indices, src_indptr = self._precision.indices, self._precision.indptr
        else:
            src_indices, src_indptr = self._block.indices, self._block.indptr
        inv_perm = self._inv_perm
        cols = np.repeat(np.arange(m, dtype=np.intp), np.diff(src_indptr))
        permuted = cast(
            csc_array,
            coo_array(
                (
                    np.arange(src_indices.size, dtype=np.float64),
                    (inv_perm[src_indices], inv_perm[cols]),
                ),
                shape=(m, m),
            ).tocsc(),
        )
        permuted.sort_indices()
        g = permuted.data.astype(np.intp)
        if self._block is not None:
            g = self._block.g[g]
        return _BlockPattern(g, permuted.indices, permuted.indptr)

    def get_L_csc(self) -> csc_array:
        """Return the lower triangular factor as CSC, in factor-row
        order: the block's ``L`` (``D`` folded in, rows in SuperLU's
        permutation) and then ``sqrt`` of the trivial diagonal, times
        ``sqrt`` of the scale."""
        L = _stacked_L(csc_array(self._L), self._trivial_diag)
        return L if self._scale == 1.0 else csc_array(L * np.sqrt(self._scale))


def _superlu_factor(precision: csc_array) -> SuperLUSparseFactor:
    """Factor ``precision`` with SuperLU, retaining the decomposition.

    The ``SuperLU`` object itself is kept, not just its ``L``: it is what
    :meth:`SuperLUSparseFactor.solve` drives, and the symmetric ``L`` the
    sampling operators want is derived from it on demand.
    """
    observed, trivial, diag = _partition(precision)
    block, pattern = _block_of(precision, observed)
    # Equil (a rescaling for unsymmetric systems) would only cost a pass
    splu_ = splu(
        block,
        diag_pivot_thresh=0,
        permc_spec="MMD_AT_PLUS_A",
        options=dict(SymmetricMode=True, Equil=False),
    )
    if (splu_.perm_r != splu_.perm_c).any():
        raise ValueError("Matrix must be symmetric")
    return SuperLUSparseFactor(
        _lu=splu_,
        _inv_perm=splu_.perm_r.copy(),
        _precision=precision,
        _observed=observed,
        _trivial=trivial,
        _trivial_diag=diag,
        _block=pattern,
    )


SparseFactor = CholmodSparseFactor | SuperLUSparseFactor


@dataclass
class DenseFactor(MemoryUsageMixin):
    """Wraps a LAPACK Cholesky factorization for solving and sampling.

    Stores the upper-triangular factor U where Λ = U^T U.  Every
    operation solves against ``U`` directly; the explicit ``U^{-1}`` is
    built lazily and only for :meth:`trace_inv`, which genuinely needs
    the inverse.  ``fit``/``partial_fit`` (which only need ``solve``)
    therefore pay no extra cost, and neither does sampling.
    """

    _U: NDArray[np.float64]
    _n_features: int
    _scale: float = 1.0

    @property
    def solve_cost(self) -> int:
        """Zero: the gates price a dense solve off the feature count
        instead, so they never read this. See
        :attr:`CholmodSparseFactor.solve_cost`."""
        return 0

    @property
    def n_factored(self) -> int:
        """Every feature: a dense factor has nothing trivial to set aside.
        See :attr:`CholmodSparseFactor.n_factored`."""
        return self._n_features

    @cached_property
    def _U_inv(self) -> NDArray[np.float64]:
        """Explicit ``U^{-1}``, for :meth:`trace_inv` only.

        ``dtrtri`` inverts the triangle at a third of the flops of
        solving against a dense identity, but writes only the upper
        triangle of its result, leaving whatever ``cho_factor`` left
        below the diagonal.  The strict lower triangle is therefore
        garbage; every consumer reads the upper one only.  ``_U``
        itself is not overwritten.

        A singular ``U`` is reported through ``info`` rather than
        raised, and leaves the triangle untouched -- so this
        would return ``U`` itself, and :meth:`trace_inv` a plausible
        number rather than an error.  Raise instead, matching the
        ``solve_triangular`` this replaced.
        """
        U_inv, info = dtrtri(self._U, lower=0)
        if info > 0:
            raise LinAlgError(
                f"singular precision factor: U[{info - 1}, {info - 1}] is zero"
            )
        if info < 0:
            raise ValueError(f"dtrtri: illegal value in argument {-info}")
        return cast(NDArray[np.float64], U_inv)

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        out = cho_solve((self._U, False), b, check_finite=False)
        return cast(NDArray[np.float64], _unscale(out, self._scale))

    def sample_at(
        self,
        features: Optional[NDArray[np.intp]],
        size: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Zero-mean draws ``U⁻¹ z ~ N(0, Λ⁻¹)`` at ``features``, shape
        ``(|features|, size)``; every feature when ``features`` is None.
        Same contract as :meth:`CholmodSparseFactor.sample_at`; a dense
        factor has no trivial features to skip, so it always draws all
        of them.

        Solves against ``U`` rather than multiplying by an explicit
        inverse.  The factor is rebuilt after every ``partial_fit``, so
        forming ``U^{-1}`` charged an O(d³) triangular inversion to each
        update -- more than an order of magnitude above the Cholesky
        that preceded it -- to save nothing: ``dtrsm`` and the ``dgemm``
        against the inverse cost the same per draw.  ``dtrsm`` also
        keeps the draw inside scipy's BLAS pool (see
        :func:`~bayesianbandits._blas_helpers.lower_predictive_sqrt`).
        """
        # the scale rides in dtrsm's alpha: alpha * U^{-1} Z in one call
        alpha = 1.0 / np.sqrt(self._scale)
        out = dtrsm(
            alpha, self._U, _normals(rng, size, self._n_features), lower=0, side=0
        )
        return cast(NDArray[np.float64], out if features is None else out[features])

    def half_solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Solve U^T x = b, the transpose of the ``sample_at`` operator.

        Same contract as :meth:`CholmodSparseFactor.half_solve`; one
        triangular solve, without building ``U^{-1}``. A sparse ``b`` is
        densified: a dense factor has no trivial rows to leave out.
        """
        rhs = cast(Any, b).toarray() if issparse(b) else b
        out = solve_triangular(self._U, rhs, trans=1, lower=False, check_finite=False)
        return cast(NDArray[np.float64], _unscale(out, np.sqrt(self._scale)))

    def logdet(self) -> float:
        """log|sΛ| = 2·sum(log(diag(U))) + d·log s."""
        return 2.0 * float(np.sum(np.log(np.diag(self._U)))) + self._n_features * float(
            np.log(self._scale)
        )

    def trace_inv(self) -> float:
        """tr((sΛ)⁻¹) = ||U⁻¹||²_F / s.

        ``dlantr`` reads the triangle in place, so the Frobenius norm
        costs neither the ``triu`` copy nor the squared temporary that
        ``sum(U_inv ** 2)`` allocates -- two p x p arrays that dominate
        this call once the inverse itself is in cache.
        """
        norm = _dlantr("F", self._U_inv, uplo="U", diag="N")
        return float(norm) ** 2 / self._scale


PrecisionFactor = SparseFactor | DenseFactor


_Factor = TypeVar("_Factor", bound="PrecisionFactor")


def scale_factor(factor: _Factor, scale: float) -> _Factor:
    """A factor of ``scale`` times what ``factor`` factors, sharing its
    factorization: ``decay`` and the Normal-Inverse-Gamma shape
    precision both need a scalar multiple of an existing factor, and
    neither should pay a refactorization for it.

    A shallow copy with the scale composed in, so the caller's factor is
    untouched and lazily built state (permutations, triangular views)
    carries over. Views made this way share one numeric factorization;
    :meth:`~CholmodSparseFactor.refactorize` on any of them invalidates
    the others.
    """
    if scale == 1.0:
        return factor
    scaled = copy.copy(factor)
    scaled._scale = factor._scale * scale
    return scaled


def create_sparse_factor(
    precision: csc_array, solver: Union[SparseSolver, None] = None
) -> SparseFactor:
    """Create a SparseFactor from a precision matrix.

    Only the observed block is factored; the trivial features (see
    :func:`trivial_columns`) are carried as a diagonal.
    """
    if solver is None:
        solver = globals()["solver"]
    if not issparse(precision):
        raise TypeError("precision must be a sparse array")
    if solver == SparseSolver.CHOLMOD:
        observed, trivial, diag = _partition(precision)
        block, pattern = _block_of(precision, observed)
        return CholmodSparseFactor(
            _factor=cholmod_cho_factor(csc_matrix(block)),
            _precision=precision,
            _observed=observed,
            _trivial=trivial,
            _trivial_diag=diag,
            _block=pattern,
        )
    else:
        return _superlu_factor(precision)
