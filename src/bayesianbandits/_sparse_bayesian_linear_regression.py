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


def centered_predictive_draws(
    factor: PrecisionFactor,
    X: Any,
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Centered joint draws of ``X @ w`` for ``w ~ N(0, Λ^{-1})``.

    Samples in predictive space rather than weight space: one
    half-solve against the cached factor gives ``B = M^T X^T``
    (``M`` being the colorize operator, so ``B^T B = X Λ^{-1} X^T``),
    an economy SVD of ``B`` yields a stable square root of the
    n_rows × n_rows predictive covariance, and each draw then costs
    O(n_rows · min(n_rows, p)) instead of O(p²). The joint law across
    rows is exactly that of weight-space sampling; only the random
    stream differs. Worthwhile whenever ``size`` exceeds ``n_rows``.

    Returns an array of shape ``(size, n_rows)``.
    """
    Xt = X.T.toarray() if issparse(X) else np.asarray(X.T, dtype=np.float64)
    B = np.asarray(factor.half_solve(Xt), dtype=np.float64)
    # B may be (p,) for a single row; keep columns = rows of X
    if B.ndim == 1:
        B = B[:, None]
    _, s, vt = np.linalg.svd(B, full_matrices=False)
    z = rng.standard_normal((size, s.shape[0]))
    return cast(NDArray[np.float64], (z * s) @ vt)


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
