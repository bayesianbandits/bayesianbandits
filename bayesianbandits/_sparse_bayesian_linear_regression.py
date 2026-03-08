import os
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import (  # type: ignore  # type: ignore
    csc_array,
    csc_matrix,
    csr_matrix,
    diags,  # type: ignore
    issparse,  # type: ignore
)
from scipy.sparse.linalg import splu, spsolve, use_solver  # type: ignore

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

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return self._factor.solve(b)

    def colorize(self, z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Solve L^T x = z, undo permutation."""
        inv_perm = np.argsort(self._factor.perm)
        return self._factor.solve(z, system="Lt")[inv_perm]

    def logdet(self) -> float:
        """Log-determinant of the factored matrix via CHOLMOD."""
        return float(self._factor.logdet())

    def get_L_csc(self) -> csc_array:
        """Return the lower triangular Cholesky factor as CSC.

        The factor satisfies PΛP' = LL' where P is the fill-reducing
        permutation.  The returned L includes the permutation implicitly
        (rows/cols are in permuted order).
        """
        L = self._factor.L  # property, not method
        if not isinstance(L, csc_array):
            return csc_array(L)
        return L


@dataclass
class SuperLUSparseFactor:
    """Wraps SuperLU decomposition for solving and sampling."""

    _L: csr_matrix
    _Pr: csc_matrix
    _precision: csc_array

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], spsolve(self._precision, b))

    def colorize(self, z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Solve L^T x = z, undo permutation."""
        return cast(NDArray[np.float64], self._Pr.T @ spsolve(self._L.T, z))

    def logdet(self) -> float:
        """Log-determinant of the factored matrix.

        L already has D folded in (L = L_splu @ diag(sqrt(D))), so
        |P| = |L|^2  =>  log|P| = 2 * sum(log|diag(L)|).
        """
        return float(2.0 * np.sum(np.log(np.abs(self._L.diagonal()))))

    def get_L_csc(self) -> csc_array:
        """Return the lower triangular factor as CSC.

        The SuperLU factor already has D folded in: L = L_splu @ diag(√D).
        Rows are in permuted order (matching _Pr).
        """
        L = self._L
        if not isinstance(L, csc_array):
            return csc_array(L)
        return L


ConcreteFactor = CholmodSparseFactor | SuperLUSparseFactor


@dataclass
class ScaledSparseFactor:
    """A factor scaled by a scalar. Avoids refactorizing for scalar precision changes.

    If L L^T = P, then (s*P) has factor (√s * L).
    - solve(s*P, b) = (1/s) * solve(P, b)
    - colorize(s*P, z) = (1/√s) * colorize(P, z)
    """

    _inner: ConcreteFactor
    _scale: float
    _precision: csc_array  # the inner's precision; only used for .shape

    def solve(self, b: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self._inner.solve(b) / self._scale)

    def colorize(self, z: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self._inner.colorize(z) / np.sqrt(self._scale))

    def logdet(self) -> float:
        """Log-determinant: log|s*P| = p*log(s) + log|P|."""
        p = cast(tuple[int, int], self._precision.shape)[0]
        return float(p * np.log(self._scale) + self._inner.logdet())

    def get_L_csc(self) -> csc_array:
        """Return the scaled lower triangular factor as CSC.

        If inner factor has LL' = P, then scaled factor has (√s·L)(√s·L)' = s·P.
        """
        L = self._inner.get_L_csc()
        return csc_array(L * np.sqrt(self._scale))


SparseFactor = CholmodSparseFactor | SuperLUSparseFactor | ScaledSparseFactor


def scale_factor(factor: SparseFactor, scale: float) -> SparseFactor:
    """Scale a factor by a scalar, composing rather than nesting."""
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
        Pr = csc_matrix(
            (
                np.ones(splu_.L.shape[0]),
                (splu_.perm_r, np.arange(splu_.L.shape[0])),
            )
        )
        return SuperLUSparseFactor(_L=L, _Pr=Pr, _precision=precision)


def multivariate_normal_sample_from_sparse_precision(
    mean: Union[csc_array, NDArray[np.float64], None],
    factor: SparseFactor,
    size: int = 1,
    random_state: Union[int, None, np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Sample from a multivariate normal distribution with mean mu (default 0)
    and sparse precision matrix Q.

    Parameters
    ----------
    mean : array_like, optional
        Mean of the distribution. Default is 0.
    factor : SparseFactor
        Factored precision matrix.
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
    n_features = cast(tuple[int, int], factor._precision.shape)[0]
    _Z = rng.standard_normal((size, n_features))
    samples = factor.colorize(_Z.T).T
    if samples.shape[0] == 1:
        samples = samples[0]
    if mean is not None:
        samples = samples + mean
    return samples


def multivariate_t_sample_from_sparse_precision(
    loc: Union[csc_array, NDArray[np.float64], None],
    factor: SparseFactor,
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
    factor : SparseFactor
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

    z = multivariate_normal_sample_from_sparse_precision(
        mean=None, factor=factor, size=size, random_state=random_state
    )
    if loc is None:
        loc = np.zeros_like(z)
    samples = cast(NDArray[np.float64], loc + z / np.sqrt(x)[..., None])
    from scipy.stats._multivariate import _squeeze_output  # type: ignore

    samples = _squeeze_output(samples)

    return samples
