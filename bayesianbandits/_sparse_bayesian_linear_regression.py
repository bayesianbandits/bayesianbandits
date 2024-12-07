import os
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Callable,
    Final,
    Literal,
    Protocol,
    Tuple,
    TypeGuard,
    TypeVar,
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
    eye,  # type: ignore
    issparse,  # type: ignore
)
from scipy.sparse.linalg import splu, spsolve, use_solver  # type: ignore
from scipy.stats import Covariance  # type: ignore
from scipy.stats._multivariate import _squeeze_output  # type: ignore

use_solver(useUmfpack=False)

try:
    from sksparse.cholmod import cholesky as cholmod_cholesky  # type: ignore

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
    from sksparse.cholmod import cholesky as cholmod_cholesky  # type: ignore

    # This helps Pylance understand that solver can be either SparseSolver.SUPERLU or SparseSolver.CHOLMOD.
    # For some reason, without this cast, it thinks solver is always SparseSolver.SUPERLU.
    solver: Literal[SparseSolver.SUPERLU] | Literal[SparseSolver.CHOLMOD] = cast(
        Literal[SparseSolver.SUPERLU, SparseSolver.CHOLMOD], solver
    )  # type: ignore

SM = TypeVar("SM", bound="Union[csc_matrix, csr_matrix]")
SM_T = TypeVar("SM_T", bound="Union[csc_matrix, csr_matrix]")


class SparseMatrixProtocol(Protocol[SM, SM_T]):
    shape: tuple[int, int]
    T: SM_T

    def dot(self, other: Union[SM, NDArray[np.float64]]) -> SM: ...
    def diagonal(self) -> NDArray[np.float64]: ...


class SparseCholeskyDecompositionOBject(Protocol):
    def solve_Lt(
        self, b: NDArray[np.float64], use_LDLt_decomposition: bool
    ) -> NDArray[np.float64]: ...
    def apply_Pt(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


class SuperLUObject(Protocol):
    L: SparseMatrixProtocol[csr_matrix, csc_matrix]
    U: SparseMatrixProtocol[csr_matrix, csc_matrix]
    perm_r: NDArray[np.int_]
    perm_c: NDArray[np.int_]


@dataclass
class LUObject:
    L: SparseMatrixProtocol[csr_matrix, csc_matrix]
    Pr: SparseMatrixProtocol[csc_matrix, csr_matrix]

    def solve_system(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self.Pr.T @ spsolve(self.L.T, x))


class CovViaSparsePrecision(Covariance):
    """Covariance class for sparse precision matrices.

    Parameters
    ----------
    prec : csc_array
        Sparse precision matrix.
    use_suitesparse : bool, optional
        Whether to use the sksparse.cholmod module for solving linear systems.
        If False, use scipy.sparse.linalg.splu instead. Default is True if
        sksparse.cholmod is installed, False otherwise.

    Raises
    ------
    ValueError
        If prec is not a sparse array.
    ValueError
        If prec is not symmetric.

    Notes
    -----
    This class is used to colorize samples from a standard normal distribution
    with a sparse precision matrix. The precision matrix is assumed to be
    positive-definite, because our Bayesian linear regression models always
    have positive-definite precision matrices (due to the prior).

    A coloring transform from a precision matrix is performed by first finding
    a factor W such that W W^T = P. Then, the colorizing transform is given by
    W^T X = Z, where Z is a standard normal random variable. The resulting
    random variable X has the precision matrix P.

    If using suitesparse, this is done by computing the Cholesky decomposition
    of P, which may or may not be permuted. CHOLMOD's solve_Lt solves W^T X = Z
    to color X with the permuted precision matrix P'. This P' is the precision
    matrix we would have gotten if we'd reordered the features of the training
    data to be in the order given by the permutation. Therefore, we need only
    apply the inverse permutation to X to get samples colored with our actual
    precision matrix P.

    If not using suitesparse, this is done by computing the LU decomposition
    using SuperLU. By telling SuperLU that the matrix is symmetric by setting
    options=dict(SymmetricMode=True), diag_pivot_thresh=0, and
    permc_spec="MMD_AT_PLUS_A", we can prevent partial pivoting and get a
    symmetric factorization L L^T = P', where P' is some permutation of P.
    L^T X = Z is solved to color X with the permuted precision matrix P', same
    as with suitesparse. The inverse permutation is applied to X to get samples
    colored with the original precision matrix P.
    """

    def __init__(self, prec: csc_array, solver: SparseSolver = solver):
        if not issparse(prec):
            raise TypeError("prec must be a sparse array")

        self.solver: Final[SparseSolver] = solver
        self._precision: Final[csc_array] = prec

        if self.solver == SparseSolver.CHOLMOD:
            self._W: LUObject | SparseCholeskyDecompositionOBject = cast(
                SparseCholeskyDecompositionOBject, cholmod_cholesky(csc_matrix(prec))
            )

        else:
            # Tells SuperLU that we have a symmetric matrix and we only want to pivot on the diagonal
            splu_ = cast(
                SuperLUObject,
                splu(
                    prec,
                    diag_pivot_thresh=0,
                    permc_spec="MMD_AT_PLUS_A",
                    options=dict(SymmetricMode=True),
                ),
            )
            if (splu_.perm_r != splu_.perm_c).any():
                raise ValueError("W must be symmetric")

            self._W = LUObject(  # type: ignore
                L=cast(
                    SparseMatrixProtocol[csr_matrix, csc_matrix],
                    splu_.L.dot(diags(np.sqrt(splu_.U.diagonal()))),  # type: ignore
                ),
                Pr=cast(
                    SparseMatrixProtocol[csc_matrix, csr_matrix],
                    csc_matrix(
                        (
                            np.ones(splu_.L.shape[0]),
                            (splu_.perm_r, np.arange(splu_.L.shape[0])),
                        )
                    ),
                ),
            )

        self._rank = cast(Tuple[int, ...], prec.shape)[
            -1
        ]  # must be full rank for cholesky
        self._shape = prec.shape
        self._allow_singular = False

    @property
    def colorize_solve(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        w: LUObject | SparseCholeskyDecompositionOBject = self._W
        if self._is_cholmod(w):
            return lambda x: w.apply_Pt(w.solve_Lt(x, False))
        else:
            assert isinstance(w, LUObject)
            return lambda x: w.solve_system(x)

    @cached_property
    def _covariance(self) -> csc_array:
        prec_shape: Tuple[int, ...] = cast(Tuple[int, ...], self._precision.shape)
        return cast(
            csc_array, spsolve(self._precision, eye(prec_shape[0], format="csc"))
        )

    def _whiten(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("Not implemented for sparse matrices")

    def _colorize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        samples = self.colorize_solve(x.T).T
        # csc_arrray and csc_matrix have different behavior, so CHOLMOD doesn't
        # auto-squeeze the output. We do it here.
        if samples.shape[0] == 1:
            return samples[0]
        return samples

    def _is_cholmod(
        self, obj: SparseCholeskyDecompositionOBject | LUObject
    ) -> TypeGuard[SparseCholeskyDecompositionOBject]:
        return self.solver == SparseSolver.CHOLMOD


def multivariate_normal_sample_from_sparse_covariance(
    mean: Union[csc_array, NDArray[np.float64], None],
    cov: Covariance,
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
    prec : array_like
        Precision matrix of the distribution. Ideally a csc sparse array.
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
    # Set the random state
    rng = np.random.default_rng(random_state)

    # Compute size from the shape of Q plus the size parameter
    gen_size = cast(Tuple[int, int], (size,) + (cov.shape[-1],))  # type: ignore

    # Sample Z from a standard multivariate normal distribution
    _Z = rng.standard_normal(gen_size)

    # Colorize Z
    _y: NDArray[np.float64] = cast(NDArray[np.float64], cov.colorize(_Z))  # type: ignore

    # Add the mean vector to each sample if provided
    if mean is not None:
        _y += mean  # type: ignore

    return cast(NDArray[np.float64], _y)


def multivariate_t_sample_from_sparse_covariance(
    loc: Union[csc_array, NDArray[np.float64], None],
    shape: Covariance,
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
    shape_inv_ : array_like
        Inverse of the shape matrix of the distribution. Ideally a csc sparse array.
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

    # Set the random state
    rng = np.random.default_rng(random_state)

    x = rng.chisquare(df, size) / df

    z = multivariate_normal_sample_from_sparse_covariance(
        mean=None, cov=shape, size=size, random_state=random_state
    )
    if loc is None:
        loc = np.zeros_like(z)
    samples = cast(NDArray[np.float64], loc + z / np.sqrt(x)[..., None])
    samples = _squeeze_output(samples)

    return samples
