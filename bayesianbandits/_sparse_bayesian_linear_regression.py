import os
from enum import Enum
from functools import cached_property
from typing import Any, Tuple, Union, cast

import numpy as np
from attr import dataclass
from numpy.typing import NDArray
from scipy.sparse import csc_array, csc_matrix, csr_matrix, diags, eye, issparse
from scipy.sparse.linalg import splu, spsolve, use_solver
from scipy.stats import Covariance
from scipy.stats._multivariate import _squeeze_output

use_solver(useUmfpack=False)

try:
    from sksparse.cholmod import cholesky as cholmod_cholesky

    use_cholmod = True
except ImportError:
    use_cholmod = False

try:
    import scikits.umfpack as um

    use_umfpack = True
except ImportError:
    use_umfpack = False


if os.environ.get("BB_NO_SUITESPARSE", "0") == "1":
    use_cholmod = False
    use_umfpack = False


class SparseSolver(Enum):
    """Enum for sparse solvers."""

    SUPERLU = 0
    CHOLMOD = 1
    UMFPACK = 2


if use_cholmod:
    solver = SparseSolver.CHOLMOD
elif use_umfpack:
    solver = SparseSolver.UMFPACK
else:
    solver = SparseSolver.SUPERLU


@dataclass
class LUObject:
    L: csr_matrix
    Pr: csc_array


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

    def __init__(self, prec: csc_array, solver=solver):
        if not issparse(prec):
            raise ValueError("prec must be a sparse array")

        self.solver = solver

        self._precision = prec

        if self.solver == SparseSolver.CHOLMOD:
            self._W = cholmod_cholesky(csc_matrix(prec))

        elif self.solver == SparseSolver.UMFPACK:
            umc = um.UmfpackContext()
            # Tells umfpack that we have a symmetric matrix and we only want to pivot on the diagonal
            umc.control[um.UMFPACK_STRATEGY] = um.UMFPACK_STRATEGY_SYMMETRIC  # type: ignore
            umc.control[um.UMFPACK_SYM_PIVOT_TOLERANCE] = 0.0  # type: ignore
            umc.control[um.UMFPACK_SCALE] = um.UMFPACK_SCALE_NONE  # type: ignore

            L, U, P, Q, _, _ = umc.lu(csc_matrix(prec))
            if not (P == Q).all():
                raise ValueError("P != Q. Was a diagonal element of prec zero?")

            self._W = LUObject(
                L=L.dot(diags(np.sqrt(U.diagonal()))),
                Pr=csc_array((np.ones(L.shape[0]), (P, np.arange(L.shape[0])))),
            )

        else:
            # Tells SuperLU that we have a symmetric matrix and we only want to pivot on the diagonal
            splu_ = splu(
                prec,
                diag_pivot_thresh=0,
                permc_spec="MMD_AT_PLUS_A",
                options=dict(SymmetricMode=True),
            )
            if (splu_.perm_r != splu_.perm_c).any():
                raise ValueError("W must be symmetric")

            self._W = LUObject(
                L=splu_.L.dot(diags(np.sqrt(splu_.U.diagonal()))),
                Pr=csc_array(
                    (
                        np.ones(splu_.L.shape[0]),
                        (splu_.perm_r, np.arange(splu_.L.shape[0])),
                    )
                ),
            )

        self._rank = prec.shape[-1]  # must be full rank for cholesky
        self._shape = prec.shape
        self._allow_singular = False

    @property
    def colorize_solve(self) -> Any:
        if self.solver == SparseSolver.CHOLMOD:
            return lambda x: self._W.apply_Pt(  # type: ignore
                self._W.solve_Lt(x, False)  # type: ignore
            )

        elif self.solver == SparseSolver.UMFPACK:
            return lambda x: self._W.Pr @ um.spsolve(self._W.L.T, x)
        return lambda x: self._W.Pr.T @ spsolve(csc_array(self._W.L.T), x)

    @cached_property
    def _covariance(self) -> csc_array:
        return spsolve(self._precision, eye(self._precision.shape[0], format="csc"))

    def _whiten(self, x):
        raise NotImplementedError("Not implemented for sparse matrices")

    def _colorize(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        samples = self.colorize_solve(x.T).T
        # csc_arrray and csc_matrix have different behavior, so CHOLMOD doesn't
        # auto-squeeze the output. We do it here.
        if samples.shape[0] == 1:
            return samples[0]
        return samples


def multivariate_normal_sample_from_sparse_covariance(
    mean: Union[csc_array, NDArray[np.float_], None],
    cov: Covariance,
    size: int = 1,
    random_state: Union[int, None, np.random.Generator] = None,
) -> NDArray[np.float_]:
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
    gen_size = cast(Tuple[int, int], (size,) + (cov.shape[-1],))

    # Sample Z from a standard multivariate normal distribution
    Z = rng.standard_normal(gen_size)

    # Colorize Z
    Y = cast(NDArray[np.float_], cov.colorize(Z))

    # Add the mean vector to each sample if provided
    if mean is not None:
        Y += mean

    return cast(NDArray[np.float_], Y)


def multivariate_t_sample_from_sparse_covariance(
    loc: Union[csc_array, NDArray[np.float_], None],
    shape: Covariance,
    df: float = 1.0,
    size: int = 1,
    random_state: Union[int, None, np.random.Generator] = None,
) -> NDArray[np.float_]:
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
    samples = loc + z / np.sqrt(x)[..., None]
    samples = _squeeze_output(samples)

    return cast(NDArray[np.float_], samples)
