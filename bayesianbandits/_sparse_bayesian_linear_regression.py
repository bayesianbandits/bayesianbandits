from functools import cached_property
from typing import Union

import numpy as np
from scipy.sparse import csc_array, diags, eye, issparse
from scipy.sparse.linalg import factorized, splu, spsolve
from scipy.stats import Covariance
from scipy.stats._multivariate import _squeeze_output


class CovViaSparsePrecision(Covariance):
    def __init__(self, prec: csc_array):
        if not issparse(prec):
            raise ValueError("prec must be a sparse array")

        self._precision = prec
        # Compute the Covariance matrix from the precision matrix
        self._chol_P = sparse_cholesky(prec)
        self._log_pdet = 2 * np.log(self._chol_P.diagonal()).sum(axis=-1)
        self._rank = prec.shape[-1]  # must be full rank for cholesky
        self._shape = prec.shape
        self._allow_singular = False

    @cached_property
    def colorize_solve(self):
        return factorized(self._chol_P)

    @cached_property
    def _covariance(self):
        return spsolve(self._precision, eye(self._precision.shape[0], format="csc"))

    def _whiten(self, x):
        return x @ self._chol_P

    def _colorize(self, x):
        return self.colorize_solve(x.T).T


def sparse_cholesky(A: csc_array) -> csc_array:
    """Compute the Cholesky decomposition of a sparse, positive-definite matrix.

    In Bayesian linear regression, the precision matrix is always positive-definite.

    Parameters
    ----------
    A : csc_array
        Sparse, positive-definite matrix.

    Returns
    -------
    csc_array
        Cholesky decomposition of A (lower triangular matrix)

    Raises
    ------
    ValueError
        If A is not positive-definite.
    """
    # The input matrix A must be sparse and symmetric positive-definite.
    # Fortunately, any reasonable precision matrix in a Bayesian model is
    # sparse and symmetric positive-definite.
    n = A.shape[0]
    LU = splu(A, permc_spec="NATURAL", diag_pivot_thresh=0)  # sparse LU decomposition

    if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():
        return LU.L.dot(diags(np.sqrt(LU.U.diagonal())))
    else:
        raise ValueError("A is not positive-definite")


def multivariate_normal_sample_from_sparse_covariance(
    mean: Union[csc_array, np.ndarray, None],
    cov: Covariance,
    size: int = 1,
    random_state: Union[int, None, np.random.Generator] = None,
):
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
    if isinstance(size, int):
        gen_size = (size,) + (cov.shape[-1],)
    else:
        raise ValueError("size must be an int")

    # Sample Z from a standard multivariate normal distribution
    Z = rng.standard_normal(gen_size)

    # Colorize Z
    Y = cov.colorize(Z)

    # Add the mean vector to each sample if provided
    if mean is not None:
        Y += mean

    return Y


def multivariate_t_sample_from_sparse_covariance(
    loc: Union[csc_array, np.ndarray],
    shape: Covariance,
    df: float = 1.0,
    size: int = 1,
    random_state: Union[int, None, np.random.Generator] = None,
):
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

    samples = loc + z / np.sqrt(x)[..., None]
    samples = _squeeze_output(samples)

    return samples
