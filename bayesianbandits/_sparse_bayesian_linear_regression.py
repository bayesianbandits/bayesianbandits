import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats._multivariate import _squeeze_output


def multivariate_normal_sample_from_sparse_precision(
    mean, prec, size=1, random_state=None
):
    """
    Sample from a multivariate normal distribution with mean mu (default 0)
    and sparse precision matrix Q.

    Parameters
    ----------
    mean : array_like, optional
        Mean of the distribution. Default is 0.
    prec : array_like
        Precision matrix of the distribution. Ideally a csc sparse matrix.
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

    Examples
    --------
    >>> mean = [1, 2]
    >>> cov = [[1, 0], [0, 1]]
    >>> x = multivariate_normal_sample_from_sparse_precision(mean, cov, 1000)
    >>> x.shape
    (1000, 2)
    """
    # Ensure Q is in CSC format for efficient solving
    Q = csc_matrix(prec)

    # Set the random state
    rng = np.random.default_rng(random_state)

    # Compute size from the shape of Q plus the size parameter
    if isinstance(size, int):
        size = (Q.shape[0], size)
    elif isinstance(size, tuple):
        size = (Q.shape[0],) + size
    else:
        raise ValueError("size must be an int or tuple")

    # Sample Z from a standard multivariate normal distribution
    Z = rng.standard_normal(size)

    # Solve the linear system QX = Z
    X = spsolve(Q, Z)

    # Solve the linear system QY = X to get Y = LZ
    # Transpose so the output has the same shape as multivariate_normal.rvs
    Y = spsolve(Q, X).T

    # Add the mean vector to each sample if provided
    if mean is not None:
        Y += mean

    return Y


def multivariate_t_sample_from_sparse_precision(
    loc, shape, df=1, size=1, random_state=None
):
    """
    Sample from a multivariate t distribution with mean loc, shape matrix
    shape, and degrees of freedom df.

    Parameters
    ----------
    loc : array_like
        Mean of the distribution.
    shape : array_like
        Shape matrix of the distribution. Ideally a csc sparse matrix.
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

    Examples
    --------
    >>> loc = np.array([1, 2])
    >>> shape = np.array([[1, 0], [0, 1]])
    >>> x = multivariate_t_sample_from_sparse_precision(loc, shape, size=1000)
    >>> x.shape
    (1000, 2)
    """

    # Set the random state
    rng = np.random.default_rng(random_state)

    x = rng.chisquare(df, size) / df

    z = multivariate_normal_sample_from_sparse_precision(
        mean=None, prec=shape, size=size, random_state=random_state
    )

    samples = loc + z / np.sqrt(x)[:, np.newaxis]
    samples = _squeeze_output(samples)

    return samples
