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

    Parameters:
    - Q: Sparse precision matrix (inverse of the covariance matrix).
    - mu: Mean vector. Default is 0.
    - size: Number of samples to draw.

    Returns:
    - Samples from the multivariate normal distribution with shape (Q.shape[0], size).
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
    # Set the random state
    rng = np.random.default_rng(random_state)

    x = rng.chisquare(df, size) / df

    z = multivariate_normal_sample_from_sparse_precision(
        mean=None, prec=shape, size=size, random_state=random_state
    )

    samples = loc[:, np.newaxis] + z / np.sqrt(x)
    samples = _squeeze_output(samples)

    return samples
