"""Shared plain helpers for the test suite.

Kept separate from ``conftest.py`` (which holds fixtures) because the
module name ``conftest`` is ambiguous when pytest is invoked from the
repo root: ``benchmarks/conftest.py`` shadows it. ``tests`` is a
package, so import these as ``from tests._helpers import ...``.
"""

import numpy as np
import scipy.sparse as sp


def symmetrize(A) -> np.ndarray:
    """Mirror the upper triangle to the lower, producing a full symmetric
    matrix.

    Dense-mode routines store only the upper triangle (dsyrk convention);
    tests that need full-matrix operations (eigvalsh, matmul, element
    access in the lower triangle) should symmetrize first.
    """
    A = np.asarray(A)
    return np.triu(A) + np.triu(A, 1).T


def cov_inv_dense(est) -> np.ndarray:
    """Return ``est.cov_inv_`` as a full symmetric dense array.

    Dense fits store only the upper triangle of the precision for speed;
    reference computations need the full symmetric matrix.
    """
    if sp.issparse(est.cov_inv_):
        return est.cov_inv_.toarray()
    return symmetrize(est.cov_inv_)
