"""Shared plain helpers for the test suite.

Kept separate from ``conftest.py`` (which holds fixtures) because the
module name ``conftest`` is ambiguous when pytest is invoked from the
repo root: ``benchmarks/conftest.py`` shadows it. ``tests`` is a
package, so import these as ``from tests._helpers import ...``.
"""

import numpy as np
import scipy.sparse as sp

from bayesianbandits import NormalRegressor


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


def fit_dense(cls=NormalRegressor, d=40, rows=200, seed=0, **kwargs) -> tuple:
    """Fit an estimator on random dense data; returns ``(est, rng)``."""
    rng = np.random.default_rng(seed)
    if cls is NormalRegressor:
        kwargs.setdefault("alpha", 1.0)
        kwargs.setdefault("beta", 1.0)
    est = cls(**kwargs)
    X_train = rng.standard_normal((rows, d))
    y = X_train @ rng.standard_normal(d) + rng.standard_normal(rows)
    est.fit(X_train, y)
    return est, rng


def fit_sparse(d=400, rows=300, seed=0, **kwargs):
    """Fit a sparse NormalRegressor on random data; returns ``(est, rng)``."""
    rng = np.random.default_rng(seed)
    est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True, **kwargs)
    X_train = sp.csc_array(
        sp.random(rows, d, density=10 / d, random_state=1)  # type: ignore[call-arg]
    )
    est.fit(X_train, rng.standard_normal(rows))
    return est, rng
