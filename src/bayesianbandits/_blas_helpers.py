"""Thin wrappers around BLAS routines used by the Bayesian regression estimators."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.blas import dgemv, dsymv, dsyrk  # type: ignore[attr-defined]
from scipy.sparse import csc_array

__all__ = ["dgemv", "dsymv", "dsyrk", "update_precision_dense", "compute_eta_dense"]

_Array = Union[NDArray[Any], csc_array]


def update_precision_dense(
    alpha: float,
    X_weighted: _Array,
    prior_scaled: _Array,
) -> NDArray[np.float64]:
    """Compute ``alpha * X_weighted.T @ X_weighted + prior_scaled`` in-place.

    ``prior_scaled`` must be F-contiguous and is overwritten.  Only the
    **upper triangle** of the result is meaningful — ``dsyrk`` does not
    fill the lower triangle.  All downstream consumers (``dsymv``,
    ``cho_factor(lower=False)``, and ``dsyrk`` itself) read only the
    upper triangle, so no symmetrisation is needed.
    """
    return dsyrk(alpha, X_weighted, trans=1, beta=1.0, c=prior_scaled, overwrite_c=True)


def compute_eta_dense(
    prior_decay: float,
    cov_inv: _Array,
    coef: _Array,
    alpha: float,
    X: _Array,
    y_weighted: _Array,
) -> NDArray[np.float64]:
    """Compute ``prior_decay * cov_inv @ coef + alpha * X.T @ y_weighted``.

    Fuses two BLAS calls (``dsymv`` then ``dgemv``) into one buffer
    with no extra allocation.
    """
    eta = dsymv(prior_decay, cov_inv, coef)
    return dgemv(alpha, X, y_weighted, trans=1, beta=1.0, y=eta, overwrite_y=True)
