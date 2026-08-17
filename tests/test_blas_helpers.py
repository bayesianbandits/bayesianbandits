"""Unit checks for ``lower_predictive_sqrt`` (including the zero padding a
rank-deficient ``B`` needs) and ``affine_lower_factor``. The rest of the
module is layout plumbing that the estimator-level identity checks cover.
"""

import numpy as np
import pytest

from bayesianbandits._blas_helpers import (
    affine_lower_factor,
    lower_predictive_sqrt,
    standard_normal_f,
)


@pytest.mark.parametrize(
    ("n_features", "n_rows"),
    [(10, 1), (10, 4), (10, 10), (4, 10), (1, 3)],
    ids=["single-row", "tall", "square", "wide", "degenerate"],
)
def test_reproduces_the_gram_matrix(n_features: int, n_rows: int):
    """``L Lᵀ = Bᵀ B`` exactly, including when ``B`` has fewer rows than
    columns and ``L`` must be zero-padded."""
    rng = np.random.default_rng(0)
    B = rng.standard_normal((n_features, n_rows))
    L = lower_predictive_sqrt(B.copy(), n_rows)
    assert L.shape == (n_rows, n_rows)
    np.testing.assert_allclose(L @ L.T, B.T @ B, atol=1e-12)
    np.testing.assert_allclose(L, np.tril(L), atol=1e-15)


def test_exact_for_duplicated_rows():
    """Repeated prediction rows make ``Bᵀ B`` singular; the QR route
    represents that exactly where a Cholesky would fail."""
    rng = np.random.default_rng(2)
    col = rng.standard_normal((6, 1))
    B = np.hstack([col, col, rng.standard_normal((6, 1))])
    L = lower_predictive_sqrt(B.copy(), 3)
    np.testing.assert_allclose(L @ L.T, B.T @ B, atol=1e-12)


def test_affine_lower_factor_matches_the_unfused_expression():
    """The draw step: ``mean + z @ Lᵀ``, fused into one dgemm."""
    rng = np.random.default_rng(6)
    mean = rng.standard_normal(4)
    z = standard_normal_f(rng, 9, 4)
    L = np.tril(rng.standard_normal((4, 4)))
    np.testing.assert_allclose(
        affine_lower_factor(mean, z, L), mean + z @ L.T, atol=1e-13
    )
