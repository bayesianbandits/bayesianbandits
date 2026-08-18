"""Unit check for ``lower_predictive_sqrt``, including the zero padding a
rank-deficient ``B`` needs. The rest of the module is layout plumbing that
the estimator-level identity checks cover.
"""

import numpy as np
import pytest

from bayesianbandits._blas_helpers import lower_predictive_sqrt


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
