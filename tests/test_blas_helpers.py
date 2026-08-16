"""Direct tests for the raw BLAS/LAPACK wrappers.

The reward-space path routes every step through ``scipy.linalg`` rather
than ``numpy`` (see ``lower_predictive_sqrt``), so these check that the
hand-rolled calls still agree with the obvious ``numpy`` spelling.
"""

import numpy as np
import pytest

from bayesianbandits._blas_helpers import (
    affine_lower_factor,
    dense_matvec,
    lower_predictive_sqrt,
    standard_normal_f,
)


class TestLowerPredictiveSqrt:
    @pytest.mark.parametrize(
        ("n_features", "n_rows"),
        [(10, 1), (10, 4), (10, 10), (4, 10), (1, 3)],
        ids=["single-row", "tall", "square", "wide", "degenerate"],
    )
    def test_reproduces_the_gram_matrix(self, n_features: int, n_rows: int):
        """``L Lᵀ = Bᵀ B`` exactly, including when ``B`` has fewer rows
        than columns and ``L`` must be zero-padded."""
        rng = np.random.default_rng(0)
        B = rng.standard_normal((n_features, n_rows))
        L = lower_predictive_sqrt(B.copy(), n_rows)
        assert L.shape == (n_rows, n_rows)
        np.testing.assert_allclose(L @ L.T, B.T @ B, atol=1e-12)

    def test_is_lower_triangular_with_nonnegative_diagonal(self):
        rng = np.random.default_rng(1)
        B = rng.standard_normal((8, 5))
        L = lower_predictive_sqrt(B.copy(), 5)
        np.testing.assert_allclose(L, np.tril(L), atol=1e-15)
        assert (np.diagonal(L) >= 0.0).all()

    def test_exact_for_duplicated_rows(self):
        """Repeated prediction rows make ``Bᵀ B`` singular; the QR route
        represents that exactly where a Cholesky would fail."""
        rng = np.random.default_rng(2)
        col = rng.standard_normal((6, 1))
        B = np.hstack([col, col, rng.standard_normal((6, 1))])
        L = lower_predictive_sqrt(B.copy(), 3)
        np.testing.assert_allclose(L @ L.T, B.T @ B, atol=1e-12)

    def test_overwrites_its_input(self):
        """Documented contract: ``B`` is scratch space for ``dgeqrf``."""
        rng = np.random.default_rng(3)
        B = np.asfortranarray(rng.standard_normal((6, 3)))
        before = B.copy()
        lower_predictive_sqrt(B, 3)
        assert not np.array_equal(B, before)


class TestDenseMatvec:
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_matches_numpy_for_either_memory_order(self, order: str):
        """C-contiguous input is passed transposed with ``trans=1`` to
        avoid a copy, so both layouts need covering."""
        rng = np.random.default_rng(4)
        A = np.asarray(rng.standard_normal((7, 5)), order=order)
        x = rng.standard_normal(5)
        np.testing.assert_allclose(dense_matvec(A, x), A @ x, atol=1e-13)

    def test_accepts_a_column_vector(self):
        rng = np.random.default_rng(5)
        A = rng.standard_normal((4, 3))
        x = rng.standard_normal((3, 1))
        np.testing.assert_allclose(dense_matvec(A, x), (A @ x).ravel(), atol=1e-13)


class TestAffineLowerFactor:
    def test_matches_the_unfused_expression(self):
        rng = np.random.default_rng(6)
        mean = rng.standard_normal(4)
        z = standard_normal_f(rng, 9, 4)
        L = np.tril(rng.standard_normal((4, 4)))
        np.testing.assert_allclose(
            affine_lower_factor(mean, z, L), mean + z @ L.T, atol=1e-13
        )

    def test_draws_are_fortran_ordered(self):
        """``dgemm`` takes both operands without copying only if the
        draw buffer is already Fortran-ordered."""
        z = standard_normal_f(np.random.default_rng(7), 5, 3)
        assert z.shape == (5, 3)
        assert z.flags.f_contiguous
