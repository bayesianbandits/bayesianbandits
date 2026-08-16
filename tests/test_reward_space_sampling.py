"""Tests for ``sample_reward_space``: joint posterior predictive draws in reward space.

The routes are exact reformulations, so the contract is an identity, not
a distribution: the QR square root must satisfy ``L Lᵀ = X Λ⁻¹ Xᵀ``, and
blocked mode must satisfy it block by block. That identity is checked
directly against a dense reference, which subsumes any moment or
goodness-of-fit check on the draws and cannot flake.

What that identity does *not* cover, and what is therefore tested
separately: applying the factor to normals, the per-estimator scaling
composed on top of it (NIG's chi-square mixing, the GLM link), and the
guards that raise.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from bayesianbandits import (
    BayesianGLM,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from tests._helpers import cov_inv_dense
from tests._helpers import fit_dense as _fit_dense
from tests._helpers import fit_sparse as _fit_sparse


def _reference_S(est, X_dense: np.ndarray) -> np.ndarray:
    return X_dense @ np.linalg.solve(cov_inv_dense(est), X_dense.T)


def _blocked_factor_covariance(L: np.ndarray) -> np.ndarray:
    """Assemble the full covariance implied by per-block factors."""
    n_blocks, k, _ = L.shape
    out = np.zeros((n_blocks * k, n_blocks * k))
    for b in range(n_blocks):
        out[b * k : (b + 1) * k, b * k : (b + 1) * k] = L[b] @ L[b].T
    return out


def _block_diagonal(S: np.ndarray, block_size: int) -> np.ndarray:
    out = np.zeros_like(S)
    for start in range(0, S.shape[0], block_size):
        stop = start + block_size
        out[start:stop, start:stop] = S[start:stop, start:stop]
    return out


def _relative_error(got: np.ndarray, want: np.ndarray) -> float:
    sd = np.sqrt(np.diag(want))
    return float(np.abs((got - want) / np.outer(sd, sd)).max())


# -- the exactness identity ---------------------------------------------------


class TestPredictiveCholesky:
    def test_factor_reproduces_the_predictive_covariance(self):
        est, rng = _fit_dense()
        X = rng.standard_normal((10, 40))
        mean, L = est._predictive_cholesky(X)
        assert _relative_error(L @ L.T, _reference_S(est, X)) < 1e-12
        assert_allclose(mean, X @ est.coef_)

    def test_blocked_factor_reproduces_the_diagonal_blocks(self):
        """Blocked mode is joint within a block and independent across
        blocks, which is exactly a block-diagonal covariance."""
        est, rng = _fit_dense()
        X = rng.standard_normal((12, 40))
        mean, L = est._predictive_cholesky(X, block_size=4)
        assert L.shape == (3, 4, 4)
        S_ref = _reference_S(est, X)
        got = _blocked_factor_covariance(L)
        assert _relative_error(got, _block_diagonal(S_ref, 4)) < 1e-12
        assert_allclose(mean, X @ est.coef_)

    def test_rank_deficient_rows_are_exact_and_padded(self):
        """Duplicate rows make the covariance singular. The QR route
        represents that exactly, where a Cholesky of the explicit
        covariance would fail, and pads R to a full ``(n, n)`` factor
        when there are more rows than features."""
        est, rng = _fit_dense(d=4, rows=80)
        X = rng.standard_normal((12, 4))
        X[6:9] = X[0:3]
        mean, L = est._predictive_cholesky(X)

        assert L.shape == (12, 12)
        assert_allclose(L, np.tril(L), atol=0)
        target = X @ np.linalg.inv(cov_inv_dense(est)) @ X.T
        assert_allclose(L @ L.T, target, atol=1e-10)
        assert_allclose(mean, X @ est.coef_)

    def test_duplicate_rows_draw_identically(self):
        """The covariance identity above implies it, but this is the
        user-visible consequence: repeated contexts share a draw."""
        est, rng = _fit_dense()
        X = np.tile(rng.standard_normal(40), (4, 1))
        draws = est.sample_reward_space(X, size=3)
        assert draws.shape == (3, 4)
        assert np.abs(draws - draws[:, [0]]).max() < 1e-8 * np.abs(draws).max()

    def test_sparse_factor_matches_the_dense_reference(self, sparse_solver):
        est, rng = _fit_sparse()
        X = sp.csc_array(
            sp.random(10, 400, density=0.05, random_state=2)  # type: ignore[call-arg]
        )
        mean, L = est._predictive_cholesky(X)
        assert _relative_error(L @ L.T, _reference_S(est, X.toarray())) < 1e-9
        assert_allclose(mean, np.asarray(X @ est.coef_).ravel())

    def test_sparse_after_decay_solves_through_the_scaling(self, sparse_solver):
        """``decay`` wraps the factor in ``ScaledSparseFactor``."""
        est, rng = _fit_sparse()
        X = sp.csc_array(
            sp.random(6, 400, density=0.05, random_state=3)  # type: ignore[call-arg]
        )
        est.decay(X.toarray(), decay_rate=0.9)
        _, L = est._predictive_cholesky(X)
        assert _relative_error(L @ L.T, _reference_S(est, X.toarray())) < 1e-9

    def test_blocked_sparse_chunks_give_the_same_factor(self, sparse_solver):
        """Blocked mode chunks the dense half-solve scratch; a chunked run
        must produce the same factors as an unchunked one."""
        from unittest import mock

        from bayesianbandits import _estimators

        est, _ = _fit_sparse(d=400, rows=300)
        X = sp.csc_array(
            sp.random(24, 400, density=0.05, random_state=7)  # type: ignore[call-arg]
        )
        with mock.patch.object(_estimators, "_MARGINAL_SD_BLOCK_ELEMS", 8):
            _, chunked = est._predictive_cholesky(X, 4)
        _, whole = est._predictive_cholesky(X, 4)
        assert chunked.shape == (6, 4, 4)
        assert_allclose(chunked, whole, atol=1e-10)

    def test_sparse_model_accepts_dense_X(self, sparse_solver):
        """``accept_sparse`` only permits sparse input; a sparse model can
        legitimately receive a dense ndarray."""
        est, rng = _fit_sparse()
        draws = est.sample_reward_space(rng.standard_normal((4, 400)), size=50)
        assert draws.shape == (50, 4)
        assert np.isfinite(draws).all()

    def test_zero_rows_yield_exact_mean(self):
        est, _ = _fit_dense()
        draws = est.sample_reward_space(np.zeros((3, 40)), size=5)
        assert np.all(draws == 0.0)

    def test_unfitted_model_uses_the_prior_predictive(self):
        est = NormalRegressor(alpha=2.0, beta=1.0, random_state=0)
        X = np.random.default_rng(0).standard_normal((5, 7))
        _, L = est._predictive_cholesky(est._validated_for_sampling(X))
        assert_allclose(L @ L.T, X @ X.T / 2.0, atol=1e-10)

    def test_block_size_must_divide_rows_and_be_positive(self):
        est, rng = _fit_dense()
        X = rng.standard_normal((10, 40))
        with pytest.raises(ValueError, match="divide"):
            est._predictive_cholesky(X, block_size=4)
        with pytest.raises(ValueError, match="positive"):
            est._predictive_cholesky(X, block_size=-2)


# -- what the identity does not cover -----------------------------------------


class TestComposition:
    """Per-estimator scaling applied on top of the shared factor."""

    def test_nig_scales_the_shape_matrix_by_b_over_a(self):
        """``_predictive_cholesky`` returns the unscaled shape matrix;
        ``sample_reward_space`` applies ``b/a`` itself, so a dropped
        scale factor shows up only here."""
        est, rng = _fit_dense(cls=NormalInverseGammaRegressor, random_state=123)
        X = rng.standard_normal((6, 40))
        _, L = est._predictive_cholesky(X)
        assert_allclose(L @ L.T, _reference_S(est, X), atol=1e-8)

        a, b = float(est.a_), float(est.b_)
        draws = est.sample_reward_space(X, 40_000)
        want = (b / (a - 1.0)) * _reference_S(est, X)
        assert_allclose(np.cov(draws.T), want, rtol=0.1, atol=0.05 * np.abs(want).max())

    @pytest.mark.parametrize("block_size", [None, 3])
    def test_glm_applies_the_link_to_eta_draws(self, block_size):
        """The GLM draws the linear predictor through the shared factor
        and applies the inverse link elementwise. With a log link the eta
        draws come back exactly under ``log``."""
        rng = np.random.default_rng(0)
        est = BayesianGLM(alpha=1.0, link="log", random_state=123)
        est.fit(rng.standard_normal((100, 10)) * 0.3, rng.poisson(1.0, 100) * 1.0)
        X = rng.standard_normal((6, 10)) * 0.3

        draws = est.sample_reward_space(X, 40_000, block_size=block_size)
        assert (draws > 0.0).all()  # the link's range
        eta = np.log(draws)
        want = _reference_S(est, X)
        if block_size is not None:
            want = _block_diagonal(want, block_size)
        sd = np.sqrt(np.diag(want))
        z = np.abs(eta.mean(axis=0) - X @ est.coef_) / (sd / np.sqrt(40_000))
        assert z.max() < 5.0
        assert_allclose(np.cov(eta.T), want, atol=0.05 * np.abs(want).max())

    def test_nig_blocks_mix_independent_chi_squares(self):
        """Within a block the draws share one chi-square; across blocks
        the mixing variables are independent. A shared-g implementation
        would silently couple contexts, and no factor check sees it."""
        est, rng = _fit_dense(cls=NormalInverseGammaRegressor, random_state=3)
        X = np.tile(rng.standard_normal(40), (4, 1))
        draws = est.sample_reward_space(X, 20_000, block_size=2)
        log_sq = np.log((draws - (X @ est.coef_)) ** 2 + 1e-300)
        assert np.corrcoef(log_sq[:, 0], log_sq[:, 1])[0, 1] > 0.9
        assert abs(np.corrcoef(log_sq[:, 0], log_sq[:, 2])[0, 1]) < 0.05
