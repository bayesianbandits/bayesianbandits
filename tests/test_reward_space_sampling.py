"""Tests for ``sample_reward_space``.

The contract is an identity, checked against a dense reference:
``L Lᵀ = X Λ⁻¹ Xᵀ``, block by block in blocked mode. Tested separately
is what the identity does not cover: the per-estimator scaling on top
of the factor (NIG's chi-square mixing, the GLM link) and the guards.
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
    @pytest.mark.parametrize("block_size", [None, 4])
    def test_factor_reproduces_the_predictive_covariance(self, block_size):
        """Blocked mode is joint within a block and independent across
        blocks, which is exactly a block-diagonal covariance."""
        est, rng = _fit_dense()
        X = rng.standard_normal((12, 40))
        mean, draw = est._predictive_cholesky(X, block_size)
        L = draw.L
        assert draw.M.shape[1] == 0
        want = _reference_S(est, X)
        if block_size is None:
            got = L @ L.T
        else:
            assert L.shape == (3, 4, 4)
            got, want = _blocked_factor_covariance(L), _block_diagonal(want, 4)
        assert _relative_error(got, want) < 1e-12
        assert_allclose(mean, X @ est.coef_)

    def test_rank_deficient_rows_are_exact_and_padded(self):
        """Duplicate rows make the covariance singular; the QR route is
        exact where a Cholesky would fail, pads R to ``(n, n)`` when rows
        outnumber features, and repeated contexts share a draw."""
        est, rng = _fit_dense(d=4, rows=80)
        X = rng.standard_normal((12, 4))
        X[6:9] = X[0:3]
        mean, draw = est._predictive_cholesky(X)
        L = draw.L
        assert L.shape == (12, 12)
        assert_allclose(L, np.tril(L), atol=0)
        target = X @ np.linalg.inv(cov_inv_dense(est)) @ X.T
        assert_allclose(L @ L.T, target, atol=1e-10)
        assert_allclose(mean, X @ est.coef_)
        draws = est.sample_reward_space(X, size=3)
        assert_allclose(draws[:, 6:9], draws[:, 0:3], rtol=1e-8)

    def test_sparse_factor_matches_the_dense_reference(self, sparse_solver):
        """Through the sparse factor: sparse ``X``, dense ``X`` handed to
        a sparse model, and after ``decay`` (which scales the cached
        factor rather than refactoring)."""
        est, rng = _fit_sparse()
        X = sp.csc_array(
            sp.random(10, 400, density=0.05, random_state=2)  # type: ignore[call-arg]
        )
        mean, draw = est._predictive_cholesky(X)
        L = draw.L
        assert _relative_error(L @ L.T, _reference_S(est, X.toarray())) < 1e-9
        assert_allclose(mean, np.asarray(X @ est.coef_).ravel())
        _, dense = est._predictive_cholesky(X.toarray())
        assert dense.M.shape[1] == 0
        assert_allclose(dense.L, L, atol=1e-10)

        est.decay(X.toarray(), decay_rate=0.9)
        _, draw = est._predictive_cholesky(X)
        assert _relative_error(draw.L @ draw.L.T, _reference_S(est, X.toarray())) < 1e-9

    def test_degenerate_inputs(self):
        """All-zero rows draw exactly the mean; ``size=0`` returns an
        empty ``(0, n_rows)`` array on both paths."""
        est, rng = _fit_dense()
        assert np.all(est.sample_reward_space(np.zeros((3, 40)), size=5) == 0.0)
        X = rng.standard_normal((6, 40))
        for block_size in (None, 3):
            assert est.sample_reward_space(X, 0, block_size=block_size).shape == (0, 6)

    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (NormalRegressor, dict(alpha=1.0, beta=1.0)),
            (NormalInverseGammaRegressor, {}),
            (BayesianGLM, {}),
        ],
    )
    def test_wrong_feature_count_raises(self, cls, kwargs):
        """A too-narrow or too-wide ``X`` raises from every sampling entry
        point (the sparse weight-space path would otherwise read a narrow
        ``X`` as if its missing features were zero)."""
        rng = np.random.default_rng(0)
        X_train, y = rng.standard_normal((30, 8)), rng.standard_normal(30)
        for sparse in (False, True):
            est = cls(sparse=sparse, random_state=0, **kwargs)
            est.fit(sp.csc_array(X_train) if sparse else X_train, y)
            for width in (6, 10):
                Xq = rng.standard_normal((3, width))
                for method in (est.sample, est.sample_reward_space):
                    with pytest.raises(ValueError, match="expecting 8 features"):
                        method(sp.csc_array(Xq) if sparse else Xq)

    def test_unfitted_model_uses_the_prior_predictive(self):
        est = NormalRegressor(alpha=2.0, beta=1.0, random_state=0)
        X = np.random.default_rng(0).standard_normal((5, 7))
        _, draw = est._predictive_cholesky(est._validated_for_sampling(X))
        L = draw.L
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
        _, draw = est._predictive_cholesky(X)
        assert_allclose(draw.L @ draw.L.T, _reference_S(est, X), atol=1e-8)

        a, b = float(est.a_), float(est.b_)
        draws = est.sample_reward_space(X, 40_000)
        want = (b / (a - 1.0)) * _reference_S(est, X)
        assert_allclose(np.cov(draws.T), want, rtol=0.1, atol=0.05 * np.abs(want).max())

    @pytest.mark.parametrize("block_size", [None, 3])
    def test_glm_applies_the_link_to_eta_draws(self, block_size):
        """The GLM draws eta through the shared draw and applies the
        inverse link elementwise: with a log link, ``log`` recovers the
        eta draws exactly."""
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


# -- never-observed features stay a sparse map, not rows of the QR -----------


def _fit_cold(cls=NormalRegressor, p=300, m=20, rows=60, seed=0, **kwargs):
    """A sparse estimator whose training data touched only the first ``m``
    of ``p`` features, so the factor partitions ``p - m`` as never-observed."""
    rng = np.random.default_rng(seed)
    X_train = sp.csc_array(
        sp.hstack(
            [
                sp.random(rows, m, density=0.4, random_state=seed),  # type: ignore[call-arg]
                sp.csc_array((rows, p - m)),
            ]
        )
    )
    if cls is NormalRegressor:
        kwargs = dict(alpha=0.7, beta=1.0, **kwargs)
    est = cls(sparse=True, random_state=seed, **kwargs)
    est.fit(X_train, rng.standard_normal(rows))
    assert est._precision_factor.n_factored == m
    return est, rng


def _query_touching_unobserved(rng, n_rows, p=300, m=20, nnz=6):
    """Rows whose entries fall on both sides of the partition; several
    rows share never-observed features so ``M`` must share their normal."""
    X = sp.random(n_rows, p, density=nnz / p, random_state=rng).tolil()  # type: ignore[call-arg]
    for i in range(0, n_rows, 2):
        X[i, m + (i % 7)] = 1.0 + i / n_rows  # shared never-observed feature
        X[i, i % m] = -0.5  # and an observed one
    return sp.csc_array(X)


def _two_part_covariance(draw):
    L, M = draw.L, draw.M
    S = L @ L.T if L.ndim == 2 else _blocked_factor_covariance(L)
    return S + (M @ M.T).toarray()


class TestNeverObservedFeatures:
    @pytest.mark.parametrize("block_size", [None, 4])
    def test_split_square_root_reproduces_the_covariance(
        self, sparse_solver, block_size
    ):
        """``L Lᵀ + M Mᵀ = X Λ⁻¹ Xᵀ`` with ``L`` from the observed block
        only and ``M`` exactly the never-observed part
        ``X_T diag(1/Λ_jj) X_Tᵀ``, one column per distinct feature (per
        block); chunking the blocked half-solve changes nothing."""
        from unittest import mock

        from bayesianbandits import _estimators

        est, rng = _fit_cold()
        X = _query_touching_unobserved(rng, 24)
        Xd = X.toarray()
        want = _reference_S(est, Xd)
        Xt, diag_t = Xd[:, 20:], est.cov_inv_.diagonal()[20:]
        want_M = (Xt / diag_t) @ Xt.T
        n_pairs = np.count_nonzero(np.abs(Xt).sum(axis=0))
        if block_size is not None:
            want = _block_diagonal(want, block_size)
            want_M = _block_diagonal(want_M, block_size)
            n_pairs = sum(
                np.count_nonzero(np.abs(Xt[b : b + block_size]).sum(axis=0))
                for b in range(0, 24, block_size)
            )
        mean, draw = est._predictive_cholesky(X, block_size)
        M = draw.M
        assert M.shape == (24, n_pairs)
        assert _relative_error(_two_part_covariance(draw), want) < 1e-9
        assert_allclose((M @ M.T).toarray(), want_M, atol=1e-12)
        assert_allclose(mean, Xd @ est.coef_)
        if block_size is not None:
            with mock.patch.object(_estimators, "_MARGINAL_SD_BLOCK_ELEMS", 8 * 20):
                _, chunked = est._predictive_cholesky(X, block_size)
            assert_allclose(chunked.L, draw.L, atol=1e-10)
            assert_allclose((chunked.M @ chunked.M.T).toarray(), want_M, atol=1e-12)

    @pytest.mark.parametrize("block_size", [None, 3])
    def test_draws_scale_both_parts(self, block_size):
        """The draws add the two parts, and NIG's per-draw (per
        draw-and-block) chi-square scale multiplies the never-observed
        part too."""
        est, rng = _fit_cold(NormalInverseGammaRegressor)
        X = _query_touching_unobserved(rng, 6)
        draws = est.sample_reward_space(X, 60_000, block_size=block_size)
        a, b = float(est.a_), float(est.b_)
        want = (b / (a - 1.0)) * _reference_S(est, X.toarray())
        if block_size is not None:
            want = _block_diagonal(want, block_size)
        assert_allclose(np.cov(draws.T), want, rtol=0.1, atol=0.05 * np.abs(want).max())
        _, draw = est._predictive_cholesky(X, block_size)
        M = draw.M
        assert (M @ M.T).toarray().max() > 0.1 * want.max()  # part is not negligible

    def test_row_route_carries_the_split_root(self):
        """``sample`` at large ``size`` takes the row-side reduction, which
        holds the same two parts."""
        from unittest import mock

        from bayesianbandits import _estimators
        from bayesianbandits import _support_covariance as sc

        est, rng = _fit_cold()
        X = _query_touching_unobserved(rng, 4)
        with mock.patch.object(sc, "build", lambda *a, **k: None):
            draw = _estimators.build_joint_reduction(
                est._precision_factor, X, 300, 40_000, True
            )
        assert isinstance(draw, _estimators._RowSpaceDraw)
        want = _reference_S(est, X.toarray())
        assert _relative_error(_two_part_covariance(draw), want) < 1e-9

    def test_scratch_is_bounded_by_the_observed_block(self):
        """Many rows each touching many never-observed features: the dense
        scratch is ``n_factored`` by the rows, not ``n_factored + t``, on
        the factor's return and end to end (``tracemalloc``)."""
        import tracemalloc

        from bayesianbandits._estimators import _marginal_predictive_sd

        p, m = 20_000, 20
        est, rng = _fit_cold(p=p, m=m, rows=40, seed=9)
        n_rows = 1000
        X = sp.csc_array(
            sp.random(n_rows, p, density=30 / p, random_state=9)  # type: ignore[call-arg]
        )
        t = np.count_nonzero(np.diff(X.indptr)[m:])
        assert t > 5 * n_rows  # the (t, n_rows) dense block would be > 40 MB
        factor = est._precision_factor
        half = factor.half_solve(X.T)
        assert half.block.shape == (m, n_rows)
        assert half.trivial.shape == (t, n_rows) and half.trivial.nnz <= X.nnz

        # reference through the sparse solve, before measuring
        want = np.sqrt(np.asarray(X.multiply(factor.solve(X.T).T).sum(axis=1)).ravel())
        tracemalloc.start()
        got = _marginal_predictive_sd(factor, X)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert_allclose(got, want, rtol=1e-10)
        assert peak < 20 * 2**20, f"peak {peak / 2**20:.0f} MB"
