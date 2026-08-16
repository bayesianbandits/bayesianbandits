"""Tests for ``sample_reward_space``: joint posterior predictive draws in reward space.

Covers five fronts:

1. Predictive covariance correctness: the QR square root agrees with a
   dense linear-algebra reference at close to machine precision, for
   dense and sparse (CHOLMOD and SuperLU) factors, in full and blocked
   modes.
2. Moment equivalence: ``sample`` and ``sample_reward_space`` reproduce
   the closed-form predictive mean and covariance (times ``b/(a-1)``
   for the NIG multivariate t) within statistically scaled tolerances;
   blocked mode additionally zeroes cross-block covariance.
3. Planted-bug power suite: two-sample KS tests on random projections
   (Cramer-Wold device) must reject small, realistic implementation
   bugs while NOT rejecting the weight-space vs reward-space comparison.
4. Resolution: ``resolve_reward_space_sampler`` honors the flop model
   (size=1 and n >> d stay on weight-space), the QR row guard, the
   MRO-safety rule for subclasses overriding ``sample``, and unfitted
   models.
5. Agent wiring: ``_sample_context_major_blocks``,
   ``LipschitzContextualAgent`` with an IDS-style policy, and
   ``LearnerPipeline.sample_reward_space`` all route through the
   reward-space path with correct shapes and unchanged decisions.
"""

from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.stats import ks_2samp

from bayesianbandits import (
    Arm,
    ArmColumnFeaturizer,
    BayesianGLM,
    InformationDirectedSampling,
    LipschitzContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
    _estimators,
)
from bayesianbandits._arm import (
    _sample_context_major_blocks,
    _take_rows,
    resolve_marginal_sampler,
    resolve_reward_space_sampler,
)
from bayesianbandits._estimators import _reward_space_is_cheaper
from bayesianbandits.pipelines import LearnerPipeline
from tests._helpers import cov_inv_dense
from tests._helpers import fit_dense as _fit_dense
from tests._helpers import fit_sparse as _fit_sparse


def _reference_S(est, X_dense: np.ndarray) -> np.ndarray:
    return X_dense @ np.linalg.solve(cov_inv_dense(est), X_dense.T)


def _block_diagonal(S: np.ndarray, block_size: int) -> np.ndarray:
    """Zero the cross-block entries of a covariance reference."""
    out = np.zeros_like(S)
    for start in range(0, S.shape[0], block_size):
        stop = start + block_size
        out[start:stop, start:stop] = S[start:stop, start:stop]
    return out


def _assert_moments_match(draws, mean, cov, z_tol=5.0, cov_tol_factor=7.0):
    """Check empirical mean/covariance against closed-form targets with
    tolerances scaled to the Monte Carlo standard error."""
    n_draws = draws.shape[0]
    sd = np.sqrt(np.diag(cov))
    z_max = np.abs((draws.mean(axis=0) - mean) / (sd / np.sqrt(n_draws))).max()
    assert z_max < z_tol, f"mean off by {z_max:.2f} MC standard errors"
    cov_err = np.abs((np.cov(draws.T) - cov) / np.outer(sd, sd)).max()
    cov_tol = cov_tol_factor / np.sqrt(n_draws)
    assert cov_err < cov_tol, f"cov err {cov_err:.4f} exceeds {cov_tol:.4f}"


def _blocked_factor_covariance(L: np.ndarray) -> np.ndarray:
    """Assemble the full covariance implied by per-block factors."""
    n_blocks, k, _ = L.shape
    out = np.zeros((n_blocks * k, n_blocks * k))
    for b in range(n_blocks):
        out[b * k : (b + 1) * k, b * k : (b + 1) * k] = L[b] @ L[b].T
    return out


# -- 1. Predictive covariance correctness -------------------------------------


class TestPredictiveCholesky:
    def test_dense_cholesky_covariance_matches_reference(self):
        est, rng = _fit_dense()
        X = rng.standard_normal((10, 40))
        mean, L = est._predictive_cholesky(X)
        S_ref = _reference_S(est, X)
        sd = np.sqrt(np.diag(S_ref))
        # The QR square root is exact -- no regularization term
        assert np.abs((L @ L.T - S_ref) / np.outer(sd, sd)).max() < 1e-12
        assert_allclose(mean, X @ est.coef_)

    def test_blocked_cholesky_matches_reference_blockwise(self):
        est, rng = _fit_dense()
        X = rng.standard_normal((12, 40))
        mean, L = est._predictive_cholesky(X, block_size=4)
        assert L.shape == (3, 4, 4)
        S_ref = _reference_S(est, X)
        sd = np.sqrt(np.diag(S_ref))
        S_blk = _blocked_factor_covariance(L)
        ref_blk = _block_diagonal(S_ref, 4)
        assert np.abs((S_blk - ref_blk) / np.outer(sd, sd)).max() < 1e-12
        assert_allclose(mean, X @ est.coef_)

    def test_blocked_wider_than_features_pads_factor(self):
        """More rows per block than features: the factor is rank-deficient
        and must be padded, not truncated."""
        est, rng = _fit_dense(d=3, rows=50)
        X = rng.standard_normal((10, 3))
        _, L = est._predictive_cholesky(X, block_size=5)
        assert L.shape == (2, 5, 5)
        S_ref = _reference_S(est, X)
        sd = np.sqrt(np.diag(S_ref))
        err = np.abs(
            (_blocked_factor_covariance(L) - _block_diagonal(S_ref, 5))
            / np.outer(sd, sd)
        )
        assert err.max() < 1e-12

    def test_block_size_must_divide_rows(self):
        est, rng = _fit_dense()
        X = rng.standard_normal((10, 40))
        with pytest.raises(ValueError, match="divide"):
            est._predictive_cholesky(X, block_size=4)
        with pytest.raises(ValueError, match="positive"):
            est._predictive_cholesky(X, block_size=-2)

    def test_mixed_scale_rows_are_exact(self):
        """Rows differing by many orders of magnitude must not contaminate
        each other (the failure mode of any shared regularization term)."""
        est, rng = _fit_dense()
        X = np.vstack([rng.standard_normal(40) * 1e6, rng.standard_normal(40) * 1e-3])
        _, L = est._predictive_cholesky(X)
        S_ref = _reference_S(est, X)
        sd = np.sqrt(np.diag(S_ref))
        assert np.abs((L @ L.T - S_ref) / np.outer(sd, sd)).max() < 1e-10

    def test_sparse_S_matches_dense_reference(self, sparse_solver):
        est, rng = _fit_sparse()
        X = sp.csc_array(
            sp.random(10, 400, density=0.05, random_state=2)  # type: ignore[call-arg]
        )
        mean, L = est._predictive_cholesky(X)
        X_dense = X.toarray()
        S_ref = _reference_S(est, X_dense)
        sd = np.sqrt(np.diag(S_ref))
        assert np.abs((L @ L.T - S_ref) / np.outer(sd, sd)).max() < 1e-9
        assert_allclose(mean, np.asarray(X @ est.coef_).ravel())

    def test_sparse_model_accepts_dense_X(self, sparse_solver):
        """check_array's accept_sparse only *permits* sparse input, so a
        sparse model can legitimately receive a dense ndarray."""
        est, rng = _fit_sparse()
        X = rng.standard_normal((4, 400))
        draws = est.sample_reward_space(X, size=500)
        assert draws.shape == (500, 4)
        assert np.isfinite(draws).all()

    def test_nig_cholesky_is_unscaled_shape_matrix(self):
        """NIG inherits _predictive_cholesky unchanged: it factors the raw
        X @ inv(Lambda) @ X.T shape matrix, and sample_reward_space
        applies the b/a scaling itself."""
        est, rng = _fit_dense(cls=NormalInverseGammaRegressor)
        X = rng.standard_normal((6, 40))
        _, L = est._predictive_cholesky(X)
        S_ref = _reference_S(est, X)
        scale = np.mean(np.diag(S_ref))
        assert np.abs(L @ L.T - S_ref).max() < 1e-8 * scale

    def test_unfitted_prior_predictive(self):
        est = NormalRegressor(alpha=2.0, beta=1.0, random_state=0)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 7))
        draws = est.sample_reward_space(X, size=40_000)
        _assert_moments_match(draws, np.zeros(5), X @ X.T / 2.0)

    def test_glm_eta_space_cholesky(self):
        rng = np.random.default_rng(0)
        est = BayesianGLM(alpha=1.0, random_state=0)
        X_train = rng.standard_normal((50, 8))
        est.fit(X_train, (rng.random(50) > 0.5).astype(np.float64))
        X = rng.standard_normal((5, 8))
        mean, L = est._predictive_cholesky(X)
        S_ref = _reference_S(est, X)
        scale = np.mean(np.diag(S_ref))
        assert np.abs(L @ L.T - S_ref).max() < 1e-8 * scale
        assert_allclose(mean, X @ est.coef_)

    def test_degenerate_rows_yield_exactly_correlated_draws(self):
        """Repeated contexts make S rank-deficient; the QR square root
        represents that exactly, so duplicate rows get identical draws
        (matching the weight-space branch's behavior)."""
        est, rng = _fit_dense()
        x = rng.standard_normal(40)
        X = np.tile(x, (4, 1))
        draws = est.sample_reward_space(X, size=3)
        assert np.isfinite(draws).all()
        assert draws.shape == (3, 4)
        spread = np.abs(draws - draws[:, [0]]).max()
        assert spread < 1e-8 * np.abs(draws).max()

    def test_zero_rows_yield_exact_mean(self):
        """An all-zero X has zero predictive covariance; draws are exactly
        the (zero) mean."""
        est, _ = _fit_dense()
        X = np.zeros((3, 40))
        draws = est.sample_reward_space(X, size=5)
        assert draws.shape == (5, 3)
        assert np.all(draws == 0.0)

    def test_sparse_after_decay_uses_scaled_factor(self, sparse_solver):
        """decay wraps the sparse factor in ScaledSparseFactor; the
        predictive covariance must solve through the scaling."""
        est, rng = _fit_sparse()
        X = sp.csc_array(
            sp.random(6, 400, density=0.05, random_state=3)  # type: ignore[call-arg]
        )
        est.decay(X.toarray(), decay_rate=0.9)
        mean, L = est._predictive_cholesky(X)
        X_dense = X.toarray()
        S_ref = _reference_S(est, X_dense)
        sd = np.sqrt(np.diag(S_ref))
        assert np.abs((L @ L.T - S_ref) / np.outer(sd, sd)).max() < 1e-9
        assert np.isfinite(est.sample_reward_space(X, size=3)).all()


# -- 2. Moment equivalence ----------------------------------------------------


class TestMomentEquivalence:
    N_DRAWS = 40_000

    @pytest.mark.parametrize("block_size", [None, 4])
    def test_dense_normal(self, block_size):
        est, rng = _fit_dense(random_state=123)
        X = rng.standard_normal((8, 40))
        mean = X @ est.coef_
        S = _reference_S(est, X)
        if block_size is not None:
            S = _block_diagonal(S, block_size)
        draws = est.sample_reward_space(X, self.N_DRAWS, block_size=block_size)
        _assert_moments_match(draws, mean, S)

    @pytest.mark.parametrize("block_size", [None, 4])
    def test_sparse_normal(self, block_size, sparse_solver):
        est, rng = _fit_sparse(random_state=123)
        X = sp.csc_array(
            sp.random(8, 400, density=0.05, random_state=2)  # type: ignore[call-arg]
        )
        X_dense = X.toarray()
        mean = X_dense @ est.coef_
        S = _reference_S(est, X_dense)
        if block_size is not None:
            S = _block_diagonal(S, block_size)
        draws = est.sample_reward_space(X, self.N_DRAWS, block_size=block_size)
        _assert_moments_match(draws, mean, S)

    @pytest.mark.parametrize("block_size", [None, 4])
    def test_nig_student_t(self, block_size):
        est, rng = _fit_dense(cls=NormalInverseGammaRegressor, random_state=123)
        X = rng.standard_normal((8, 40))
        mean = X @ est.coef_
        a, b = float(est.a_), float(est.b_)
        # Marginal t covariance: b/(a-1) * X inv(Lambda) X^T
        cov = (b / (a - 1.0)) * _reference_S(est, X)
        if block_size is not None:
            cov = _block_diagonal(cov, block_size)
        draws = est.sample_reward_space(X, self.N_DRAWS, block_size=block_size)
        _assert_moments_match(draws, mean, cov)

    def test_nig_blocks_mix_independent_chi_squares(self):
        """Within a block, standardized draws share one chi-square per
        draw (squared draws are strongly correlated); across blocks the
        mixing variables are independent (near-zero correlation).
        Catches a shared-g implementation that would silently couple
        contexts."""
        est, rng = _fit_dense(cls=NormalInverseGammaRegressor, random_state=3)
        x = rng.standard_normal(40)
        # identical rows: within-block correlation of squared draws is
        # then purely the mixing variable's
        X = np.tile(x, (4, 1))
        draws = est.sample_reward_space(X, 20_000, block_size=2)
        centered = draws - (X @ est.coef_)
        log_sq = np.log(centered**2 + 1e-300)
        within = np.corrcoef(log_sq[:, 0], log_sq[:, 1])[0, 1]
        across = np.corrcoef(log_sq[:, 0], log_sq[:, 2])[0, 1]
        assert within > 0.9
        assert abs(across) < 0.05

    @pytest.mark.parametrize("block_size", [None, 3])
    def test_glm_eta_space(self, block_size):
        """The GLM samples eta and applies the link; with a log link the
        eta draws are recovered exactly by log(), and are Gaussian with
        the predictive moments."""
        rng = np.random.default_rng(0)
        est = BayesianGLM(alpha=1.0, link="log", random_state=123)
        X_train = rng.standard_normal((100, 10)) * 0.3
        y = rng.poisson(1.0, size=100).astype(np.float64)
        est.fit(X_train, y)
        X = rng.standard_normal((6, 10)) * 0.3
        mean = X @ est.coef_
        S = _reference_S(est, X)
        if block_size is not None:
            S = _block_diagonal(S, block_size)
        draws = est.sample_reward_space(X, self.N_DRAWS, block_size=block_size)
        _assert_moments_match(np.log(draws), mean, S)


# -- 3. Planted-bug power suite (KS on random projections) --------------------

_KS_N = 500_000
_KS_N_PROJECTIONS = 20
# Per-projection threshold 2.5e-4: 0.05 Bonferroni-split across the 20
# projections, then tightened 10x. A non-reject assertion (min p > threshold
# over 20 projections) therefore fails spuriously with probability
# ~20 * 2.5e-4 = 0.5% under the null if a numpy/BLAS change re-rolls the draws.
_KS_ALPHA = 0.05 / _KS_N_PROJECTIONS / 10


def _ks_min_p(a, b, n, seed=7):
    """Minimum two-sample KS p-value over random unit projections.

    By the Cramer-Wold device, equality of all 1-D projections is
    equality of the joint distribution, so this tests the full
    distribution (all moments, correlations, tails)."""
    rng = np.random.default_rng(seed)
    p_values = []
    for _ in range(_KS_N_PROJECTIONS):
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        p_values.append(ks_2samp(a @ v, b @ v)[1])
    return min(p_values)


@pytest.fixture(scope="module")
def ks_draws():
    """Weight-space baseline plus reward-space and bug-injected draw sets.

    The bugs are small, realistic failure modes of a reward-space
    implementation, injected at magnitudes the KS pipeline must detect
    at this sample size (calibrated with margin; all seeds fixed):

    - scale: covariance inflated 2% (wrong factor scaling)
    - indep: exact marginals, correlations dropped (lost row pairing)
    - shift: one coordinate's mean off by 0.02 sd (mean mishandling)
    - tails: t(30) rescaled to identical mean AND covariance (wrong
      distribution family; invisible to moment checks)
    """
    rng = np.random.default_rng(0)
    d, n = 60, 10
    est = NormalRegressor(alpha=1.0, beta=1.0, random_state=0)
    X_train = rng.standard_normal((300, d))
    est.fit(X_train, X_train @ rng.standard_normal(d) + rng.standard_normal(300))
    X = rng.standard_normal((n, d))
    mean, L = est._predictive_cholesky(X)
    sd = np.sqrt(np.diag(L @ L.T))

    est.random_state_ = np.random.default_rng(100)
    weight = np.vstack([est.sample(X, 50_000) for _ in range(_KS_N // 50_000)])
    est.random_state_ = np.random.default_rng(200)
    reward = est.sample_reward_space(X, _KS_N)
    # blocked mode against the block-diagonal of the weight-space draws:
    # marginals within one block of 5 rows must match jointly
    est.random_state_ = np.random.default_rng(300)
    reward_blocked = est.sample_reward_space(X, _KS_N, block_size=5)

    def bug_scale(seed):
        z = np.random.default_rng(seed).standard_normal((_KS_N, n))
        return mean + 1.01 * (z @ L.T)

    def bug_indep(seed):
        z = np.random.default_rng(seed).standard_normal((_KS_N, n))
        return mean + sd * z

    def bug_shift(seed):
        z = np.random.default_rng(seed).standard_normal((_KS_N, n))
        out = mean + z @ L.T
        out[:, 0] += 0.02 * sd[0]
        return out

    def bug_tails(seed):
        g = np.random.default_rng(seed)
        df = 30.0
        z = g.standard_normal((_KS_N, n))
        u = g.chisquare(df, size=_KS_N)
        # Rescale to unit variance so mean and covariance match exactly
        t = z * (np.sqrt(df / u) * np.sqrt((df - 2.0) / df))[:, None]
        return mean + t @ L.T

    return {
        "n": n,
        "weight": weight,
        "reward": reward,
        "reward-blocked": reward_blocked,
        "bug-scale": bug_scale(300),
        "bug-indep": bug_indep(301),
        "bug-shift": bug_shift(302),
        "bug-tails": bug_tails(303),
    }


@pytest.mark.slow
class TestDistributionalEquivalence:
    def test_weight_vs_reward_not_rejected(self, ks_draws):
        min_p = _ks_min_p(ks_draws["weight"], ks_draws["reward"], ks_draws["n"])
        assert min_p > _KS_ALPHA, (
            f"weight-space vs reward-space rejected (min p={min_p:.2e}); "
            "the branches no longer sample the same distribution"
        )

    def test_weight_vs_blocked_reward_not_rejected_within_block(self, ks_draws):
        """Projections restricted to one block: the blocked draw's joint
        distribution within a block must match weight-space exactly."""
        block = slice(0, 5)
        min_p = _ks_min_p(
            ks_draws["weight"][:, block], ks_draws["reward-blocked"][:, block], 5
        )
        assert min_p > _KS_ALPHA, (
            f"weight-space vs blocked reward-space rejected within a block "
            f"(min p={min_p:.2e})"
        )

    @pytest.mark.parametrize(
        "bug", ["bug-scale", "bug-indep", "bug-shift", "bug-tails"]
    )
    def test_planted_bugs_are_rejected(self, ks_draws, bug):
        """Power check: if these stop rejecting, the equivalence test
        above has lost the sensitivity that gives it meaning."""
        min_p = _ks_min_p(ks_draws["weight"], ks_draws[bug], ks_draws["n"])
        assert min_p < _KS_ALPHA, f"{bug} not detected (min p={min_p:.2e})"


# -- 4. Resolution ------------------------------------------------------------


class TestResolution:
    def test_rng_cost_keeps_weight_space_at_n_over_d(self):
        """RNG throughput dominates at large size: reward-space draws
        n*size normals vs weight-space's d*size, so n > d stays on
        weight-space even in blocked mode (measured in
        benchmarks/test_bench_reward_space_sampling.py)."""
        assert not _reward_space_is_cheaper(320, 101, 1000, False, 0, block_size=10)
        # n < d: blocked reward-space wins (the IDS Lipschitz shape)
        assert _reward_space_is_cheaper(80, 100, 1000, False, 0, block_size=10)

    def test_row_guard_applies_per_qr_block(self):
        assert not _reward_space_is_cheaper(5000, 101, 10**9, False, 0)
        assert not _reward_space_is_cheaper(10_000, 101, 10**9, False, 0, 5000)
        # per-block cap: many rows are fine when each QR block is small
        # and the model is high-dimensional sparse
        assert _reward_space_is_cheaper(5000, 100_000, 10**6, True, 1_000_000, 10)

    def test_sparse_dispatch_uses_nnz(self):
        nnz = 1_000_000
        assert not _reward_space_is_cheaper(10, 100_000, 1, True, nnz)
        assert _reward_space_is_cheaper(10, 100_000, 1000, True, nnz)

    def test_resolver_honors_flop_model(self):
        est, _ = _fit_dense(d=101, rows=300)
        assert resolve_reward_space_sampler(est, 10, 1) is None
        assert resolve_reward_space_sampler(est, 10, 1000) is not None
        assert resolve_reward_space_sampler(est, 320, 1000) is None
        assert resolve_reward_space_sampler(est, 80, 1000, block_size=10) is not None

    def test_resolver_skips_unfitted_models(self):
        est = NormalRegressor(alpha=1.0, beta=1.0)
        assert resolve_reward_space_sampler(est, 10, 1000) is None

    def test_resolver_respects_sample_override(self):
        """A subclass overriding ``sample`` without ``sample_reward_space``
        must never have its custom sampling bypassed."""

        class ClippedRegressor(NormalRegressor):
            def sample(self, X, size=1):
                return np.clip(super().sample(X, size), 0.0, None)

        est = ClippedRegressor(alpha=1.0, beta=1.0)
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((300, 101))
        est.fit(X_train, rng.standard_normal(300))
        assert resolve_reward_space_sampler(est, 10, 1000) is None
        # and the draws callers do get are the clipped ones
        assert (est.sample(rng.standard_normal((10, 101)), size=50) >= 0.0).all()

    def test_resolver_sampler_applies_block_size(self):
        est, rng = _fit_dense(d=101, rows=300)
        X = rng.standard_normal((8, 101))
        sampler = resolve_reward_space_sampler(est, 8, 1000, block_size=4)
        assert sampler is not None
        est.random_state_ = np.random.default_rng(5)
        via_resolver = sampler(X, 1000)
        est.random_state_ = np.random.default_rng(5)
        direct = est.sample_reward_space(X, 1000, block_size=4)
        assert_allclose(via_resolver, direct)


# -- 5. Agent wiring ----------------------------------------------------------


class _SharedLearner:
    """Minimal batchable learner: shared final estimator, per-arm shift.

    ``transform`` appends nothing but shifts one feature per arm so the
    arms' feature rows differ (a realistic shared-model setup) while
    ``final_estimator`` stays one instance, which is all
    ``can_batch_arms`` requires.
    """

    def __init__(self, model, shift):
        self.final_estimator = model
        self._shift = shift

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        X[:, 0] += self._shift
        return X


def _shared_model_arms(model, n_arms):
    return [
        Arm(i, learner=_SharedLearner(model, float(i)))  # type: ignore[arg-type]
        for i in range(n_arms)
    ]


class TestAgentWiring:
    def test_context_major_blocks_match_the_joint_distribution(self):
        """Blocked reward-space draws must have the same per-(arm,
        context) mean and sd as plain joint ``sample`` draws."""
        rng = np.random.default_rng(0)
        d, n_arms, n_ctx, size = 101, 4, 3, 5000
        model = NormalRegressor(alpha=1.0, beta=1.0, random_state=0)
        model.fit(rng.standard_normal((300, d)), rng.standard_normal(300))
        # arm-major stack, the layout LipschitzContextualAgent produces
        X_stacked = rng.standard_normal((n_arms * n_ctx, d))

        joint = _sample_context_major_blocks(
            lambda X, size: model.sample_reward_space(X, size, block_size=n_arms),
            X_stacked,
            n_arms,
            n_ctx,
            size,
        )
        plain = np.asarray(model.sample(X_stacked, size=size)).T.reshape(
            n_arms, n_ctx, size
        )
        assert joint.shape == plain.shape == (n_arms, n_ctx, size)
        sd = plain.std(axis=-1)
        z = np.abs(joint.mean(axis=-1) - plain.mean(axis=-1)) / (sd / np.sqrt(size))
        assert z.max() < 6.0
        assert np.abs(joint.std(axis=-1) / sd - 1.0).max() < 0.1

    def test_gate_keeps_size_one_on_weight_space(self):
        """The solve-count gate must never pick reward space at
        ``size=1``: one weight draw costs one solve, which no reduction
        can beat, and Thompson sampling lives here."""
        rng = np.random.default_rng(0)
        model = NormalRegressor(alpha=1.0, beta=1.0, random_state=0)
        model.fit(rng.standard_normal((300, 101)), rng.standard_normal(301 - 1))
        assert resolve_reward_space_sampler(model, n_rows=3, size=1) is None
        assert resolve_reward_space_sampler(model, n_rows=1, size=1) is None

    def test_context_major_blocks_are_joint_within_a_context(self):
        """Each context's arm rows form one joint block: with identical
        feature rows across arms, draws for one context stay perfectly
        correlated, while different contexts decorrelate."""
        rng = np.random.default_rng(0)
        d, n_arms, n_ctx, size = 101, 3, 2, 4000
        model = NormalRegressor(alpha=1.0, beta=1.0, random_state=0)
        model.fit(rng.standard_normal((300, d)), rng.standard_normal(300))
        # every arm sees the same row for a given context, so within a
        # context the arms are perfectly correlated under a joint law
        per_context = rng.standard_normal((n_ctx, d))
        X_stacked = np.tile(per_context, (n_arms, 1))  # arm-major

        joint = _sample_context_major_blocks(
            lambda X, size: model.sample_reward_space(X, size, block_size=n_arms),
            X_stacked,
            n_arms,
            n_ctx,
            size,
        )
        assert_allclose(joint[0, 0], joint[1, 0])
        r = np.corrcoef(joint[0, 0], joint[0, 1])[0, 1]
        assert abs(r) < 0.06

    def _make_lipschitz(self, learner, n_arms=4, seed=0):
        arms = [Arm(i, reward_function=None, learner=None) for i in range(n_arms)]
        return LipschitzContextualAgent(
            arms=arms,
            policy=InformationDirectedSampling(samples=500),
            arm_featurizer=ArmColumnFeaturizer(column_name="product_id"),
            learner=learner,
            random_seed=seed,
        )

    def test_lipschitz_agent_routes_through_reward_space(self, monkeypatch):
        rng = np.random.default_rng(0)
        d = 100  # enriched d + 1 for the arm column
        agent = self._make_lipschitz(NormalRegressor(alpha=1.0, beta=1.0))
        X_train = rng.standard_normal((300, d))
        agent.pull(X_train[:1])
        agent.update(X_train, rng.standard_normal(300))

        calls = {"reward_space": 0}
        original = type(agent.learner).sample_reward_space  # type: ignore[attr-defined]

        def counting(self, X, size=1, *, block_size=None):
            calls["reward_space"] += 1
            assert block_size == len(agent.arms)
            return original(self, X, size, block_size=block_size)

        monkeypatch.setattr(type(agent.learner), "sample_reward_space", counting)
        tokens = agent.pull(rng.standard_normal((3, d)))
        assert len(tokens) == 3
        assert calls["reward_space"] == 1

    def test_lipschitz_agent_falls_back_for_small_models(self):
        """Tiny d: the flop gate declines and pulls still work via
        weight-space sample."""
        rng = np.random.default_rng(0)
        agent = self._make_lipschitz(NormalRegressor(alpha=1.0, beta=1.0))
        X = rng.standard_normal((3, 2))
        agent.pull(X)
        agent.update(X, rng.standard_normal(3))
        assert len(agent.pull(X)) == 3

    def test_lipschitz_decisions_match_between_paths(self):
        """Same posterior, both sampling paths: IDS decision frequencies
        must agree within Monte Carlo tolerance."""
        rng = np.random.default_rng(0)
        d = 100
        X_train = rng.standard_normal((60, d))
        y_train = rng.standard_normal(60)
        X_test = rng.standard_normal((1, d))
        n_pulls = 300

        def pull_freqs(force_weight_space):
            agent = self._make_lipschitz(
                NormalRegressor(alpha=1.0, beta=1.0, random_state=7), seed=7
            )
            agent.pull(X_train[:1])
            agent.update(X_train, y_train)
            if force_weight_space:
                agent.learner._use_reward_space = (  # type: ignore[attr-defined]
                    lambda *a, **k: False
                )
            counts = np.zeros(len(agent.arms))
            for _ in range(n_pulls):
                counts[agent.pull(X_test)[0]] += 1
            return counts / n_pulls

        f_rs = pull_freqs(False)
        f_ws = pull_freqs(True)
        # binomial MC noise at n=300 per path: generous 4-sigma band
        tol = 4.0 * np.sqrt(0.25 / n_pulls) * np.sqrt(2)
        assert np.abs(f_rs - f_ws).max() < tol

    def test_pipeline_forwards_reward_space(self):
        rng = np.random.default_rng(0)
        model = NormalRegressor(alpha=1.0, beta=1.0, random_state=0)
        model.fit(rng.standard_normal((300, 101)), rng.standard_normal(300))
        pipeline = LearnerPipeline(steps=[], learner=model)
        X = rng.standard_normal((4, 101))
        model.random_state_ = np.random.default_rng(3)
        via_pipeline = pipeline.sample_reward_space(X, 1000, block_size=2)
        model.random_state_ = np.random.default_rng(3)
        direct = model.sample_reward_space(X, 1000, block_size=2)
        assert_allclose(via_pipeline, direct)

    def test_pipeline_falls_back_to_sample(self):
        """A learner without sample_reward_space: the pipeline falls back
        to fully joint sample draws."""

        class PlainLearner:
            def sample(self, X, size=1):
                return np.zeros((size, X.shape[0]))

            def predict(self, X):
                return np.full(X.shape[0], 3.0)

            def partial_fit(self, X, y, sample_weight=None):
                return self

            def decay(self, X, *, decay_rate=None):
                self.decayed = True

        learner = PlainLearner()
        pipeline = LearnerPipeline(steps=[], learner=learner)  # type: ignore[arg-type]
        X = np.zeros((4, 3))
        draws = pipeline.sample_reward_space(X, 7, block_size=2)
        assert draws.shape == (7, 4)
        # the gate declines too, so nothing routes to reward space
        assert not pipeline._use_reward_space(4, 7, 2)
        # and the rest of the protocol still forwards to the learner
        assert pipeline.predict(X).tolist() == [3.0] * 4
        assert pipeline.partial_fit(X, np.zeros(4)) is pipeline
        pipeline.decay(X)
        assert learner.decayed


# -- 6. Resolver and factor edge paths ----------------------------------------


class TestResolverEdgePaths:
    def test_learner_with_no_class_level_sample(self):
        """_defines_before_sample walks the MRO; a learner whose sample is
        an instance attribute defines neither name at class level, so the
        walk falls off the end and the resolver declines."""

        class InstanceSampler:
            pass

        learner = InstanceSampler()
        learner.sample = lambda X, size=1: np.zeros((size, len(X)))  # type: ignore[attr-defined]
        assert resolve_reward_space_sampler(learner, 10, 1000) is None
        assert resolve_marginal_sampler(learner) is learner.sample  # type: ignore[attr-defined]

    def test_take_rows_uses_iloc_for_dataframes(self):
        """Plain X[indices] selects columns by label on a DataFrame, so
        positional row selection must go through iloc."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(np.arange(9.0).reshape(3, 3), columns=["a", "b", "c"])
        taken = _take_rows(df, np.array([2, 0, 1]))
        assert taken.values.tolist() == [[6, 7, 8], [0, 1, 2], [3, 4, 5]]
        # and the dense/sparse branch still selects rows positionally
        assert _take_rows(np.arange(9.0).reshape(3, 3), np.array([2, 0])).tolist() == [
            [6, 7, 8],
            [0, 1, 2],
        ]


class TestPredictiveCholeskyEdgePaths:
    def test_full_mode_pads_when_rows_exceed_features(self):
        """QR of the (d, n) half-solve returns an (min(d, n), n) R, which
        must be zero-padded to (n, n) before transposing to a lower
        factor."""
        est, rng = _fit_dense(d=4, rows=80)
        X = rng.standard_normal((12, 4))
        X[6:9] = X[0:3]  # linearly dependent rows must stay exact
        mean, L = est._predictive_cholesky(X)

        assert L.shape == (12, 12)
        assert_allclose(L, np.tril(L), atol=0)
        assert (np.diag(L) >= -1e-12).all()
        target = X @ np.linalg.inv(cov_inv_dense(est)) @ X.T
        assert_allclose(L @ L.T, target, atol=1e-10)
        assert_allclose(mean, X @ est.coef_)

    def test_blocked_sparse_chunks_via_csr(self, sparse_solver):
        """Blocked mode bounds the dense half-solve scratch by chunking
        over blocks; a sparse X is copied to CSR first, since rows are
        CSC's minor axis."""
        est, rng = _fit_sparse(d=400, rows=300)
        X = sp.csc_array(
            sp.random(24, 400, density=0.05, random_state=7)  # type: ignore[call-arg]
        )
        target = X.toarray() @ np.linalg.inv(cov_inv_dense(est)) @ X.toarray().T

        with mock.patch.object(_estimators, "_MARGINAL_SD_BLOCK_ELEMS", 8):
            _, L_chunked = est._predictive_cholesky(X, 4)
        _, L_whole = est._predictive_cholesky(X, 4)

        assert L_chunked.shape == (6, 4, 4)
        assert_allclose(L_chunked, L_whole, atol=1e-10)
        for b in range(6):
            block = target[b * 4 : (b + 1) * 4, b * 4 : (b + 1) * 4]
            assert_allclose(L_chunked[b] @ L_chunked[b].T, block, atol=1e-10)


class TestContextMajorBlocks:
    def test_identity_permutation_is_skipped(self):
        """With one arm or one context the context-major gather is the
        identity, so the sampler must see ``X_stacked`` itself."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 2))
        seen: list[object] = []

        def sampler(X_in, size):
            seen.append(X_in)
            return rng.standard_normal((size, X_in.shape[0]))

        _sample_context_major_blocks(sampler, X, n_arms=1, n_contexts=5, size=4)
        assert seen[-1] is X
        _sample_context_major_blocks(sampler, X, n_arms=5, n_contexts=1, size=4)
        assert seen[-1] is X

    def test_axes_are_restored_from_context_major_order(self):
        """The sampler is handed context-major rows; the result must come
        back indexed (arm, context, draw)."""
        n_arms, n_ctx, size = 3, 2, 4
        # arm-major row ``a * n_ctx + c`` carries the value ``a * 10 + c``
        X = np.array(
            [[a * 10 + c] for a in range(n_arms) for c in range(n_ctx)], dtype=float
        )

        def sampler(X_in, size):
            # every draw simply echoes the row's payload
            return np.tile(X_in[:, 0], (size, 1))

        out = _sample_context_major_blocks(sampler, X, n_arms, n_ctx, size)
        assert out.shape == (n_arms, n_ctx, size)
        for a in range(n_arms):
            for c in range(n_ctx):
                assert_allclose(out[a, c], a * 10 + c)
