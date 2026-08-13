"""Tests for iid marginal posterior predictive sampling (``sample_marginal``).

Covers four fronts:

1. Marginal sd correctness: per-row predictive standard deviations agree
   with ``sqrt(diag(X @ inv(Lambda) @ X.T))`` from a dense linear-algebra
   reference at close to machine precision, for dense and sparse
   (CHOLMOD and SuperLU) factors, including decay-scaled factors.
2. Distributional equivalence: each row's marginal from
   ``sample_marginal`` matches the same row's marginal under joint
   weight-space ``sample`` (two-sample KS per row), and NIG standardized
   draws match the exact Student t -- with a planted df bug confirming
   the t test has power.
3. The independence contract: marginal draws are iid across rows, while
   ``sample`` rows within a draw are correlated.
4. Plumbing: ``Arm``/``LearnerPipeline``/``batch_sample_arms``
   forwarding, the policies' ``marginal_ok`` opt-in (and subclass
   opt-out), and the fallback to joint ``sample`` for learners without
   ``sample_marginal`` or whose class overrides ``sample`` without it.
"""

from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.stats import ks_2samp, kstest
from scipy.stats import t as t_dist

from bayesianbandits import (
    EXP3A,
    Arm,
    BayesianGLM,
    ContextualAgent,
    EpsilonGreedy,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits._arm import batch_sample_arms, resolve_marginal_sampler
from bayesianbandits.pipelines import LearnerPipeline
from tests._helpers import cov_inv_dense


def _fit_dense(cls=NormalRegressor, d=40, rows=200, seed=0, **kwargs) -> tuple:
    rng = np.random.default_rng(seed)
    if cls is NormalRegressor:
        kwargs.setdefault("alpha", 1.0)
        kwargs.setdefault("beta", 1.0)
    est = cls(random_state=0, **kwargs)
    X_train = rng.standard_normal((rows, d))
    y = X_train @ rng.standard_normal(d) + rng.standard_normal(rows)
    est.fit(X_train, y)
    return est, rng


def _fit_sparse(d=400, rows=300, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True, random_state=0, **kwargs)
    X_train = sp.csc_array(
        sp.random(rows, d, density=10 / d, random_state=1)  # type: ignore[call-arg]
    )
    est.fit(X_train, rng.standard_normal(rows))
    return est, rng


def _reference_sd(est, X_dense: np.ndarray) -> np.ndarray:
    S = X_dense @ np.linalg.solve(cov_inv_dense(est), X_dense.T)
    return np.sqrt(np.diag(S))


# -- 1. Marginal sd correctness -----------------------------------------------


class TestMarginalSd:
    def test_dense_sd_matches_reference(self):
        est, rng = _fit_dense()
        X = rng.standard_normal((10, 40))
        draws = est.sample_marginal(X, size=200_000)
        sd_ref = _reference_sd(est, X)
        assert_allclose(draws.mean(axis=0), X @ est.coef_, atol=5e-2)
        assert_allclose(draws.std(axis=0), sd_ref, rtol=2e-2)

    def test_sparse_sd_matches_dense_reference(self, sparse_solver):
        est, rng = _fit_sparse()
        X = sp.csc_array(
            sp.random(10, 400, density=0.05, random_state=2)  # type: ignore[call-arg]
        )
        draws = est.sample_marginal(X, size=100_000)
        sd_ref = _reference_sd(est, X.toarray())
        assert_allclose(draws.std(axis=0), sd_ref, rtol=3e-2)
        assert_allclose(
            draws.mean(axis=0), np.asarray(X @ est.coef_).ravel(), atol=5e-2
        )

    def test_sparse_after_decay_uses_scaled_factor(self, sparse_solver):
        """decay wraps the sparse factor in ScaledSparseFactor; the
        marginal sd must solve through the scaling."""
        est, rng = _fit_sparse()
        X = sp.csc_array(
            sp.random(6, 400, density=0.05, random_state=3)  # type: ignore[call-arg]
        )
        est.decay(X.toarray(), decay_rate=0.9)
        draws = est.sample_marginal(X, size=100_000)
        sd_ref = _reference_sd(est, X.toarray())
        assert_allclose(draws.std(axis=0), sd_ref, rtol=3e-2)

    def test_mixed_scale_rows_are_exact(self):
        """Rows differing by many orders of magnitude must not contaminate
        each other."""
        est, rng = _fit_dense()
        X = np.vstack([rng.standard_normal(40) * 1e6, rng.standard_normal(40) * 1e-3])
        draws = est.sample_marginal(X, size=200_000)
        sd_ref = _reference_sd(est, X)
        assert_allclose(draws.std(axis=0), sd_ref, rtol=2e-2)

    def test_zero_rows_yield_exact_mean(self):
        """An all-zero X has zero predictive variance; draws are exactly
        the (zero) mean."""
        est, _ = _fit_dense()
        X = np.zeros((3, 40))
        draws = est.sample_marginal(X, size=5)
        assert draws.shape == (5, 3)
        assert np.all(draws == 0.0)

    def test_sparse_model_accepts_dense_X(self, sparse_solver):
        """check_array's accept_sparse only *permits* sparse input; a
        dense ndarray on a sparse model must work (regression: an
        earlier implementation crashed on .toarray())."""
        est, rng = _fit_sparse()
        X = rng.standard_normal((4, 400))
        draws = est.sample_marginal(X, size=500)
        assert draws.shape == (500, 4)
        assert np.isfinite(draws).all()

    def test_unfitted_prior_predictive_accepts_list_X(self):
        """Unfitted models draw from the prior predictive, and plain
        Python lists are validated before the prior is initialized."""
        est = NormalRegressor(alpha=2.0, beta=1.0, random_state=0)
        draws = est.sample_marginal([[1.0, 2.0]], size=100_000)  # type: ignore[arg-type]
        # prior predictive variance: x @ x.T / alpha = (1 + 4) / 2
        assert_allclose(draws.mean(), 0.0, atol=2e-2)
        assert_allclose(draws.var(), 2.5, rtol=3e-2)

    def test_glm_log_link_and_unknown_link(self):
        rng = np.random.default_rng(0)
        est = BayesianGLM(alpha=1.0, link="log", random_state=0)
        X_train = rng.standard_normal((50, 6))
        est.fit(X_train, rng.poisson(1.0, 50).astype(np.float64))
        X = rng.standard_normal((3, 6))
        draws = est.sample_marginal(X, size=100)
        assert draws.shape == (100, 3)
        assert (draws > 0).all()

        est.link = "cauchit"
        with pytest.raises(ValueError, match="Unknown link"):
            est.sample_marginal(X, size=2)

    def test_nig_variance_scaling(self):
        """NIG marginal variance is b/(a-1) times diag(X inv(Lambda) X.T)."""
        est, rng = _fit_dense(cls=NormalInverseGammaRegressor)
        X = rng.standard_normal((6, 40))
        draws = est.sample_marginal(X, size=400_000)
        sd_ref = _reference_sd(est, X)
        expected_var = est.b_ / (est.a_ - 1) * sd_ref**2
        assert_allclose(draws.var(axis=0), expected_var, rtol=3e-2)


# -- 2. Distributional equivalence --------------------------------------------


_KS_ALPHA = 1e-3


@pytest.mark.slow
class TestMarginalDistribution:
    def test_rows_match_weight_space_marginals(self):
        est, rng = _fit_dense()
        X = rng.standard_normal((5, 40))
        marginal = est.sample_marginal(X, size=40_000)
        joint = est.sample(X, size=40_000)
        min_p = min(
            ks_2samp(marginal[:, i], joint[:, i]).pvalue  # type: ignore[union-attr]
            for i in range(X.shape[0])
        )
        assert min_p > _KS_ALPHA, (
            f"marginal vs weight-space rejected (min p={min_p:.2e}); "
            "the paths no longer share per-row marginals"
        )

    def test_glm_rows_match_weight_space_marginals(self):
        rng = np.random.default_rng(0)
        est = BayesianGLM(alpha=1.0, random_state=0)
        X_train = rng.standard_normal((100, 8))
        est.fit(X_train, (rng.random(100) > 0.5).astype(np.float64))
        X = rng.standard_normal((4, 8))
        marginal = est.sample_marginal(X, size=40_000)
        joint = est.sample(X, size=40_000)
        min_p = min(
            ks_2samp(marginal[:, i], joint[:, i]).pvalue  # type: ignore[union-attr]
            for i in range(X.shape[0])
        )
        assert min_p > _KS_ALPHA

    def test_nig_standardized_draws_match_student_t(self):
        """One-sample KS against the exact Student t: the test with power
        to catch df bugs in the chi-square mixing (a df off-by-one shifts
        excess kurtosis by ~0.5, invisible to moment checks)."""
        # few rows -> small df (2a = 7.2): tail differences are visible,
        # so the planted bug below actually has power
        est, rng = _fit_dense(cls=NormalInverseGammaRegressor, d=5, rows=7)
        X = rng.standard_normal((1, 5))
        df = 2.0 * est.a_
        sd_ref = _reference_sd(est, X)[0]
        scale = np.sqrt(est.b_ / est.a_) * sd_ref
        loc = (X @ est.coef_).item()

        draws = est.sample_marginal(X, size=200_000).ravel()
        p_correct = kstest((draws - loc) / scale, t_dist(df=df).cdf).pvalue
        assert p_correct > _KS_ALPHA, f"correct draws rejected (p={p_correct:.2e})"

        # planted df bug: same construction with df+2 must be rejected,
        # otherwise this test has no power
        g = np.random.default_rng(0).chisquare(df + 2.0, size=200_000)
        z = np.random.default_rng(1).standard_normal(200_000)
        bugged = z * np.sqrt((df + 2.0) / g)
        p_bug = kstest(bugged, t_dist(df=df).cdf).pvalue
        assert p_bug < _KS_ALPHA, f"planted df bug not detected (p={p_bug:.2e})"


# -- 3. Independence contract -------------------------------------------------


class TestIndependenceContract:
    def test_marginal_rows_are_independent(self):
        """Duplicate contexts: joint ``sample`` gives identical rows within
        a draw, marginal draws are independent across rows."""
        est, rng = _fit_dense()
        x = rng.standard_normal(40)
        X = np.tile(x, (2, 1))

        joint = est.sample(X, size=2_000)
        assert_allclose(joint[:, 0], joint[:, 1], rtol=1e-10)

        marginal = est.sample_marginal(X, size=2_000)
        corr = np.corrcoef(marginal[:, 0], marginal[:, 1])[0, 1]
        assert abs(corr) < 0.1

    def test_nig_marginal_rows_are_independent(self):
        """A chi-square mixing variable shared across rows would leave
        rows tail-dependent (correlated absolute deviations) even though
        each marginal stays exactly Student t; each (draw, row) cell must
        mix its own chi-square variable."""
        est, rng = _fit_dense(cls=NormalInverseGammaRegressor, d=5, rows=7)
        x = rng.standard_normal(5)
        X = np.tile(x, (2, 1))
        marginal = est.sample_marginal(X, size=50_000)
        dev = np.abs(marginal - np.median(marginal, axis=0))
        corr = np.corrcoef(dev[:, 0], dev[:, 1])[0, 1]
        assert abs(corr) < 0.05


# -- 4. Plumbing --------------------------------------------------------------


class _JointOnlyLearner:
    """Minimal learner stub without ``sample_marginal``."""

    def __init__(self):
        self.calls = 0

    def sample(self, X, size=1):
        self.calls += 1
        return np.zeros((size, len(X)))


class TestPlumbing:
    def test_arm_uses_learner_marginal_and_reward_function(self):
        est, rng = _fit_dense()
        arm = Arm("a", learner=est, reward_function=lambda s: 2.0 * s)  # type: ignore[arg-type]
        X = rng.standard_normal((3, 40))
        with mock.patch.object(
            est, "sample_marginal", wraps=est.sample_marginal
        ) as spy:
            est.random_state_ = np.random.default_rng(5)
            via_arm = arm.sample_marginal(X, size=7)
        spy.assert_called_once()
        est.random_state_ = np.random.default_rng(5)
        direct = est.sample_marginal(X, size=7)
        assert_allclose(via_arm, 2.0 * direct)

    def test_arm_falls_back_to_joint_sample(self):
        learner = _JointOnlyLearner()
        arm = Arm("a", learner=learner)  # type: ignore[arg-type]
        draws = arm.sample_marginal([[1.0]], size=4)
        assert learner.calls == 1
        assert draws.shape == (4, 1)

    def test_learner_subclass_sample_override_falls_back(self):
        """A learner subclass overriding ``sample`` but inheriting
        ``sample_marginal`` must not be bypassed by the marginal path
        (regression: the guard once only covered Arm subclasses)."""

        class Clipped(NormalRegressor):
            def sample(self, X, size=1):
                return np.clip(super().sample(X, size), 0.0, None)

        rng = np.random.default_rng(0)
        est = Clipped(alpha=1.0, beta=1.0, random_state=0)
        X_train = rng.standard_normal((30, 4))
        est.fit(X_train, rng.standard_normal(30) - 5.0)
        X = rng.standard_normal((4, 4))

        assert resolve_marginal_sampler(est) == est.sample
        plain = NormalRegressor(alpha=1.0, beta=1.0, random_state=0)
        assert resolve_marginal_sampler(plain) == plain.sample_marginal

        arm = Arm("a", learner=est)
        assert (arm.sample_marginal(X, size=500) >= 0.0).all()

        pipeline = LearnerPipeline(steps=[], learner=est)
        assert (pipeline.sample_marginal(X, size=500) >= 0.0).all()

    def test_policy_marginal_opt_out_uses_joint_sampling(self):
        """A policy subclass setting ``marginal_ok = False`` must get
        joint draws (regression: ``__call__`` once hardcoded
        ``marginal=True`` regardless of the flag)."""

        class JointUCB(UpperConfidenceBound):
            marginal_ok = False

        est, rng = _fit_dense()
        arms = [Arm(i, learner=est) for i in range(2)]
        X = rng.standard_normal((1, 40))
        with (
            mock.patch.object(est, "sample", wraps=est.sample) as joint_spy,
            mock.patch.object(
                est, "sample_marginal", wraps=est.sample_marginal
            ) as marginal_spy,
        ):
            JointUCB(alpha=0.8)(arms, X, np.random.default_rng(0))
        assert joint_spy.called
        assert not marginal_spy.called

    def test_arm_subclass_sample_override_is_respected(self):
        """A subclass overriding ``sample`` keeps its behavior on the
        marginal path (regression: the learner's ``sample_marginal``
        must not bypass the override)."""

        class FixedArm(Arm):
            def sample(self, X, size=1):
                return np.full((size, len(X)), 42.0)

        est, _ = _fit_dense()
        arm = FixedArm("a", learner=est)
        draws = arm.sample_marginal([[1.0] * 40], size=3)
        assert np.all(draws == 42.0)

    def test_pipeline_forwards_marginal(self):
        est, rng = _fit_dense()
        pipeline = LearnerPipeline(steps=[], learner=est)
        X = rng.standard_normal((3, 40))
        with mock.patch.object(
            est, "sample_marginal", wraps=est.sample_marginal
        ) as spy:
            draws = pipeline.sample_marginal(X, size=5)
        spy.assert_called_once()
        assert draws.shape == (5, 3)

    def test_batch_sample_arms_marginal_falls_back_without_pipeline(self):
        """Non-batchable arms return None regardless of the marginal flag."""
        est, _ = _fit_dense()
        arms = [Arm(i, learner=est) for i in range(3)]
        assert batch_sample_arms(arms, np.ones((2, 40)), size=4, marginal=True) is None

    def test_policy_marginal_flags(self):
        assert UpperConfidenceBound().marginal_ok
        assert EXP3A().marginal_ok
        assert EpsilonGreedy().marginal_ok
        assert not ThompsonSampling().marginal_ok

    @pytest.mark.parametrize(
        "policy",
        [UpperConfidenceBound(alpha=0.8), EXP3A(eta=1.0), EpsilonGreedy(epsilon=0.1)],
    )
    def test_marginal_policies_pull_end_to_end(self, policy):
        rng = np.random.default_rng(0)
        learners = [
            NormalRegressor(alpha=1.0, beta=1.0, random_state=i) for i in range(3)
        ]
        arms = [Arm(i, learner=learner) for i, learner in enumerate(learners)]
        agent = ContextualAgent(arms, policy, random_seed=0)
        X = rng.standard_normal((2, 3))
        for learner in learners:
            learner.fit(rng.standard_normal((20, 3)), rng.standard_normal(20))
        tokens = agent.pull(X)
        assert len(tokens) == 2
