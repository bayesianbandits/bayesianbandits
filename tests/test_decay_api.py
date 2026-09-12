"""``decay(rule, steps=)``: the clock-tick forgetting API."""

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse as sp
from sklearn.preprocessing import FunctionTransformer

from bayesianbandits import (
    Agent,
    Arm,
    ContextualAgent,
    DirichletClassifier,
    EmpiricalBayesDirichletClassifier,
    EmpiricalBayesGammaRegressor,
    EmpiricalBayesNormalRegressor,
    ExponentialForgetting,
    FeatureWiseForgetting,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
    SiftForgetting,
    StabilizedForgetting,
    ThompsonSampling,
)
from bayesianbandits.pipelines import AgentPipeline, LearnerPipeline


def _dense(P):
    return P.toarray() if sp.issparse(P) else np.asarray(P)


def _learner(arm) -> Any:
    return arm.learner


def _fit_normal(sparse, learning_rate=1.0, **kwargs):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    y = X @ np.array([1.0, -1.0, 0.5, 0.0]) + rng.normal(0, 0.1, 30)
    est = NormalRegressor(
        alpha=2.0, beta=1.0, sparse=sparse, learning_rate=learning_rate, **kwargs
    )
    est.fit(sp.csc_array(X) if sparse else X, y)
    return est, X


class TestTickRules:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_steps_raise_the_rate_to_that_power(self, sparse):
        est, _ = _fit_normal(sparse)
        before = _dense(est.cov_inv_)
        est.decay(ExponentialForgetting(0.9), steps=3)
        assert_allclose(_dense(est.cov_inv_), 0.9**3 * before)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_decay_rate_is_shorthand_for_the_default_rule(self, sparse):
        est, _ = _fit_normal(sparse)
        twin, _ = _fit_normal(sparse)
        est.decay(decay_rate=0.8, steps=2)
        twin.decay(ExponentialForgetting(0.8), steps=2)
        assert_allclose(_dense(est.cov_inv_), _dense(twin.cov_inv_))

    @pytest.mark.parametrize("sparse", [False, True])
    def test_stabilized_floors_at_the_estimator_alpha(self, sparse):
        est, _ = _fit_normal(sparse)
        before = _dense(est.cov_inv_)
        est.decay(StabilizedForgetting(0.9), steps=2)
        g = 0.9**2
        assert_allclose(_dense(est.cov_inv_), g * before + (1 - g) * 2.0 * np.eye(4))

    def test_stabilized_with_its_own_alpha(self):
        est, _ = _fit_normal(False)
        before = _dense(est.cov_inv_)
        est.decay(StabilizedForgetting(0.9, alpha=7.0))
        assert_allclose(_dense(est.cov_inv_), 0.9 * before + 0.1 * 7.0 * np.eye(4))

    @pytest.mark.parametrize("sparse", [False, True])
    def test_stabilized_drops_the_cached_factor_and_sampling_still_matches(
        self, sparse
    ):
        est, X = _fit_normal(sparse)
        est.sample(sp.csc_array(X[:3]) if sparse else X[:3])  # builds the factor
        assert "_precision_factor" in est.__dict__
        est.decay(StabilizedForgetting(0.9))
        assert "_precision_factor" not in est.__dict__
        # The rebuilt factor agrees with the precision it factors.
        S = np.linalg.inv(_dense(est.cov_inv_))
        Xq = X[:5]
        draws = est.sample(sp.csc_array(Xq) if sparse else Xq, size=20_000)
        assert_allclose(np.var(draws, axis=0), np.diag(Xq @ S @ Xq.T), rtol=0.1)

    def test_exponential_scales_the_cached_factor_in_place(self):
        est, X = _fit_normal(False)
        est.sample(X[:3])
        factor = est._precision_factor
        est.decay(ExponentialForgetting(0.5), steps=2)
        assert est._precision_factor._scale == pytest.approx(factor._scale * 0.25)

    def test_rule_and_decay_rate_together_is_an_error(self):
        est, _ = _fit_normal(False)
        with pytest.raises(TypeError, match="not both"):
            est.decay(ExponentialForgetting(0.9), decay_rate=0.9)

    @pytest.mark.parametrize("rule", [FeatureWiseForgetting(0.9), SiftForgetting(0.9)])
    def test_directional_rules_are_refused_with_a_pointer(self, rule):
        est, _ = _fit_normal(False)
        with pytest.raises(TypeError, match="forgetting="):
            est.decay(rule)

    def test_unfitted_estimator_ignores_decay(self):
        est = NormalRegressor(alpha=1.0, beta=1.0)
        est.decay(StabilizedForgetting(0.5))
        assert not hasattr(est, "coef_")


class TestDeprecatedCallingConvention:
    def test_context_array_means_steps_from_its_rows(self):
        est, X = _fit_normal(False)
        twin, _ = _fit_normal(False)
        with pytest.warns(FutureWarning, match="steps="):
            est.decay(X[:4], decay_rate=0.9)
        twin.decay(decay_rate=0.9, steps=4)
        assert_allclose(_dense(est.cov_inv_), _dense(twin.cov_inv_))

    def test_falling_back_to_learning_rate_warns(self):
        est, _ = _fit_normal(False, learning_rate=0.7)
        before = _dense(est.cov_inv_)
        with pytest.warns(FutureWarning, match="learning_rate"):
            est.decay(steps=2)
        assert_allclose(_dense(est.cov_inv_), 0.7**2 * before)


class TestNormalInverseGamma:
    def test_stabilized_needs_a_scalar_prior_precision(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        est = NormalInverseGammaRegressor(lam=np.array([1.0, 2.0, 3.0]))
        est.fit(X, y)
        with pytest.raises(TypeError, match="alpha="):
            est.decay(StabilizedForgetting(0.9))
        est.decay(StabilizedForgetting(0.9, alpha=1.0))  # explicit floor works

    def test_tick_forgets_the_inverse_gamma_too(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        est = NormalInverseGammaRegressor(lam=1.0)
        est.fit(X, y)
        a, b = est.a_, est.b_
        est.decay(StabilizedForgetting(0.8), steps=2)
        assert est.a_ == pytest.approx(0.8**2 * a)
        assert est.b_ == pytest.approx(0.8**2 * b)


class TestEmpiricalBayes:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_default_is_stabilized_at_the_tuned_alpha(self, sparse):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3))
        y = X @ np.array([1.0, 0.0, -1.0]) + rng.normal(0, 0.3, 40)
        est = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        est.fit(sp.csc_array(X) if sparse else X, y)
        before = _dense(est.cov_inv_)
        s_before = est._prior_scalar
        est.decay(decay_rate=0.9, steps=2)
        g = 0.9**2
        assert_allclose(
            _dense(est.cov_inv_), g * before + (1 - g) * est.alpha * np.eye(3)
        )
        assert est._prior_scalar == pytest.approx(g * s_before + (1 - g) * est.alpha)

    def test_exponential_is_accepted_and_books_the_prior_scalar(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 3))
        y = rng.standard_normal(40)
        est = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0)
        est.fit(X, y)
        before = _dense(est.cov_inv_)
        s_before = est._prior_scalar
        est.decay(ExponentialForgetting(0.9))
        assert_allclose(_dense(est.cov_inv_), 0.9 * before)
        assert est._prior_scalar == pytest.approx(0.9 * s_before)

    def test_directional_is_refused(self):
        est = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0)
        est.fit(np.eye(3), np.ones(3))
        with pytest.raises(TypeError, match="forgetting="):
            est.decay(SiftForgetting(0.9))


class TestGroupedModels:
    def test_dirichlet_ticks_every_seen_group(self):
        clf = DirichletClassifier({0: 1.0, 1: 1.0}, random_state=0)
        clf.fit(np.array([[1], [1], [2]]), np.array([0, 1, 1]))
        a1, a2 = clf.known_alphas_[1].copy(), clf.known_alphas_[2].copy()
        clf.decay(ExponentialForgetting(0.5), steps=2)
        assert_allclose(clf.known_alphas_[1], 0.25 * a1)
        assert_allclose(clf.known_alphas_[2], 0.25 * a2)

    def test_dirichlet_stabilized_mixes_the_prior_back_in(self):
        clf = DirichletClassifier({0: 1.0, 1: 3.0}, random_state=0)
        clf.fit(np.array([[1], [1]]), np.array([0, 0]))
        a1 = clf.known_alphas_[1].copy()
        clf.decay(StabilizedForgetting(0.5))
        assert_allclose(clf.known_alphas_[1], 0.5 * a1 + 0.5 * np.array([1.0, 3.0]))

    def test_gamma_stabilized_returns_to_the_prior(self):
        model = GammaRegressor(alpha=2.0, beta=3.0, random_state=0)
        model.fit(np.array([[1], [1]]), np.array([4, 6]))
        for _ in range(200):
            model.decay(StabilizedForgetting(0.8))
        assert_allclose(model.coef_[1], [2.0, 3.0], atol=1e-6)

    def test_legacy_array_ticks_only_the_listed_groups(self):
        clf = DirichletClassifier({0: 1.0, 1: 1.0}, random_state=0)
        clf.fit(np.array([[1], [2]]), np.array([0, 1]))
        a1, a2 = clf.known_alphas_[1].copy(), clf.known_alphas_[2].copy()
        with pytest.warns(FutureWarning):
            clf.decay(np.array([[1], [1]]), decay_rate=0.5)
        assert_allclose(clf.known_alphas_[1], 0.25 * a1)
        assert_allclose(clf.known_alphas_[2], a2)

    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (EmpiricalBayesDirichletClassifier, {"alphas": {0: 1.0, 1: 1.0}}),
            (EmpiricalBayesGammaRegressor, {"alpha": 2.0, "beta": 3.0}),
        ],
    )
    def test_eb_grouped_defaults_to_stabilized(self, cls, kwargs):
        model = cls(**kwargs, random_state=0)
        table = model.known_alphas_ if hasattr(model, "known_alphas_") else None
        model.decay(decay_rate=0.5)  # before any fit: nothing to tick, no error
        if table is not None:
            assert len(table) == 0


class TestPassThrough:
    def test_contextual_agent_passes_the_rule_and_steps_to_each_arm(self):
        arms = [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0)) for i in range(2)]
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=0)
        X = np.array([[1.0, 0.0]])
        for arm in arms:
            arm.update(X, np.array([1.0]))
        before = [_dense(_learner(arm).cov_inv_) for arm in arms]
        agent.decay(StabilizedForgetting(0.5), steps=2)
        for arm, b in zip(arms, before):
            assert_allclose(_dense(_learner(arm).cov_inv_), 0.25 * b + 0.75 * np.eye(2))

    def test_agent_decay_takes_no_context(self):
        agent = Agent(
            [Arm(i, learner=GammaRegressor(alpha=1.0, beta=1.0)) for i in range(2)],
            ThompsonSampling(),
            random_seed=0,
        )
        agent.select_for_update(0).update(np.array([3.0]))
        before = _learner(agent.arm(0)).coef_[1].copy()
        agent.decay(decay_rate=0.5, steps=2)
        assert_allclose(_learner(agent.arm(0)).coef_[1], 0.25 * before)

    def test_pipelines_pass_through_and_transform_a_legacy_array(self):
        arms = [Arm(0, learner=NormalRegressor(alpha=1.0, beta=1.0))]
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=0)
        pipeline = AgentPipeline([("id", FunctionTransformer())], agent)
        X = np.array([[1.0, 2.0]])
        arms[0].update(X, np.array([1.0]))
        before = _dense(_learner(arms[0]).cov_inv_)
        pipeline.decay(ExponentialForgetting(0.5))
        assert_allclose(_dense(_learner(arms[0]).cov_inv_), 0.5 * before)
        with pytest.warns(FutureWarning):
            pipeline.decay(np.vstack([X, X]), decay_rate=0.5)
        assert_allclose(_dense(_learner(arms[0]).cov_inv_), 0.125 * before)

        learner = LearnerPipeline(
            [("id", FunctionTransformer())], NormalRegressor(alpha=1.0, beta=1.0)
        )
        learner.partial_fit(X, np.array([1.0]))
        before = _dense(cast(Any, learner.learner).cov_inv_)
        learner.decay(StabilizedForgetting(0.5))
        assert_allclose(
            _dense(cast(Any, learner.learner).cov_inv_), 0.5 * before + 0.5 * np.eye(2)
        )
