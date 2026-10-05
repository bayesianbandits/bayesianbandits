"""Tests for rng property with setter for reseeding after deserialization."""

import pickle

import numpy as np
import pytest
from sklearn.preprocessing import FunctionTransformer

from bayesianbandits import (
    Agent,
    Arm,
    ContextualAgent,
    NormalRegressor,
    ThompsonSampling,
)
from bayesianbandits.api import LipschitzContextualAgent
from bayesianbandits.featurizers._arm_column import ArmColumnFeaturizer
from bayesianbandits.pipelines import LearnerPipeline
from bayesianbandits.pipelines._agent import (
    AgentPipeline,
)

X = np.ones((1, 2))


def _make_contextual_agent(seed=None):
    arms = [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0)) for i in range(3)]
    return ContextualAgent(arms, ThompsonSampling(), random_seed=seed)


def _make_agent(seed=None):
    arms = [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0)) for i in range(3)]
    return Agent(arms, ThompsonSampling(), random_seed=seed)


def _make_lipschitz_agent(seed=None):
    arms = [Arm(i, learner=None) for i in range(3)]
    learner = NormalRegressor(alpha=1.0, beta=1.0)
    featurizer = ArmColumnFeaturizer(column_name="arm_id")
    return LipschitzContextualAgent(
        arms, ThompsonSampling(), featurizer, learner, random_seed=seed
    )


class TestContextualAgentRngSetter:
    def test_rng_setter_propagates_to_arms(self):
        agent = _make_contextual_agent(seed=0)
        new_rng = np.random.default_rng(42)
        agent.rng = new_rng

        # All arm learners should share the same Generator object
        for arm in agent.arms:
            assert arm.learner.random_state is agent.rng

    def test_rng_setter_accepts_int(self):
        agent = _make_contextual_agent(seed=0)
        agent.rng = 42
        assert isinstance(agent.rng, np.random.Generator)

    def test_rng_setter_accepts_generator(self):
        agent = _make_contextual_agent(seed=0)
        gen = np.random.default_rng(42)
        agent.rng = gen
        assert isinstance(agent.rng, np.random.Generator)

    def test_rng_setter_accepts_none(self):
        agent = _make_contextual_agent(seed=0)
        agent.rng = None
        assert isinstance(agent.rng, np.random.Generator)

    def test_setstate_migrates_old_pickle(self):
        agent = _make_contextual_agent(seed=0)
        state = agent.__dict__.copy()
        # Simulate old pickle format: 'rng' instead of '_rng'
        state["rng"] = state.pop("_rng")
        assert "rng" in state and "_rng" not in state

        new_agent = ContextualAgent.__new__(ContextualAgent)
        new_agent.__setstate__(state)

        assert isinstance(new_agent.rng, np.random.Generator)
        assert "_rng" in new_agent.__dict__
        assert "rng" not in new_agent.__dict__


class TestAgentRngSetter:
    def test_agent_rng_setter_delegates(self):
        agent = _make_agent(seed=0)
        agent.rng = 42
        assert isinstance(agent.rng, np.random.Generator)
        # Verify it propagated to inner agent's arms
        for arm in agent.arms:
            assert arm.learner.random_state is agent.rng


class TestLipschitzContextualAgentRngSetter:
    def test_lipschitz_rng_propagates_to_shared_learner(self):
        agent = _make_lipschitz_agent(seed=0)
        agent.rng = 99
        assert isinstance(agent.rng, np.random.Generator)
        assert agent.learner.random_state is agent.rng

    def test_lipschitz_setstate_migrates_old_pickle(self):
        agent = _make_lipschitz_agent(seed=0)
        state = agent.__dict__.copy()
        state["rng"] = state.pop("_rng")

        new_agent = LipschitzContextualAgent.__new__(LipschitzContextualAgent)
        new_agent.__setstate__(state)

        assert isinstance(new_agent.rng, np.random.Generator)
        assert "_rng" in new_agent.__dict__


class TestPipelineRngSetter:
    def test_contextual_pipeline_rng_setter_delegates(self):
        agent = _make_contextual_agent(seed=0)
        pipeline = AgentPipeline(
            steps=[("noop", _NoopTransformer())], final_agent=agent
        )
        pipeline.rng = 42
        assert isinstance(pipeline.rng, np.random.Generator)
        assert pipeline.rng is agent.rng


class TestReseedProducesNewState:
    def test_reseed_changes_rng_state(self):
        agent = _make_agent(seed=0)

        # Record rng output before reseed
        vals_before = agent.rng.random(10)

        # Reseed and verify different output
        agent.rng = 999
        vals_after = agent.rng.random(10)
        assert not np.array_equal(vals_before, vals_after)

        # All arm learners should share the new rng
        for arm in agent.arms:
            assert arm.learner.random_state is agent.rng


def _trained():
    agent = _make_contextual_agent(seed=0)
    for arm in agent.arms:
        agent.select_for_update(arm.action_token).update(X, np.array([1.0]))
    return agent


def _sampled_only():
    agent = _make_contextual_agent(seed=0)
    agent.pull(X)
    return agent


def _nested_pipeline():
    def learner():
        inner = LearnerPipeline(
            [("a", FunctionTransformer())], NormalRegressor(alpha=1.0, beta=1.0)
        )
        return LearnerPipeline([("b", FunctionTransformer())], inner)

    arms = [Arm(i, learner=learner()) for i in range(3)]
    agent = ContextualAgent(arms, ThompsonSampling(), random_seed=0)
    agent.pull(X)
    return agent


def _added_trained_arm():
    agent = _make_contextual_agent(seed=0)
    learner = NormalRegressor(alpha=1.0, beta=1.0).fit(X, np.array([1.0]))
    agent.add_arm(Arm(99, learner=learner))
    return agent


def _lipschitz():
    agent = _make_lipschitz_agent(seed=0)
    agent.pull(X)
    return agent


def _draw(agent):
    if isinstance(agent, LipschitzContextualAgent):
        return agent.learner.sample(np.ones((1, 3)), size=5)
    return agent.arms[-1].sample(X, size=5)


@pytest.mark.parametrize(
    "make_agent",
    [_trained, _sampled_only, _nested_pipeline, _added_trained_arm, _lipschitz],
)
def test_reseed_after_load_changes_draws(make_agent):
    blob = pickle.dumps(make_agent())
    a, b, c = (pickle.loads(blob) for _ in range(3))
    a.rng, b.rng, c.rng = 111, 222, 111
    da, db, dc = _draw(a), _draw(b), _draw(c)

    assert not np.array_equal(da, db)
    assert np.array_equal(da, dc)


class _NoopTransformer:  # pragma: no cover
    def transform(self, X):
        return X
