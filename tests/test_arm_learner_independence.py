"""Agents built on per-arm learners require those learners to be distinct.

``ContextualAgent`` and ``Agent`` model arms as independent: they give
arms no way to differ in features, so a shared learner would leave the
arms statistically indistinguishable, and the policies -- which sample
one arm at a time -- would draw a separate weight vector per arm and
silently discard the dependence a shared posterior implies.

Sharing one model across arms is ``LipschitzContextualAgent``'s job; it
distinguishes arms with an ``arm_featurizer`` and samples the shared
learner once for all of them.
"""

import numpy as np
import pytest

from bayesianbandits import (
    EXP3A,
    Agent,
    Arm,
    ContextualAgent,
    EpsilonGreedy,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits.pipelines import LearnerPipeline

POLICIES = [
    ("thompson", ThompsonSampling()),
    ("ucb", UpperConfidenceBound(alpha=0.68, samples=17)),
    ("epsilon_greedy", EpsilonGreedy(epsilon=0.1, samples=17)),
    ("exp3a", EXP3A(samples=17)),
]


def _learner(seed=0):
    est = NormalRegressor(alpha=1.0, beta=1.0, random_state=seed)
    rng = np.random.default_rng(seed)
    est.fit(rng.standard_normal((30, 2)), rng.standard_normal(30))
    return est


def _independent_arms(n=3):
    return [Arm(i, learner=_learner(i)) for i in range(n)]


# -- the invariant ------------------------------------------------------------


def test_contextual_agent_rejects_a_shared_learner():
    shared = _learner()
    with pytest.raises(ValueError, match="own learner"):
        ContextualAgent(
            [Arm(0, learner=shared), Arm(1, learner=shared)], ThompsonSampling()
        )


def test_agent_rejects_a_shared_learner():
    shared = _learner()
    with pytest.raises(ValueError, match="own learner"):
        Agent([Arm(0, learner=shared), Arm(1, learner=shared)], ThompsonSampling())


def test_add_arm_rejects_a_learner_already_in_use():
    """The invariant holds for arms added after construction, too."""
    arms = _independent_arms(2)
    agent = ContextualAgent(arms, ThompsonSampling())
    with pytest.raises(ValueError, match="own learner"):
        agent.add_arm(Arm(99, learner=arms[0].learner))


def test_distinct_pipelines_wrapping_one_estimator_are_rejected():
    """Two pipelines can differ as objects and still share a posterior."""
    shared = _learner()
    arms = [
        Arm(0, learner=LearnerPipeline(steps=[], learner=shared)),
        Arm(1, learner=LearnerPipeline(steps=[], learner=shared)),
    ]
    with pytest.raises(ValueError, match="own learner"):
        ContextualAgent(arms, ThompsonSampling())


def test_error_points_at_the_agent_that_supports_sharing():
    shared = _learner()
    with pytest.raises(ValueError, match="LipschitzContextualAgent"):
        ContextualAgent(
            [Arm(0, learner=shared), Arm(1, learner=shared)], ThompsonSampling()
        )


def test_independent_learners_are_accepted():
    agent = ContextualAgent(_independent_arms(), ThompsonSampling())
    assert len(agent.arms) == 3


def test_reusing_a_learner_after_removing_its_arm_is_allowed():
    """The check is about *concurrent* sharing, not history."""
    arms = _independent_arms(2)
    agent = ContextualAgent(arms, ThompsonSampling())
    agent.remove_arm(0)
    agent.add_arm(Arm(99, learner=arms[0].learner))
    assert {a.action_token for a in agent.arms} == {1, 99}


# -- drawing over independent arms --------------------------------------------


@pytest.mark.parametrize("size", [1, 4])
def test_draw_samples_is_always_3d(size):
    """``select`` relies on a ``(n_arms, n_contexts, size)`` array."""
    arms = _independent_arms()
    X = np.random.default_rng(2).standard_normal((5, 2))
    drawn = ThompsonSampling()._draw_samples(arms, X, size)
    assert drawn.shape == (3, 5, size)


@pytest.mark.parametrize("name,policy", POLICIES, ids=[p[0] for p in POLICIES])
def test_every_policy_pulls_over_independent_arms(name, policy):
    agent = ContextualAgent(_independent_arms(), policy, random_seed=0)
    X = np.random.default_rng(3).standard_normal((5, 2))
    chosen = agent.pull(X)
    assert len(chosen) == 5
    assert set(chosen) <= {0, 1, 2}


def test_independent_arms_draw_independently():
    """With independent learners, per-arm draws *are* the joint law, so
    arms are uncorrelated -- which is correct here, unlike the shared
    case this agent now refuses."""
    arms = _independent_arms(2)
    X = np.ones((1, 2))  # nonzero, so the predictive variance is nonzero
    drawn = ThompsonSampling()._draw_samples(arms, X, 4000)
    corr = np.corrcoef(drawn[0, 0], drawn[1, 0])[0, 1]
    assert abs(corr) < 0.1
