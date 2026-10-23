"""Tests for ``select_with_probabilities``: the distribution a policy drew
its decision from, conditional on the samples it was handed."""

from typing import Any, Callable, Dict, List, Sequence, Sized

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.preprocessing import FunctionTransformer

from bayesianbandits import (
    EXP3A,
    Agent,
    AgentPipeline,
    Arm,
    ContextualAgent,
    EpsilonGreedy,
    InformationDirectedSampling,
    LipschitzContextualAgent,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits.featurizers import FunctionArmFeaturizer
from bayesianbandits.policies._base import PolicyDefaultUpdate

# Policies putting mass on every arm, which IPS of a fixed target needs
FULL_SUPPORT: List[Any] = [
    EXP3A(eta=1.0, gamma=0.1, samples=8),
    EpsilonGreedy(epsilon=0.3, samples=8),
]
POLICIES = FULL_SUPPORT + [
    UpperConfidenceBound(alpha=0.68, samples=8),
    InformationDirectedSampling(samples=8),
]


def _arms(n_arms: int) -> List[Arm[Any, int]]:
    return [Arm(i, learner=None) for i in range(n_arms)]


def _tokens(arms: List[Arm[Any, int]]) -> List[int]:
    return [arm.action_token for arm in arms]


@pytest.mark.parametrize("policy", POLICIES, ids=repr)
def test_agrees_with_select(policy: Any) -> None:
    samples = np.random.default_rng(0).normal(size=(4, 5, policy.samples_needed))
    arms = _arms(4)

    plain = policy.select(samples, arms, np.random.default_rng(1))
    chosen, _ = policy.select_with_probabilities(
        samples, arms, np.random.default_rng(1)
    )

    assert _tokens(chosen) == _tokens(plain)


@pytest.mark.parametrize("policy", POLICIES, ids=repr)
def test_calibrated(policy: Any) -> None:
    """Each arm's play count matches the sum of its reported probabilities."""
    rng = np.random.default_rng(3)
    n_rounds = 20_000
    means = np.array([1.0, 0.0, -1.0])
    samples = means[:, None, None] + rng.normal(
        size=(3, n_rounds, policy.samples_needed)
    )

    chosen, probs = policy.select_with_probabilities(samples, _arms(3), rng)
    played = np.array(_tokens(chosen))

    assert np.allclose(probs.sum(axis=1), 1.0)
    for arm in range(3):
        residual = np.sum(played == arm) - probs[:, arm].sum()
        scale = np.sqrt(np.sum(probs[:, arm] * (1 - probs[:, arm])))
        assert abs(residual) <= 4 * scale  # exact when the policy is deterministic


@pytest.mark.parametrize("policy", FULL_SUPPORT, ids=repr)
def test_ips_is_unbiased(policy: Any) -> None:
    """The target always plays the logger's least favourite arm, so the
    importance weights are large and a wrong propensity shows up as bias."""
    rng = np.random.default_rng(2)
    means = np.array([1.0, 0.0, -1.0])
    n_rounds = 20_000
    samples = means[:, None, None] + rng.normal(
        size=(3, n_rounds, policy.samples_needed)
    )
    arms = _arms(3)

    chosen, probs = policy.select_with_probabilities(samples, arms, rng)
    played = np.array(_tokens(chosen))
    rewards = means[played] + rng.normal(size=n_rounds)

    assert probs.shape == (n_rounds, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)

    target = 2
    terms = (played == target) * rewards / probs[np.arange(n_rounds), played]
    estimate = terms.mean()
    stderr = terms.std(ddof=1) / np.sqrt(n_rounds)

    assert abs(estimate - means[target]) < 4 * stderr


def _learned_arms() -> List[Arm[Any, int]]:
    return [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0)) for i in range(3)]


def _one_hot(X: Sized, action_tokens: Sequence[int]) -> NDArray[np.float64]:
    out = np.zeros((len(X), len(action_tokens), len(action_tokens)))
    out[:, np.arange(len(action_tokens)), np.arange(len(action_tokens))] = 1.0
    return out


AGENTS: Dict[str, Callable[[Any, int], Any]] = {
    "Agent": lambda policy, seed: Agent(_learned_arms(), policy, random_seed=seed),
    "ContextualAgent": lambda policy, seed: ContextualAgent(
        _learned_arms(), policy, random_seed=seed
    ),
    "AgentPipeline": lambda policy, seed: AgentPipeline(
        [("double", FunctionTransformer(lambda x: 2 * x))],
        ContextualAgent(_learned_arms(), policy, random_seed=seed),
    ),
    "LipschitzContextualAgent": lambda policy, seed: LipschitzContextualAgent(
        _arms(3),
        policy,
        arm_featurizer=FunctionArmFeaturizer(_one_hot),
        learner=NormalRegressor(alpha=1.0, beta=1.0),
        random_seed=seed,
    ),
}


@pytest.mark.parametrize("build", AGENTS.values(), ids=AGENTS.keys())
def test_agents_agree_with_pull(build: Callable[[Any, int], Any]) -> None:
    policy = EpsilonGreedy(epsilon=0.3, samples=8)
    X = () if build is AGENTS["Agent"] else (np.ones((5, 2)),)

    plain = build(policy, 1).pull(*X)
    agent = build(policy, 1)
    tokens, probs = agent.pull_with_probabilities(*X)

    assert tokens == plain
    assert probs.shape == (len(tokens), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs[np.arange(len(tokens)), tokens] > 0).all()
    assert agent.arm_to_update.action_token == tokens[-1]


def test_thompson_sampling_raises() -> None:
    agent = Agent(_learned_arms(), ThompsonSampling(), random_seed=0)
    with pytest.raises(NotImplementedError, match="largest"):
        agent.pull_with_probabilities()


def test_default_raises() -> None:
    with pytest.raises(NotImplementedError, match="PolicyDefaultUpdate does not"):
        PolicyDefaultUpdate[Any, int]().select_with_probabilities(
            np.zeros((2, 1, 1)), _arms(2), np.random.default_rng(0)
        )
