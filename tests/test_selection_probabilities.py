"""Tests for ``select_with_probabilities``: the distribution a policy drew
its decision from, conditional on the samples it was handed."""

from typing import Any, List

import numpy as np
import pytest

from bayesianbandits import (
    EXP3A,
    Arm,
    EpsilonGreedy,
    InformationDirectedSampling,
    UpperConfidenceBound,
)

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
